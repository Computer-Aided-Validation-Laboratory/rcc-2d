#!/usr/bin/env python3
"""Apply the grid method to every completed Exp3 eggbox sequence.

The driver discovers bespoke and Riley output naming conventions and processes
rigid, affine and finite-star/chirp cases.  Each sequence receives full-field
plots below ``out/exp3_gridmethod/<case>/<render-root>/<config>/``.  Rigid
cases additionally receive a prescribed-motion and unwrap cross-check report.

Use ``EXP3_GRIDMETHOD_CASE=<case>`` or ``EXP3_GRIDMETHOD_LIMIT=<n>`` to run a
focused subset while developing.
"""
from __future__ import annotations

import csv
import gc
import os
import re
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp0params_common import GRIDMETHOD_CASES
from exp3params import EGGBOX_PERIOD_FINAL_PX, GRIDMETHOD_WINDOW, GRIDMETHOD_WINDOW_WIDTH_PERIODS, MEASUREMENT_BIT_DEPTHS
from modules.gridmethod import GridMethodConfig, analyse_sequence
from modules.render_selection import analysis_enabled, measurement_enabled
from modules.output_naming import data_case_name, is_rigid_case
from modules.analysis_parallel import run_analysis_jobs
from modules.render_outputs import quantise_camera


OUT_ROOT = Path("out")
RESULT_ROOT = OUT_ROOT / "exp3_gridmethod"
CASE_FILTER = os.environ.get("EXP3_GRIDMETHOD_CASE")
LIMIT = int(os.environ.get("EXP3_GRIDMETHOD_LIMIT", "0"))
FIELD_PLOT_LAYOUT_VERSION = "stacked-finite-star-v1"
GRIDMETHOD_RESULT_VERSION = f"window-{GRIDMETHOD_WINDOW}-{GRIDMETHOD_WINDOW_WIDTH_PERIODS:g}-v3-bit-depth"


def image_frames(directory: Path) -> list[Path]:
    files = list(directory.glob("frame*.npy"))
    if not files:
        files = list(directory.glob("image_c00_f*.npy"))
    def number(path: Path) -> int:
        match = re.search(r"(?:frame|_f)(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Cannot determine frame number from {path}")
        return int(match.group(1))
    return sorted(files, key=number)


def measurement_bit_depths() -> list[int]:
    """Allow a focused depth run without changing Exp3's parameter file."""
    value = os.environ.get("EXP3_MEASUREMENT_BIT_DEPTHS")
    return list(MEASUREMENT_BIT_DEPTHS) if not value else [int(item) for item in value.split(",") if item.strip()]


def read_quantised_images(files: list[Path], bit_depth: int) -> np.ndarray:
    """Digitise canonical [0, 1] camera floats for one measurement depth."""
    images = np.stack([np.load(path, mmap_mode="r") for path in files]).astype(np.float64, copy=False)
    # Keep compatibility with historical Riley NPY outputs expressed in 8-bit
    # codes.  Current canonical outputs are already normalised floats.
    if images.size and float(np.nanmax(np.abs(images))) > 1.0 + 1e-8:
        images /= 255.0
    return quantise_camera(images, bit_depth).astype(np.float64, copy=False) / float((1 << bit_depth) - 1)


def sequences() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for directory in OUT_ROOT.glob("exp3_*render*/*/*"):
        if not directory.is_dir() or not directory.name.startswith("eggb_"):
            continue
        case, root = directory.parent.name, directory.parent.parent.name
        if not analysis_enabled(root, "eggb"):
            continue
        if not measurement_enabled(root, GRIDMETHOD_CASES):
            continue
        if CASE_FILTER and case != CASE_FILTER:
            continue
        if len(image_frames(directory)) >= 2:
            found.append((case, root, directory))
    return sorted(found, key=lambda item: tuple(str(x) for x in item))


def result_dir(case: str, root: str, directory: Path, bit_depth: int) -> Path:
    """Return this render sequence's dedicated Grid Method result directory."""
    return RESULT_ROOT / case / root / directory.name / f"b{bit_depth:02d}"


def sequence_complete(case: str, root: str, directory: Path, frame_count: int, bit_depth: int) -> bool:
    """Require all fields and rigid diagnostics before skipping a sequence."""
    out_dir = result_dir(case, root, directory, bit_depth)
    fields_complete = all(
        (out_dir / f"displacement_frame{frame:02d}.png").is_file()
        and (out_dir / f"displacement_frame{frame:02d}.npz").is_file()
        for frame in range(frame_count)
    )
    if not fields_complete:
        return False
    config_marker = out_dir / ".gridmethod_result_version"
    if not config_marker.is_file() or config_marker.read_text().strip() != GRIDMETHOD_RESULT_VERSION:
        return False
    return (
        not is_rigid_case(case)
        or ((out_dir / "rigid_motion_summary.csv").is_file()
            and (out_dir / "rigid_motion_summary.png").is_file())
    )


def field_plot_marker(out_dir: Path) -> Path:
    return out_dir / ".field_plot_layout"


def refresh_field_figures(case: str, root: str, directory: Path, frame_count: int, bit_depth: int) -> bool:
    """Regenerate field PNGs from stored results when only styling changed."""
    out_dir = result_dir(case, root, directory, bit_depth)
    marker = field_plot_marker(out_dir)
    if marker.is_file() and marker.read_text().strip() == FIELD_PLOT_LAYOUT_VERSION:
        return False
    paths = [out_dir / f"displacement_frame{frame:02d}.npz" for frame in range(frame_count)]
    if not all(path.is_file() for path in paths):
        return False
    for frame, path in enumerate(paths):
        with np.load(path) as data:
            save_field(
                out_dir / f"displacement_frame{frame:02d}.png",
                np.asarray(data["ux"]), np.asarray(data["uy"]), frame,
                EGGBOX_PERIOD_FINAL_PX, f"{directory.name}, {bit_depth}-bit",
            )
    marker.write_text(f"{FIELD_PLOT_LAYOUT_VERSION}\n")
    return True


def clear_generated_artifacts(case: str, root: str, directory: Path, bit_depth: int) -> None:
    """Clear only Grid Method artifacts owned by an incomplete sequence."""
    out_dir = result_dir(case, root, directory, bit_depth)
    for pattern in (
        "displacement_frame*.png", "displacement_frame*.npz",
        "rigid_motion_summary.csv", "rigid_motion_summary.png",
        ".gridmethod_result_version",
    ):
        for path in out_dir.glob(pattern):
            path.unlink()


def expected_motion(case: str, frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not is_rigid_case(case):
        return None
    root = Path("data") / data_case_name(case)
    x = np.loadtxt(root / "field_disp_x.csv", delimiter=",")
    y = np.loadtxt(root / "field_disp_y.csv", delimiter=",")
    if x.ndim == 1:
        x, y = x[None, :], y[None, :]
    return x[:, :frames].mean(axis=0), y[:, :frames].mean(axis=0)


def crop(field: np.ndarray, config: GridMethodConfig) -> np.ndarray:
    halo = int(np.ceil(4.0 * config.period_px * config.window_width_periods + 2.0 * config.period_px))
    return field[halo:-halo, halo:-halo]


def field_title(label: str, frame: int, component: str, width: int = 42) -> str:
    """Wrap long renderer/configuration labels before they exceed a panel."""
    if len(label) <= width:
        return f"{label}, frame {frame:02d}: {component}"
    readable = label.replace("_", " ")
    return "\n".join(textwrap.wrap(readable, width=width, break_long_words=False) + [f"frame {frame:02d}: {component}"])


def save_field(path: Path, ux: np.ndarray, uy: np.ndarray, frame: int, pitch: float, label: str) -> None:
    # The finite-star image is about 4:1.  Stack its components, matching the
    # DIC presentation, rather than squeezing two wide maps side by side.
    stack_fields = ux.shape[1] / ux.shape[0] > 2.0
    fig, axes = (
        plt.subplots(2, 1, figsize=(10, 7.0), constrained_layout=True)
        if stack_fields
        else plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    )
    for axis, field, name in zip(np.ravel(axes), (ux, uy), ("$u_x$", "$u_y$")):
        center = float(np.nanmean(field)) if np.any(np.isfinite(field)) else 0.0
        if np.any(np.isfinite(field)):
            max_dev = float(np.nanmax(np.abs(field - center)))
            r = max(max_dev, 0.05)
        else:
            r = 0.05
        image = axis.imshow(
            field, cmap="coolwarm", origin="upper",
            vmin=center - r, vmax=center + r
        )
        rows, cols = field.shape
        axis.add_patch(Rectangle((pitch, pitch), cols - 1 - 2*pitch, rows - 1 - 2*pitch, fill=False, edgecolor="black", linewidth=1.2))
        axis.set_title(field_title(label, frame, name), fontsize=9)
        axis.set_xlabel("column [px]")
        axis.set_ylabel("row [px]")
        fig.colorbar(image, ax=axis, label="displacement [px]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rigid_summary(path: Path, rows: list[dict[str, float | int]], label: str) -> None:
    frames = np.asarray([row["frame"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, constrained_layout=True)
    for axis, component in zip(axes, ("ux", "uy")):
        axis.plot(frames, [row[f"expected_{component}_px"] for row in rows], "k--", label="prescribed")
        axis.plot(frames, [row[f"measured_{component}_px"] for row in rows], "o-", label="grid method")
        axis.set_title(component)
        axis.set_xlabel("frame")
        axis.set_ylabel("displacement [px]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(label)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def analyse(case: str, root: str, directory: Path, bit_depth: int) -> None:
    files = image_frames(directory)
    images = read_quantised_images(files, bit_depth)
    config = GridMethodConfig(
        period_px=EGGBOX_PERIOD_FINAL_PX,
        window_width_periods=GRIDMETHOD_WINDOW_WIDTH_PERIODS,
        window=GRIDMETHOD_WINDOW, displacement_method="iterative", unwrap="reliability",
    )
    result = analyse_sequence(images, config)
    out_dir = result_dir(case, root, directory, bit_depth)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_motion(case, len(files))
    rows: list[dict[str, float | int]] = []
    crosscheck = analyse_sequence(images, replace(config, unwrap="skimage")) if expected is not None else None
    for frame in range(len(files)):
        ux, uy = result.displacement_x[frame], -result.displacement_y[frame]
        save_field(out_dir / f"displacement_frame{frame:02d}.png", ux, uy, frame, config.period_px, f"{directory.name}, {bit_depth}-bit")
        np.savez_compressed(out_dir / f"displacement_frame{frame:02d}.npz", ux=ux, uy=uy)
        if expected is not None:
            ux_roi, uy_roi = crop(ux, config), crop(uy, config)
            sx, sy = crop(crosscheck.displacement_x[frame], config), crop(-crosscheck.displacement_y[frame], config)
            rows.append({"frame": frame, "expected_ux_px": float(expected[0][frame]), "expected_uy_px": float(expected[1][frame]), "measured_ux_px": float(np.nanmean(ux_roi)), "measured_uy_px": float(np.nanmean(uy_roi)), "error_ux_px": float(np.nanmean(ux_roi)-expected[0][frame]), "error_uy_px": float(np.nanmean(uy_roi)-expected[1][frame]), "unwrap_max_delta_px": float(max(np.nanmax(abs(ux_roi-sx)), np.nanmax(abs(uy_roi-sy))) )})
    if rows:
        with (out_dir / "rigid_motion_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        save_rigid_summary(out_dir / "rigid_motion_summary.png", rows, f"{directory.name}, {bit_depth}-bit")
    field_plot_marker(out_dir).write_text(f"{FIELD_PLOT_LAYOUT_VERSION}\n")
    (out_dir / ".gridmethod_result_version").write_text(f"{GRIDMETHOD_RESULT_VERSION}\n")
    del images, result, crosscheck
    gc.collect()


def _analyse_job(item: tuple[str, str, Path, int]) -> tuple[str, str, str, int]:
    case, root, directory, bit_depth = item
    analyse(case, root, directory, bit_depth)
    return case, root, directory.name, bit_depth


def main() -> None:
    jobs = sequences()
    if LIMIT:
        jobs = jobs[:LIMIT]
    if not jobs:
        raise FileNotFoundError("No completed Exp3 eggbox sequences matched the requested filter.")
    bit_depths = measurement_bit_depths()
    print(f"Exp3 grid method: {len(jobs)} sequences; bits={bit_depths}; families={','.join(GRIDMETHOD_CASES)}")
    pending: list[tuple[str, str, Path, int]] = []
    for index, (case, root, directory) in enumerate(jobs, start=1):
        frame_count = len(image_frames(directory))
        for bit_depth in bit_depths:
            print(f"[{index}/{len(jobs)}] {root}/{case}/{directory.name}/b{bit_depth:02d}")
            if refresh_field_figures(case, root, directory, frame_count, bit_depth):
                print("  refreshed displacement figures from stored fields")
            if sequence_complete(case, root, directory, frame_count, bit_depth):
                print("  complete: Grid Method fields and diagnostics already exist; skipping")
                continue
            print("  incomplete: clearing generated Grid Method artifacts and reprocessing all frames")
            clear_generated_artifacts(case, root, directory, bit_depth)
            pending.append((case, root, directory, bit_depth))
    if not pending:
        return
    # Sequences share no output files, so each process owns a full
    # load/analyse/write/release lifecycle without synchronisation.
    for index, (case, root, name, bit_depth) in enumerate(
        run_analysis_jobs("Exp3 Grid Method", pending, _analyse_job), start=1
    ):
        print(f"  completed [{index}/{len(pending)}] {root}/{case}/{name}/b{bit_depth:02d}", flush=True)


if __name__ == "__main__":
    main()
