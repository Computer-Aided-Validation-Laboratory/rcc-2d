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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from exp0params_common import GRIDMETHOD_JOBS
from exp3params import EGGBOX_PERIOD_FINAL_PX
from modules.gridmethod import GridMethodConfig, analyse_sequence


OUT_ROOT = Path("out")
RESULT_ROOT = OUT_ROOT / "exp3_gridmethod"
CASE_FILTER = os.environ.get("EXP3_GRIDMETHOD_CASE")
LIMIT = int(os.environ.get("EXP3_GRIDMETHOD_LIMIT", "0"))
WORKERS = max(1, int(os.environ.get("EXP3_GRIDMETHOD_JOBS", str(GRIDMETHOD_JOBS))))


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


def sequences() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for directory in OUT_ROOT.glob("exp3_*render*/*/*"):
        if not directory.is_dir() or not directory.name.startswith("eggbox_"):
            continue
        case, root = directory.parent.name, directory.parent.parent.name
        if CASE_FILTER and case != CASE_FILTER:
            continue
        if len(image_frames(directory)) >= 2:
            found.append((case, root, directory))
    return sorted(found, key=lambda item: tuple(str(x) for x in item))


def result_dir(case: str, root: str, directory: Path) -> Path:
    """Return this render sequence's dedicated Grid Method result directory."""
    return RESULT_ROOT / case / root / directory.name


def sequence_complete(case: str, root: str, directory: Path, frame_count: int) -> bool:
    """Require all fields and rigid diagnostics before skipping a sequence."""
    out_dir = result_dir(case, root, directory)
    fields_complete = all(
        (out_dir / f"displacement_frame{frame:02d}.png").is_file()
        and (out_dir / f"displacement_frame{frame:02d}.npz").is_file()
        for frame in range(frame_count)
    )
    if not fields_complete:
        return False
    return (
        "rigid" not in case
        or ((out_dir / "rigid_motion_summary.csv").is_file()
            and (out_dir / "rigid_motion_summary.png").is_file())
    )


def clear_generated_artifacts(case: str, root: str, directory: Path) -> None:
    """Clear only Grid Method artifacts owned by an incomplete sequence."""
    out_dir = result_dir(case, root, directory)
    for pattern in (
        "displacement_frame*.png", "displacement_frame*.npz",
        "rigid_motion_summary.csv", "rigid_motion_summary.png",
    ):
        for path in out_dir.glob(pattern):
            path.unlink()


def expected_motion(case: str, frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    if "rigid" not in case:
        return None
    root = Path("data") / case
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, field, name in zip(axes, (ux, uy), ("$u_x$", "$u_y$")):
        # Natural per-field limits expose variation about the nominal motion.
        image = axis.imshow(field, cmap="coolwarm", origin="upper")
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


def analyse(case: str, root: str, directory: Path) -> None:
    files = image_frames(directory)
    images = np.stack([np.load(path, mmap_mode="r") for path in files]).astype(np.float64, copy=False)
    config = GridMethodConfig(period_px=EGGBOX_PERIOD_FINAL_PX, window_width_periods=2.0, window="gaussian", displacement_method="iterative", unwrap="reliability")
    result = analyse_sequence(images, config)
    out_dir = result_dir(case, root, directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_motion(case, len(files))
    rows: list[dict[str, float | int]] = []
    crosscheck = analyse_sequence(images, replace(config, unwrap="skimage")) if expected is not None else None
    for frame in range(len(files)):
        ux, uy = result.displacement_x[frame], -result.displacement_y[frame]
        save_field(out_dir / f"displacement_frame{frame:02d}.png", ux, uy, frame, config.period_px, directory.name)
        np.savez_compressed(out_dir / f"displacement_frame{frame:02d}.npz", ux=ux, uy=uy)
        if expected is not None:
            ux_roi, uy_roi = crop(ux, config), crop(uy, config)
            sx, sy = crop(crosscheck.displacement_x[frame], config), crop(-crosscheck.displacement_y[frame], config)
            rows.append({"frame": frame, "expected_ux_px": float(expected[0][frame]), "expected_uy_px": float(expected[1][frame]), "measured_ux_px": float(np.nanmean(ux_roi)), "measured_uy_px": float(np.nanmean(uy_roi)), "error_ux_px": float(np.nanmean(ux_roi)-expected[0][frame]), "error_uy_px": float(np.nanmean(uy_roi)-expected[1][frame]), "unwrap_max_delta_px": float(max(np.nanmax(abs(ux_roi-sx)), np.nanmax(abs(uy_roi-sy))) )})
    if rows:
        with (out_dir / "rigid_motion_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        save_rigid_summary(out_dir / "rigid_motion_summary.png", rows, directory.name)
    del images, result, crosscheck
    gc.collect()


def main() -> None:
    jobs = sequences()
    if LIMIT:
        jobs = jobs[:LIMIT]
    if not jobs:
        raise FileNotFoundError("No completed Exp3 eggbox sequences matched the requested filter.")
    print(f"Exp3 grid method: {len(jobs)} sequences; workers={WORKERS}")
    pending: list[tuple[str, str, Path]] = []
    for index, (case, root, directory) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {root}/{case}/{directory.name}")
        frame_count = len(image_frames(directory))
        if sequence_complete(case, root, directory, frame_count):
            print("  complete: Grid Method fields and diagnostics already exist; skipping")
            continue
        print("  incomplete: clearing generated Grid Method artifacts and reprocessing all frames")
        clear_generated_artifacts(case, root, directory)
        pending.append((case, root, directory))
    if not pending:
        return
    if WORKERS == 1 or len(pending) == 1:
        for case, root, directory in pending:
            analyse(case, root, directory)
        return
    # Sequences share no output files, so each process can safely own a full
    # load/analyse/write/release lifecycle without synchronisation.
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(pending))) as executor:
        futures = {
            executor.submit(analyse, case, root, directory): (case, root, directory)
            for case, root, directory in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case, root, directory = futures[future]
            future.result()
            print(f"  completed [{index}/{len(pending)}] {root}/{case}/{directory.name}", flush=True)


if __name__ == "__main__":
    main()
