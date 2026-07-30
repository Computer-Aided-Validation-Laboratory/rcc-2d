#!/usr/bin/env python3
"""Run resumable two-stage PyVale 2D DIC for Exp3 additive-speckle sequences.

Completed render directories are discovered, rather than encoded here, so a
rerun automatically includes custom/Riley, analytic/SSAA/PSF, float/uint
renders and all three Exp3 deformation cases.  Results are written below
``out/exp3_dic/<case>/<render-root>/<pattern-config>/``.

Use ``EXP3_DIC_CASE=<case>``, ``EXP3_DIC_MATCH=<text>``, or
``EXP3_DIC_LIMIT=<n>`` for a focused run.

Stage 1 calculates only missing per-frame DIC CSVs.  Stage 2 imports only
CSVs lacking displacement figures in independent spawned processes, then
writes any missing rigid-bias summaries.  Thus restarting never repeats a
completed DIC correlation, field figure, or summary.
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pyvale.dic as dic
from exp3params import (
    DIC_CORRELATION_THRESHOLD,
    DIC_POSTPROCESS_JOBS,
    DIC_SHAPE_FUNCTION,
    DIC_SUBSET_SIZE_PX,
    DIC_SUBSET_STEP_PX,
)


OUT_ROOT = Path("out")
RESULT_ROOT = OUT_ROOT / "exp3_dic"
CASE_FILTER = os.environ.get("EXP3_DIC_CASE")
MATCH_FILTER = os.environ.get("EXP3_DIC_MATCH")
LIMIT = int(os.environ.get("EXP3_DIC_LIMIT", "0"))
# DIC is CPU-bound on this workstation; use physical cores only rather than
# adding SMT contention.
DIC_NUM_THREADS = 8


def render_sequences() -> list[tuple[str, str, Path]]:
    """Return every complete additive disk/Gaussian image sequence."""
    found: list[tuple[str, str, Path]] = []
    for config_dir in OUT_ROOT.glob("exp3_*render*/*/*"):
        if not config_dir.is_dir():
            continue
        case_dir = config_dir.parent
        render_root = case_dir.parent
        case, config = case_dir.name, config_dir.name
        if CASE_FILTER and case != CASE_FILTER:
            continue
        if not (config.startswith("diskaddsat_") or config.startswith("gausscont_")):
            continue
        if MATCH_FILTER and MATCH_FILTER not in str(config_dir):
            continue
        frames = image_frames(config_dir)
        if len(frames) < 2:
            continue
        found.append((case, render_root.name, config_dir))
    return sorted(found, key=lambda item: tuple(str(x) for x in item))


def image_frames(directory: Path) -> list[Path]:
    """Read both bespoke ``frameNN`` and Riley ``image_c00_fNN`` naming."""
    files = list(directory.glob("frame*.npy"))
    if not files:
        files = list(directory.glob("image_c00_f*.npy"))
    def frame_number(path: Path) -> int:
        match = re.search(r"(?:frame|_f)(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Cannot determine frame number from {path}")
        return int(match.group(1))
    return sorted(files, key=frame_number)


def read_uint8(path: Path) -> np.ndarray:
    """DIC receives the actual clamped 8-bit camera image, not a float view."""
    image = np.load(path, mmap_mode="r")
    # Bespoke float renders are [0, 1], whereas Riley's corresponding NPY
    # output is already in camera-code units.  Preserve both conventions.
    if float(np.nanmax(image)) > 1.0 + 1e-8:
        return np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def field_title(label: str, frame: int, component: str, width: int = 42) -> str:
    """Keep renderer/configuration identifiers readable within a panel."""
    if len(label) <= width:
        return f"{label}, frame {frame:02d}: {component}"
    readable = label.replace("_", " ")
    return "\n".join(textwrap.wrap(readable, width=width, break_long_words=False) + [f"frame {frame:02d}: {component}"])


def physical_expected_rigid(case: str, frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    if "rigid" not in case:
        return None
    data_dir = Path("data") / case
    ux = np.loadtxt(data_dir / "field_disp_x.csv", delimiter=",")
    uy = np.loadtxt(data_dir / "field_disp_y.csv", delimiter=",")
    if ux.ndim == 1:
        ux, uy = ux[None, :], uy[None, :]
    return ux[:, :frames].mean(axis=0), uy[:, :frames].mean(axis=0)


def save_field(path: Path, result: object, frame: int, label: str) -> None:
    # PyVale reports image-space V (positive down); convert to Exp3 physical Y.
    ux = result.u_px[frame]
    uy = -result.v_px[frame]
    x, y = result.ss_x, result.ss_y
    # The finite-star camera is approximately 4:1.  Stacking its two fields
    # preserves useful plot height; square-camera cases remain compact beside
    # one another.
    stack_fields = ux.shape[1] / ux.shape[0] > 2.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), constrained_layout=True) if stack_fields else plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    axes = np.ravel(axes)
    for ax, field, name in zip(axes, (ux, uy), ("$u_x$", "$u_y$")):
        # The subset grid is regular; imshow avoids building a 250k-cell
        # QuadMesh for every frame and is substantially faster in large runs.
        im = ax.imshow(field, extent=(x.min(), x.max(), y.max(), y.min()), cmap="coolwarm", aspect="equal")
        ax.set_aspect("equal")
        ax.set_xlabel("column [px]")
        ax.set_ylabel("row [px]")
        ax.set_title(field_title(label, frame + 1, name), fontsize=9)
        fig.colorbar(im, ax=ax, label="displacement [px]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rigid_bias(path: Path, rows: list[dict[str, float | int]], title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True, constrained_layout=True)
    for ax, component, label in zip(axes, ("ux", "uy"), ("$u_x$", "$u_y$")):
        expected = np.asarray([row[f"expected_{component}_px"] for row in rows])
        bias = np.asarray([row[f"bias_{component}_px"] for row in rows])
        ax.plot(expected, bias, "o-", label="mean DIC bias")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("prescribed rigid shift [px]")
        ax.set_ylabel(f"{label} bias [px]")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_pair(config_dir: Path, output_dir: Path, index: int) -> None:
    """Run one native-engine call; intended to execute in a fresh process."""
    frames = image_frames(config_dir)
    reference = read_uint8(frames[0])
    prefix = f"dic_frame{index:02d}_"
    dic.calculate_2d(
        reference=reference,
        deformed=read_uint8(frames[index]),
        roi_mask=np.ones(reference.shape, dtype=np.uint8),
        seed=[reference.shape[1] // 2, reference.shape[0] // 2],
        subset_size=DIC_SUBSET_SIZE_PX,
        subset_step=DIC_SUBSET_STEP_PX,
        shape_function=DIC_SHAPE_FUNCTION,
        correlation_criteria="ZNSSD",
        threshold=DIC_CORRELATION_THRESHOLD,
        max_displacement=8,
        method="MULTIWINDOW_RG",
        num_threads=DIC_NUM_THREADS,
        output_basepath=output_dir,
        output_prefix=prefix,
        output_delimiter=",",
        output_below_threshold=True,
        debug_level=0,
    )


def frame_csv(output_dir: Path, index: int) -> list[Path]:
    """Return this script's result CSV for one deformed frame."""
    return sorted(output_dir.glob(f"dic_frame{index:02d}_*.csv"))


def output_dir_for(case: str, render_root: str, config_dir: Path) -> Path:
    return RESULT_ROOT / case / render_root / config_dir.name


def run_dic_stage(case: str, render_root: str, config_dir: Path) -> Path:
    """Stage 1: calculate only missing DIC CSVs, serially and resumably."""
    frames = image_frames(config_dir)
    output_dir = output_dir_for(case, render_root, config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, len(frames)):
        csv_files = frame_csv(output_dir, index)
        if len(csv_files) == 1:
            print(f"  DIC frame {index:02d}: CSV exists; skipping")
            continue
        if len(csv_files) > 1:
            raise RuntimeError(
                f"Expected at most one DIC CSV for frame {index:02d} in {output_dir}; found {len(csv_files)}."
            )
        # PyVale's native engine retains process-global buffers across calls in
        # the installed branch.  A child per frame gives deterministic memory
        # release and makes an interrupted batch naturally resumable.
        print(f"  DIC frame {index:02d}: calculating", flush=True)
        worker_env = os.environ | {
            "EXP3_DIC_WORKER": "1",
            "EXP3_DIC_WORKER_DIR": str(config_dir.resolve()),
            "EXP3_DIC_WORKER_OUT": str(output_dir.resolve()),
            "EXP3_DIC_WORKER_FRAME": str(index),
        }
        subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True, env=worker_env)
    return output_dir


def postprocess_frame(csv_path: str, image_path: str, label: str, index: int) -> tuple[str, int]:
    """Stage-2 worker: import one CSV and write only its missing field figure."""
    result = dic.import_2d(Path(csv_path), delimiter=",", layout="matrix")
    save_field(Path(image_path), result, 0, label)
    del result
    return image_path, index


def rigid_rows(case: str, output_dir: Path, frame_count: int) -> list[dict[str, float | int]]:
    """Read one completed rigid sequence for its compact bias summary only."""
    expected = physical_expected_rigid(case, frame_count)
    if expected is None:
        return []
    rows: list[dict[str, float | int]] = []
    for index in range(1, frame_count):
        csv_files = frame_csv(output_dir, index)
        if len(csv_files) != 1:
            raise RuntimeError(f"Expected exactly one DIC CSV for frame {index:02d} in {output_dir}")
        result = dic.import_2d(csv_files[0], delimiter=",", layout="matrix")
        mean_x = float(np.nanmean(result.u_px[0]))
        mean_y = float(-np.nanmean(result.v_px[0]))
        rows.append({
            "frame": index,
            "expected_ux_px": float(expected[0][index]),
            "expected_uy_px": float(expected[1][index]),
            "mean_ux_px": mean_x,
            "mean_uy_px": mean_y,
            "bias_ux_px": mean_x - float(expected[0][index]),
            "bias_uy_px": mean_y - float(expected[1][index]),
        })
        del result
    return rows


def run_postprocess_stage(sequences: list[tuple[str, str, Path]]) -> None:
    """Stage 2: parallel CSV import/field figures, then compact summaries."""
    tasks: list[tuple[str, str, str, int]] = []
    summaries: list[tuple[str, Path, int, str]] = []
    for case, render_root, config_dir in sequences:
        frames = image_frames(config_dir)
        output_dir = output_dir_for(case, render_root, config_dir)
        for index in range(1, len(frames)):
            csv_files = frame_csv(output_dir, index)
            if len(csv_files) != 1:
                raise RuntimeError(f"Stage 1 did not produce one CSV for frame {index:02d} in {output_dir}")
            image_path = output_dir / f"displacement_frame{index:02d}.png"
            if not image_path.is_file():
                tasks.append((str(csv_files[0]), str(image_path), config_dir.name, index))
        if "rigid" in case:
            summaries.append((case, output_dir, len(frames), config_dir.name))

    if tasks:
        workers = min(max(1, DIC_POSTPROCESS_JOBS), len(tasks))
        print(f"Stage 2: generating {len(tasks)} displacement maps with {workers} processes")
        # Spawn avoids inheriting PyVale's native state from the DIC controller.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(postprocess_frame, *task) for task in tasks]
            for future in as_completed(futures):
                image_path, index = future.result()
                print(f"  postprocess frame {index:02d}: wrote {Path(image_path).name}", flush=True)
    else:
        print("Stage 2: all displacement maps exist; skipping figure workers")

    for case, output_dir, frame_count, label in summaries:
        summary_path = output_dir / "rigid_bias_summary.csv"
        figure_path = output_dir / "rigid_interpolation_bias.png"
        if summary_path.is_file() and figure_path.is_file():
            print("  rigid summary exists; skipping")
            continue
        rows = rigid_rows(case, output_dir, frame_count)
        with summary_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_rigid_bias(figure_path, rows, title=label)


def main() -> None:
    if os.environ.get("EXP3_DIC_WORKER") == "1":
        run_pair(
            Path(os.environ["EXP3_DIC_WORKER_DIR"]),
            Path(os.environ["EXP3_DIC_WORKER_OUT"]),
            int(os.environ["EXP3_DIC_WORKER_FRAME"]),
        )
        return
    sequences = render_sequences()
    if LIMIT:
        sequences = sequences[:LIMIT]
    if not sequences:
        raise FileNotFoundError("No completed additive Exp3 render sequences matched the requested filter.")
    print(f"Exp3 DIC stage 1: {len(sequences)} completed sequences; subset={DIC_SUBSET_SIZE_PX}, step={DIC_SUBSET_STEP_PX}, shape={DIC_SHAPE_FUNCTION}, threshold={DIC_CORRELATION_THRESHOLD}, threads={DIC_NUM_THREADS}")
    for index, (case, root, directory) in enumerate(sequences, start=1):
        print(f"[{index}/{len(sequences)}] {root}/{case}/{directory.name}")
        run_dic_stage(case, root, directory)
    run_postprocess_stage(sequences)


if __name__ == "__main__":
    main()
