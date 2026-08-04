#!/usr/bin/env python3
"""Run resumable two-stage PyVale 2D DIC for Exp3 additive-speckle sequences.

Completed render directories are discovered, rather than encoded here, so a
rerun automatically includes custom/Riley, analytic/SSAA/PSF, float/uint
renders and all three Exp3 deformation cases.  Results are written below
``out/exp3_dic/<case>/<render-root>/<pattern-config>/``.

Use ``EXP3_DIC_CASE=<case>``, ``EXP3_DIC_MATCH=<text>``, or
``EXP3_DIC_LIMIT=<n>`` for a focused run.

Stage 1 calculates only missing per-frame PyVale binary results.  Stage 2
imports each temporary binary, atomically saves the compact NPZ required by
downstream analysis, validates it, then removes the binary.  It also writes
only missing displacement figures and rigid-bias summaries.  Thus restarting
never repeats a completed DIC correlation, compact conversion, field figure,
or summary.
"""
from __future__ import annotations

import csv
from multiprocessing import get_context
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modules.exp3_dic_data import load_result, read_pyvale_binary, result_path, save_arrays
from modules.render_selection import analysis_enabled, measurement_enabled
from modules.output_naming import data_case_name, is_rigid_case
from modules.analysis_parallel import run_analysis_jobs
from modules.render_outputs import quantise_camera
from exp0params_common import DIC_CASES
from exp3params import (
    DIC_CORRELATION_THRESHOLD,
    DIC_SHAPE_FUNCTION,
    DIC_SUBSET_SIZE_PX,
    DIC_SUBSET_STEP_PX,
    MEASUREMENT_BIT_DEPTHS,
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
        if not measurement_enabled(render_root.name, DIC_CASES):
            continue
        if CASE_FILTER and case != CASE_FILTER:
            continue
        if not (config.startswith("diskadd_") or config.startswith("gaussadd_")):
            continue
        if not analysis_enabled(render_root.name, "diskadd" if config.startswith("diskadd_") else "gaussadd"):
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
    files = [path for path in files if not path.stem.endswith("_raw")]
    def frame_number(path: Path) -> int:
        match = re.search(r"(?:frame|_f)(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Cannot determine frame number from {path}")
        return int(match.group(1))
    return sorted(files, key=frame_number)


def measurement_bit_depths() -> list[int]:
    value = os.environ.get("EXP3_MEASUREMENT_BIT_DEPTHS")
    return list(MEASUREMENT_BIT_DEPTHS) if not value else [int(item) for item in value.split(",") if item.strip()]


def read_camera_codes(path: Path, bit_depth: int) -> np.ndarray:
    """Digitise the canonical float camera image for one DIC bit depth."""
    image = np.load(path, mmap_mode="r")
    # Bespoke float renders are [0, 1], whereas Riley's corresponding NPY
    # output is already in camera-code units.  Preserve both conventions.
    values = np.asarray(image, dtype=np.float64)
    if values.size and float(np.nanmax(np.abs(values))) > 1.0 + 1e-8:
        values /= 255.0
    return quantise_camera(values, bit_depth)


def field_title(label: str, frame: int, component: str, width: int = 42) -> str:
    """Keep renderer/configuration identifiers readable within a panel."""
    if len(label) <= width:
        return f"{label}, frame {frame:02d}: {component}"
    readable = label.replace("_", " ")
    return "\n".join(textwrap.wrap(readable, width=width, break_long_words=False) + [f"frame {frame:02d}: {component}"])


def physical_expected_rigid(case: str, frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not is_rigid_case(case):
        return None
    data_dir = Path("data") / data_case_name(case)
    ux = np.loadtxt(data_dir / "field_disp_x.csv", delimiter=",")
    uy = np.loadtxt(data_dir / "field_disp_y.csv", delimiter=",")
    if ux.ndim == 1:
        ux, uy = ux[None, :], uy[None, :]
    return ux[:, :frames].mean(axis=0), uy[:, :frames].mean(axis=0)


def save_field_arrays(path: Path, ux: np.ndarray, uy: np.ndarray, x: np.ndarray, y: np.ndarray, frame: int, label: str) -> None:
    # PyVale reports image-space V (positive down); convert to Exp3 physical Y.
    uy = -uy
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


def save_field(path: Path, result: object, frame: int, label: str) -> None:
    save_field_arrays(path, result.u_px[frame], result.v_px[frame], result.ss_x, result.ss_y, frame, label)


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


def run_pair(config_dir: Path, output_dir: Path, index: int, bit_depth: int) -> None:
    """Run one native-engine call; intended to execute in a fresh process."""
    # Keep the optional PyVale/Blender import out of spawned postprocess
    # workers.  Those workers only read the documented ``.dic2d`` format.
    import pyvale.dic as dic
    frames = image_frames(config_dir)
    reference = read_camera_codes(frames[0], bit_depth)
    deformed = read_camera_codes(frames[index], bit_depth)
    prefix = f"dic_frame{index:02d}_"
    # PyVale's ndarray convenience path converts each deformed image to an
    # 8-bit ``L`` TIFF internally.  Use temporary, explicitly uint16 TIFFs
    # instead, so a requested >8-bit DIC analysis truly reaches its backend.
    with tempfile.TemporaryDirectory(prefix="exp3_dic_") as temp:
        temp_dir = Path(temp)
        reference_path, deformed_path = temp_dir / "reference.tiff", temp_dir / "deformed.tiff"
        Image.fromarray(reference).save(reference_path)
        Image.fromarray(deformed).save(deformed_path)
        dic.calculate_2d(
            reference=reference_path,
            deformed=deformed_path,
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
            output_binary=True,
            output_delimiter=",",
            output_below_threshold=True,
            debug_level=0,
        )


def frame_csv(output_dir: Path, index: int) -> list[Path]:
    """Return this script's result CSV for one deformed frame."""
    return sorted(output_dir.glob(f"dic_frame{index:02d}_*.csv"))


def frame_binary(output_dir: Path, index: int) -> list[Path]:
    return sorted(output_dir.glob(f"dic_frame{index:02d}_*.dic2d"))


def raw_result(output_dir: Path, index: int) -> Path | None:
    """Return one temporary PyVale result, accepting legacy CSVs too."""
    candidates = frame_binary(output_dir, index) + frame_csv(output_dir, index)
    if len(candidates) > 1:
        raise RuntimeError(f"Expected at most one raw DIC result for frame {index:02d} in {output_dir}")
    return candidates[0] if candidates else None


def import_raw(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".dic2d":
        return read_pyvale_binary(path)
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 5, 6))
    x_values, y_values = np.unique(raw[:, 0]), np.unique(raw[:, 1])
    shape = (len(y_values), len(x_values))
    if raw.shape[0] != shape[0] * shape[1]:
        raise ValueError(f"DIC CSV does not describe a rectangular subset grid: {path}")
    return {
        "ss_x": np.meshgrid(x_values, y_values)[0], "ss_y": np.meshgrid(x_values, y_values)[1],
        "u_px": raw[:, 2].reshape((1, *shape)), "v_px": raw[:, 3].reshape((1, *shape)),
        "converged": raw[:, 4].astype(bool).reshape((1, *shape)), "cost_zncc": raw[:, 5].reshape((1, *shape)),
    }


def output_dir_for(case: str, render_root: str, config_dir: Path, bit_depth: int) -> Path:
    return RESULT_ROOT / case / render_root / config_dir.name / f"b{bit_depth:02d}"


def run_dic_stage(case: str, render_root: str, config_dir: Path, bit_depth: int) -> Path:
    """Stage 1: calculate only missing temporary PyVale binary results."""
    frames = image_frames(config_dir)
    output_dir = output_dir_for(case, render_root, config_dir, bit_depth)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, len(frames)):
        compact = result_path(output_dir, index)
        raw = raw_result(output_dir, index)
        if compact.is_file():
            print(f"  DIC frame {index:02d}: compact NPZ exists; skipping")
            continue
        if raw is not None:
            print(f"  DIC frame {index:02d}: temporary {raw.suffix} exists; awaiting postprocess")
            continue
        # PyVale's native engine retains process-global buffers across calls in
        # the installed branch.  A child per frame gives deterministic memory
        # release and makes an interrupted batch naturally resumable.
        print(f"  DIC frame {index:02d}: calculating", flush=True)
        worker_env = os.environ | {
            "EXP3_DIC_WORKER": "1",
            "EXP3_DIC_WORKER_DIR": str(config_dir.resolve()),
            "EXP3_DIC_WORKER_OUT": str(output_dir.resolve()),
            "EXP3_DIC_WORKER_FRAME": str(index),
            "EXP3_DIC_WORKER_BITS": str(bit_depth),
        }
        subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True, env=worker_env)
    return output_dir


def postprocess_frame(raw_path: str | None, compact_path: str, image_path: str, label: str, index: int) -> tuple[str, int]:
    """Stage-2 worker: compact one temporary PyVale result and plot if needed."""
    compact = Path(compact_path)
    if not compact.exists():
        if raw_path is None:
            raise RuntimeError(f"No raw DIC result available to create {compact}")
        raw = Path(raw_path)
        result = import_raw(raw)
        save_arrays(compact, **result)
        del result
        # ``save_result`` atomically reopens and validates before this removal.
        raw.unlink()
    if not Path(image_path).is_file():
        data = load_result(compact)
        save_field_arrays(Path(image_path), data["u_px"][0], data["v_px"][0], data["ss_x"], data["ss_y"], 0, label)
    return image_path, index


def _postprocess_job(task: tuple[str | None, str, str, str, int]) -> tuple[str, int]:
    return postprocess_frame(*task)


def rigid_rows(case: str, output_dir: Path, frame_count: int) -> list[dict[str, float | int]]:
    """Read one completed rigid sequence for its compact bias summary only."""
    expected = physical_expected_rigid(case, frame_count)
    if expected is None:
        return []
    rows: list[dict[str, float | int]] = []
    for index in range(1, frame_count):
        compact = result_path(output_dir, index)
        if not compact.is_file():
            raise RuntimeError(f"Expected compact DIC NPZ for frame {index:02d} in {output_dir}")
        result = load_result(compact)
        mean_x = float(np.nanmean(result["u_px"][0]))
        mean_y = float(-np.nanmean(result["v_px"][0]))
        rows.append({
            "frame": index,
            "expected_ux_px": float(expected[0][index]),
            "expected_uy_px": float(expected[1][index]),
            "mean_ux_px": mean_x,
            "mean_uy_px": mean_y,
            "bias_ux_px": mean_x - float(expected[0][index]),
            "bias_uy_px": mean_y - float(expected[1][index]),
        })
    return rows


def run_postprocess_stage(sequences: list[tuple[str, str, Path]], bit_depths: list[int]) -> None:
    """Stage 2: compact raw results in parallel, then write figures/summaries."""
    tasks: list[tuple[str | None, str, str, str, int]] = []
    summaries: list[tuple[str, Path, int, str]] = []
    for case, render_root, config_dir in sequences:
        frames = image_frames(config_dir)
        for bit_depth in bit_depths:
            output_dir = output_dir_for(case, render_root, config_dir, bit_depth)
            label = f"{config_dir.name}, {bit_depth}-bit"
            for index in range(1, len(frames)):
                compact = result_path(output_dir, index)
                raw = raw_result(output_dir, index)
                if not compact.is_file() and raw is None:
                    raise RuntimeError(f"Stage 1 did not produce a DIC result for frame {index:02d} in {output_dir}")
                image_path = output_dir / f"displacement_frame{index:02d}.png"
                if not compact.is_file() or not image_path.is_file():
                    tasks.append((str(raw) if raw is not None else None, str(compact), str(image_path), label, index))
            if is_rigid_case(case):
                summaries.append((case, output_dir, len(frames), label))

    if tasks:
        # Spawn avoids inheriting PyVale's native state from the DIC controller.
        for image_path, index in run_analysis_jobs(
            "Exp3 DIC stage 2", tasks, _postprocess_job, mp_context=get_context("spawn"),
        ):
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
            int(os.environ["EXP3_DIC_WORKER_BITS"]),
        )
        return
    sequences = render_sequences()
    if LIMIT:
        sequences = sequences[:LIMIT]
    if not sequences:
        raise FileNotFoundError("No completed additive Exp3 render sequences matched the requested filter.")
    bit_depths = measurement_bit_depths()
    print(f"Exp3 DIC stage 1: {len(sequences)} completed sequences; bits={bit_depths}; families={','.join(DIC_CASES)}; subset={DIC_SUBSET_SIZE_PX}, step={DIC_SUBSET_STEP_PX}, shape={DIC_SHAPE_FUNCTION}, threshold={DIC_CORRELATION_THRESHOLD}, threads={DIC_NUM_THREADS}")
    for index, (case, root, directory) in enumerate(sequences, start=1):
        print(f"[{index}/{len(sequences)}] {root}/{case}/{directory.name}")
        for bit_depth in bit_depths:
            run_dic_stage(case, root, directory, bit_depth)
    run_postprocess_stage(sequences, bit_depths)


if __name__ == "__main__":
    main()
