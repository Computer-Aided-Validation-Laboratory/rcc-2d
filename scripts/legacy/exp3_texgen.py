# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Stream Exp3 f64 and digitised eggbox/additive-speckle textures to disk."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from PIL import Image

from exp1common import evaluate_eggbox_analytic_average
from exp2speckint2d import make_speckle_pattern
from exp3params import (
    ANALYTIC_SPECKLE_TYPES,
    BIT_DEPTHS,
    BLACK_AREA_FRACTIONS,
    FORCE_RENDER_OVER,
    GAMMA,
    GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
    GAUSSIAN_CUTOFF_SIGMAS,
    GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,
    I0,
    NUM_PROCESSES,
    PERTURBATION_DISTRIBUTIONS,
    PERTURBATION_FRACTIONS,
    P_PIXELS,
    PX_PER_SPECK,
    RANDOM_SEED,
    TARG_PX_X,
    TARG_PX_Y,
    TEXGEN_MAX_TEXELS_PER_BATCH,
    TEX_OVERSAMPLES,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
)
from script_timing import ScriptTimer


Image.MAX_IMAGE_PIXELS = None
_WORKER_PATTERN = None


def _selected_ints(env_name: str, configured: list[int]) -> list[int]:
    value = os.environ.get(env_name)
    return list(configured) if not value else [
        int(item.strip()) for item in value.split(",") if item.strip()
    ]


def _selected_types() -> list[str]:
    value = os.environ.get("EXP3_TEXTURE_TYPES")
    selected = ["eggbox", "speckle"] if not value else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    invalid = set(selected).difference({"eggbox", "speckle"})
    if invalid:
        raise ValueError(f"Unsupported EXP3_TEXTURE_TYPES: {', '.join(sorted(invalid))}")
    return selected


def _tag(pattern_type: str, black_fraction: float, distribution: str, fraction: float) -> str:
    return (
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_seed{RANDOM_SEED}"
    )


def _shape(oversamp: int) -> tuple[int, int]:
    return (
        oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD),
        oversamp * (TARG_PX_X + 2 * TEX_PX_PAD),
    )


def _tiff_path(directory: Path, prefix: str, depth: int, suffix: str = "") -> Path:
    return directory / f"{prefix}_b{depth}{suffix}.tiff"


def _complete(raw_path: Path, tiff_prefix: str, bits: list[int], tiff_suffix: str = "") -> bool:
    return raw_path.exists() and all(
        _tiff_path(raw_path.parent, tiff_prefix, depth, tiff_suffix).exists()
        for depth in bits
    )


def _quantise_tiffs(
    raw_path: Path,
    tiff_prefix: str,
    bits: list[int],
    coverage_to_intensity: bool,
    tiff_suffix: str = "",
) -> None:
    """Produce missing TIFFs from the f64 memmap one bounded row band at a time."""
    raw = np.load(raw_path, mmap_mode="r")
    rows_per_batch = max(1, TEXGEN_MAX_TEXELS_PER_BATCH // raw.shape[1])
    for depth in bits:
        output = _tiff_path(raw_path.parent, tiff_prefix, depth, tiff_suffix)
        if not FORCE_RENDER_OVER and output.exists():
            continue
        dtype = np.uint8 if depth == 8 else np.uint16
        staging_path = raw_path.parent / f".{output.stem}.raw"
        staging = np.memmap(staging_path, mode="w+", dtype=dtype, shape=raw.shape)
        scale = float(2**depth - 1)
        try:
            for start in range(0, raw.shape[0], rows_per_batch):
                stop = min(start + rows_per_batch, raw.shape[0])
                values = raw[start:stop]
                if coverage_to_intensity:
                    inverse = 1.0 - np.clip(values, 0.0, 1.0)
                    values = np.clip(I0 + GAMMA * (2.0 * inverse - 1.0), 0.0, 1.0)
                staging[start:stop] = np.rint(np.clip(values, 0.0, 1.0) * scale).astype(dtype)
            staging.flush()
            Image.fromarray(staging).save(output, format="TIFF", big_tiff=staging.nbytes > 2**32)
        finally:
            del staging
            staging_path.unlink(missing_ok=True)


def _eggbox_rows(task: tuple[int, int, int, float, float, float, float]) -> tuple[int, int, np.ndarray]:
    start, stop, width, x_start, y_start, texel_size, period = task
    values = evaluate_eggbox_analytic_average(
        start_x=x_start,
        start_y=y_start + start * texel_size,
        pixel_size=texel_size,
        num_px_x=width,
        num_px_y=stop - start,
        p_phys=period,
        i0=I0,
        gamma=GAMMA,
    )
    return start, stop, values


def _initialise_speckle_worker(pattern) -> None:
    global _WORKER_PATTERN
    _WORKER_PATTERN = pattern


def _speckle_rows(task: tuple[int, int, np.ndarray, float, float]) -> tuple[int, int, np.ndarray]:
    if _WORKER_PATTERN is None:
        raise RuntimeError("Speckle texture worker was not initialised.")
    start, stop, x, y_start, texel_size = task
    y = y_start + np.arange(start, stop, dtype=np.float64) * texel_size
    xx, yy = np.meshgrid(x, y)
    if _WORKER_PATTERN.pattern_type == "diskaddsat":
        coverage = _WORKER_PATTERN.evaluate_diskaddsat_box_average(xx, yy, texel_size, texel_size)
    else:
        coverage = _WORKER_PATTERN.evaluate_gausscont_box_average(xx, yy, texel_size, texel_size)
    return start, stop, coverage


def _generate_raw(
    raw_path: Path,
    shape: tuple[int, int],
    tasks: list[tuple],
    worker,
    initializer=None,
    initargs=(),
) -> None:
    """Fill an f64 NPY memmap atomically, flipping world-y into image rows."""
    partial = raw_path.with_name(f".{raw_path.stem}.partial.npy")
    partial.unlink(missing_ok=True)
    raw = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float64, shape=shape)
    workers = max(1, min(NUM_PROCESSES, int(os.environ.get("EXP3_NUM_PROCESSES", NUM_PROCESSES))))
    print(f"    {len(tasks)} row batches, {workers} workers, output={raw.nbytes / 2**30:.2f} GiB")
    try:
        with mp.Pool(workers, initializer=initializer, initargs=initargs) as pool:
            for start, stop, values in pool.imap_unordered(worker, tasks):
                raw[shape[0] - stop:shape[0] - start] = values[::-1]
        raw.flush()
    except Exception:
        del raw
        partial.unlink(missing_ok=True)
        raise
    del raw
    partial.replace(raw_path)


def _generate_eggbox(oversamp: int, bits: list[int]) -> None:
    shape = _shape(oversamp)
    raw_path = TEXTURE_OUTPUT_DIR / (
        f"tex_px{TARG_PX_X}_int_analytic_param_0_pad{TEX_PX_PAD}_oversamp{oversamp}.npy"
    )
    tiff_prefix = f"tex_px{TARG_PX_X}_int_analytic_param_0"
    tiff_suffix = f"_pad{TEX_PX_PAD}_oversamp{oversamp}"
    if not FORCE_RENDER_OVER and _complete(raw_path, tiff_prefix, bits, tiff_suffix):
        print("    outputs exist; skipping.")
        return
    if not raw_path.exists() or FORCE_RENDER_OVER:
        texel_size = 1.0 / oversamp
        rows = max(1, TEXGEN_MAX_TEXELS_PER_BATCH // shape[1])
        start = -0.5 * TARG_PX_X - TEX_PX_PAD
        tasks = [
            (row, min(row + rows, shape[0]), shape[1], start, start, texel_size, P_PIXELS)
            for row in range(0, shape[0], rows)
        ]
        _generate_raw(raw_path, shape, tasks, _eggbox_rows)
    _quantise_tiffs(raw_path, tiff_prefix, bits, coverage_to_intensity=False, tiff_suffix=tiff_suffix)


def _generate_speckle(oversamp: int, bits: list[int], pattern_type: str, black_fraction: float, distribution: str, fraction: float) -> None:
    shape = _shape(oversamp)
    tag = _tag(pattern_type, black_fraction, distribution, fraction)
    tiff_prefix = f"tex_px{TARG_PX_X}_{tag}_pad{TEX_PX_PAD}_oversamp{oversamp}_analytic"
    raw_path = TEXTURE_OUTPUT_DIR / f"{tiff_prefix}.npy"
    if not FORCE_RENDER_OVER and _complete(raw_path, tiff_prefix, bits):
        print("    outputs exist; skipping.")
        return
    if not raw_path.exists() or FORCE_RENDER_OVER:
        texel_size = 1.0 / oversamp
        extent = 0.5 * TARG_PX_X + TEX_PX_PAD
        bounds = (-extent, extent, -extent, extent)
        pattern = make_speckle_pattern(
            pattern_type, PX_PER_SPECK, black_fraction, distribution, fraction,
            RANDOM_SEED, GAUSSIAN_CUTOFF_SIGMAS, bounds, I0, GAMMA,
            GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION, GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
        )
        rows = max(1, TEXGEN_MAX_TEXELS_PER_BATCH // shape[1])
        x = bounds[0] + np.arange(shape[1], dtype=np.float64) * texel_size
        tasks = [
            (row, min(row + rows, shape[0]), x, bounds[2], texel_size)
            for row in range(0, shape[0], rows)
        ]
        _generate_raw(raw_path, shape, tasks, _speckle_rows, _initialise_speckle_worker, (pattern,))
    _quantise_tiffs(raw_path, tiff_prefix, bits, coverage_to_intensity=True)


def main() -> None:
    print("Experiment 3: f64 and digitised analytic texture generator")
    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bits = _selected_ints("EXP3_BIT_DEPTHS", BIT_DEPTHS)
    timer = ScriptTimer(__file__)
    for texture_type in _selected_types():
        for oversamp in _selected_ints("EXP3_TEX_OVERSAMPLES", TEX_OVERSAMPLES):
            if texture_type == "eggbox":
                print(f"  eggbox, oversamp={oversamp}")
                with timer.case(f"eggbox_oversamp{oversamp}"):
                    _generate_eggbox(oversamp, bits)
                continue
            for pattern_type in ANALYTIC_SPECKLE_TYPES:
                for black_fraction in BLACK_AREA_FRACTIONS:
                    for distribution in PERTURBATION_DISTRIBUTIONS:
                        for fraction in PERTURBATION_FRACTIONS:
                            print(f"  {_tag(pattern_type, black_fraction, distribution, fraction)}, oversamp={oversamp}")
                            case = f"{_tag(pattern_type, black_fraction, distribution, fraction)}_oversamp{oversamp}"
                            with timer.case(case):
                                _generate_speckle(oversamp, bits, pattern_type, black_fraction, distribution, fraction)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
