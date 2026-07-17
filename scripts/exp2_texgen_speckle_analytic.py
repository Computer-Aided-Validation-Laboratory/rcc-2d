# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Generate exact texel averages for additive-saturation speckle fields."""

import multiprocessing
import os

import numpy as np

from exp2params import (
    ANALYTIC_SPECKLE_TYPES,
    BLACK_AREA_FRACTIONS,
    GAUSSIAN_CUTOFF_SIGMAS,
    GAMMA,
    I0,
    NUM_PROCESSES,
    PERTURBATION_DISTRIBUTIONS,
    PERTURBATION_FRACTIONS,
    PX_PER_SPECK,
    RANDOM_SEED,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_OVERSAMPLES,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
)
from exp2speckint2d import (
    MAX_PIXELS_PER_CHUNK,
    make_speckle_pattern,
    save_image,
)

NUM_PROCESSES_RUN = max(1, min(
    NUM_PROCESSES,
    int(os.environ.get("EXP2_NUM_PROCESSES", str(NUM_PROCESSES))),
))

_worker_pattern = None
_worker_pattern_type: str | None = None
_worker_x: np.ndarray | None = None
_worker_start_y: float | None = None
_worker_texel_size: float | None = None


def _init_texture_worker(
    pattern_type: str,
    speckle_size: float,
    black_fraction: float,
    distribution: str,
    fraction: float,
    bounds: tuple[float, float, float, float],
    tex_w: int,
    texel_size: float,
) -> None:
    """Create one immutable pattern per pixel-batch worker."""
    global _worker_pattern, _worker_pattern_type, _worker_x
    global _worker_start_y, _worker_texel_size
    _worker_pattern = make_speckle_pattern(
        pattern_type,
        speckle_size,
        black_fraction,
        distribution,
        fraction,
        RANDOM_SEED,
        GAUSSIAN_CUTOFF_SIGMAS,
        bounds,
        I0,
        GAMMA,
    )
    _worker_pattern_type = pattern_type
    _worker_x = bounds[0] + np.arange(tex_w) * texel_size
    _worker_start_y = bounds[2]
    _worker_texel_size = texel_size


def _process_texture_rows(
    task: tuple[int, int],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Evaluate exact coverage and intensity for one texture row batch."""
    if (
        _worker_pattern is None
        or _worker_x is None
        or _worker_start_y is None
        or _worker_texel_size is None
        or _worker_pattern_type is None
    ):
        raise RuntimeError("Analytic texture worker was not initialised.")
    start_row, end_row = task
    y = _worker_start_y + np.arange(start_row, end_row) * _worker_texel_size
    xx, yy = np.meshgrid(_worker_x, y)
    if _worker_pattern_type == "diskaddsat":
        coverage = _worker_pattern.evaluate_diskaddsat_box_average(
            xx, yy, _worker_texel_size, _worker_texel_size
        )
    else:
        coverage = _worker_pattern.evaluate_gausscont_box_average(
            xx, yy, _worker_texel_size, _worker_texel_size
        )
    return start_row, end_row, coverage, _worker_pattern.intensity_from_coverage(coverage)


def tag(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
) -> str:
    return (
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_seed{RANDOM_SEED}"
    )


def generate_texture(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
    oversample: int,
) -> None:
    """Generate exact axis-aligned texel averages for one analytic model."""
    roi_size = float(max(TARG_PX_X, TARG_PX_Y))
    pixel_size = roi_size / max(TARG_PX_X, TARG_PX_Y)
    texel_size = pixel_size / oversample
    tex_w = oversample * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h = oversample * (TARG_PX_Y + 2 * TEX_PX_PAD)
    bounds = (
        -0.5 * roi_size - TEX_PX_PAD * pixel_size,
        0.5 * roi_size + TEX_PX_PAD * pixel_size,
        -0.5 * roi_size - TEX_PX_PAD * pixel_size,
        0.5 * roi_size + TEX_PX_PAD * pixel_size,
    )
    image = np.empty((tex_h, tex_w), dtype=np.float64)
    raw_coverage = np.empty((tex_h, tex_w), dtype=np.float64)
    rows_per_batch = max(1, MAX_PIXELS_PER_CHUNK // tex_w)
    tasks = [
        (start_row, min(start_row + rows_per_batch, tex_h))
        for start_row in range(0, tex_h, rows_per_batch)
    ]
    print(
        f"    {len(tasks)} pixel batches, {NUM_PROCESSES_RUN} workers, "
        f"up to {rows_per_batch} rows/batch"
    )
    initargs = (
        pattern_type,
        PX_PER_SPECK * pixel_size,
        black_fraction,
        distribution,
        fraction,
        bounds,
        tex_w,
        texel_size,
    )
    with multiprocessing.Pool(
        NUM_PROCESSES_RUN,
        initializer=_init_texture_worker,
        initargs=initargs,
    ) as pool:
        for start_row, end_row, coverage, intensity in pool.imap_unordered(
            _process_texture_rows, tasks
        ):
            raw_coverage[start_row:end_row] = coverage
            image[start_row:end_row] = intensity

    prefix = (
        f"tex_px{TARG_PX_X}_"
        f"{tag(pattern_type, black_fraction, distribution, fraction)}"
        f"_pad{TEX_PX_PAD}_oversamp{oversample}_analytic"
    )
    # Save pixel-integrated coverage as the primary f64 texture.  It is not
    # clamped: overlapping disks/Gaussians can and should exceed one.
    save_image(image, TEXTURE_OUTPUT_DIR, prefix, float_texture=raw_coverage)


def main() -> None:
    print("Experiment 2: analytic additive-saturation texture generator")
    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern_type in ANALYTIC_SPECKLE_TYPES:
        if pattern_type not in {"diskaddsat", "gausscont"}:
            raise ValueError(f"Unsupported analytic type: {pattern_type}")
        for black_fraction in BLACK_AREA_FRACTIONS:
            for distribution in PERTURBATION_DISTRIBUTIONS:
                for fraction in PERTURBATION_FRACTIONS:
                    for oversample in TEX_OVERSAMPLES:
                        pattern_name = tag(
                            pattern_type,
                            black_fraction,
                            distribution,
                            fraction,
                        )
                        print(f"  {pattern_name}, oversamp={oversample}")
                        generate_texture(
                            pattern_type,
                            black_fraction,
                            distribution,
                            fraction,
                            oversample,
                        )


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
