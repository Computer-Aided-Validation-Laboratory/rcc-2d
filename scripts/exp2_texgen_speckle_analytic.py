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
    save_raw_coverage,
)

NUM_PROCESSES_RUN = max(1, min(
    NUM_PROCESSES,
    int(os.environ.get("EXP2_NUM_PROCESSES", str(NUM_PROCESSES))),
))


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
    pattern = make_speckle_pattern(
        pattern_type,
        PX_PER_SPECK * pixel_size,
        black_fraction,
        distribution,
        fraction,
        RANDOM_SEED,
        GAUSSIAN_CUTOFF_SIGMAS,
        bounds,
        I0,
        GAMMA,
    )
    image = np.empty((tex_h, tex_w), dtype=np.float64)
    raw_coverage = np.empty((tex_h, tex_w), dtype=np.float64)
    rows_per_batch = max(1, MAX_PIXELS_PER_CHUNK // tex_w)
    x = bounds[0] + np.arange(tex_w) * texel_size

    for start_row in range(0, tex_h, rows_per_batch):
        end_row = min(start_row + rows_per_batch, tex_h)
        y = bounds[2] + np.arange(start_row, end_row) * texel_size
        xx, yy = np.meshgrid(x, y)
        if pattern_type == "diskaddsat":
            coverage = pattern.evaluate_diskaddsat_box_average(
                xx,
                yy,
                texel_size,
                texel_size,
            )
        else:
            coverage = pattern.evaluate_gausscont_box_average(
                xx,
                yy,
                texel_size,
                texel_size,
            )
        raw_coverage[start_row:end_row] = coverage
        image[start_row:end_row] = pattern.intensity_from_coverage(coverage)

    prefix = (
        f"tex_px{TARG_PX_X}_"
        f"{tag(pattern_type, black_fraction, distribution, fraction)}"
        f"_pad{TEX_PX_PAD}_oversamp{oversample}_analytic"
    )
    save_image(image, TEXTURE_OUTPUT_DIR, prefix)
    save_raw_coverage(raw_coverage, TEXTURE_OUTPUT_DIR, prefix)


def main() -> None:
    print("Experiment 2: analytic additive-saturation texture generator")
    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
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
                        jobs.append(
                            (
                                pattern_type,
                                black_fraction,
                                distribution,
                                fraction,
                                oversample,
                            )
                        )
    with multiprocessing.Pool(NUM_PROCESSES_RUN) as pool:
        pool.starmap(generate_texture, jobs)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
