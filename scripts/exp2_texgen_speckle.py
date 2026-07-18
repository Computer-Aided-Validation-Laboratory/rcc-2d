# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Generate oversampled analytic disk and Gaussian speckle textures."""

from pathlib import Path

import numpy as np

from exp2params import (
    BIT_DEPTHS,
    BLACK_AREA_FRACTIONS,
    FORCE_RENDER_OVER,
    GAUSSIAN_CUTOFF_SIGMAS,
    I0,
    GAMMA,
    PERTURBATION_DISTRIBUTIONS,
    PERTURBATION_FRACTIONS,
    PX_PER_SPECK,
    RANDOM_SEED,
    SPECKLE_TYPES,
    TEX_SSAA_LEVELS,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_OVERSAMPLES,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
)
from exp2speckint2d import image_outputs_complete, make_speckle_pattern, save_image


def tag(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
) -> str:
    return (
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_"
        f"seed{RANDOM_SEED}"
    )


def generate_texture(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
    oversample: int,
    ssaa: int,
) -> None:
    """Render a texture with ``ssaa`` squared midpoint samples per texel."""
    prefix = (
        f"tex_px{TARG_PX_X}_"
        f"{tag(pattern_type, black_fraction, distribution, fraction)}"
        f"_pad{TEX_PX_PAD}_oversamp{oversample}_ssaa{ssaa}"
    )
    if not FORCE_RENDER_OVER and image_outputs_complete(TEXTURE_OUTPUT_DIR, prefix):
        print("    outputs exist; skipping.")
        return
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
    coverage_image = np.zeros((tex_h, tex_w), dtype=np.float64)
    rows_per_batch = max(1, 1_000_000 // tex_w)
    x_indices = np.arange(tex_w, dtype=np.float64)
    y_indices = np.arange(tex_h, dtype=np.float64)
    for sample_y in range(ssaa):
        y_offset = (sample_y + 0.5) / ssaa
        for sample_x in range(ssaa):
            x_offset = (sample_x + 0.5) / ssaa
            x = bounds[0] + (x_indices + x_offset) * texel_size
            for start_row in range(0, tex_h, rows_per_batch):
                end_row = min(start_row + rows_per_batch, tex_h)
                y = (
                    bounds[2]
                    + (y_indices[start_row:end_row] + y_offset) * texel_size
                )
                xx, yy = np.meshgrid(x, y)
                coverage_image[start_row:end_row] += pattern.evaluate_coverage(xx, yy)
    coverage_image /= float(ssaa * ssaa)
    image = pattern.intensity_from_coverage(coverage_image)
    save_image(image, TEXTURE_OUTPUT_DIR, prefix, float_texture=coverage_image)


def main() -> None:
    print("Experiment 2: analytic speckle texture generator")
    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern_type in SPECKLE_TYPES:
        for black_fraction in BLACK_AREA_FRACTIONS:
            for distribution in PERTURBATION_DISTRIBUTIONS:
                for fraction in PERTURBATION_FRACTIONS:
                    for oversample in TEX_OVERSAMPLES:
                        for ssaa in TEX_SSAA_LEVELS:
                            pattern_name = tag(
                                pattern_type,
                                black_fraction,
                                distribution,
                                fraction,
                            )
                            print(
                                f"  {pattern_name}, "
                                f"oversamp={oversample}, ssaa={ssaa}"
                            )
                            generate_texture(
                                pattern_type,
                                black_fraction,
                                distribution,
                                fraction,
                                oversample,
                                ssaa,
                            )


if __name__ == "__main__":
    main()
