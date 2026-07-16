# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Render analytic Experiment 2 speckles through the UV mesh mapping."""

import multiprocessing
import os
import sys
from pathlib import Path

from exp1common import parse_case_params
from exp2params import (
    ACTIVE_FRAMES,
    ANALYTIC_SPECKLE_TYPES,
    BLACK_AREA_FRACTIONS,
    DEFORMATION_CASES,
    GAUSSIAN_CUTOFF_SIGMAS,
    I0,
    GAMMA,
    INTEGRATION_METHODS,
    OUTPUT_DIR,
    PERTURBATION_DISTRIBUTIONS,
    PERTURBATION_FRACTIONS,
    PX_PER_SPECK,
    RANDOM_SEED,
    SPECKLE_TYPES,
    TEX_PX_PAD,
)
from exp2speckint2d import make_speckle_pattern, render_case


def get_active_frames() -> set[int]:
    value = os.environ.get("EXP2_ACTIVE_FRAMES")
    if not value:
        return set(ACTIVE_FRAMES)

    return {int(frame) for frame in value.split(",") if frame.strip()}


def get_methods() -> list[tuple[str, int]]:
    value = os.environ.get("EXP2_METHODS")
    if not value:
        return INTEGRATION_METHODS

    methods = []
    for item in value.split(","):
        method, param = item.split(":")
        methods.append((method.strip(), int(param.strip())))
    return methods


def pattern_tag(
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


def main() -> None:
    print("Experiment 2: gridint2d analytic speckle render (UVs)")
    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / name for name in DEFORMATION_CASES]

    for case_dir in cases:
        if not case_dir.exists():
            print(f"Warning: {case_dir} does not exist. Skipping.")
            continue
        camera_pixels, roi_size = parse_case_params(case_dir)
        pixel_size = roi_size / camera_pixels
        bounds = (
            -0.5 * roi_size - TEX_PX_PAD * pixel_size,
            0.5 * roi_size + TEX_PX_PAD * pixel_size,
            -0.5 * roi_size - TEX_PX_PAD * pixel_size,
            0.5 * roi_size + TEX_PX_PAD * pixel_size,
        )
        pattern_types = SPECKLE_TYPES + ANALYTIC_SPECKLE_TYPES
        for pattern_type in pattern_types:
            for black_fraction in BLACK_AREA_FRACTIONS:
                for distribution in PERTURBATION_DISTRIBUTIONS:
                    for fraction in PERTURBATION_FRACTIONS:
                        tag = pattern_tag(
                            pattern_type,
                            black_fraction,
                            distribution,
                            fraction,
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
                        for method, param in get_methods():
                            out_dir = OUTPUT_DIR / (
                                f"{case_dir.name}_{tag}_int_"
                                f"{method}_param_{param}"
                            )
                            print(f"  {out_dir.name}")
                            render_case(
                                case_dir,
                                out_dir,
                                pattern,
                                method,
                                param,
                                get_active_frames(),
                            )


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
