"""Experiment 3 constants: DIC-convolution render matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import riley

from exp0params_common import FORCE_RENDER_OVER, NUM_PROCESSES, RILEY_RASTER_THREADS, TEST_RUN

BIT_DEPTHS = [8]
TEX_PX_PAD = 4
BACKGROUND = 0.5
I0 = 0.5
GAMMA = 0.4
EGGBOX_PX_PERIOD = 5.0
PX_PER_SPECK = 5.0
BLACK_AREA_FRACTIONS = [0.6]
RANDOM_SEED = 3
ADDITIVE_DISK_JITTER_DISTRIBUTION = "uniform"
ADDITIVE_DISK_JITTER_FRACTION = 0.25
ADDITIVE_GAUSS_JITTER_DISTRIBUTION = "gaussian"
ADDITIVE_GAUSS_JITTER_FRACTION = 0.12
GAUSSIAN_CUTOFF_SIGMAS = 4.0
GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION = 0.4
GAUSSIAN_CONTINUOUS_TAIL_SIGMAS = 8.0
PSF_SIGMA_FINAL_PX = 1.0
PSF_SUPPORT_SIGMAS = 4.0

if TEST_RUN:
    SSAA_LEVELS = [4, 8, 16]
    TEX_OVERSAMPLES = [4, 8, 16]
else:
    SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
    TEX_OVERSAMPLES = [1, 2, 4, 8, 16, 32, 64, 128]

TEX_INTERPOLATORS = {
    "nearest": riley.TextureSample.nearest,
    "linear": riley.TextureSample.linear,
    "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
}

CASE_CAMERA_PIXELS: Final[dict[str, tuple[int, int]]] = {
    "plate516_cam512_quad9_rigid": (512, 512),
    "plate516_cam512_quad9_affine": (512, 512),
    "plate260x65_cam256_quad9_chirp": (1020, 252),
}
CASE_ROI_SIZES: Final[dict[str, tuple[float, float]]] = {
    "plate516_cam512_quad9_rigid": (512.0, 512.0),
    "plate516_cam512_quad9_affine": (512.0, 512.0),
    # The current manual finite-star plate is 260 x 65 physical units.  The
    # 1020 x 252 camera preserves the existing 256 x 64 camera ROI at four
    # times the pixel resolution, with a two-unit plate border.
    "plate260x65_cam256_quad9_chirp": (256.0, 64.0),
}
DEFORMATION_CASES = list(CASE_CAMERA_PIXELS)
ACTIVE_FRAMES = {case: list(range(11)) if "chirp" not in case else [0, 1] for case in DEFORMATION_CASES}
# The two single-element cases have a globally affine inverse map.  The
# finite-star/chirp mesh is a structured rectangular Quad9 field.  Its
# dedicated Numba inverse uses exact Quad9 shape functions; this avoids the
# general-purpose VTK sampler without changing Exp1/2 mapping paths.
MAPPING_MODES = {
    case: ("structured_newton" if "chirp" in case else "affine")
    for case in DEFORMATION_CASES
}


def output_dir(name: str, case: str | None = None) -> Path:
    """Return an Exp3 output root, optionally size-qualified per case."""
    if case is None:
        return Path("out") / name
    width, height = CASE_CAMERA_PIXELS[case]
    return Path("out") / f"{name}_im{width}x{height}"


def additive_jitter_for(pattern: str) -> tuple[str, float]:
    if pattern == "diskaddsat":
        return ADDITIVE_DISK_JITTER_DISTRIBUTION, ADDITIVE_DISK_JITTER_FRACTION
    if pattern == "gausscont":
        return ADDITIVE_GAUSS_JITTER_DISTRIBUTION, ADDITIVE_GAUSS_JITTER_FRACTION
    raise ValueError(f"Unknown additive pattern {pattern!r}")
