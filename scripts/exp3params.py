"""Experiment 3 constants: DIC-convolution render matrix."""

from __future__ import annotations

from pathlib import Path
from modules.output_naming import output_root, case_name
from typing import Final

import riley

from exp0params_common import CORES, FORCE_RENDER_OVER, NUM_PROCESSES, RILEY_RASTER_THREADS, TEST_RUN

BIT_DEPTHS = [8]
TEX_PX_PAD = 4
BACKGROUND = 0.5
I0 = 0.5
GAMMA = 0.4
# Always specified in final-camera pixels.  ``eggbox_pitch_world`` converts
# this independently in X/Y for rectangular or supersampled cameras.
EGGBOX_PERIOD_FINAL_PX = 5.0
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

# DIC analysis controls.  The subset step is deliberately one final camera
# pixel so interpolation-bias fields and their spatial structure are visible.
DIC_SUBSET_SIZE_PX = 15
DIC_SUBSET_STEP_PX = 1
DIC_SHAPE_FUNCTION = "AFFINE"
# Reliability-guided DIC requires this minimum ZNCC correlation.  A value of
# 0.8 tolerates locally weak/sharp speckle subsets while retaining their cost
# and convergence diagnostics in the saved results.
DIC_CORRELATION_THRESHOLD = 0.8
# DIC itself owns the physical cores.  Once CSV results exist, this many
# independent processes import them and generate displacement figures.
DIC_POSTPROCESS_JOBS = CORES

if TEST_RUN:
    SSAA_LEVELS = [1, 2, 4, 8, 16]
    TEX_OVERSAMPLES = [1, 2, 4, 8, 16]
else:
    # Needed for the 8-bit convergence study; sharp texture cases may need
    # a higher SSAA reference once their streamed texture path is available.
    SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64]
    TEX_OVERSAMPLES = [1, 2, 4, 8, 16, 32, 64]

RILEY_TEXTURE_SAMPLERS = {
    "nearest": riley.TextureSample.nearest,
    "linear": riley.TextureSample.linear,
    "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
    "cubic_mitchell_netravali": riley.TextureSample.cubic_mitchell_netravali,
    "cubic_bspline": riley.TextureSample.cubic_bspline,
    "quintic_bspline": riley.TextureSample.quintic_bspline,
    "lanczos2": riley.TextureSample.lanczos2,
    "lanczos3": riley.TextureSample.lanczos3,
}
# Conservative default run matrix.  Set EXP3_TEX_INTERPOLATORS to a
# comma-separated subset of RILEY_TEXTURE_SAMPLERS to enable others.
TEX_INTERPOLATORS = {name: RILEY_TEXTURE_SAMPLERS[name] for name in ("linear", "cubic_catmull_rom")}

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
    """Return a canonical Exp3 output path; camera size lives in ``case``."""
    root = output_root(name)
    return root if case is None else root / case_name(case)


def additive_jitter_for(pattern: str) -> tuple[str, float]:
    if pattern == "diskaddsat":
        return ADDITIVE_DISK_JITTER_DISTRIBUTION, ADDITIVE_DISK_JITTER_FRACTION
    if pattern == "gausscont":
        return ADDITIVE_GAUSS_JITTER_DISTRIBUTION, ADDITIVE_GAUSS_JITTER_FRACTION
    raise ValueError(f"Unknown additive pattern {pattern!r}")
