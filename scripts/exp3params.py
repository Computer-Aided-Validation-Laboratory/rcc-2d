"""Experiment 3 constants: DIC-convolution render matrix."""

from __future__ import annotations

from pathlib import Path
from modules.output_naming import output_root, case_name
from typing import Final

import riley

from exp0params_common import CORES, FORCE_RENDER_OVER, NUM_PROCESSES, RILEY_RASTER_THREADS, RUN_MODE, RunMode

BIT_DEPTHS = [8, 12]
# Legacy native u8/u16 source textures are a bounded display-texture model.
# Keep them disabled by default; use the Exp0 ``texfq`` switch for simulated
# b-bit-equivalent precision of unbounded raw additive f64 coverage.
ENABLE_TRUE_UINT_TEXTURES = False
# Measurement analyses re-quantise the canonical post-integration float image;
# adding a depth here never triggers a rerender or regenerates non-preview TIFFs.
MEASUREMENT_BIT_DEPTHS = [8, 12]
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

# Grid-method local-spectrum window.  The bi-triangular window has compact,
# symmetric support of one grid pitch either side of its centre, avoiding the
# broad Gaussian window that spatially averaged the 32-pixel finite-star wave.
GRIDMETHOD_WINDOW = "triangular"
GRIDMETHOD_WINDOW_WIDTH_PERIODS = 1.0

TEST_SSAA_LEVELS = [1, 2, 4, 8, 16, 32]
TEST_TEX_OVERSAMPLES = [1, 2, 4, 8, 16, 32]
# Needed for the 8-bit convergence study; sharp texture cases may need a
# higher SSAA reference once their streamed texture path is available.
ALL_SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALL_TEX_OVERSAMPLES = [1, 2, 4, 8, 16, 32, 64, 128]

if RUN_MODE is RunMode.TEST:
    SSAA_LEVELS = list(TEST_SSAA_LEVELS)
    TEX_OVERSAMPLES = list(TEST_TEX_OVERSAMPLES)
elif RUN_MODE is RunMode.BIG:
    # One-dimensional studies (bespoke and function shader) need only the
    # levels omitted by TEST.  Texture studies use ``TEXTURE_SSAA_OS_PAIRS``
    # below, which is the full Cartesian difference rather than this product.
    SSAA_LEVELS = [level for level in ALL_SSAA_LEVELS if level not in TEST_SSAA_LEVELS]
    TEX_OVERSAMPLES = [level for level in ALL_TEX_OVERSAMPLES if level not in TEST_TEX_OVERSAMPLES]
else:
    SSAA_LEVELS = list(ALL_SSAA_LEVELS)
    TEX_OVERSAMPLES = list(ALL_TEX_OVERSAMPLES)

_test_texture_pairs = {
    (ssaa, oversamp)
    for ssaa in TEST_SSAA_LEVELS
    for oversamp in TEST_TEX_OVERSAMPLES
}
TEXTURE_SSAA_OS_PAIRS = (
    [(ssaa, oversamp) for ssaa in TEST_SSAA_LEVELS for oversamp in TEST_TEX_OVERSAMPLES]
    if RUN_MODE is RunMode.TEST else
    [
        (ssaa, oversamp)
        for ssaa in ALL_SSAA_LEVELS
        for oversamp in ALL_TEX_OVERSAMPLES
        if RUN_MODE is not RunMode.BIG or (ssaa, oversamp) not in _test_texture_pairs
    ]
)

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
TEX_INTERPOLATORS = {name: RILEY_TEXTURE_SAMPLERS[name] for name in (
    "linear",
    "cubic_catmull_rom",
    "cubic_bspline",)}

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


# Force overwrite switches for Exp3 analysis scripts
FORCE_CONV_OVERWRITE = True
FORCE_DIC_OVERWRITE = True
FORCE_GRIDMETHOD_OVERWRITE = True
FORCE_INTERP_BIAS_OVERWRITE = True
