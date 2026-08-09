#--------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
from modules.output_naming import output_root
from typing import List, Tuple

import riley
from exp0params_common import (
    CORES, FORCE_RENDER_OVER, NUM_PROCESSES, RILEY_RASTER_THREADS, RUN_MODE, RunMode,
)
from modules.exp12_geometry import ROI_PIXELS, TEXTURE_PAD_PIXELS

# All samplers available in the bound Riley build.  ``TEX_INTERPOLATORS`` is
# retained below as the conservative default matrix, while an environment
# selection may request any key in this map.
RILEY_TEXTURE_SAMPLERS: dict[str, riley.TextureSample] = {
    "nearest": riley.TextureSample.nearest,
    "linear": riley.TextureSample.linear,
    "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
    "cubic_mitchell_netravali": riley.TextureSample.cubic_mitchell_netravali,
    "cubic_bspline": riley.TextureSample.cubic_bspline,
    "quintic_bspline": riley.TextureSample.quintic_bspline,
    "lanczos2": riley.TextureSample.lanczos2,
    "lanczos3": riley.TextureSample.lanczos3,
}


if RUN_MODE is RunMode.TEST:
    TEX_SSAA_LEVELS: List[int] = [1, 2, 4, 8, 16, 32, 64, 128] 
    RILEY_SSAA_LEVLES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]
    TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]
    # Per-texel SSAA levels for the analytic speckle texture generator. 
    TEX_INTERPOLATORS: dict[str, riley.TextureSample] = {
        # "nearest": riley.TextureSample.nearest,
        "linear": riley.TextureSample.linear,
        "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
        "cubic_bspline": riley.TextureSample.cubic_bspline,
        #"lanczos3": riley.TextureSample.lanczos3,
    }
    # Integration methods and parameters
    INTEGRATION_METHODS: List[Tuple[str, int]] = [
        ("rect", 1),
        ("rect", 2),
        ("rect", 4),
        ("rect", 8),
        ("rect", 16),
        ("rect", 32),
        ("rect", 64),
        ("rect", 128),
        ("gauss", 2),
        ("gauss", 4),
        ("gauss", 8),
        ("gauss", 16),
        ("gauss", 32),
        ("gauss", 64),
        ("gauss", 128),
        ("analytic", 0),
    ]
    
    
else:
    TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Per-texel SSAA levels for the analytic speckle texture generator.
    # Used for digitised input texture creation
    TEX_SSAA_LEVELS: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Actually used for riley renders
    RILEY_SSAA_LEVLES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] 
    TEX_INTERPOLATORS: dict[str, riley.TextureSample] = {
        #"nearest": riley.TextureSample.nearest,
        "linear": riley.TextureSample.linear,
        "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
        "cubic_bspline": riley.TextureSample.cubic_bspline,
        #"lanczos3": riley.TextureSample.lanczos3,
    }
    # Integration methods and parameters
    INTEGRATION_METHODS: List[Tuple[str, int]] = [
        ("rect", 1),
        ("rect", 2),
        ("rect", 4),
        ("rect", 8),
        ("rect", 16),
        ("rect", 32),
        ("rect", 64),
        ("rect", 128),
        ("rect", 256),
        ("rect", 512),
        #("rect", 1024), # TODO: check we have the RAM for this
        ("gauss", 2),
        ("gauss", 4),
        ("gauss", 8),
        ("gauss", 16),
        ("gauss", 32),
        ("gauss", 64),
        ("gauss", 128),
        ("gauss", 256),
        ("gauss", 512),
        #("gauss", 1024),
        ("analytic", 0),
    ]

if RUN_MODE is RunMode.BIG:
    _test_levels = {1, 2, 4, 8, 16, 32, 64, 128}
    TEX_SSAA_LEVELS = [level for level in TEX_SSAA_LEVELS if level not in _test_levels]
    RILEY_SSAA_LEVLES = [level for level in RILEY_SSAA_LEVLES if level not in _test_levels]
    TEX_OVERSAMPLES = [level for level in TEX_OVERSAMPLES if level not in _test_levels]
    _test_rules = {("rect", n) for n in _test_levels} | {("gauss", n) for n in _test_levels if n >= 2} | {("analytic", 0)}
    INTEGRATION_METHODS = [rule for rule in INTEGRATION_METHODS if rule not in _test_rules]
    
    
TARG_PX_X: int = ROI_PIXELS
TARG_PX_Y: int = ROI_PIXELS

def exp2_output_dir(name: str) -> Path:
    """Return the canonical Experiment 2 output root.

    The camera size is carried by each case directory (``..._cam32_...``),
    rather than duplicated in the root name.
    """
    return output_root(name)

# Output directories for Exp 2.  The image-size suffix permits retaining
# results for several target sizes side by side.
OUTPUT_DIR: Path = exp2_output_dir("exp2_speckint2d_render_uvs")
TEXTURE_OUTPUT_DIR: Path = exp2_output_dir("exp2_analytic_speckle_textures")
# Re-render existing outputs instead of skipping completed render frames.

BACKGROUND: float = 0.5
TEX_PX_PAD: int = TEXTURE_PAD_PIXELS
# Image-plane camera PSF, expressed in final rendered-image pixels.  The
# finite support is explicit so the bespoke raster and Riley use the same
# sampled, normalised Gaussian kernel.
PSF_SIGMA_FINAL_PX: float = 1.0
PSF_SUPPORT_SIGMAS: float = 4.0
BIT_DEPTHS: List[int] = [8, 12]
# Legacy native u8/u16 source textures clamp/scale the additive field before
# Riley can integrate it.  Keep that distinct experiment disabled unless it
# is explicitly needed; the recommended finite-input-precision study is
# ``texfq`` (rounded raw f64 coverage) selected in exp0params_common.py.
ENABLE_TRUE_UINT_TEXTURES: bool = False
# Limit quadrature points held by each bespoke-renderer worker.  Exp2 keeps
# additional pattern-coverage temporaries, hence its caps are lower than
# Exp1's.  Retain the established 2M affine cap and use a lower VTK cap.
AFFINE_MAX_POINTS_PER_CHUNK: int = 2_000_000
VTK_MAX_POINTS_PER_CHUNK: int = 500_000
NEWTON_MAX_POINTS_PER_CHUNK: int = 1_000_000
# Riley uses one scratch tile per active raster worker.  For f64 builds,
# scalingpolicy uses about 154 B/sub-pixel, so per-worker scratch is
# 154 * ((tile_px + 2 * halo_px) * SSAA)^2 bytes.  With tile_size_min=1
# and no halo: SSAA 256/512/1024 uses about 9.6/38.5/154 MiB per worker.
# `RASTER_CHUNKS_PER_WORKER=4` schedules four work chunks, not four buffers.


# Speckle pattern parameters
PX_PER_SPECK: float = 5.0
I0: float = 0.5
GAMMA: float = 0.4
# Fraction of each unperturbed lattice cell covered by black disk area.
BLACK_AREA_FRACTIONS: List[float] = [0.6]
# Re-enable ``disk`` and ``gausstrunc`` here for numerical-only comparisons.
# They are excluded while the additive-saturation analytic reference is active.
SPECKLE_TYPES: List[str] = []
ANALYTIC_SPECKLE_TYPES: List[str] = ["diskaddsat", "gausscont"]
# Jitter is expressed as a fraction of the lattice pitch.  Keep separate
# controls for the additive patterns: the disk pattern tolerates more jitter,
# while the broader Gaussian pattern needs less to avoid clumping.
ADDITIVE_DISK_JITTER_DISTRIBUTION: str = "uniform"
ADDITIVE_DISK_JITTER_FRACTION: float = 0.25
ADDITIVE_GAUSS_JITTER_DISTRIBUTION: str = "gaussian"
ADDITIVE_GAUSS_JITTER_FRACTION: float = 0.12


def additive_jitter_for(pattern_type: str) -> tuple[str, float]:
    """Return the configured jitter PDF and fraction for an additive pattern."""
    if pattern_type == "diskaddsat":
        return ADDITIVE_DISK_JITTER_DISTRIBUTION, ADDITIVE_DISK_JITTER_FRACTION
    if pattern_type == "gausscont":
        return ADDITIVE_GAUSS_JITTER_DISTRIBUTION, ADDITIVE_GAUSS_JITTER_FRACTION
    raise ValueError(f"No additive jitter configuration for {pattern_type!r}")


RANDOM_SEED: int = 3
GAUSSIAN_CUTOFF_SIGMAS: float = 4.0
# For `gausscont`, this is the remaining fraction of peak coverage at the
# nominal equivalent-disk radius R (not a multiplier of sigma): 0.01 means
# 1% remains at R (R = 3.03 sigma), 0.1 means 10% remains (R = 2.15 sigma),
# and 0.5 means 50% remains (R = 1.18 sigma). Thus
# sigma = R / sqrt(-2 ln(edge_fraction)); larger values make wider blobs.
GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION: float = 0.4
# `gausscont` remains mathematically untruncated; centres beyond this many
# standard deviations are omitted as a bounded, configurable tail tolerance.
GAUSSIAN_CONTINUOUS_TAIL_SIGMAS: float = 8.0

# List of deformation cases to process (e.g. rigid, affine)
DEFORMATION_CASES: List[str] = [
    "plate42_cam32_quad9_rigid",
    "plate42_cam32_quad9_affine",
    "plate42_cam32_quad9_quadsaddle",
]

# ``affine`` is a four-corner inverse-map approximation, exact only for the
# rigid/global-affine fields.  ``newton`` is an accurate 2D inverse map for
# the current single Quad9 saddle; it remains explicit until other
# element-specific shape functions are added.
DEFORMATION_MAPPING_MODES: dict[str, str] = {
    "plate42_cam32_quad9_rigid": "affine",
    "plate42_cam32_quad9_affine": "affine",
    "plate42_cam32_quad9_quadsaddle": "newton",
}


def mapping_mode_for_case(case_name: str) -> str:
    """Return the explicitly configured reference-mapping mode for a case."""
    mode = DEFORMATION_MAPPING_MODES.get(case_name)
    if mode not in {"affine", "vtk", "newton"}:
        raise ValueError(f"No valid mapping mode configured for {case_name!r}.")
    return mode

# List of frames to generate and analyze (e.g. [0, 5])
ACTIVE_FRAMES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
