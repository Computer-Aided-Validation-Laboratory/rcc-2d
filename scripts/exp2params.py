#--------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List, Tuple

# Output directory for Exp 2
OUTPUT_DIR: Path = Path("./out/exp2_speckint2d_render_uvs")
TEXTURE_OUTPUT_DIR: Path = Path("./out/exp2_analytic_speckle_textures")
# Re-render existing outputs instead of skipping completed render frames.
FORCE_RENDER_OVER: bool = False

TARG_PX_X: int = 256
TARG_PX_Y: int = 256
BACKGROUND: float = 0.5
TEX_PX_PAD: int = 4
TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32]
BIT_DEPTHS: List[int] = [8, 12, 16]
NUM_PROCESSES: int = 8
# Riley uses one scratch tile per active raster worker.  For f64 builds,
# scalingpolicy uses about 154 B/sub-pixel, so per-worker scratch is
# 154 * ((tile_px + 2 * halo_px) * SSAA)^2 bytes.  With tile_size_min=1
# and no halo: SSAA 256/512/1024 uses about 9.6/38.5/154 MiB per worker.
# `RASTER_CHUNKS_PER_WORKER=4` schedules four work chunks, not four buffers.
RILEY_RASTER_THREADS: int = 8

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
    # ("rect", 1024), TODO: check we have the RAM for this
    # ("mc", 16),
    # ("mc", 64),
    # ("mc", 256),
    # ("mc", 1024),
    ("gauss", 2),
    ("gauss", 4),
    ("gauss", 8),
    ("gauss", 16),
    ("gauss", 32),
    ("gauss", 64),
    ("gauss", 128),
    ("gauss", 256),
    ("gauss", 512),
    ("analytic", 0),
]

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
PERTURBATION_DISTRIBUTIONS: List[str] = ["uniform"] # "gaussian"
PERTURBATION_FRACTIONS: List[float] = [0.25]
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
    "plate260_cam256_quad9_rigid",
    "plate260_cam256_quad9_affine",
]

# List of frames to generate and analyze (e.g. [0, 5])
ACTIVE_FRAMES: List[int] = [0,1,2,3,4,5,6,7,8,9,10]

# Per-texel SSAA levels for the analytic speckle texture generator.
TEX_SSAA_LEVELS: List[int] = [1, 2, 4, 8, 16]
RILEY_SSAA_LEVLES: List[int] = [1, 2, 4, 8, 16, 32]
