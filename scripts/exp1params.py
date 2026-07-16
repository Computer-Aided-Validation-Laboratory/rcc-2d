# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List, Tuple

# Output directory for Exp 1
OUTPUT_DIR: Path = Path("./out/exp1_temp_out")
TEXTURE_OUTPUT_DIR: Path = Path("./out/exp1_analytic_textures")
CLEAR_DIR: bool = False

TARG_PX_X: int = 256
TARG_PX_Y: int = 256
BACKGROUND: float = 0.5
TEX_PX_PAD: int = 4
TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32, 64]
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
    ("analytic", 0),
]

# Grid pattern parameters
P_PIXELS: float = 5.0
I0: float = 0.5
GAMMA: float = 0.4


# List of deformation cases to process (e.g. rigid, affine)
DEFORMATION_CASES: List[str] = [
    "plate260_cam256_quad9_rigid",
    "plate260_cam256_quad9_affine",
]

# List of frames to generate and analyze (e.g. [0, 5])
ACTIVE_FRAMES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# SSAA levels to render with Riley
SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
