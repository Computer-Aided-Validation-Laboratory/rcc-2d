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

TARG_PX_X: int = 256
TARG_PX_Y: int = 256
TEX_PX_PAD: int = 4
TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32]
BIT_DEPTHS: List[int] = [8, 12, 16]
NUM_PROCESSES: int = 8

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
    #("rect", 1024),
    ("mc", 16),
    ("mc", 64),
    ("mc", 256),
    ("mc", 1024),
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

# Output directory for Exp 1
OUTPUT_DIR: Path = Path("./out/exp1_analytic_grid")
TEXTURE_OUTPUT_DIR: Path = Path("./out/exp1_analytic_textures")

# List of deformation cases to process (e.g. rigid, affine)
DEFORMATION_CASES: List[str] = [
    "plate260_cam256_quad9_rigid",
    "plate260_cam256_quad9_affine",
]

# List of frames to generate and analyze (e.g. [0, 5])
ACTIVE_FRAMES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
