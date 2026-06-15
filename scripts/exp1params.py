# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List

TARG_PX_X: int = 256
TARG_PX_Y: int = 256
TEX_PX_PAD: int = 4
TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8]
BIT_DEPTHS: List[int] = [8, 12, 16]
SSAA_LEVELS: List[int] = [1, 2, 4]

# Grid pattern parameters
P_PIXELS: float = 5.0
I0: float = 0.5
GAMMA: float = 0.4

# Output directory for Exp 1
OUTPUT_DIR: Path = Path("./out/exp1_analytic_grid")
