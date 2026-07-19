# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

import exp1_riley_func_world_cos as world_render

world_render.OUTPUT_ROOT = Path("./out/exp1_riley_render_func_world")


if __name__ == "__main__":
    world_render.main()
