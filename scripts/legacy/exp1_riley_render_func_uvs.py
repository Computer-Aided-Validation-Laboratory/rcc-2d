# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

import exp1_riley_func_uv_cos as uvs_render

uvs_render.OUTPUT_ROOT = Path("./out/exp1_riley_render_func_uvs")


if __name__ == "__main__":
    uvs_render.main()
