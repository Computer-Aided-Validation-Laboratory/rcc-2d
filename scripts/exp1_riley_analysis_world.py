# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

import exp1_riley_analysis as riley_analysis

riley_analysis.OUTPUT_DIR = Path("./out/exp1_gridint2d_render_world")
riley_analysis.RILEY_FUNC_DIR = Path("./out/exp1_riley_render_func_world")
riley_analysis.RESULTS_DIR_FUNC = Path("./out/exp1_riley_analysis_world")
riley_analysis.RESULTS_DIR_TEX = Path("./out/exp1_riley_analysis_world_tex")


if __name__ == "__main__":
    riley_analysis.main()
