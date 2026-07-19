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
from exp1params import exp1_output_dir
from script_timing import ScriptTimer

riley_analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_world")
riley_analysis.RILEY_FUNC_DIR = exp1_output_dir("exp1_riley_render_func_world")
riley_analysis.RILEY_TEX_DIR = exp1_output_dir("exp1_riley_render_tex")
riley_analysis.RESULTS_DIR_FUNC = exp1_output_dir("exp1_riley_analysis_world")
riley_analysis.RESULTS_DIR_TEX = exp1_output_dir("exp1_riley_analysis_world_tex")
riley_analysis.ANALYSIS_MODE = "func"


if __name__ == "__main__":
    with ScriptTimer(__file__).case("all_configured_cases"):
        riley_analysis.main()
