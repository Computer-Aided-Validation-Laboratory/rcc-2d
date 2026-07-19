# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

import exp1_gridint2d_analysis as analysis
from exp1params import exp1_output_dir
from script_timing import ScriptTimer

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_world")
analysis.RESULTS_DIR = exp1_output_dir("exp1_gridint2d_analysis_world")


if __name__ == "__main__":
    with ScriptTimer(__file__).case("all_configured_cases"):
        analysis.main()
