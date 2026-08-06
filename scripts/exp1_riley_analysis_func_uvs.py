# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

from modules import exp1_riley_analysis_common as riley_analysis
from exp1params import exp1_output_dir
from modules.output_naming import analysis_output_root
from modules.script_timing import ScriptTimer
from modules.render_selection import riley_enabled

riley_analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs")
riley_analysis.RILEY_FUNC_DIR = exp1_output_dir("exp1_riley_render_func_uvs")
riley_analysis.RESULTS_DIR_FUNC = analysis_output_root("exp1", "riley_func_uvs")
riley_analysis.ANALYSIS_MODE = "func"


if __name__ == "__main__":
    if not riley_enabled("func"):
        print("Experiment 1 Riley function analysis disabled by RILEY_RENDER_CASES; skipping.")
    else:
        with ScriptTimer(__file__).case("all_configured_cases"):
            riley_analysis.main()
