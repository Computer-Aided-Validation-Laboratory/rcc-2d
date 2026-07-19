# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Floating-texture-shader-only analysis for Experiment 1.

Floating texture sampling is UV based, so it is compared against the
UV-integrator reference.  Results are deliberately kept separate from the
digitised (unsigned-integer) texture analysis.
"""

import exp1_riley_analysis_common as riley_analysis
from exp1params import exp1_output_dir
from script_timing import ScriptTimer

riley_analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs")
riley_analysis.RILEY_TEX_DIR = exp1_output_dir("exp1_riley_render_texfloat")
riley_analysis.RESULTS_DIR_TEX = exp1_output_dir("exp1_riley_analysis_texfloat")
riley_analysis.ANALYSIS_MODE = "tex"


if __name__ == "__main__":
    with ScriptTimer(__file__).case("all_configured_cases"):
        riley_analysis.main()
