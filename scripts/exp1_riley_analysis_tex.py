# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Texture-shader-only analysis for Experiment 1.

Texture sampling is UV based, so this compares it against the UV-integrator
reference and writes one subdirectory per named texture interpolator.
"""

from pathlib import Path

import exp1_riley_analysis as riley_analysis

riley_analysis.OUTPUT_DIR = Path("./out/exp1_gridint2d_render_uvs")
riley_analysis.RILEY_FUNC_DIR = Path("./out/exp1_riley_render_func_uvs")
riley_analysis.RILEY_TEX_DIR = Path("./out/exp1_riley_render_tex")
riley_analysis.RESULTS_DIR_FUNC = Path("./out/exp1_riley_analysis_tex_func")
riley_analysis.RESULTS_DIR_TEX = Path("./out/exp1_riley_analysis_tex")
riley_analysis.ANALYSIS_MODE = "tex"


if __name__ == "__main__":
    riley_analysis.main()
