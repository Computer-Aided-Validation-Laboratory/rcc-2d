# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

import exp1_analyse_analytic_eggbox as analytic_analysis

analytic_analysis.OUTPUT_DIR = Path("./out/exp1_2d_analytic_render_world")
analytic_analysis.RESULTS_DIR = Path("./out/exp1_2d_analytic_analysis_world")


if __name__ == "__main__":
    analytic_analysis.main()
