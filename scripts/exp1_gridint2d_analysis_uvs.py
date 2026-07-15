# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path

import exp1_pxint2d_analysis as analytic_analysis

analytic_analysis.OUTPUT_DIR = Path("./out/exp1_gridint2d_render_uvs")
analytic_analysis.RESULTS_DIR = Path("./out/exp1_gridint2d_analysis_uvs")


if __name__ == "__main__":
    analytic_analysis.main()
