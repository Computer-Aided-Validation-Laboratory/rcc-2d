# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
import multiprocessing

import exp1_gridint2d_numerical_render_world as renderer
from exp1params import exp1_output_dir
from script_timing import ScriptTimer

renderer.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_world")
renderer.INTEGRATION_METHODS = [
    ("analytic", 0),
]

if __name__ == "__main__":

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    with ScriptTimer(__file__).case("all_configured_cases"):
        renderer.main()
