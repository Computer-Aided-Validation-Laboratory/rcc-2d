# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Run the complete Experiment 1 render suite followed by Experiment 2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

# Keep this explicit: analysis and data-generation scripts are deliberately
# excluded, and each renderer must finish before the next begins.
EXP1_RENDER_SCRIPTS = (
    "exp1_eggbox_grid_texgen.py",
    "exp1_gridint2d_numerical_render_world.py",
    "exp1_gridint2d_numerical_render_uvs.py",
    "exp1_gridint2d_analytic_render_world.py",
    "exp1_gridint2d_analytic_render_uvs.py",
    "exp1_riley_render_func_world.py",
    "exp1_riley_render_func_uvs.py",
    "exp1_riley_render_tex_uvs.py",
    "exp1_riley_render_texfloat_uvs.py",
)
EXP2_RENDER_SCRIPTS = (
    # The maintained area-fraction texture generator.  The older
    # ``exp2_speckle_texgen.py`` is a superseded ratio-based variant.
    "exp2_texgen_speckle.py",
    "exp2_texgen_speckle_analytic.py",
    "exp2_speckint2d_render_uvs.py",
    "exp2_riley_render_texf.py",
)


def run_suite(name: str, scripts: tuple[str, ...]) -> None:
    """Run one experiment's render scripts serially with this interpreter."""
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    for script in scripts:
        print(f"\n--- {script} ---", flush=True)
        subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            check=True,
            cwd=SCRIPTS_DIR.parent,
        )


def main() -> None:
    run_suite("Experiment 1 renders", EXP1_RENDER_SCRIPTS)
    run_suite("Experiment 2 renders", EXP2_RENDER_SCRIPTS)
    print("\nAll Experiment 1 and Experiment 2 renders completed.")


if __name__ == "__main__":
    main()
