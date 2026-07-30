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

from modules.script_timing import ScriptTimer, timed_call
from modules.render_selection import custom_enabled, float_textures_enabled, riley_enabled


SCRIPTS_DIR = Path(__file__).resolve().parent

# Keep this explicit: analysis and data-generation scripts are deliberately
# excluded, and each renderer must finish before the next begins.
# - Excluded world coords as final analysis will be based on uvs alone - was good
# to compare both and verify they are the same.
# - Analytic render scripts for exp1 are also included in the numerical cases -
# no need to run both
EXP1_RENDER_SCRIPTS = (
    "exp1_eggbox_grid_texgen.py",
    #"exp1_gridint2d_numerical_render_world.py",
    "exp1_gridint2d_numerical_render_uvs.py",
    #"exp1_gridint2d_analytic_render_world.py",
    #"exp1_gridint2d_analytic_render_uvs.py",
    #"exp1_riley_render_func_world.py",
    "exp1_riley_render_func_uvs.py",
    "exp1_riley_render_texfloat_uvs.py",
    "exp1_riley_render_texuint_uvs.py",
    "exp1_gridint2d_numerical_render_uvs_psf.py",
    "exp1_riley_render_func_uvs_psf.py",
    "exp1_riley_render_texfloat_uvs_psf.py",
    # "exp1_riley_render_texuint_uvs_psf.py",
)
EXP2_RENDER_SCRIPTS = (
    "exp2_texgen_speckle_analytic.py",
    "exp2_speckint2d_render_uvs.py",
    "exp2_riley_render_texfloat.py",
    "exp2_riley_render_texuint.py",
    "exp2_speckint2d_render_uvs_psf.py",
    "exp2_riley_render_texfloat_psf.py",
    # "exp2_riley_render_texuint_psf.py",
)


def selected_scripts(scripts: tuple[str, ...]) -> tuple[str, ...]:
    """Apply shared render-family controls to the explicit launcher lists."""
    selected: list[str] = []
    for script in scripts:
        if "texuint" in script and not riley_enabled("texuint_psf" if "psf" in script else "texuint"):
            continue
        if "texfloat" in script and not float_textures_enabled(psf="psf" in script):
            continue
        if "riley_render_func" in script and not riley_enabled("func_psf" if "psf" in script else "func"):
            continue
        if "gridint2d" in script and not custom_enabled("eggbox"):
            continue
        if "speckint2d" in script and not any(custom_enabled(name) for name in ("disk", "gauss", "disk_psf")):
            continue
        selected.append(script)
    return tuple(selected)


def run_suite(name: str, scripts: tuple[str, ...]) -> None:
    """Run one experiment's render scripts serially with this interpreter."""
    timer = ScriptTimer(__file__)
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    for script in scripts:
        print(f"\n--- {script} ---", flush=True)
        timed_call(
            timer, script, subprocess.run,
            [sys.executable, str(SCRIPTS_DIR / script)],
            check=True, cwd=SCRIPTS_DIR.parent,
        )


def main() -> None:
    run_suite("Experiment 1 renders", selected_scripts(EXP1_RENDER_SCRIPTS))
    run_suite("Experiment 2 renders", selected_scripts(EXP2_RENDER_SCRIPTS))
    print("\nAll Experiment 1 and Experiment 2 renders completed.")


if __name__ == "__main__":
    main()
