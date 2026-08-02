"""Run the resumable Experiment 1 render suite in dependency order.

Each child owns its own completion check.  ``FORCE_RENDER_OVER`` in
``exp0params_common.py`` therefore forces the same render paths used during
an interactive invocation, while the normal setting makes this launcher a
safe resume operation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from modules.render_logging import render_log
from modules.render_selection import custom_enabled, float_textures_enabled, riley_enabled
from modules.script_timing import ScriptTimer, timed_call


SCRIPTS = (
    "exp1_eggbox_grid_texgen.py",
    "exp1_gridint2d_numerical_render_uvs.py",
    "exp1_riley_render_func_uvs.py",
    "exp1_riley_render_texfloat_uvs.py",
    "exp1_riley_render_texuint_uvs.py",
    "exp1_gridint2d_numerical_render_uvs_psf.py",
    "exp1_riley_render_func_uvs_psf.py",
    "exp1_riley_render_texfloat_uvs_psf.py",
    # The integer PSF script remains available but deliberately deferred.
    # "exp1_riley_render_texuint_uvs_psf.py",
)


def selected_scripts() -> tuple[str, ...]:
    """Apply the shared Exp0 render-family controls to the suite."""
    selected: list[str] = []
    for script in SCRIPTS:
        psf = "psf" in script
        if "texuint" in script and not riley_enabled("texuint_psf" if psf else "texuint"):
            continue
        if "texfloat" in script and not float_textures_enabled(psf=psf):
            continue
        if "riley_render_func" in script and not riley_enabled("func_psf" if psf else "func"):
            continue
        if "gridint2d" in script and not custom_enabled("eggbox_psf" if psf else "eggbox"):
            continue
        selected.append(script)
    return tuple(selected)


def main() -> None:
    here = Path(__file__).resolve().parent
    timer = ScriptTimer(__file__)
    scripts = selected_scripts()
    print(f"Running {len(scripts)} enabled Experiment 1 render scripts.", flush=True)
    for script in scripts:
        render_log("EXP1", "launcher", script, "starting")
        print(f"\n{'=' * 78}\nExperiment 1: {script}\n{'=' * 78}", flush=True)
        timed_call(timer, script, subprocess.run, [sys.executable, str(here / script)], check=True, cwd=here.parent)
        render_log("EXP1", "launcher", script, "completed")


if __name__ == "__main__":
    main()
