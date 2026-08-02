"""Run the resumable Experiment 2 render suite in dependency order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from modules.render_logging import render_log
from modules.render_selection import custom_enabled, float_textures_enabled, riley_enabled
from modules.script_timing import ScriptTimer, timed_call


SCRIPTS = (
    "exp2_texgen_speckle_analytic.py",
    "exp2_speckint2d_render_uvs.py",
    "exp2_riley_render_texfloat.py",
    "exp2_riley_render_texuint.py",
    "exp2_speckint2d_render_uvs_psf.py",
    "exp2_riley_render_texfloat_psf.py",
    # The integer PSF script remains available but deliberately deferred.
    # "exp2_riley_render_texuint_psf.py",
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
        if "speckint2d" in script:
            enabled = "disk_psf" if psf else ("disk", "gauss")
            if isinstance(enabled, tuple):
                if not any(custom_enabled(name) for name in enabled):
                    continue
            elif not custom_enabled(enabled):
                continue
        selected.append(script)
    return tuple(selected)


def main() -> None:
    here = Path(__file__).resolve().parent
    timer = ScriptTimer(__file__)
    scripts = selected_scripts()
    print(f"Running {len(scripts)} enabled Experiment 2 render scripts.", flush=True)
    for script in scripts:
        render_log("EXP2", "launcher", script, "starting")
        print(f"\n{'=' * 78}\nExperiment 2: {script}\n{'=' * 78}", flush=True)
        timed_call(timer, script, subprocess.run, [sys.executable, str(here / script)], check=True, cwd=here.parent)
        render_log("EXP2", "launcher", script, "completed")


if __name__ == "__main__":
    main()
