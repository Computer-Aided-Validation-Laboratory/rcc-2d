"""Run the resumable Experiment 3 render matrix in dependency order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from modules.render_selection import custom_enabled, float_textures_enabled, quantised_float_textures_enabled, riley_enabled
from exp3params import ENABLE_TRUE_UINT_TEXTURES
from modules.render_logging import render_log


SCRIPTS = (
    "exp3_texgen_eggbox.py",
    "exp3_texgen_speckle.py",
    "exp3_gridint2d_render_analytic.py",
    "exp3_gridint2d_render_ssaa.py",
    "exp3_speckint2d_render_analytic.py",
    "exp3_speckint2d_render_ssaa.py",
    "exp3_riley_render_func_ssaa.py",
    "exp3_riley_render_texfloat_ssaa.py",
    "exp3_riley_render_texfq_ssaa.py",
    "exp3_riley_render_texuint_ssaa.py",
    "exp3_speckint2d_render_disk_psf.py",
    "exp3_riley_render_texfloat_disk_psf.py",
    "exp3_riley_render_texfq_disk_psf.py",
    # Integer PSF texture renders are available but intentionally deferred.
    # "exp3_riley_render_texuint_disk_psf.py",
)


def main() -> None:
    here = Path(__file__).parent
    for script in SCRIPTS:
        if "texuint" in script and (not ENABLE_TRUE_UINT_TEXTURES or not riley_enabled("texuint_psf" if "psf" in script else "texuint")):
            continue
        if "texfq" in script and not quantised_float_textures_enabled(psf="psf" in script):
            continue
        if "texfloat" in script and not float_textures_enabled(psf="psf" in script):
            continue
        if "riley_render_func" in script and not riley_enabled("func"):
            continue
        if "gridint2d" in script and not custom_enabled("eggbox"):
            continue
        if "speckint2d_render_ssaa" in script and not any(custom_enabled(name) for name in ("disk", "gauss")):
            continue
        if "speckint2d_render_disk_psf" in script and not custom_enabled("disk_psf"):
            continue
        render_log("EXP3", "launcher", script, "starting")
        print(f"\n{'=' * 78}\nExperiment 3: {script}\n{'=' * 78}", flush=True)
        subprocess.run([sys.executable, str(here / script)], check=True)
        render_log("EXP3", "launcher", script, "completed")


if __name__ == "__main__":
    main()
