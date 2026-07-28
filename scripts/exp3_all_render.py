"""Run the resumable Experiment 3 render matrix in dependency order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = (
    "exp3_texgen_eggbox.py",
    "exp3_texgen_speckle.py",
    "exp3_gridint2d_render_analytic.py",
    "exp3_gridint2d_render_ssaa.py",
    "exp3_speckint2d_render_analytic.py",
    "exp3_speckint2d_render_ssaa.py",
    "exp3_riley_render_func_ssaa.py",
    "exp3_riley_render_texfloat_ssaa.py",
    "exp3_riley_render_texuint_ssaa.py",
    "exp3_speckint2d_render_disk_psf.py",
    "exp3_riley_render_texfloat_disk_psf.py",
    # Integer PSF texture renders are available but intentionally deferred.
    # "exp3_riley_render_texuint_disk_psf.py",
)


def main() -> None:
    here = Path(__file__).parent
    for script in SCRIPTS:
        print(f"\n{'=' * 78}\nExperiment 3: {script}\n{'=' * 78}", flush=True)
        subprocess.run([sys.executable, str(here / script)], check=True)


if __name__ == "__main__":
    main()
