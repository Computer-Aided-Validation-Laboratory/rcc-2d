"""Run the enabled, resumable Experiment 1 analysis suites."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from modules.render_selection import custom_enabled, riley_enabled


def selected_scripts() -> tuple[str, ...]:
    scripts: list[str] = []
    if custom_enabled("eggbox"):
        scripts.append("exp1_gridint2d_analysis_uvs.py")
    if riley_enabled("func"):
        scripts.append("exp1_riley_analysis_func_uvs.py")
    if riley_enabled("texfloat"):
        scripts.append("exp1_riley_analysis_texfloat.py")
    if riley_enabled("texuint"):
        scripts.append("exp1_riley_analysis_texuint_uvs.py")
    if custom_enabled("eggbox_psf"):
        scripts.append("exp1_gridint2d_analysis_uvs_psf.py")
    if riley_enabled("func_psf"):
        scripts.append("exp1_riley_analysis_func_uvs_psf.py")
    if riley_enabled("texfloat_psf"):
        scripts.append("exp1_riley_analysis_texfloat_psf.py")
    if riley_enabled("texuint_psf"):
        scripts.append("exp1_riley_analysis_texuint_psf.py")
    return tuple(scripts)


def child_environment() -> dict[str, str]:
    """Prevent nested numerical-library threading inside analysis workers."""
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def main() -> None:
    here = Path(__file__).resolve().parent
    scripts = selected_scripts()
    print(f"Running {len(scripts)} enabled Experiment 1 analysis scripts.", flush=True)
    for script in scripts:
        print(f"--- starting {script} ---", flush=True)
        subprocess.run([sys.executable, str(here / script)], cwd=here.parent, env=child_environment(), check=True)
        print(f"--- finished {script} ---", flush=True)


if __name__ == "__main__":
    main()
