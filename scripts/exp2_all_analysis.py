"""Run the enabled, resumable Experiment 2 analysis suites."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from modules.render_selection import custom_enabled, riley_enabled


def selected_scripts() -> tuple[str, ...]:
    scripts: list[str] = []
    if any(custom_enabled(case) for case in ("disk", "gauss")):
        scripts.append("exp2_speckint2d_analysis.py")
    if riley_enabled("texfloat"):
        scripts.append("exp2_riley_analysis_texfloat.py")
    if custom_enabled("disk_psf"):
        scripts.append("exp2_speckint2d_analysis_uvs_psf.py")
    if riley_enabled("texfloat_psf"):
        scripts.append("exp2_riley_analysis_texfloat_psf.py")
    if riley_enabled("texuint_psf"):
        scripts.append("exp2_riley_analysis_texuint_psf.py")
    return tuple(scripts)


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def main() -> None:
    here = Path(__file__).resolve().parent
    scripts = selected_scripts()
    print(f"Running {len(scripts)} enabled Experiment 2 analysis scripts.", flush=True)
    for script in scripts:
        print(f"--- starting {script} ---", flush=True)
        subprocess.run([sys.executable, str(here / script)], cwd=here.parent, env=child_environment(), check=True)
        print(f"--- finished {script} ---", flush=True)


if __name__ == "__main__":
    main()
