"""Run the complete analysis suites for Experiments 1--3."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from modules.render_selection import custom_enabled, riley_enabled

SCRIPTS_DIR = Path(__file__).resolve().parent

def selected_analysis_scripts() -> tuple[str, ...]:
    """Return only analysis wrappers enabled by the shared Exp0 controls."""
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


def _child_environment() -> dict[str, str]:
    """Keep numerical libraries single-threaded inside process-pool workers."""
    import os
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def _run(script: str) -> str:
    print(f"--- starting {script} ---", flush=True)
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script)], check=True,
        cwd=SCRIPTS_DIR.parent, env=_child_environment(),
    )
    return script


def main() -> None:
    analysis_scripts = selected_analysis_scripts()
    # Each suite uses the shared process-pool harness over cases/groups. Run
    # suites serially here so two pools never oversubscribe the workstation.
    print(f"Running {len(analysis_scripts)} enabled Exp1/2 analysis scripts sequentially; each uses ANALYSIS_WORKERS.")
    for script in analysis_scripts:
        _run(script)
        print(f"--- finished {script} ---", flush=True)
    # Exp3 analysis contains its own sequential orchestration for DIC and Grid
    # Method workloads.  Run it after the independent Exp1/2 batch to avoid
    # oversubscribing the machine.
    print("--- starting exp3_all_analysis.py ---", flush=True)
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "exp3_all_analysis.py")],
        check=True, cwd=SCRIPTS_DIR.parent, env=_child_environment(),
    )
    print("--- finished exp3_all_analysis.py ---", flush=True)
    print("All Experiment 1--3 analysis scripts completed.")


if __name__ == "__main__":
    main()
