"""Run independent Exp1/Exp2 analysis scripts in bounded parallel batches."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from exp0params_common import ANALYSIS_JOBS

SCRIPTS_DIR = Path(__file__).resolve().parent

ANALYSIS_SCRIPTS = (
    "exp1_gridint2d_analysis_uvs.py",
    "exp1_riley_analysis_func_uvs.py",
    "exp1_riley_analysis_texfloat.py",
    "exp1_riley_analysis_texuint_uvs.py",
    "exp1_gridint2d_analysis_uvs_psf.py",
    "exp1_riley_analysis_func_uvs_psf.py",
    "exp1_riley_analysis_texfloat_psf.py",
    # "exp1_riley_analysis_texuint_psf.py",  # uint PSF renders are disabled.
    "exp2_speckint2d_analysis.py",
    "exp2_riley_analysis_texfloat.py",
    "exp2_speckint2d_analysis_uvs_psf.py",
    "exp2_riley_analysis_texfloat_psf.py",
    # "exp2_riley_analysis_texuint_psf.py",  # uint PSF renders are disabled.
)


def _child_environment() -> dict[str, str]:
    """Prevent each analysis process from creating a second thread pool."""
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
    jobs = max(1, int(os.environ.get("EXPALL_ANALYSIS_JOBS", ANALYSIS_JOBS)))
    jobs = min(jobs, len(ANALYSIS_SCRIPTS))
    print(f"Running {len(ANALYSIS_SCRIPTS)} analysis scripts with {jobs} concurrent jobs.")
    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_run, script): script for script in ANALYSIS_SCRIPTS}
        for future in as_completed(futures):
            script = futures[future]
            try:
                future.result()
                print(f"--- finished {script} ---", flush=True)
            except BaseException as error:
                failures.append((script, error))
                print(f"--- FAILED {script}: {error} ---", flush=True)
    if failures:
        failed = ", ".join(script for script, _ in failures)
        raise RuntimeError(f"Analysis suite failed: {failed}") from failures[0][1]
    print("All analysis scripts completed.")


if __name__ == "__main__":
    main()
