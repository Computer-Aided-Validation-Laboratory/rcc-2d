#!/usr/bin/env python3
"""Run the Exp3 analysis suite sequentially to avoid worker oversubscription."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_SCRIPTS = (
    "exp3_analysis_conv.py",
    "exp3_analysis_dic.py",
    "exp3_analysis_gridmethod.py",
    "exp3_analysis_dic_rigid_s_interp_err.py",
)


def main() -> None:
    for script in ANALYSIS_SCRIPTS:
        print(f"--- starting {script} ---", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], cwd=SCRIPTS_DIR.parent, check=True)
        print(f"--- finished {script} ---", flush=True)
    print("All Exp3 analyses completed.")


if __name__ == "__main__":
    main()
