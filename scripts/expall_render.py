"""Run the resumable Experiment 1--3 render launchers in order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = ("exp1_all_render.py", "exp2_all_render.py", "exp3_all_render.py")


def main() -> None:
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n{'=' * 78}\nStarting {script}\n{'=' * 78}", flush=True)
        subprocess.run([sys.executable, str(here / script)], cwd=here.parent, check=True)
        print(f"--- finished {script} ---", flush=True)
    print("All Experiment 1--3 renders completed.")


if __name__ == "__main__":
    main()
