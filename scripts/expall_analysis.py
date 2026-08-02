"""Run the enabled Experiment 1--3 analysis launchers sequentially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = ("exp1_all_analysis.py", "exp2_all_analysis.py", "exp3_all_analysis.py")


def main() -> None:
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"--- starting {script} ---", flush=True)
        subprocess.run([sys.executable, str(here / script)], cwd=here.parent, check=True)
        print(f"--- finished {script} ---", flush=True)
    print("All Experiment 1--3 analyses completed.")


if __name__ == "__main__":
    main()
