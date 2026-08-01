"""Resumable analysis-suite completion markers shared by Exp1--3."""
from __future__ import annotations

import os
from pathlib import Path

from exp0params_common import FORCE_ANALYSIS_OVER


MARKER = ".analysis_complete"


def force_analysis() -> bool:
    value = os.environ.get("FORCE_ANALYSIS_OVER")
    return FORCE_ANALYSIS_OVER if value is None else value.strip().lower() in {"1", "true", "yes"}


def analysis_should_run(results_dir: Path, label: str) -> bool:
    marker = results_dir / MARKER
    if not force_analysis() and marker.is_file():
        print(f"{label}: completed analysis exists; skipping.")
        return False
    return True


def mark_analysis_complete(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / MARKER).write_text("complete\n")
