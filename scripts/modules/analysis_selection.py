"""Resumable analysis-suite completion markers shared by Exp1--3."""
from __future__ import annotations

import os
from pathlib import Path

from exp0params_common import FORCE_ANALYSIS_OVER


MARKER = ".analysis_complete"


def force_analysis() -> bool:
    value = os.environ.get("FORCE_ANALYSIS_OVER")
    return FORCE_ANALYSIS_OVER if value is None else value.strip().lower() in {"1", "true", "yes"}


def analysis_should_run(
    results_dir: Path,
    label: str,
    force_overwrite: bool = False,
) -> bool:
    marker = results_dir / MARKER
    if not force_analysis() and not force_overwrite and marker.is_file():
        print(f"{label}: completed analysis exists; skipping.")
        return False
    return True


def mark_analysis_complete(results_dir: Path) -> None:
    """Mark a suite only when it actually produced analysis data.

    Disabled render families and absent source data previously left a root
    containing only this marker.  Remove such speculative empty directories
    so ``out/`` reflects real analysis products only.
    """
    if not results_dir.exists():
        return
    has_output = any(
        path.is_file() and path.name != MARKER
        for path in results_dir.rglob("*")
    )
    if not has_output:
        (results_dir / MARKER).unlink(missing_ok=True)
        for path in sorted(results_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            results_dir.rmdir()
        except OSError:
            pass
        print(f"No analysis files were produced for {results_dir}; removed empty output directory.")
        return
    (results_dir / MARKER).write_text("complete\n")
