"""Small, consistent progress messages for long-running render launchers."""
from __future__ import annotations

from datetime import datetime


def case_label(case: str) -> str:
    """Return a compact deformation-case label suitable for a live log."""
    lower = case.lower()
    for label in ("quadsaddle", "finitestar", "chirp", "affine", "rigid"):
        if label in lower:
            return label
    return case


def render_log(experiment: str, renderer: str, case: str, message: str) -> None:
    """Print one flush-safe progress line without intercepting renderer output.

    This is deliberately an explicit call rather than a ``stdout`` wrapper:
    Riley's native diagnostic stream must pass through unchanged.
    """
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} {experiment} {renderer} {case}] {message}", flush=True)
