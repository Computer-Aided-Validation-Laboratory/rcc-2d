"""Experiment-independent analysis plotting and memory utilities.

Experiment modules own discovery, reference selection and labels.  This file
owns the invariant mechanics: Agg figures, deterministic explicit log ticks,
and prompt release of Matplotlib/Numpy allocations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from modules.analysis_memory import make_agg_figure, release_batch, release_figure


def image_error_metrics(image: np.ndarray, reference: np.ndarray, bit_depth: int, quantise) -> dict[str, float]:
    """Return the common float and camera-code convergence statistics.

    ``quantise`` is supplied by the caller to keep this module independent of
    render persistence while ensuring every experiment measures exactly the
    same signed difference convention: candidate minus reference.
    """
    difference = np.asarray(image, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    code_difference = (
        quantise(image, bit_depth).astype(np.int64)
        - quantise(reference, bit_depth).astype(np.int64)
    )
    absolute_codes = np.abs(code_difference)
    result = {
        "e_f64": float(np.sqrt(np.mean(difference**2))),
        "mean_f64": float(np.mean(difference)),
        "e_inf": float(np.max(np.abs(difference))),
        "e_b": float(np.sqrt(np.mean(code_difference**2))),
        "mean_eb": float(np.mean(code_difference)),
        "delta_b": float(np.mean(absolute_codes >= 1)),
        "severe_b": float(np.mean(absolute_codes >= 2)),
        "p95_eb": float(np.percentile(absolute_codes, 95)),
        "p99_eb": float(np.percentile(absolute_codes, 99)),
        "max_eb": float(np.max(absolute_codes)),
    }
    del difference, code_difference, absolute_codes
    return result


def explicit_log_ticks(axis, values: Sequence[float]) -> None:
    """Use supplied sample/oversampling values as readable log-axis ticks."""
    ticks = np.unique(np.sort(np.asarray(values, dtype=float)))
    ticks = ticks[np.isfinite(ticks) & (ticks > 0)]
    if not len(ticks):
        return
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in ticks]))
    axis.set_xlim(float(ticks[0]) * 0.85, float(ticks[-1]) * 1.15)


def samples_along_axis(total_samples: Sequence[float]) -> np.ndarray:
    """Convert N² per-pixel samples to the N samples shown on figures."""
    return np.sqrt(np.asarray(total_samples, dtype=float))


__all__ = [
    "explicit_log_ticks", "make_agg_figure", "release_batch",
    "release_figure", "samples_along_axis", "image_error_metrics",
]
