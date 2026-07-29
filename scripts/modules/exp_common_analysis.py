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
    "release_figure", "samples_along_axis",
]
