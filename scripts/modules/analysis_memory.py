"""Bounded-memory helpers shared by Exp1 and Exp2 analysis scripts."""

from __future__ import annotations

import ctypes
import gc

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


try:
    _MALLOC_TRIM = ctypes.CDLL("libc.so.6").malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except OSError:
    _MALLOC_TRIM = None


def make_agg_figure(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float],
    constrained_layout: bool = False,
):
    """Create a figure without registering it in pyplot's global manager."""
    figure = Figure(figsize=figsize, constrained_layout=constrained_layout)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(nrows, ncols)


def release_figure(figure: Figure) -> None:
    """Release Agg artists/buffers and return freed pages to the OS."""
    figure.clear()
    release_batch()


def release_batch() -> None:
    """Collect cyclic arrays/artists and trim the C allocator where available."""
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)
