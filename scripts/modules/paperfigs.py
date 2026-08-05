"""Shared, journal-oriented Matplotlib helpers for paper figures.

Experiment scripts provide data and scientific labels; this module keeps the
physical page layout, typography, axes and export behaviour consistent.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg


def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


def make_figure(size_cm: tuple[float, float], *, rows: int, columns: int, tick_font_size: float) -> tuple[Figure, np.ndarray]:
    """Create an Agg-backed figure sized in physical centimetres."""
    figure = Figure(
        figsize=(cm_to_inch(size_cm[0]), cm_to_inch(size_cm[1])),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(rows, columns), dtype=object)
    # Let Matplotlib allocate the panel canvas from the actual titles, ticks
    # and labels.  Fixed fractional spacing caused label collisions whenever
    # an additional labelled column was requested.
    figure.get_layout_engine().set(w_pad=0.08, h_pad=0.08, wspace=0.08, hspace=0.12)
    for axis in axes.flat:
        axis.tick_params(labelsize=tick_font_size)
    return figure, axes.reshape(rows, columns)


def set_sample_axis(axis, samples: Iterable[int], label: str, label_font_size: float) -> None:
    ticks = sorted({int(value) for value in samples if int(value) > 0})
    if not ticks:
        return
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    # Alternate labels onto two baselines.  At the high end of a compact
    # base-2 axis this keeps adjacent labels such as 256 and 512 legible;
    # constrained layout reserves the additional line automatically.
    axis.xaxis.set_major_formatter(
        FixedFormatter([str(value) if index % 2 == 0 else f"\n{value}" for index, value in enumerate(ticks)])
    )
    axis.set_xlim(0.85 * ticks[0], 1.15 * ticks[-1])
    axis.set_xlabel(label, fontsize=label_font_size)


def set_nonnegative_error_axis(axis, values: Iterable[float], *, bit_depth: int, ylabel: str, label_font_size: float) -> None:
    """Use zero-inclusive symlog axes for RMSE and maximum absolute error."""
    valid = [float(value) for value in values if np.isfinite(value) and value >= 0.0]
    ceiling = 1.15 * max(valid + [1.0])
    axis.set_yscale("symlog", linthresh=0.25, linscale=0.8)
    axis.set_ylim(0.0, ceiling)
    axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.8, alpha=0.8, label="1 LSB")
    axis.set_ylabel(ylabel, fontsize=label_font_size)


def finish_axis(axis, *, title: str, samples: Sequence[int], bit_depth: int, values: Iterable[float], ylabel: str, title_font_size: float, axis_label_font_size: float) -> None:
    set_nonnegative_error_axis(axis, values, bit_depth=bit_depth, ylabel=ylabel, label_font_size=axis_label_font_size)
    set_sample_axis(axis, samples, "Axis integration samples", axis_label_font_size)
    axis.set_title(title, fontsize=title_font_size, pad=4)
    axis.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)


def add_figure_legend(figure: Figure, handles: Sequence, *, font_size: float, columns: int = 3) -> None:
    if handles:
        figure.legend(
            handles=handles,
            # Keep scientific panel titles clear; ``bbox_inches='tight'`` in
            # save_figure expands the canvas to retain this external legend
            # band without overlapping the bottom-row axis labels.
            loc="lower center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=columns,
            fontsize=font_size,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.1,
        )


def annotate_no_data(axis, message: str, *, font_size: float) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, fontsize=font_size)
    axis.set_axis_off()


def save_figure(figure: Figure, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = stem.with_suffix(f".{extension.lstrip('.')}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    figure.clear()
    return written
