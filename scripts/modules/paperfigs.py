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


# Stable texture-OS styles shared by every paper figure.  The render matrix
# commonly contains ten powers of two, exceeding Matplotlib's default colour
# cycle; indexing by log2(OS) keeps a given OS visually identical everywhere.
_TEXTURE_OS_COLOURS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#003f5c", "#ffa600",
)
_TEXTURE_OS_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p")
_TEXTURE_OS_LINESTYLES = (
    "-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1)),
    (0, (1, 1)), (0, (5, 2, 1, 2)), (0, (3, 2)), (0, (1, 2)),
    (0, (6, 1, 1, 1, 1, 1)), (0, (4, 1, 1, 1)),
)


def texture_os_style(oversample: int) -> tuple[str, str, object]:
    """Return a stable, distinct style for a texture oversampling level."""
    value = max(1, int(oversample))
    exponent = int(round(np.log2(value)))
    index = exponent if 2**exponent == value else value - 1
    return (
        _TEXTURE_OS_COLOURS[index % len(_TEXTURE_OS_COLOURS)],
        _TEXTURE_OS_MARKERS[index % len(_TEXTURE_OS_MARKERS)],
        _TEXTURE_OS_LINESTYLES[index % len(_TEXTURE_OS_LINESTYLES)],
    )


def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


def make_figure(size_cm: tuple[float, float], *, rows: int, columns: int, tick_font_size: float) -> tuple[Figure, np.ndarray]:
    """Create an Agg-backed figure sized in physical centimetres."""
    figure = Figure(
        figsize=(cm_to_inch(size_cm[0]), cm_to_inch(size_cm[1])),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    axes = np.asarray(
        figure.subplots(rows, columns), dtype=object,
    ).reshape(rows, columns)
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


def set_nonnegative_error_axis(
    axis, values: Iterable[float], *, bit_depth: int, ylabel: str,
    label_font_size: float,
) -> None:
    """Use zero-inclusive symlog axes for RMSE and maximum absolute error."""
    valid = [
        float(value) for value in values
        if np.isfinite(value) and value >= 0.0
    ]
    ceiling = 1.15 * max(valid + [1.0])
    axis.set_yscale("symlog", linthresh=0.25, linscale=0.8)
    axis.set_ylim(0.0, ceiling)
    axis.axhline(
        1.0, color="0.25", linestyle="--", linewidth=0.8, alpha=0.8,
        label="1 LSB",
    )
    axis.axhline(
        0.5, color="0.25", linestyle=":", linewidth=0.8, alpha=0.6,
        label="0.5 LSB",
    )

    from matplotlib.ticker import FuncFormatter

    def bit_formatter(x, pos):
        if x == 0:
            return "0"
        rounded = round(x, 6)
        if rounded == 0:
            return f"{x:g}"
        return f"{rounded:g}"

    axis.yaxis.set_major_formatter(FuncFormatter(bit_formatter))
    axis.set_ylabel(ylabel, fontsize=label_font_size)


def finish_axis(axis, *, title: str, samples: Sequence[int], bit_depth: int, values: Iterable[float], ylabel: str, title_font_size: float, axis_label_font_size: float) -> None:
    set_nonnegative_error_axis(axis, values, bit_depth=bit_depth, ylabel=ylabel, label_font_size=axis_label_font_size)
    set_sample_axis(axis, samples, "Axis integration samples", axis_label_font_size)
    axis.set_title(title, fontsize=title_font_size, pad=4)
    axis.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)


def set_signed_error_axis(
    axis, values: Iterable[float], *, bit_depth: int, ylabel: str,
    label_font_size: float,
) -> None:
    """Use zero-inclusive symmetric symlog axes for signed errors."""
    valid = [float(v) for v in values if np.isfinite(v)]
    ceiling = 1.15 * max([abs(v) for v in valid] + [1.0])
    axis.set_yscale("symlog", linthresh=0.25, linscale=0.8)
    axis.set_ylim(-ceiling, ceiling)
    axis.axhline(0.0, color="0.25", linestyle=":", linewidth=0.8, alpha=0.8)
    axis.axhline(
        1.0, color="0.25", linestyle="--", linewidth=0.8, alpha=0.4,
        label="1 LSB",
    )
    axis.axhline(-1.0, color="0.25", linestyle="--", linewidth=0.8, alpha=0.4)
    axis.axhline(
        0.5, color="0.25", linestyle=":", linewidth=0.8, alpha=0.6,
    )
    axis.axhline(
        -0.5, color="0.25", linestyle=":", linewidth=0.8, alpha=0.6,
    )

    from matplotlib.ticker import FuncFormatter

    def bit_formatter(x, pos):
        if x == 0:
            return "0"
        rounded = round(x, 6)
        if rounded == 0:
            return f"{x:g}"
        return f"{rounded:g}"

    axis.yaxis.set_major_formatter(FuncFormatter(bit_formatter))
    axis.set_ylabel(ylabel, fontsize=label_font_size)


def finish_signed_axis(
    axis, *, title: str, samples: Sequence[int], bit_depth: int,
    values: Iterable[float], ylabel: str, title_font_size: float,
    axis_label_font_size: float,
) -> None:
    set_signed_error_axis(
        axis, values, bit_depth=bit_depth, ylabel=ylabel,
        label_font_size=axis_label_font_size,
    )
    set_sample_axis(
        axis, samples, "Axis integration samples", axis_label_font_size,
    )
    axis.set_title(title, fontsize=title_font_size, pad=4)
    axis.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)



def add_figure_legend(
    figure: Figure, handles: Sequence, *, font_size: float, columns: int = 3,
    y_offset: float = -0.09,
) -> None:
    if handles:
        figure.legend(
            handles=handles,
            # Keep scientific panel titles clear; ``bbox_inches='tight'`` in
            # save_figure expands the canvas to retain this external legend
            # band without overlapping the bottom-row axis labels.
            loc="lower center",
            bbox_to_anchor=(0.5, y_offset),
            ncol=columns,
            fontsize=font_size,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.1,
        )


def annotate_no_data(axis, message: str, *, font_size: float) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, fontsize=font_size)
    axis.set_axis_off()


def paper_output_directories() -> tuple[Path, ...]:
    """Return the repository and manuscript destinations for paper assets."""
    # Local import keeps this general plotting module independent of the
    # experiment scripts while making the destinations centrally configurable.
    from paperparams import PAPER_DIR, PAPER_OUTPUT_DIR
    return tuple(dict.fromkeys((Path(PAPER_OUTPUT_DIR), Path(PAPER_DIR))))


def _mirrored_stems(stem: Path) -> tuple[Path, ...]:
    """Mirror standard paper outputs to ``PAPER_DIR`` without caller changes."""
    from paperparams import PAPER_OUTPUT_DIR
    stem = Path(stem)
    if stem.parent == Path(PAPER_OUTPUT_DIR):
        return tuple(directory / stem.name for directory in paper_output_directories())
    return (stem,)


def save_figure(figure: Figure, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    written: list[Path] = []
    for output_stem in _mirrored_stems(stem):
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in formats:
            path = output_stem.with_suffix(f".{extension.lstrip('.')}")
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
    figure.clear()
    return written


def write_latex_preview(
    stems: Sequence[str], captions: dict[str, str], labels: dict[str, str],
) -> list[Path]:
    """Write mirrored ``\\input`` blocks and compile the repository preview.

    ``PAPER_DIR`` is the live manuscript repository, so its own ``article.tex``
    must never be replaced.  It receives only figures and self-contained
    figure blocks for the manuscript to input.  ``out/paper`` additionally
    retains the generated preview article and PDF.
    """
    import subprocess
    from paperparams import PAPER_OUTPUT_DIR

    written: list[Path] = []
    for output_dir in paper_output_directories():
        output_dir.mkdir(parents=True, exist_ok=True)
        blocks: list[Path] = []
        for stem in stems:
            block = output_dir / f"{stem}.tex"
            block.write_text(
                "\\begin{figure}[p]\n"
                "  \\centering\n"
                "  \\includegraphics[height=0.9\\textheight,width=\\textwidth,"
                f"keepaspectratio]{{{stem}.pdf}}\n"
                f"  \\caption{{{captions[stem]}}}\n"
                f"  \\label{{{labels[stem]}}}\n"
                "\\end{figure}\n",
                encoding="utf-8",
            )
            blocks.append(block)
        written.extend(blocks)
        if output_dir != Path(PAPER_OUTPUT_DIR):
            continue
        article = output_dir / "article.tex"
        inputs = "\n".join(f"\\input{{{block.stem}}}\n\\clearpage" for block in blocks)
        article.write_text(
            "\\documentclass[10pt,a4paper]{article}\n"
            "\\usepackage[a4paper,margin=2.5cm]{geometry}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{lmodern}\n"
            "\\usepackage{graphicx}\n"
            "\\begin{document}\n"
            f"{inputs}\n"
            "\\end{document}\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", article.name],
            cwd=output_dir,
            check=True,
        )
        written.extend([article, output_dir / "article.pdf"])
    return written
