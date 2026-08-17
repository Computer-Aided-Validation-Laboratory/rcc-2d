"""Shared, journal-oriented Matplotlib helpers for paper figures.

Experiment scripts provide data and scientific labels; this module keeps the
physical page layout, typography, axes and export behaviour consistent.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from paperfiglabels import LABEL_025_LSB, LABEL_AXIS_INTEGRATION
from paperparams import LINE_COLOURS, PaperLayout


def configure_paper_matplotlib() -> None:
    """Apply the article-matching LaTeX typography before figure creation."""
    from paperparams import (
        PAPER_FONT_FAMILY, PAPER_SERIF_FONT, PAPER_TEX_PREAMBLE,
        PAPER_USE_TEX,
    )

    mpl.rcParams.update({
        "font.family": PAPER_FONT_FAMILY,
        "font.serif": [PAPER_SERIF_FONT],
        "text.usetex": PAPER_USE_TEX,
        "text.latex.preamble": PAPER_TEX_PREAMBLE if PAPER_USE_TEX else "",
        # Keep all line-series markers hollow in both axes and legends.  This
        # improves overlap readability without per-figure styling switches.
        "lines.markerfacecolor": "none",
    })


# Stable texture-OS styles shared by every paper figure.  The render matrix
# commonly contains ten powers of two, exceeding Matplotlib's default colour
# cycle; indexing by log2(OS) keeps a given OS visually identical everywhere.
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
        LINE_COLOURS[index % len(LINE_COLOURS)],
        _TEXTURE_OS_MARKERS[index % len(_TEXTURE_OS_MARKERS)],
        _TEXTURE_OS_LINESTYLES[index % len(_TEXTURE_OS_LINESTYLES)],
    )


def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


def make_figure(layout: PaperLayout, *, rows: int, columns: int, tick_font_size: float) -> tuple[Figure, np.ndarray]:
    """Create an Agg-backed figure sized in physical centimetres."""
    configure_paper_matplotlib()
    canvas_width_cm, canvas_height_cm = layout.canvas_cm(rows)
    figure = Figure(
        figsize=(
            cm_to_inch(canvas_width_cm),
            cm_to_inch(canvas_height_cm),
        ),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    axes = np.asarray(
        figure.subplots(rows, columns), dtype=object,
    ).reshape(rows, columns)
    # Let Matplotlib allocate the panel canvas from the actual titles, ticks
    # and labels.  Fixed fractional spacing caused label collisions whenever
    # an additional labelled column was requested.
    figure.get_layout_engine().set(
        w_pad=layout.w_pad, h_pad=layout.h_pad,
        wspace=layout.wspace, hspace=layout.hspace,
    )
    figure._paper_layout = layout
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
    if "rmse" in ylabel.lower():
        axis.axhline(
            0.25, color="0.25", linestyle=":", linewidth=0.8, alpha=0.6,
            label=LABEL_025_LSB,
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
    set_sample_axis(
        axis, samples, LABEL_AXIS_INTEGRATION, axis_label_font_size
    )
    axis.set_title(title, fontsize=title_font_size, pad=4)
    axis.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)


def finish_floating_error_axis(
    axis, *, title: str, samples: Sequence[int], values: Iterable[float],
    ylabel: str, title_font_size: float, axis_label_font_size: float,
) -> None:
    """Format a zero-inclusive floating-point image-error convergence axis.

    Pixel-integral errors can reach exact floating-point zero, so a pure log
    axis would silently lose valid samples.  A positive symlog axis preserves
    those points while retaining the useful logarithmic view of convergence.
    """
    valid = [float(value) for value in values if np.isfinite(value) and value >= 0.0]
    positive = [value for value in valid if value > 0.0]
    ceiling = 1.15 * max(valid + [1.0e-16])
    linthresh = min(positive) if positive else max(ceiling * 1.0e-3, 1.0e-16)
    axis.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
    axis.set_ylim(0.0, ceiling)
    axis.set_ylabel(ylabel, fontsize=axis_label_font_size)
    set_sample_axis(axis, samples, LABEL_AXIS_INTEGRATION, axis_label_font_size)
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
        axis, samples, LABEL_AXIS_INTEGRATION, axis_label_font_size,
    )
    axis.set_title(title, fontsize=title_font_size, pad=4)
    axis.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)



def add_figure_legend(
    figure: Figure, handles: Sequence, *, font_size: float, columns: int = 3,
    auto_position: bool = True,
) -> None:
    if handles:
        # Reserve a real band inside the fixed physical canvas.  This avoids
        # ``bbox_inches='tight'`` changing the PDF dimensions according to
        # legend content, which would otherwise make TeX rescale its fonts.
        reserve = figure._paper_layout.legend_band
        # ``rect`` is (left, bottom, width, height), not (left, bottom,
        # right, top).  Reduce its height with the reserved legend band so
        # constrained layout never sends top-row titles beyond the canvas.
        figure.get_layout_engine().set(
            rect=(0.0, reserve, 1.0, 1.0 - reserve)
        )
        legend = figure.legend(
            handles=handles,
            # The legend lives in the canvas band reserved above.
            loc="lower center",
            bbox_to_anchor=(0.5, figure._paper_layout.legend_anchor_y),
            ncol=columns,
            fontsize=font_size,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.1,
        )
        if not auto_position:
            # Keep the existing panel geometry intact, but remove excess white
            # canvas beneath a manually managed legend.  Figure 4 uses this
            # path because its equal-aspect maps must not be repositioned.
            figure.canvas.draw()
            font_height = (font_size / 72.0) / figure.get_figheight()
            legend.set_bbox_to_anchor(
                (0.5, 0.5 * font_height), transform=figure.transFigure,
            )
            figure.canvas.draw()
            return

        # Finalise title leading before measuring the tight axes boxes; this
        # gives every line-plot legend a consistent, small separation from the
        # lowest x-axis label even when typography is changed globally.
        prepare_panel_titles(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes = [axis for axis in figure.axes if axis.get_in_layout()]
        if not axes:
            return
        legend_height = legend.get_window_extent(renderer).height / figure.bbox.height
        font_height = (font_size / 72.0) / figure.get_figheight()
        # Publication rule: half a legend-font height below the legend, and
        # one full legend-font height from the legend to the x-axis label.
        legend_bottom = 0.5 * font_height
        target_axis_bottom = legend_bottom + legend_height + font_height
        legend.set_bbox_to_anchor((0.5, legend_bottom), transform=figure.transFigure)

        # The supplied layouts contain a conservative initial legend reserve.
        # Iteratively reduce/increase it until the final tight axes bounding
        # box meets the rule above, eliminating both excessive white space and
        # label/legend collisions after a global font-size change.
        legend_band = figure._paper_layout.legend_band
        for _ in range(3):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            axis_bottom = min(
                axis.get_tightbbox(renderer).y0 / figure.bbox.height
                for axis in axes
            )
            legend_band = min(
                0.45, max(0.0, legend_band + target_axis_bottom - axis_bottom)
            )
            figure.get_layout_engine().set(
                rect=(0.0, legend_band, 1.0, 1.0 - legend_band)
            )
        figure.canvas.draw()


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


def save_figure(
    figure: Figure, stem: Path, formats: Sequence[str], dpi: int
) -> list[Path]:
    prepare_panel_titles(figure)

    written: list[Path] = []
    for output_stem in _mirrored_stems(stem):
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in formats:
            path = output_stem.with_suffix(f".{extension.lstrip('.')}")
            figure.savefig(
                path,
                dpi=dpi,
                # Do not crop the canvas: its configured dimensions are the
                # physical dimensions used by the corresponding TeX block.
                bbox_inches=None,
            )
            written.append(path)
    figure.clear()
    return written


def prepare_panel_titles(figure: Figure) -> None:
    """Apply publication title leading before a constrained-layout draw."""
    from paperparams import (
        PANEL_TITLE_LINE_GAP_EX, PANEL_TITLE_LINE_SPACING, PAPER_USE_TEX,
    )

    # Apply one configurable leading value to every panel title immediately
    # before layout/export.  This includes titles assigned directly by each
    # experiment script as well as those assigned through ``finish_axis``.
    for axis in figure.axes:
        title = axis.title
        text = title.get_text()
        if PAPER_USE_TEX and "\n" in text and "\\shortstack{" not in text:
            # Matplotlib's ``linespacing`` is ignored by several usetex
            # backends.  TeX's ``\\[...ex]`` is deterministic and prevents
            # panel prefixes such as ``(a)`` colliding with the next line.
            separator = rf"\\[{PANEL_TITLE_LINE_GAP_EX:g}ex]"
            title.set_text(r"\shortstack{" + separator.join(text.splitlines()) + "}")
        title.set_linespacing(PANEL_TITLE_LINE_SPACING)


def write_latex_preview(
    stems: Sequence[str],
) -> list[Path]:
    """Write mirrored ``\\input`` blocks and compile the repository preview.

    ``PAPER_DIR`` is the live manuscript repository, so its own ``article.tex``
    must never be replaced.  It receives only figures and self-contained
    figure blocks for the manuscript to input.  ``out/paper`` additionally
    retains the generated preview article and PDF.
    """
    import subprocess
    from paperfigtex import FIGURE_PLACEMENT
    from paperparams import (
        PAGE_MARGIN_CM, PAPER_FIGURES, PAPER_OUTPUT_DIR,
        PAPER_PREVIEW_CLEARPAGE,
    )

    written: list[Path] = []
    for output_dir in paper_output_directories():
        output_dir.mkdir(parents=True, exist_ok=True)
        blocks: list[Path] = []
        for stem in stems:
            figure_spec = PAPER_FIGURES[stem]
            block = output_dir / f"{stem}.tex"
            block.write_text(
                f"\\begin{{figure}}[{FIGURE_PLACEMENT}]\n"
                "  \\centering\n"
                f"  \\includegraphics[width={figure_spec.layout.width_cm:g}cm]{{{stem}.pdf}}\n"
                f"  \\caption{{{figure_spec.caption}}}\n"
                f"  \\label{{{figure_spec.label}}}\n"
                "\\end{figure}\n",
                encoding="utf-8",
            )
            blocks.append(block)
        written.extend(blocks)
        if output_dir != Path(PAPER_OUTPUT_DIR):
            continue
        article = output_dir / "article.tex"
        separator = "\n\\clearpage\n" if PAPER_PREVIEW_CLEARPAGE else "\n"
        inputs = separator.join(f"\\input{{{block.stem}}}" for block in blocks)
        article.write_text(
            "\\documentclass[10pt,a4paper]{article}\n"
            f"\\usepackage[a4paper,margin={PAGE_MARGIN_CM:g}cm]{{geometry}}\n"
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
