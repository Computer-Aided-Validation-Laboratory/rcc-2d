#!/usr/bin/env python3
"""Create the four journal-ready Experiment 1 convergence figures.

The script computes digitised metrics directly from canonical floating-point
renders.  TIFF previews are deliberately not inputs, so each figure uses the
same digitisation definition as the analysis scripts.
"""
from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from modules.exp_common_analysis import image_error_metrics
from modules.paperfigs import add_figure_legend, annotate_no_data, finish_axis, make_figure, save_figure
from modules.render_outputs import quantise_camera
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT, FIGURE_2X3_CM, FONT_SIZE_PT, LEGEND_FONT_SIZE_PT,
    GRID_LINE_WIDTH_PT, GRID_MARKER_SIZE_PT, RILEY_LINE_WIDTH_PT, RILEY_MARKER_SIZE_PT,
    PAPER_DPI, PAPER_FORMATS, PAPER_FRAME, PAPER_OUTPUT_DIR,
    PAPER_TEXFLOAT_BIT_DEPTH, PAPER_TEXTURE_INTERPOLATOR,
    TICK_FONT_SIZE_PT,
    FIGURE_CAPTIONS, FIGURE_LABELS,
)

OUT = Path("out")
GRID_SUMMARY = OUT / "exp1_grid2d_analysis_uvs" / "summary.csv"
GRID_RENDER = OUT / "exp1_grid2d_render_uvs"
FUNC_RENDER = OUT / "exp1_riley_render_func_uvs"
TEXFLOAT_RENDER = OUT / "exp1_riley_render_texf"
TEXUINT_RENDER = OUT / "exp1_riley_render_texuint"
TEXTURE_STUDIES = (
    ("pt42_cam32_q9_rig", 0, "Undeformed"),
    ("pt42_cam32_q9_rig", 3, "Rigid 0.3px"),
    ("pt42_cam32_q9_aff", 3, "Affine 0.3px"),
)
METRICS = (("e_b", "Digitised RMSE [bits]"), ("max_eb", "Max. digitised err. [bits]"))


@dataclass(frozen=True)
class Series:
    label: str
    samples: tuple[int, ...]
    values: tuple[float, ...]
    bit_depth: int
    colour: str
    marker: str
    linestyle: str


def panel_prefix(index: int) -> str:
    """Return the conventional journal sub-panel prefix, e.g. ``(a)``."""
    return f"({chr(ord('a') + index)})"


def display_reference(value: str) -> str:
    """Turn analysis-internal reference tokens into article-facing labels."""
    if value.strip().lower() in {"analytic", "analytic:0", "analytic reference"}:
        return "Analytic"
    return value.replace("_", " ")


def load_normalised(path: Path) -> np.ndarray:
    image = np.asarray(np.load(path), dtype=np.float64)
    if image.size and np.nanmax(np.abs(image)) > 1.0 + 1e-12:
        image /= 255.0
    return image


def analytic_reference(case: str, frame: int = PAPER_FRAME) -> tuple[np.ndarray, str]:
    direct = sorted((GRID_RENDER / case).glob(f"targ_px*_int_analytic_param_0_frame{frame:02d}.npy"))
    if direct:
        return load_normalised(direct[0]), "Analytic"
    candidates = []
    for path in (GRID_RENDER / case).glob(f"targ_px*_int_gauss_param_*_frame{frame:02d}.npy"):
        match = re.search(r"param_(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No analytic or Gauss reference for {case} frame {frame:02d}")
    order, path = max(candidates)
    return load_normalised(path), f"Highest Gauss ({order}×{order})"


def grid_metric_series(case: str, metric: str, frame: int = PAPER_FRAME) -> tuple[list[Series], str]:
    if not GRID_SUMMARY.is_file():
        return [], "No grid summary"
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    ref = "Analytic"
    with GRID_SUMMARY.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Case"] == case and int(row["Frame"]) == frame and row["Method"] != "analytic":
                groups[(row["Method"], int(row["BitDepth"]))].append(row); ref = row["Reference"]
    colours = {"gauss": "#1b9e77", "rect": "#377eb8"}; markers = {"gauss": "s", "rect": "o"}
    return [Series(f"Grid2D {method.title()}, {bits}-bit", tuple(int(round(float(r["Samples"]) ** .5)) for r in sorted(rows, key=lambda r: float(r["Samples"]))), tuple(float(r[metric]) for r in sorted(rows, key=lambda r: float(r["Samples"]))), bits, colours[method], markers[method], "-" if bits <= 8 else "--") for (method, bits), rows in sorted(groups.items())], ref


def function_series(case: str, reference: np.ndarray, metric: str, bit_depths: set[int], frame: int = PAPER_FRAME) -> list[Series]:
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for directory in (FUNC_RENDER / case).glob("ss*_f"):
        match = re.fullmatch(r"ss(\d+)_f", directory.name)
        path = directory / f"image_c00_f{frame:02d}.npy"
        if not match or not path.is_file():
            continue
        image = load_normalised(path)
        if image.shape != reference.shape:
            continue
        for bits in bit_depths:
            groups[bits].append((int(match.group(1)), image_error_metrics(image, reference, bits, quantise_camera)[metric]))
    return [Series(f"Riley Rect, {bits}-bit", tuple(x for x, _ in sorted(points)), tuple(y for _, y in sorted(points)), bits, "#d95f02", "^", "-" if bits <= 8 else "--") for bits, points in sorted(groups.items())]


def texture_series(case: str, root: Path, *, source_bits: int | None, camera_bits: int, metric: str, frame: int) -> list[Series]:
    reference, _ = analytic_reference(case, frame)
    pattern = re.compile(r"ss(\d+)_(?:b(\d+)_)?os(\d+)(?:_f)?")
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for case_dir in root.glob(f"{case}_{PAPER_TEXTURE_INTERPOLATOR}"):
        for directory in case_dir.iterdir():
            match = pattern.fullmatch(directory.name)
            path = directory / f"image_c00_f{frame:02d}.npy"
            if not match or not path.is_file():
                continue
            texture_bits = int(match.group(2)) if match.group(2) else None
            if source_bits is not None and texture_bits != source_bits:
                continue
            image = load_normalised(path)
            if image.shape == reference.shape:
                grouped[int(match.group(3))].append((int(match.group(1)), image_error_metrics(image, reference, camera_bits, quantise_camera)[metric]))
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    return [Series(f"Tex-OS={osamp}", tuple(x for x, _ in sorted(points)), tuple(y for _, y in sorted(points)), camera_bits, colours[index % len(colours)], "o", "-") for index, (osamp, points) in enumerate(sorted(grouped.items()))]


def draw_series(axis, series: list[Series], metric_label: str, reference: str) -> list[Line2D]:
    handles = []
    all_samples: list[int] = []; all_values: list[float] = []
    for item in series:
        is_grid = item.label.startswith("Grid2D")
        # Riley is drawn after Grid2D.  The heavier Grid2D trace remains
        # visible around the narrower Riley trace where parity makes them
        # coincide.
        linewidth, markersize = (
            (GRID_LINE_WIDTH_PT, GRID_MARKER_SIZE_PT)
            if is_grid
            else (RILEY_LINE_WIDTH_PT, RILEY_MARKER_SIZE_PT)
        )
        axis.plot(item.samples, item.values, color=item.colour, marker=item.marker, linestyle=item.linestyle, linewidth=linewidth, markersize=markersize)
        handles.append(Line2D([], [], color=item.colour, marker=item.marker, linestyle=item.linestyle, linewidth=linewidth, markersize=markersize, label=item.label))
        all_samples.extend(item.samples); all_values.extend(item.values)
    if not series:
        annotate_no_data(axis, "No completed render data", font_size=FONT_SIZE_PT)
        return handles
    finish_axis(axis, title=f"Reference: {reference}", samples=all_samples, bit_depth=max(item.bit_depth for item in series), values=all_values, ylabel=metric_label, title_font_size=FONT_SIZE_PT, axis_label_font_size=AXIS_LABEL_FONT_SIZE_PT)
    return handles


def figure_function_shaders() -> list[Path]:
    """Figure 1: undeformed and 0.3 px rigid/affine comparisons."""
    figure, axes = make_figure(FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    all_handles: list[Line2D] = []
    studies = (
        ("pt42_cam32_q9_rig", 0, "Undeformed"),
        ("pt42_cam32_q9_rig", 3, "Rigid 0.3px"),
        ("pt42_cam32_q9_aff", 3, "Affine 0.3px"),
    )
    for column, (case, frame, subtitle) in enumerate(studies):
        reference, ref_label = analytic_reference(case, frame)
        for row, (metric, ylabel) in enumerate(METRICS):
            grid, summary_ref = grid_metric_series(case, metric, frame)
            bits = {item.bit_depth for item in grid}
            data = grid + function_series(case, reference, metric, bits, frame)
            label = display_reference(summary_ref or ref_label)
            all_handles.extend(draw_series(axes[row, column], data, ylabel, label))
            axes[row, column].set_title(
                f"{panel_prefix(row * 3 + column)} {subtitle}, Ref: {label}",
                fontsize=FONT_SIZE_PT,
            )
            # Keep a complete vertical scale on every panel: the deformation
            # columns are compared independently in the paper figure.
    unique = {handle.get_label(): handle for handle in all_handles}
    add_figure_legend(figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT, columns=3)
    return save_figure(figure, PAPER_OUTPUT_DIR / "exp1_fig1_eggbox_function_shaders", PAPER_FORMATS, PAPER_DPI)


def figure_texture_convergence(
    stem: str, texture_label: str, root: Path, source_bits: int | None,
    output_bits: int,
) -> list[Path]:
    """A Fig. 1-style texture convergence figure with Tex-OS traces."""
    figure, axes = make_figure(FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    handles: list[Line2D] = []
    for column, (case, frame, deformation) in enumerate(TEXTURE_STUDIES):
        _, ref_label = analytic_reference(case, frame)
        for row, (metric, ylabel) in enumerate(METRICS):
            data = texture_series(
                case, root, source_bits=source_bits, camera_bits=output_bits,
                metric=metric, frame=frame,
            )
            handles.extend(draw_series(axes[row, column], data, ylabel, ref_label))
            axes[row, column].set_title(
                f"{panel_prefix(row * 3 + column)} {deformation}, Ref: {ref_label}",
                fontsize=FONT_SIZE_PT,
            )
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT, columns=4)
    return save_figure(figure, PAPER_OUTPUT_DIR / stem, PAPER_FORMATS, PAPER_DPI)


def exp1_figure_stems() -> tuple[str, ...]:
    return (
        "exp1_fig1_eggbox_function_shaders",
        "exp1_fig2_riley_texf_b8",
        "exp1_fig3_riley_texu8_b8",
        "exp1_fig4_riley_texu12_b8",
        "exp1_fig5_riley_texf_b12",
        "exp1_fig6_affine_eggbox_difference_maps",
    )


def generate_texture_figures() -> list[Path]:
    """Generate the four source-texture/output-depth convergence figures."""
    written = figure_texture_convergence("exp1_fig2_riley_texf_b8", "Texture f64", TEXFLOAT_RENDER, None, 8)
    # These calls intentionally remain valid when the uint campaign has not
    # completed: ``draw_series`` renders an explicit no-data panel instead.
    written.extend(figure_texture_convergence("exp1_fig3_riley_texu8_b8", "Texture u8", TEXUINT_RENDER, 8, 8))
    written.extend(figure_texture_convergence("exp1_fig4_riley_texu12_b8", "Texture u12", TEXUINT_RENDER, 12, 8))
    written.extend(figure_texture_convergence("exp1_fig5_riley_texf_b12", "Texture f64", TEXFLOAT_RENDER, None, 12))
    return written


def figure_affine_function_difference_maps() -> list[Path]:
    """Fig. 6: signed 8-bit affine Eggbox errors for selected Riley SSAA."""
    case, frame = "pt42_cam32_q9_aff", 3
    reference, _ = analytic_reference(case, frame)
    levels = (1, 2, 4, 8, 64, 256)
    figure, axes = make_figure(FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    differences: list[np.ndarray | None] = []
    for samples in levels:
        path = FUNC_RENDER / case / f"ss{samples}_f" / f"image_c00_f{frame:02d}.npy"
        if path.is_file():
            image = load_normalised(path)
            if image.shape == reference.shape:
                differences.append(
                    quantise_camera(image, 8).astype(np.float64)
                    - quantise_camera(reference, 8).astype(np.float64)
                )
                continue
        differences.append(None)
    scale = max((float(np.max(np.abs(value))) for value in differences if value is not None), default=1.0)
    images = []
    for index, (samples, difference) in enumerate(zip(levels, differences, strict=True)):
        axis = axes.flat[index]
        axis.set_title(f"{panel_prefix(index)} SSAA={samples}", fontsize=FONT_SIZE_PT)
        if difference is None:
            annotate_no_data(axis, "No completed render data", font_size=FONT_SIZE_PT)
            continue
        images.append(axis.imshow(difference, cmap="gray", vmin=-scale, vmax=scale, interpolation="nearest", origin="upper"))
        axis.set_xlabel("Pixel x", fontsize=AXIS_LABEL_FONT_SIZE_PT)
        axis.set_ylabel("Pixel y", fontsize=AXIS_LABEL_FONT_SIZE_PT)
    if images:
        colourbar = figure.colorbar(images[0], ax=list(axes.flat), shrink=0.86, pad=0.02)
        colourbar.set_label("Digitised difference [bits]", fontsize=AXIS_LABEL_FONT_SIZE_PT)
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(figure, PAPER_OUTPUT_DIR / "exp1_fig6_affine_eggbox_difference_maps", PAPER_FORMATS, PAPER_DPI)


def remove_superseded_figures() -> None:
    """Remove only previously generated Exp1 paper figures no longer used."""
    stems = (
        "exp1_fig1_1_rigid_eggbox_function_shaders",
        "exp1_fig1_2_affine_eggbox_function_shaders",
        "exp1_fig2_riley_textures_pt42_cam32_q9_rig",
        "exp1_fig3_riley_textures_pt42_cam32_q9_aff",
        "exp1_fig4_riley_textures_pt42_cam32_q9_qsadd",
    )
    for stem in stems:
        for extension in (*PAPER_FORMATS, "tex"):
            (PAPER_OUTPUT_DIR / stem).with_suffix(f".{extension}").unlink(missing_ok=True)


def write_tex_preview() -> list[Path]:
    """Write editable figure blocks and compile a minimal A4 preview article."""
    stems = exp1_figure_stems()
    blocks: list[Path] = []
    for stem in stems:
        block = PAPER_OUTPUT_DIR / f"{stem}.tex"
        block.write_text(
            "\\begin{figure}[p]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n"
            f"  \\caption{{{FIGURE_CAPTIONS[stem]}}}\n"
            f"  \\label{{{FIGURE_LABELS[stem]}}}\n"
            "\\end{figure}\n",
            encoding="utf-8",
        )
        blocks.append(block)
    article = PAPER_OUTPUT_DIR / "article.tex"
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
        cwd=PAPER_OUTPUT_DIR,
        check=True,
    )
    return [article, *blocks, PAPER_OUTPUT_DIR / "article.pdf"]


def main() -> None:
    remove_superseded_figures()
    written = figure_function_shaders()
    written.extend(generate_texture_figures())
    written.extend(figure_affine_function_difference_maps())
    written.extend(write_tex_preview())
    print("Wrote paper figures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
