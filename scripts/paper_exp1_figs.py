#!/usr/bin/env python3
"""Create the four journal-ready Experiment 1 convergence figures.

The script computes digitised metrics directly from canonical floating-point
renders.  TIFF previews are deliberately not inputs, so each figure uses the
same digitisation definition as the analysis scripts.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from modules.exp_common_analysis import image_error_metrics
from modules.paperfigs import (
    add_figure_legend, annotate_no_data, finish_axis,
    make_figure, paper_output_directories, save_figure, texture_os_style,
    write_latex_preview,
)
from paperfiglabels import (
    LABEL_DIGITISED_RMSE, LABEL_MAX_DIGITISED_ERROR, LABEL_DIGITISED_DIFF,
    TITLE_UNDEFORMED, TITLE_RIGID_03PX, TITLE_AFFINE_03PX,
    LABEL_PIXEL_X, LABEL_PIXEL_Y, LABEL_NO_DATA, PANEL_PREFIX_TEMPLATE,
    TITLE_REFERENCE_TEMPLATE, TITLE_PANEL_CASE_REFERENCE_TEMPLATE,
    TITLE_PANEL_PX_SS_TEMPLATE, TITLE_PANEL_PX_SS_TEX_OS_TEMPLATE,
    TITLE_EXP1_TEXTURE_ROW_F64_U8, TITLE_EXP1_TEXTURE_ROW_U8_U8,
    TITLE_EXP1_TEXTURE_ROW_F64_U12, TITLE_EXP1_TEXTURE_ROW_U12_U12,
    LABEL_TEX_OS_TEMPLATE, TITLE_TEXTURE_CONVERGENCE_PANEL_TEMPLATE,
    LABEL_GRID2D_METHOD_TEMPLATE, LABEL_RILEY_RECT_TEMPLATE,
    INTERPOLATOR_LABELS, TITLE_EXP1_DIAGONAL_PANEL_TEMPLATE,
    LABEL_DIAGONAL_ANALYTIC_TEMPLATE, LABEL_DIAGONAL_H2_TEMPLATE,
)
from modules.render_outputs import quantise_camera
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT, DIFFERENCE_CMAP,
    LAYOUT_LINE_2X3, LAYOUT_LINE_2X3_EXP1_FIG2, LAYOUT_IMAGE_1X3,
    LAYOUT_IMAGE_MATRIX, FONT_SIZE_PT,
    LEGEND_FONT_SIZE_PT, LINE_WIDTH_PT, MARKER_SIZE_PT, PAPER_DPI,
    PAPER_FORMATS,
    PAPER_FRAME, PAPER_OUTPUT_DIR,
    PAPER_TEXTURE_INTERPOLATOR, TICK_FONT_SIZE_PT,
    EXP1_FIG2_DIFF_SSAA_LEVELS, EXP2_DIFF_SSAA_LEVELS,
    EXP2_DIFF_OVERSAMPLES, EXP1_DIFF_FUNC_CASE, EXP1_DIFF_FUNC_FRAME,
    EXP1_DIFF_TEX_CASE, EXP1_DIFF_TEX_FRAME, EXP1_FIG2_DIFF_LIMIT_BITS,
    EXP1_FIG2_THIRD_COLUMN_TITLE_X, EXP1_FIG5_DIFF_LIMIT_BITS,
    DIFFERENCE_MATRIX_COLORBAR_FRACTION, DIFFERENCE_MATRIX_COLORBAR_ASPECT,
    DIFFERENCE_MATRIX_COLORBAR_SHRINK, DIFFERENCE_MATRIX_COLORBAR_PAD,
    LINE_COLOURS, PAPER_MAIN_TEXTURE_INTERPOLATORS,
    PAPER_DIAGONAL_ANALYTIC_COLOUR, PAPER_DIAGONAL_H2_COLOUR,
    PAPER_DIAGONAL_INTERPOLATOR_MARKERS,
)

OUT = Path("out")
GRID_SUMMARY = OUT / "exp1_analysis" / "grid2d_uvs" / "summary.csv"
GRID_RENDER = OUT / "exp1_grid2d_render_uvs"
FUNC_RENDER = OUT / "exp1_riley_render_func_uvs"
TEXFLOAT_RENDER = OUT / "exp1_riley_render_texf"
TEXUINT_RENDER = OUT / "exp1_riley_render_texu"
TEXTURE_STUDIES = (
    ("pt42_cam32_q9_rig", 0, TITLE_UNDEFORMED),
    ("pt42_cam32_q9_rig", 3, TITLE_RIGID_03PX),
    ("pt42_cam32_q9_aff", 3, TITLE_AFFINE_03PX),
)
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
    return PANEL_PREFIX_TEMPLATE.format(letter=chr(ord("a") + index))


def display_reference(value: str) -> str:
    """Turn analysis-internal reference tokens into article-facing labels."""
    if value.strip().lower() in {"analytic", "analytic:0", "analytic reference"}:
        return "Analytic"
    return value.replace("_", " ")


def load_normalised(path: Path, bit_depth: int | None = None) -> np.ndarray:
    image = np.asarray(np.load(path), dtype=np.float64)
    if image.size and np.nanmax(np.abs(image)) > 1.0 + 1e-12:
        scale = float(2**bit_depth - 1) if bit_depth is not None else 255.0
        image /= scale
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
    return load_normalised(path), f"Gauss {order}"


def grid_metric_series(case: str, metric: str, frame: int = PAPER_FRAME) -> tuple[list[Series], str]:
    if not GRID_SUMMARY.is_file():
        return [], "No grid summary"
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    ref = "Analytic"
    with GRID_SUMMARY.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Case"] == case and int(row["Frame"]) == frame and row["Method"] != "analytic":
                groups[(row["Method"], int(row["BitDepth"]))].append(row); ref = row["Reference"]
    colours = {"gauss": LINE_COLOURS[0], "rect": LINE_COLOURS[1]}; markers = {"gauss": "s", "rect": "o"}
    return [Series(LABEL_GRID2D_METHOD_TEMPLATE.format(method=method.title(), bit_depth=bits), tuple(int(round(float(r["Samples"]) ** .5)) for r in sorted(rows, key=lambda r: float(r["Samples"]))), tuple(float(r[metric]) for r in sorted(rows, key=lambda r: float(r["Samples"]))), bits, colours[method], markers[method], "-" if bits <= 8 else "--") for (method, bits), rows in sorted(groups.items())], ref


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
    return [Series(LABEL_RILEY_RECT_TEMPLATE.format(bit_depth=bits), tuple(x for x, _ in sorted(points)), tuple(y for _, y in sorted(points)), bits, LINE_COLOURS[2], "^", "-" if bits <= 8 else "--") for bits, points in sorted(groups.items())]


def texture_series(
    case: str, root: Path, *, source_bits: int | None, camera_bits: int,
    metric: str, frame: int, interpolator: str | None = None,
) -> list[Series]:
    """Collect one f64/quantised-texture OS family for a camera bit depth."""
    reference, _ = analytic_reference(case, frame)
    interpolator = interpolator or PAPER_TEXTURE_INTERPOLATOR
    pattern = re.compile(r"ss(\d+)_(?:b(\d+)_)?os(\d+)(?:_f)?")
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for case_dir in root.glob(f"{case}_{interpolator}"):
        for directory in case_dir.iterdir():
            match = pattern.fullmatch(directory.name)
            path = directory / f"image_c00_f{frame:02d}.npy"
            if not match or not path.is_file():
                continue
            texture_bits = int(match.group(2)) if match.group(2) else None
            if source_bits is not None and texture_bits != source_bits:
                continue
            image = load_normalised(path, texture_bits)
            if image.shape == reference.shape:
                grouped[int(match.group(3))].append((
                    int(match.group(1)),
                    image_error_metrics(
                        image, reference, camera_bits, quantise_camera,
                    )[metric],
                ))
    return [
        Series(
            LABEL_TEX_OS_TEMPLATE.format(osamp=osamp),
            tuple(x for x, _ in sorted(points)),
            tuple(y for _, y in sorted(points)), camera_bits,
            *texture_os_style(osamp),
        )
        for osamp, points in sorted(grouped.items())
    ]


def diagonal_reference_series(
    case: str, root: Path, *, source_bits: int | None, camera_bits: int,
    frame: int,
) -> list[Series]:
    """Compare each diagonal texture render with analytic and 2x references."""
    analytic, _ = analytic_reference(case, frame)
    expression = re.compile(r"ss(\d+)_(?:b(\d+)_)?os(\d+)(?:_f)?")
    result: list[Series] = []
    for interpolator in PAPER_MAIN_TEXTURE_INTERPOLATORS:
        images: dict[int, np.ndarray] = {}
        for case_dir in root.glob(f"{case}_{interpolator}"):
            for directory in case_dir.iterdir():
                match = expression.fullmatch(directory.name)
                path = directory / f"image_c00_f{frame:02d}.npy"
                if not match or not path.is_file():
                    continue
                texture_bits = int(match.group(2)) if match.group(2) else None
                if texture_bits != source_bits:
                    continue
                ssaa, osamp = int(match.group(1)), int(match.group(3))
                if ssaa == osamp:
                    images[ssaa] = load_normalised(path, texture_bits)
        if not images:
            continue
        marker = PAPER_DIAGONAL_INTERPOLATOR_MARKERS[interpolator]
        name = INTERPOLATOR_LABELS[interpolator]
        analytic_points = [
            (level, image_error_metrics(
                image, analytic, camera_bits, quantise_camera,
            )["e_b"])
            for level, image in sorted(images.items())
            if image.shape == analytic.shape
        ]
        if analytic_points:
            result.append(Series(
                LABEL_DIAGONAL_ANALYTIC_TEMPLATE.format(interpolator=name),
                tuple(level for level, _ in analytic_points),
                tuple(value for _, value in analytic_points), camera_bits,
                PAPER_DIAGONAL_ANALYTIC_COLOUR, marker, "-",
            ))
        h2_points = [
            (level, image_error_metrics(
                image, images[2 * level], camera_bits, quantise_camera,
            )["e_b"])
            for level, image in sorted(images.items())
            if 2 * level in images and images[2 * level].shape == image.shape
        ]
        if h2_points:
            result.append(Series(
                LABEL_DIAGONAL_H2_TEMPLATE.format(interpolator=name),
                tuple(level for level, _ in h2_points),
                tuple(value for _, value in h2_points), camera_bits,
                PAPER_DIAGONAL_H2_COLOUR, marker, "--",
            ))
    return result


def draw_series(axis, series: list[Series], metric_label: str, reference: str) -> list[Line2D]:
    handles = []
    all_samples: list[int] = []; all_values: list[float] = []
    for item in series:
        linewidth, markersize = LINE_WIDTH_PT, MARKER_SIZE_PT
        axis.plot(item.samples, item.values, color=item.colour, marker=item.marker, linestyle=item.linestyle, linewidth=linewidth, markersize=markersize)
        handles.append(Line2D([], [], color=item.colour, marker=item.marker, linestyle=item.linestyle, linewidth=linewidth, markersize=markersize, label=item.label))
        all_samples.extend(item.samples); all_values.extend(item.values)
    if not series:
        annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
        return handles
    finish_axis(axis, title=TITLE_REFERENCE_TEMPLATE.format(reference=reference), samples=all_samples, bit_depth=max(item.bit_depth for item in series), values=all_values, ylabel=metric_label, title_font_size=FONT_SIZE_PT, axis_label_font_size=AXIS_LABEL_FONT_SIZE_PT)
    return handles





def figure_function_shaders() -> list[Path]:
    """Figure 1: RMSE and maximum digitised-error convergence."""
    studies = (
        ("pt42_cam32_q9_rig", 0, TITLE_UNDEFORMED),
        ("pt42_cam32_q9_rig", 3, TITLE_RIGID_03PX),
        ("pt42_cam32_q9_aff", 3, TITLE_AFFINE_03PX),
    )
    written = []

    figure, axes = make_figure(
        LAYOUT_LINE_2X3, rows=2, columns=3,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles: list[Line2D] = []
    metrics = (
        ("e_b", LABEL_DIGITISED_RMSE),
        ("max_eb", LABEL_MAX_DIGITISED_ERROR),
    )
    for row, (metric, metric_label) in enumerate(metrics):
        for col, (case, frame, subtitle) in enumerate(studies):
            reference, ref_label = analytic_reference(case, frame)
            grid, summary_ref = grid_metric_series(case, metric, frame)
            bits = {item.bit_depth for item in grid}
            data = grid + function_series(
                case, reference, metric, bits, frame,
            )
            label = display_reference(summary_ref or ref_label)
            handles.extend(draw_series(
                axes[row, col], data, metric_label, label,
            ))
            axes[row, col].set_title(
                TITLE_PANEL_CASE_REFERENCE_TEMPLATE.format(
                    panel=panel_prefix(row * len(studies) + col),
                    case=subtitle, reference=label,
                ),
                fontsize=FONT_SIZE_PT,
            )
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(
        figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=3,
    )
    written.extend(save_figure(
        figure,
        PAPER_OUTPUT_DIR / "exp1_fig1_eggbox_function_shaders_rmse",
        PAPER_FORMATS, PAPER_DPI,
    ))

    return written


def figure_texture_convergence() -> list[Path]:
    """Figure 2: u12 Riley texture convergence."""
    figures = (
        (
            "exp1_fig2_riley_textures_b12_rmse",
            (
                (TITLE_EXP1_TEXTURE_ROW_F64_U12, TEXFLOAT_RENDER, None, 12),
                (TITLE_EXP1_TEXTURE_ROW_U12_U12, TEXUINT_RENDER, 12, 12),
            ),
        ),
    )
    written: list[Path] = []
    for stem, rows_config in figures:
        figure, axes = make_figure(
            LAYOUT_LINE_2X3_EXP1_FIG2, rows=2, columns=3,
            tick_font_size=TICK_FONT_SIZE_PT,
        )
        handles: list[Line2D] = []
        for row, (row_label, root, source_bits, camera_bits) in enumerate(rows_config):
            for column, (case, frame, deformation) in enumerate(TEXTURE_STUDIES):
                _, reference = analytic_reference(case, frame)
                data = texture_series(
                    case, root, source_bits=source_bits,
                    camera_bits=camera_bits, metric="e_b", frame=frame,
                )
                handles.extend(draw_series(
                    axes[row, column], data, LABEL_DIGITISED_RMSE, reference,
                ))
                axes[row, column].set_title(
                    TITLE_TEXTURE_CONVERGENCE_PANEL_TEMPLATE.format(
                        panel=panel_prefix(row * len(TEXTURE_STUDIES) + column),
                        texture=row_label, deformation=deformation,
                        reference=reference,
                    ),
                    fontsize=FONT_SIZE_PT,
                )
                if stem == "exp1_fig2_riley_textures_b12_rmse" and column == 2:
                    axes[row, column].title.set_x(
                        EXP1_FIG2_THIRD_COLUMN_TITLE_X
                    )
        unique = {handle.get_label(): handle for handle in handles}
        add_figure_legend(
            figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT,
            columns=5,
        )
        written.extend(save_figure(
            figure, PAPER_OUTPUT_DIR / stem, PAPER_FORMATS, PAPER_DPI,
        ))
    return written


def figure_diagonal_refinement() -> list[Path]:
    """Figure 3: u12 diagonal refinement for all selected interpolants."""
    rows_config = (
        (TITLE_EXP1_TEXTURE_ROW_F64_U12, TEXFLOAT_RENDER, None, 12),
        (TITLE_EXP1_TEXTURE_ROW_U12_U12, TEXUINT_RENDER, 12, 12),
    )
    figure, axes = make_figure(
        LAYOUT_LINE_2X3, rows=2, columns=3,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles: list[Line2D] = []
    for row, (texture, root, source_bits, bits) in enumerate(rows_config):
        for column, (case, frame, deformation) in enumerate(TEXTURE_STUDIES):
            data = diagonal_reference_series(
                case, root, source_bits=source_bits, camera_bits=bits,
                frame=frame,
            )
            handles.extend(draw_series(
                axes[row, column], data, LABEL_DIGITISED_RMSE, "",
            ))
            axes[row, column].set_title(
                TITLE_EXP1_DIAGONAL_PANEL_TEMPLATE.format(
                    panel=panel_prefix(row * len(TEXTURE_STUDIES) + column),
                    texture=texture, deformation=deformation,
                ),
                fontsize=FONT_SIZE_PT,
            )
            if column == 2:
                axes[row, column].title.set_x(EXP1_FIG2_THIRD_COLUMN_TITLE_X)
    add_figure_legend(
        figure, list({item.get_label(): item for item in handles}.values()),
        font_size=LEGEND_FONT_SIZE_PT, columns=3,
    )
    return save_figure(
        figure,
        PAPER_OUTPUT_DIR / "exp1_fig3_riley_textures_u12_diagonal_refinement_rmse",
        PAPER_FORMATS, PAPER_DPI,
    )


def exp1_figure_stems() -> tuple[str, ...]:
    return (
        "exp1_fig1_eggbox_function_shaders_rmse",
        "exp1_fig2_riley_textures_b12_rmse",
        "exp1_fig3_riley_textures_u12_diagonal_refinement_rmse",
    )


def figure_rigid_function_difference_maps(
    *, output_dir: Path = PAPER_OUTPUT_DIR,
    stem: str = "exp1_fig2_rigid_eggbox_difference_maps",
) -> list[Path]:
    """Create signed 8-bit rigid Eggbox difference maps."""
    case, frame = EXP1_DIFF_FUNC_CASE, EXP1_DIFF_FUNC_FRAME
    reference, _ = analytic_reference(case, frame)
    levels = EXP1_FIG2_DIFF_SSAA_LEVELS
    figure, axes = make_figure(
        LAYOUT_IMAGE_1X3, rows=1, columns=len(levels),
        tick_font_size=TICK_FONT_SIZE_PT,
    )
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
    scale = EXP1_FIG2_DIFF_LIMIT_BITS
    images = []
    for index, (samples, difference) in enumerate(zip(levels, differences, strict=True)):
        axis = axes.flat[index]
        axis.set_title(
            TITLE_PANEL_PX_SS_TEMPLATE.format(
                panel=panel_prefix(index), ssaa=samples,
            ), fontsize=FONT_SIZE_PT,
        )
        if difference is None:
            annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
            continue
        images.append(axis.imshow(
            difference, cmap=DIFFERENCE_CMAP, vmin=-scale, vmax=scale,
            interpolation="nearest", origin="upper"
        ))
        axis.set_xlabel(LABEL_PIXEL_X, fontsize=AXIS_LABEL_FONT_SIZE_PT)
        axis.set_ylabel(LABEL_PIXEL_Y, fontsize=AXIS_LABEL_FONT_SIZE_PT)
    if images:
        colourbar = figure.colorbar(
            images[0], ax=list(axes.flat), shrink=0.86, pad=0.02
        )
        colourbar.set_label(
            LABEL_DIGITISED_DIFF, fontsize=AXIS_LABEL_FONT_SIZE_PT
        )
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(
        figure,
        output_dir / stem,
        PAPER_FORMATS, PAPER_DPI,
    )


def figure_texture_difference_maps(
    *, output_dir: Path = PAPER_OUTPUT_DIR,
    stem: str = "exp1_fig5_riley_texf_difference_maps",
) -> list[Path]:
    """Create selected signed 8-bit texture-shader difference maps."""
    case, frame = EXP1_DIFF_TEX_CASE, EXP1_DIFF_TEX_FRAME
    reference_image, _ = analytic_reference(case, frame)
    ssaa_levels = EXP2_DIFF_SSAA_LEVELS
    oversamples = EXP2_DIFF_OVERSAMPLES
    root = TEXFLOAT_RENDER / f"{case}_{PAPER_TEXTURE_INTERPOLATOR}"
    # Never hard-code the grid: this figure follows the selected SSAA/OS
    # matrix exactly, so changing either list cannot leave blank panels.
    rows, columns = len(ssaa_levels), len(oversamples)
    figure, axes = make_figure(
        LAYOUT_IMAGE_MATRIX, rows=rows, columns=columns,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    differences: dict[tuple[int, int], np.ndarray | None] = {}
    for ssaa in ssaa_levels:
        for oversamp in oversamples:
            found_path = None
            for suffix in ("", "_f"):
                path = (
                    root
                    / f"ss{ssaa}_os{oversamp}{suffix}"
                    / f"image_c00_f{frame:02d}.npy"
                )
                if path.is_file():
                    found_path = path
                    break
            if found_path is not None:
                image = load_normalised(found_path)
                if image.shape == reference_image.shape:
                    differences[(ssaa, oversamp)] = (
                        quantise_camera(image, 8).astype(np.float64)
                        - quantise_camera(reference_image, 8).astype(np.float64)
                    )
                    continue
            differences[(ssaa, oversamp)] = None
    scale = EXP1_FIG5_DIFF_LIMIT_BITS
    images = []
    for row, ssaa in enumerate(ssaa_levels):
        for column, oversamp in enumerate(oversamples):
            axis = axes[row, column]
            difference = differences[(ssaa, oversamp)]
            axis.set_title(
                TITLE_PANEL_PX_SS_TEX_OS_TEMPLATE.format(
                    panel=panel_prefix(row * len(oversamples) + column),
                    ssaa=ssaa, osamp=oversamp,
                ),
                fontsize=FONT_SIZE_PT,
            )
            if difference is None:
                annotate_no_data(
                    axis, LABEL_NO_DATA,
                    font_size=FONT_SIZE_PT,
                )
                continue
            images.append(axis.imshow(
                difference, cmap=DIFFERENCE_CMAP, vmin=-scale, vmax=scale,
                interpolation="nearest", origin="upper",
            ))
            if row == len(ssaa_levels) - 1:
                axis.set_xlabel(LABEL_PIXEL_X, fontsize=AXIS_LABEL_FONT_SIZE_PT)
            if column == 0:
                axis.set_ylabel(LABEL_PIXEL_Y, fontsize=AXIS_LABEL_FONT_SIZE_PT)
    if images:
        colourbar = figure.colorbar(
            images[0], ax=list(axes.flat),
            fraction=DIFFERENCE_MATRIX_COLORBAR_FRACTION,
            aspect=DIFFERENCE_MATRIX_COLORBAR_ASPECT,
            shrink=DIFFERENCE_MATRIX_COLORBAR_SHRINK,
            pad=DIFFERENCE_MATRIX_COLORBAR_PAD,
        )
        colourbar.set_label(
            LABEL_DIGITISED_DIFF, fontsize=AXIS_LABEL_FONT_SIZE_PT,
        )
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(
        figure,
        output_dir / stem,
        PAPER_FORMATS, PAPER_DPI,
    )


def remove_superseded_figures() -> None:
    """Remove only previously generated Exp1 paper figures no longer used."""
    stems = (
        # Difference maps are supplementary-only as of the current paper
        # layout.  Remove stale main-paper copies when figures are rebuilt.
        "exp1_fig2_rigid_eggbox_difference_maps",
        "exp1_fig5_riley_texf_difference_maps",
        "exp1_fig3_riley_textures_b8_rmse",
        "exp1_fig4_riley_textures_b12_rmse",
        "exp1_fig1_1_rigid_eggbox_function_shaders",
        "exp1_fig1_2_affine_eggbox_function_shaders",
        "exp1_fig2_riley_textures_pt42_cam32_q9_rig",
        "exp1_fig3_riley_textures_pt42_cam32_q9_aff",
        "exp1_fig4_riley_textures_pt42_cam32_q9_qsadd",
        "exp1_fig2_riley_texf_b8_rmse",
        "exp1_fig2_riley_texf_b8_max_eb",
        "exp1_fig3_riley_texu8_b8_rmse",
        "exp1_fig3_riley_texu8_b8_max_eb",
        "exp1_fig4_riley_texu12_b12_rmse",
        "exp1_fig4_riley_texu12_b12_max_eb",
        "exp1_fig5_riley_texf_b12_rmse",
        "exp1_fig5_riley_texf_b12_max_eb",
        "exp2_fig3_riley_texf_rmse",
        "exp2_fig3_riley_texf_max_eb",
        "exp1_fig2_riley_textures_rmse",
        "exp1_fig2_riley_textures_max_eb",
        "exp2_fig3_riley_textures_disk_rmse",
        "exp2_fig3_riley_textures_disk_max_eb",
        "exp2_fig3_riley_textures_gauss_rmse",
        "exp2_fig3_riley_textures_gauss_max_eb",
        "exp1_fig4_affine_eggbox_difference_maps",
        "exp1_fig2_affine_eggbox_difference_maps",
        "exp1_fig3_riley_texf_difference_maps",
    )
    for output_dir in paper_output_directories():
        for stem in stems:
            for extension in (*PAPER_FORMATS, "tex"):
                (output_dir / stem).with_suffix(f".{extension}").unlink(missing_ok=True)


def write_tex_preview() -> list[Path]:
    """Write editable figure blocks and compile a minimal A4 preview article."""
    return write_latex_preview(exp1_figure_stems())


def main() -> None:
    remove_superseded_figures()
    written = figure_function_shaders()
    written.extend(figure_texture_convergence())
    written.extend(figure_diagonal_refinement())
    written.extend(write_tex_preview())
    print("Wrote paper figures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
