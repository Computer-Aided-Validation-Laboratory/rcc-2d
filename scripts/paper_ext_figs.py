#!/usr/bin/env python3
"""Supplementary convergence figures, kept separate from the journal article.

The journal figures compare every render with an analytic/highest reference.
This module supplements them with max-code-error and mismatch-fraction views,
and with local h/2 comparisons.  An h/2 point is compared only with the next
available factor-of-two refinement, never with a global finest reference.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from matplotlib.lines import Line2D

import paper_exp1_figs as exp1
import paper_exp2_figs as exp2
import paper_exp3_figs as exp3
from exp3_analysis_conv_rmse import discover_dic, discover_grid
from modules.exp_common_analysis import image_error_metrics
from modules.paperfigs import (
    add_figure_legend, annotate_no_data, finish_axis, make_figure,
    save_figure, set_sample_axis, texture_os_style,
)
from modules.render_outputs import quantise_camera
from paperfiglabels import (
    LABEL_DIGITISED_RMSE, LABEL_MAX_DIGITISED_ERROR,
    LABEL_NO_DATA, LABEL_DISP_RMSE_PX,
    LABEL_AXIS_REFINEMENT_LEVEL,
    TITLE_H2_DIAGONAL, TITLE_H2_DISPLACEMENT, TITLE_H2_PX_SS,
    TITLE_EXT_DIAGONAL_TEXTURE_PANEL_TEMPLATE,
    LABEL_EXT_ANALYTIC_REFERENCE, LABEL_EXT_H2_DIAGONAL_REFERENCE,
)
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT, FONT_SIZE_PT, LINE_WIDTH_PT, MARKER_SIZE_PT,
    LAYOUT_LINE_1X3, LAYOUT_LINE_2X2_BALANCED,
    LAYOUT_LINE_2X3, LEGEND_FONT_SIZE_PT, LINE_COLOURS, PAPER_DPI,
    PAPER_EXT_OUTPUT_DIR, PAPER_FORMATS, TICK_FONT_SIZE_PT,
    PAPER_EXT_INSET_BOUNDS,
    PAPER_EXT_INSET_MIN_LEVEL,
    PAPER_TEXTURE_INTERPOLATOR,
    EXP2_DIFF_TEX_CASE, EXP2_DIFF_TEX_FRAME,
    EXP2_FIG7_DIFF_LIMIT_BITS, EXP2_FIG8_DIFF_LIMIT_BITS,
)


METRICS = (("max_eb", LABEL_MAX_DIGITISED_ERROR, "max_digitised_error"),)
H2_METRICS = (("e_b", LABEL_DIGITISED_RMSE, "rmse"), *METRICS)
# Supplementary self-convergence uses only simultaneous texture/pixel
# refinement.  The former fixed-OS and fixed-SSAA h/2 figures were removed.
H2_MODES = (("diagonal", TITLE_H2_DIAGONAL, "diagonal"),)

# Filesystem identifiers are intentionally used to discover completed render
# families.  Output stems use the shorter article-facing forms below.
INTERPOLATOR_TOKENS = {
    "line": "line",
    "cubic_bspline": "cubicbs",
    "cubiccm": "cubiccm",
}


def _interpolator_token(interpolator: str) -> str:
    """Return a concise, stable output token for an interpolant."""
    return INTERPOLATOR_TOKENS.get(interpolator, interpolator)


def _exp1_texture_interpolators() -> tuple[str, ...]:
    """Discover Exp. 1 interpolants that have at least one image result."""
    found: set[str] = set()
    expression = re.compile(r"^pt42_cam32_q9_(?:rig|aff|qsadd)_(.+)$")
    for root in (exp1.TEXFLOAT_RENDER, exp1.TEXUINT_RENDER):
        if not root.is_dir():
            continue
        for directory in root.iterdir():
            match = expression.fullmatch(directory.name)
            if match and any(directory.glob("ss*_os*/*image_c00_f*.npy")):
                found.add(match.group(1))
    return tuple(sorted(found, key=lambda value: (_interpolator_token(value), value)))


def _exp2_texture_interpolators() -> tuple[str, ...]:
    """Discover Exp. 2 interpolants that have at least one image result."""
    found: set[str] = set()
    expression = re.compile(
        r"^pt42_cam32_q9_(?:rig|aff|qsadd)_(?:gaussadd|diskadd)_seed3_(.+)$"
    )
    if not exp2.RILEY_TEXF_RENDER.is_dir():
        return ()
    for directory in exp2.RILEY_TEXF_RENDER.iterdir():
        match = expression.fullmatch(directory.name)
        if match and any(directory.glob("ss*_os*/image_c00_f*_clamped.npy")):
            found.add(match.group(1))
    return tuple(sorted(found, key=lambda value: (_interpolator_token(value), value)))


def _save(figure, stem: str) -> list[Path]:
    """Save only to the extension directory; do not create TeX/article files."""
    return save_figure(figure, Path(PAPER_EXT_OUTPUT_DIR) / stem, PAPER_FORMATS, PAPER_DPI)


def _dedupe(handles: list[Line2D]) -> list[Line2D]:
    return list({handle.get_label(): handle for handle in handles}.values())


def _draw(axis, rows, ylabel: str, title: str) -> list[Line2D]:
    """Draw Exp1/2-style rows with the common paper axis treatment."""
    handles: list[Line2D] = []
    samples: list[int] = []
    values: list[float] = []
    for row in rows:
        width, size = LINE_WIDTH_PT, MARKER_SIZE_PT
        axis.plot(row.samples, row.values, color=row.colour, marker=row.marker,
                  linestyle=row.linestyle, linewidth=width, markersize=size)
        handles.append(Line2D([], [], color=row.colour, marker=row.marker,
                              linestyle=row.linestyle, linewidth=width,
                              markersize=size, label=row.label))
        samples.extend(row.samples); values.extend(row.values)
    if not rows:
        annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
        return handles
    finish_axis(axis, title=title, samples=samples,
                bit_depth=max(row.bit_depth for row in rows), values=values,
                ylabel=ylabel, title_font_size=FONT_SIZE_PT,
                axis_label_font_size=AXIS_LABEL_FONT_SIZE_PT)
    return handles


def _finish_displacement_axis(axis, *, title: str, samples: list[int]) -> None:
    """Format a free, linear displacement-RMSE axis for Exp3 extensions.

    ``finish_axis`` is intentionally not used here: it applies the paper's
    zero-inclusive symlog convention for digitised image errors, which is not
    appropriate for physical displacement RMSE in pixels.
    """
    set_sample_axis(axis, samples, LABEL_AXIS_REFINEMENT_LEVEL, AXIS_LABEL_FONT_SIZE_PT)
    axis.set_ylabel(LABEL_DISP_RMSE_PX, fontsize=AXIS_LABEL_FONT_SIZE_PT)
    axis.set_title(title, fontsize=FONT_SIZE_PT)
    axis.grid(True, which="both", linestyle=":", alpha=0.6)


def _h2_rows(
    images: dict[tuple[int, int], np.ndarray], *, bit_depth: int,
    relation: str, label_prefix: str = r"$r_{tex}$",
) -> list[exp1.Series]:
    """Make h/2 image-metric rows for independent or diagonal refinement."""
    groups: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (ssaa, osamp), image in images.items():
        if relation == "pxss":
            target, group, x_value = (2 * ssaa, osamp), osamp, ssaa
        elif relation == "texos":
            target, group, x_value = (ssaa, 2 * osamp), ssaa, osamp
        else:
            if ssaa != osamp:
                continue
            target, group, x_value = (2 * ssaa, 2 * osamp), 0, ssaa
        reference = images.get(target)
        if reference is not None and reference.shape == image.shape:
            groups[group].append((x_value, image, reference))
    rows_by_metric: dict[str, list[exp1.Series]] = {key: [] for key, _, _ in H2_METRICS}
    for group, points in sorted(groups.items()):
        ordered = sorted(points, key=lambda item: item[0])
        if relation == "pxss":
            label = f"{label_prefix}={group}"
        elif relation == "texos":
            label = rf"$r_{{px}}$={group}"
        else:
            label = r"$r_{px}$=$r_{tex}$"
        colour, marker, linestyle = texture_os_style(max(1, group))
        for key, _, _ in H2_METRICS:
            rows_by_metric[key].append(exp1.Series(
                label, tuple(point[0] for point in ordered),
                tuple(image_error_metrics(point[1], point[2], bit_depth, quantise_camera)[key]
                      for point in ordered),
                bit_depth, colour, marker, linestyle,
            ))
    return rows_by_metric


def _reference_rows(
    rows: list[exp1.Series], *, label: str, linestyle: object,
) -> list[exp1.Series]:
    """Clone series with the reference definition encoded in the legend."""
    return [
        exp1.Series(
            label, row.samples, row.values, row.bit_depth, row.colour,
            row.marker, linestyle,
        )
        for row in rows
    ]


def _diagonal_analytic_rows(
    images: dict[tuple[int, int], np.ndarray], reference: np.ndarray,
    *, bit_depth: int, metric: str,
) -> list[exp1.Series]:
    """Return analytic-reference metrics for the diagonal resolution path."""
    points = []
    for (ssaa, osamp), image in images.items():
        if ssaa == osamp and image.shape == reference.shape:
            points.append((ssaa, image_error_metrics(
                image, reference, bit_depth, quantise_camera,
            )[metric]))
    if not points:
        return []
    ordered = sorted(points)
    return [exp1.Series(
        LABEL_EXT_ANALYTIC_REFERENCE,
        tuple(level for level, _ in ordered),
        tuple(value for _, value in ordered), bit_depth,
        LINE_COLOURS[0], "o", "-",
    )]


def _h2_function_rows(case: str, frame: int, bit_depths: set[int]):
    """h/2 rows for Exp1 function shaders (Px-SS is the only resolution)."""
    methods: list[tuple[str, str, str, str, dict[int, np.ndarray]]] = []
    for method, colour, marker in (("gauss", LINE_COLOURS[0], "s"), ("rect", LINE_COLOURS[1], "o")):
        paths: dict[int, np.ndarray] = {}
        pattern = re.compile(rf"targ_px\d+_int_{method}_param_(\d+)_frame{frame:02d}\.npy")
        for path in (exp1.GRID_RENDER / case).glob(f"*{method}*frame{frame:02d}.npy"):
            match = pattern.fullmatch(path.name)
            if match:
                paths[int(match.group(1))] = exp1.load_normalised(path)
        methods.append((f"Grid2D {method.title()}", colour, marker, "-", paths))
    paths = {}
    for directory in (exp1.FUNC_RENDER / case).glob("ss*_f"):
        match = re.fullmatch(r"ss(\d+)_f", directory.name)
        path = directory / f"image_c00_f{frame:02d}.npy"
        if match and path.is_file():
            paths[int(match.group(1))] = exp1.load_normalised(path)
    methods.append(("Riley Func", LINE_COLOURS[2], "^", "-", paths))
    result = {key: [] for key, _, _ in H2_METRICS}
    for label, colour, marker, linestyle, images in methods:
        for bits in sorted(bit_depths):
            points = []
            for ssaa, image in sorted(images.items()):
                reference = images.get(2 * ssaa)
                if reference is not None and reference.shape == image.shape:
                    points.append((ssaa, image_error_metrics(image, reference, bits, quantise_camera)))
            for key, _, _ in H2_METRICS:
                if points:
                    result[key].append(exp1.Series(
                        f"{label}, {bits}bit", tuple(x for x, _ in points),
                        tuple(values[key] for _, values in points), bits,
                        colour, marker, linestyle,
                    ))
    return result


def _texture_images(
    case: str, root: Path, source_bits: int | None, frame: int,
    interpolator: str,
) -> dict[tuple[int, int], np.ndarray]:
    images = {}
    expression = re.compile(r"ss(\d+)_(?:b(\d+)_)?os(\d+)(?:_f)?")
    for case_dir in root.glob(f"{case}_{interpolator}"):
        for directory in case_dir.iterdir():
            match = expression.fullmatch(directory.name)
            path = directory / f"image_c00_f{frame:02d}.npy"
            if not match or not path.is_file():
                continue
            texture_bits = int(match.group(2)) if match.group(2) else None
            if texture_bits != source_bits:
                continue
            images[(int(match.group(1)), int(match.group(3)))] = exp1.load_normalised(path, texture_bits)
    return images


def _exp2_texture_images(
    case: str, pattern: str, frame: int, interpolator: str,
) -> dict[tuple[int, int], np.ndarray]:
    root = exp2.RILEY_TEXF_RENDER / f"{case}_{pattern}_seed3_{interpolator}"
    images = {}
    if not root.is_dir():
        return images
    expression = re.compile(r"ss(\d+)_os(\d+)")
    for directory in root.iterdir():
        match = expression.fullmatch(directory.name)
        path = directory / f"image_c00_f{frame:02d}_clamped.npy"
        if match and path.is_file():
            images[(int(match.group(1)), int(match.group(2)))] = exp2._load(path)
    return images


def _plot_exp1_function(metric: str, ylabel: str, token: str, h2: bool) -> list[Path]:
    figure, axes = make_figure(LAYOUT_LINE_1X3, rows=1, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    handles = []
    for col, (case, frame, deformation) in enumerate((
        ("pt42_cam32_q9_rig", 0, "Undeformed"),
        ("pt42_cam32_q9_rig", 3, "Rigid 0.3px"),
        ("pt42_cam32_q9_aff", 3, "Affine 0.3px"),
    )):
        if h2:
            baseline, _ = exp1.grid_metric_series(case, "e_b", frame)
            rows = _h2_function_rows(case, frame, {row.bit_depth for row in baseline})[metric]
            reference = TITLE_H2_PX_SS
        else:
            reference_image, ref = exp1.analytic_reference(case, frame)
            grid, summary_ref = exp1.grid_metric_series(case, metric, frame)
            rows = grid + exp1.function_series(case, reference_image, metric, {row.bit_depth for row in grid}, frame)
            reference = exp1.display_reference(summary_ref or ref)
        handles.extend(_draw(axes[0, col], rows, ylabel, f"{exp1.panel_prefix(col)} {deformation}\nRef: {reference}"))
    add_figure_legend(figure, _dedupe(handles), font_size=LEGEND_FONT_SIZE_PT, columns=4)
    return _save(figure, f"ext_exp1_fig1_function_{'h2_' if h2 else ''}{token}")


def _plot_exp1_texture(
    metric: str, ylabel: str, token: str, h2: bool, *, interpolator: str,
    relation: str | None = None,
) -> list[Path]:
    figures = (
        ("fig3_tex_b8", (("Texture f64, u8", exp1.TEXFLOAT_RENDER, None, 8), ("Texture u8, u8", exp1.TEXUINT_RENDER, 8, 8))),
        ("fig4_tex_b12", (("Texture f64, u12", exp1.TEXFLOAT_RENDER, None, 12), ("Texture u12, u12", exp1.TEXUINT_RENDER, 12, 12))),
    )
    written = []
    for stem, rows_config in figures:
        figure, axes = make_figure(LAYOUT_LINE_2X3, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
        handles = []
        for row, (texture, root, source_bits, bits) in enumerate(rows_config):
            for col, (case, frame, deformation) in enumerate(exp1.TEXTURE_STUDIES):
                if h2:
                    images = _texture_images(
                            case, root, source_bits, frame, interpolator,
                    )
                    diagonal_rows = _h2_rows(
                        images, bit_depth=bits, relation=relation,
                    )[metric]
                    analytic_reference, _ = exp1.analytic_reference(case, frame)
                    rows = _diagonal_analytic_rows(
                        images, analytic_reference, bit_depth=bits,
                        metric=metric,
                    ) + _reference_rows(
                        diagonal_rows,
                        label=LABEL_EXT_H2_DIAGONAL_REFERENCE,
                        linestyle="--",
                    )
                    title = TITLE_EXT_DIAGONAL_TEXTURE_PANEL_TEMPLATE.format(
                        panel=exp1.panel_prefix(row * 3 + col),
                        texture=texture, deformation=deformation,
                    )
                else:
                    rows = exp1.texture_series(
                        case, root, source_bits=source_bits, camera_bits=bits,
                        metric=metric, frame=frame, interpolator=interpolator,
                    )
                    _, ref = exp1.analytic_reference(case, frame)
                    title = (
                        f"{exp1.panel_prefix(row * 3 + col)} {texture}\n"
                        f"{deformation}, Ref: {ref}"
                    )
                handles.extend(_draw(axes[row, col], rows, ylabel, title))
        add_figure_legend(figure, _dedupe(handles), font_size=LEGEND_FONT_SIZE_PT, columns=4)
        written.extend(_save(
            figure,
            f"ext_exp1_{stem}_{_interpolator_token(interpolator)}_"
            f"{'h2_' + relation + '_' if h2 else ''}{token}",
        ))
    return written


def _exp2_speck_h2(case: str, pattern: str, frame: int, bits: int):
    result = {key: [] for key, _, _ in H2_METRICS}
    for method, colour, marker in (("gauss", LINE_COLOURS[0], "s"), ("rect", LINE_COLOURS[1], "o")):
        images = {}
        for found_method, param, directory in exp2._runs(case, pattern):
            path = exp2._image_path(directory, found_method, param, frame)
            if found_method == method and path.is_file():
                images[(param, 1)] = exp2._load(path)
        rows = _h2_rows(images, bit_depth=bits, relation="pxss", label_prefix=r"$r_{px}$")
        for key, values in rows.items():
            for value in values:
                result[key].append(exp1.Series(f"Speck2D {method.title()}, {bits}bit", value.samples, value.values, bits, colour, marker, "-"))
    return result


def _plot_exp2_speck(metric: str, ylabel: str, token: str, h2: bool) -> list[Path]:
    figure, axes = make_figure(LAYOUT_LINE_2X2_BALANCED, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT)
    handles = []
    for row, (pattern, name) in enumerate((("gaussadd", "Gauss"), ("diskadd", "Disk"))):
        for col, (case, frame, deformation) in enumerate(exp2.CASES):
            if h2:
                rows = _exp2_speck_h2(case, pattern, frame, 12)[metric]
                ref = TITLE_H2_PX_SS
            else:
                rows, reference = exp2.speck_series(case, pattern, frame, metric)
                ref = exp2.display_speck_reference(reference)
            handles.extend(_draw(axes[row, col], rows, ylabel,
                f"{exp2.panel_prefix(row * 2 + col)} {name} Speckle, {deformation}\nRef: {ref}"))
    add_figure_legend(figure, _dedupe(handles), font_size=LEGEND_FONT_SIZE_PT, columns=4)
    return _save(figure, f"ext_exp2_fig1_speck2d_{'h2_' if h2 else ''}{token}")


def _plot_exp2_tex(
    metric: str, ylabel: str, token: str, h2: bool, *, interpolator: str,
    relation: str | None = None,
) -> list[Path]:
    figure, axes = make_figure(LAYOUT_LINE_2X2_BALANCED, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT)
    handles = []
    for row, (pattern, name) in enumerate((("gaussadd", "Gauss"), ("diskadd", "Disk"))):
        for col, (case, frame, deformation) in enumerate(exp2.CASES):
            if h2:
                images = _exp2_texture_images(
                    case, pattern, frame, interpolator,
                )
                diagonal_rows = _h2_rows(
                    images,
                    bit_depth=12, relation=relation,
                )[metric]
                analytic_reference, _ = exp2.reference(case, pattern, frame)
                rows = _diagonal_analytic_rows(
                    images, analytic_reference, bit_depth=12, metric=metric,
                ) + _reference_rows(
                    diagonal_rows,
                    label=LABEL_EXT_H2_DIAGONAL_REFERENCE,
                    linestyle="--",
                )
                title = TITLE_EXT_DIAGONAL_TEXTURE_PANEL_TEMPLATE.format(
                    panel=exp2.panel_prefix(row * 2 + col),
                    texture=f"{name} Speckle", deformation=deformation,
                )
            else:
                rows, ref = exp2.texf_series(
                    case, pattern, frame, metric, 12,
                    interpolator=interpolator,
                )
                title = (
                    f"{exp2.panel_prefix(row * 2 + col)} {name} Speckle, "
                    f"{deformation}\nRef: {ref}"
                )
            handles.extend(_draw(axes[row, col], rows, ylabel, title))
    add_figure_legend(figure, _dedupe(handles), font_size=LEGEND_FONT_SIZE_PT, columns=4)
    return _save(
        figure,
        f"ext_exp2_fig2_texf_{_interpolator_token(interpolator)}_"
        f"{'h2_' + relation + '_' if h2 else ''}{token}",
    )


def _h2_displacement(rec, records, relation: str, is_dic: bool) -> float | None:
    if relation == "pxss":
        reference = exp3.find_rec(records, "riley_render_texf", rec.interpolator, 2 * rec.ssaa, rec.osamp)
    elif relation == "texos":
        reference = exp3.find_rec(records, "riley_render_texf", rec.interpolator, rec.ssaa, 2 * rec.osamp)
    else:
        if rec.ssaa != rec.osamp:
            return None
        reference = exp3.find_rec(records, "riley_render_texf", rec.interpolator, 2 * rec.ssaa, 2 * rec.osamp)
    if reference is None:
        return None
    errors = exp3.get_rmse_vs_ref(rec, reference, is_dic=is_dic)
    return errors[3] if len(errors) > 3 and np.isfinite(errors[3]) else None


def _plot_exp3_h2_figure2(records) -> list[Path]:
    figure, axes = make_figure(
        LAYOUT_LINE_1X3, rows=1, columns=1, tick_font_size=TICK_FONT_SIZE_PT,
    )
    subset = [r for r in records if r.case == exp3.EXP3_RIGID_CASE and r.pattern == "gausscont" and r.bit_depth == exp3.EXP3_BIT_DEPTH]
    samplers = (("cubic_bspline", "B-spline", LINE_COLOURS[0], "-", "o"), ("cubiccm", "Catmull-Rom", LINE_COLOURS[1], "--", "x"), ("line", "Linear", LINE_COLOURS[2], ":", "d"))
    configs = (("diagonal", r"$r_{px}$=$r_{tex}$", TITLE_H2_DIAGONAL),)
    handles = []
    for axis, (relation, xlabel, ref) in zip(axes.flat, configs, strict=True):
        samples = []
        values = []
        inset = axis.inset_axes(PAPER_EXT_INSET_BOUNDS)
        inset_samples: list[int] = []
        for interp, name, colour, style, marker in samplers:
            points = []
            for rec in subset:
                if rec.interpolator != interp or rec.analytic:
                    continue
                if relation == "pxss" and rec.osamp != 1:
                    continue
                if relation == "texos" and rec.ssaa != 1:
                    continue
                value = _h2_displacement(rec, subset, relation, True)
                x = rec.ssaa if relation != "texos" else rec.osamp
                if value is not None:
                    points.append((x, value))
            points = sorted(set(points))
            if points:
                axis.plot(*zip(*points), color=colour, marker=marker, linestyle=style, linewidth=LINE_WIDTH_PT, markersize=MARKER_SIZE_PT)
                inset_points = [point for point in points if point[0] >= PAPER_EXT_INSET_MIN_LEVEL]
                if inset_points:
                    inset.plot(*zip(*inset_points), color=colour, marker=marker,
                               linestyle=style, linewidth=LINE_WIDTH_PT * 0.8,
                               markersize=MARKER_SIZE_PT * 0.75)
                    inset_samples.extend(point[0] for point in inset_points)
                handles.append(Line2D([], [], color=colour, marker=marker, linestyle=style, label=f"Riley {name}"))
                samples.extend(x for x, _ in points); values.extend(y for _, y in points)
        if samples:
            _finish_displacement_axis(axis, title=f"Ref: {ref}", samples=samples)
            axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE_PT)
        else:
            annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
        if inset_samples:
            set_sample_axis(inset, sorted(set(inset_samples)), "", TICK_FONT_SIZE_PT - 1)
            inset.grid(True, which="both", linestyle=":", alpha=0.45)
            inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
        else:
            inset.remove()
    add_figure_legend(figure, _dedupe(handles), font_size=LEGEND_FONT_SIZE_PT, columns=3)
    return _save(figure, "ext_exp3_fig2_h2_diagonal_displacement_rmse")


def _plot_main_paper_difference_maps() -> list[Path]:
    """Write the former main-paper image-difference maps to ``paper_ext``."""
    written = exp1.figure_rigid_function_difference_maps(
        output_dir=Path(PAPER_EXT_OUTPUT_DIR),
        stem="ext_exp1_fig2_rigid_eggbox_difference_maps",
    )
    written.extend(exp1.figure_texture_difference_maps(
        output_dir=Path(PAPER_EXT_OUTPUT_DIR),
        stem="ext_exp1_fig5_riley_texf_difference_maps",
    ))
    written.extend(exp2.figure_texf_difference_maps(
        "diskadd", "ext_exp2_fig3_riley_texf_disk_difference_maps",
        EXP2_FIG7_DIFF_LIMIT_BITS, output_dir=Path(PAPER_EXT_OUTPUT_DIR),
    ))
    written.extend(exp2.figure_texf_difference_maps(
        "gaussadd", "ext_exp2_fig4_riley_texf_gauss_difference_maps",
        EXP2_FIG8_DIFF_LIMIT_BITS, output_dir=Path(PAPER_EXT_OUTPUT_DIR),
    ))
    return written


def _plot_exp3_h2_figure3(dic_records, grid_records) -> list[Path]:
    figure, axes = make_figure(LAYOUT_LINE_2X2_BALANCED, rows=1, columns=2, tick_font_size=TICK_FONT_SIZE_PT)
    samplers = (("cubic_bspline", "B-spline", LINE_COLOURS[0], "-", "o"), ("cubiccm", "Catmull-Rom", LINE_COLOURS[1], "--", "x"), ("line", "Linear", LINE_COLOURS[2], ":", "d"))
    panels = ((axes[0, 0], dic_records, "gausscont", "DIC", True), (axes[0, 1], grid_records, "eggbox", "Grid method", False))
    handles = []
    for axis, records, pattern, method, is_dic in panels:
        subset = [r for r in records if r.case == exp3.EXP3_RIGID_CASE and r.pattern == pattern and r.bit_depth == exp3.EXP3_BIT_DEPTH]
        samples = []; values = []
        inset = axis.inset_axes(PAPER_EXT_INSET_BOUNDS)
        inset_samples: list[int] = []
        for interp, name, colour, style, marker in samplers:
            points = []
            for rec in subset:
                if rec.interpolator != interp or rec.analytic or rec.ssaa != rec.osamp:
                    continue
                value = _h2_displacement(rec, subset, "diagonal", is_dic)
                if value is not None:
                    points.append((rec.ssaa, value))
            points = sorted(set(points))
            if points:
                axis.plot(*zip(*points), color=colour, marker=marker, linestyle=style, linewidth=LINE_WIDTH_PT, markersize=MARKER_SIZE_PT)
                inset_points = [point for point in points if point[0] >= PAPER_EXT_INSET_MIN_LEVEL]
                if inset_points:
                    inset.plot(*zip(*inset_points), color=colour, marker=marker,
                               linestyle=style, linewidth=LINE_WIDTH_PT * 0.8,
                               markersize=MARKER_SIZE_PT * 0.75)
                    inset_samples.extend(point[0] for point in inset_points)
                handles.append(Line2D([], [], color=colour, marker=marker, linestyle=style, label=f"Riley {name}"))
                samples.extend(x for x, _ in points); values.extend(y for _, y in points)
        if samples:
            _finish_displacement_axis(
                axis,
                title=f"{method}, {TITLE_H2_DISPLACEMENT}\nRef: {TITLE_H2_DIAGONAL}",
                samples=samples,
            )
        else:
            annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
        if inset_samples:
            set_sample_axis(inset, sorted(set(inset_samples)), "", TICK_FONT_SIZE_PT - 1)
            inset.grid(True, which="both", linestyle=":", alpha=0.45)
            inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
        else:
            inset.remove()
    add_figure_legend(figure, _dedupe(handles), font_size=LEGEND_FONT_SIZE_PT, columns=3)
    return _save(figure, "ext_exp3_fig3_h2_displacement_rmse")


def generate_figures() -> list[Path]:
    """Generate all extension figures without touching journal article assets."""
    Path(PAPER_EXT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    exp1_interpolators = _exp1_texture_interpolators()
    exp2_interpolators = _exp2_texture_interpolators()
    # The journal paper already includes the selected Catmull--Rom RMSE
    # figures.  Supplementary copies are generated only for the remaining
    # available interpolants.
    ext_exp1_interpolators = tuple(
        value for value in exp1_interpolators
        if value != PAPER_TEXTURE_INTERPOLATOR
    )
    ext_exp2_interpolators = tuple(
        value for value in exp2_interpolators
        if value != PAPER_TEXTURE_INTERPOLATOR
    )
    for interpolator in ext_exp1_interpolators:
        written.extend(_plot_exp1_texture(
            "e_b", LABEL_DIGITISED_RMSE, "rmse", False,
            interpolator=interpolator,
        ))
    for interpolator in ext_exp2_interpolators:
        written.extend(_plot_exp2_tex(
            "e_b", LABEL_DIGITISED_RMSE, "rmse", False,
            interpolator=interpolator,
        ))
    for metric, ylabel, token in METRICS:
        written.extend(_plot_exp1_function(metric, ylabel, token, False))
        for interpolator in exp1_interpolators:
            written.extend(_plot_exp1_texture(
                metric, ylabel, token, False, interpolator=interpolator,
            ))
        written.extend(_plot_exp2_speck(metric, ylabel, token, False))
        for interpolator in exp2_interpolators:
            written.extend(_plot_exp2_tex(
                metric, ylabel, token, False, interpolator=interpolator,
            ))
    for metric, ylabel, token in H2_METRICS:
        written.extend(_plot_exp1_function(metric, ylabel, token, True))
        for _, _, relation in H2_MODES:
            for interpolator in exp1_interpolators:
                written.extend(_plot_exp1_texture(
                    metric, ylabel, token, True, interpolator=interpolator,
                    relation=relation,
                ))
        written.extend(_plot_exp2_speck(metric, ylabel, token, True))
        for _, _, relation in H2_MODES:
            for interpolator in exp2_interpolators:
                written.extend(_plot_exp2_tex(
                    metric, ylabel, token, True, interpolator=interpolator,
                    relation=relation,
                ))
    dic_records = discover_dic()
    grid_records = discover_grid()
    written.extend(_plot_exp3_h2_figure2(dic_records))
    written.extend(_plot_exp3_h2_figure3(dic_records, grid_records))
    written.extend(_plot_main_paper_difference_maps())
    return written


def main() -> None:
    for path in generate_figures():
        print(path)


if __name__ == "__main__":
    main()
