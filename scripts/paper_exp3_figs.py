#!/usr/bin/env python3
"""Create the journal-ready Experiment 3 paper figures.

The script computes and plots displacement convergence, subpixel bias,
independent Tex-OS/Px-SS refinement paths, self-convergence, and
finite-star frequency-based diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from exp3_analysis_conv_rmse import (
    EDGE_EXCLUSION_DIC,
    EDGE_EXCLUSION_GRID,
    discover_dic,
    discover_grid,
    load_dic,
    load_grid,
    select_dic_reference,
)
from modules.paperfigs import (
    add_figure_legend,
    annotate_no_data,
    make_figure,
    prepare_panel_titles,
    save_figure,
    set_sample_axis,
    write_latex_preview,
)
from paperfiglabels import (
    LABEL_HORIZ_COORD_PX,
    LABEL_COLUMN_RMSE_PX,
    LABEL_DISP_RMSE_PX,
    LABEL_SELF_CONV_RMSE,
    LABEL_NO_REFERENCE,
    LABEL_RILEY_TEMPLATE,
    LABEL_ANALYTIC_REFERENCE,
    LABEL_RILEY_BSPLINE,
    LABEL_RILEY_CATMULL_ROM,
    METHOD_DIC,
    METHOD_GRID,
    INTERPOLATOR_LABELS,
    LABEL_MISSING_METHOD_RECORDS_TEMPLATE,
    LABEL_ERROR_LOADING_METHOD_DATA_TEMPLATE,
    LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE,
    TITLE_FIG1_A,
    TITLE_FIG1_B,
    TITLE_FIG1_C,
    TITLE_FIG1_D,
    LABEL_FIG1_RILEY_TEMPLATE,
    LABEL_IMPOSED_TRANSLATION_PX,
    LABEL_MEAN_BIAS_PX,
    LABEL_PX_INTEGRATION,
    LABEL_TEX_OVERSAMPLING,
    LABEL_REF_LEVEL_OS_SS,
    LABEL_RMSE_AT_03PX,
    TITLE_FIG2_PANEL_TEMPLATE,
    TITLE_FIG2_A_ACTION,
    TITLE_FIG2_A_CONSTRAINT,
    TITLE_FIG2_B_ACTION,
    TITLE_FIG2_B_CONSTRAINT,
    TITLE_FIG2_C_ACTION,
    TITLE_FIG2_C_CONSTRAINT,
    LABEL_FIG2_A_TEMPLATE,
    LABEL_FIG2_B_TEMPLATE,
    LABEL_FIG2_C_TEMPLATE,
    TITLE_FIG3_PANEL_TEMPLATE,
    TITLE_FIG3_H2_PANEL_TEMPLATE,
    LABEL_FIG3_DIC_TEMPLATE,
    LABEL_FIG3_GRID_TEMPLATE,
    TITLE_FIG4_A_TEMPLATE,
    TITLE_FIG4_B_TEMPLATE,
    TITLE_FIG4_C_TEMPLATE,
    TITLE_FIG4_D,
    TITLE_FIG4_E_TEMPLATE,
    TITLE_FIG4_F_TEMPLATE,
    TITLE_FIG4_G_TEMPLATE,
    TITLE_FIG4_H,
    LABEL_FIG4_5_PROFILE_TEMPLATE,
)
from paperparams import (
    COLORBAR_FONT_SIZE_PT,
    EXP3_AFFINE_CASE,
    EXP3_BIT_DEPTH,
    EXP3_CHIRP_CASE,
    EXP3_FIG1_ZOOM_BIAS_YLIM,
    EXP3_FIG1_ZOOM_RMSE_YLIM,
    EXP3_FIG2_TEX_INSET_YLIM,
    EXP3_FIG2_TEX_INSET_YTICKS,
    EXP3_FIG4_REFERENCE_OSAMP,
    EXP3_FIG4_REFERENCE_SSAA,
    EXP3_FIG4_MAP_LEVEL,
    EXP3_FIG4_PROFILE_LEVELS,
    EXP3_FIG4_COLORBAR_PAD_FIG,
    EXP3_FIG4_COLORBAR_WIDTH_FIG,
    EXP3_FIG4_FIELD_LIMIT_PX,
    LAYOUT_FIELD_4X2,
    LAYOUT_LINE_1X3,
    LAYOUT_LINE_2X3,
    LAYOUT_LINE_2X2_WIDE,
    LAYOUT_LINE_2X2_WIDE_DETACHED,
    LAYOUT_LINE_2X2_TITLED,
    LINE_WIDTH_PT as EXP3_ANALYTIC_LINE_WIDTH_PT,
    LINE_WIDTH_PT as EXP3_LINE_WIDTH_PT,
    MARKER_SIZE_PT as EXP3_MARKER_SIZE_PT,
    EXP3_RIGID_CASE,
    FONT_SIZE_PT,
    LEGEND_FONT_SIZE_PT,
    LINE_COLOURS,
    PAPER_DPI,
    PAPER_FORMATS,
    PAPER_OUTPUT_DIR,
    TICK_FONT_SIZE_PT,
)

# Colors matching the Experiment 1/2 paper convention
COLOR_BSPLINE = LINE_COLOURS[0]
COLOR_CUBICCM = LINE_COLOURS[1]

INTERPOLATOR_NAMES = INTERPOLATOR_LABELS

# Cases for Figure 1: (interpolator, Px-SS, Tex-OS, colour, linestyle, marker)
# Every requested combination is shown for both texture interpolants.
EXP3_FIG1_CASES = (
    ("cubic_bspline", 1, 1, LINE_COLOURS[0], "-", "o"),
    ("cubic_bspline", 8, 1, LINE_COLOURS[1], "-", "v"),
    ("cubic_bspline", 1, 8, LINE_COLOURS[2], "-", "^"),
    ("cubic_bspline", 8, 8, LINE_COLOURS[3], "-", "s"),
    # Match colour by (Px-SS, Tex-OS); use dashed X/plus traces to
    # distinguish Catmull--Rom from the solid geometric B-spline markers.
    ("cubiccm", 1, 1, LINE_COLOURS[0], "--", "X"),
    ("cubiccm", 8, 1, LINE_COLOURS[1], "--", "+"),
    ("cubiccm", 1, 8, LINE_COLOURS[2], "--", "X"),
    ("cubiccm", 8, 8, LINE_COLOURS[3], "--", "+"),
)

def find_rec(
    records,
    root_pat: str,
    interp: str | None = None,
    ssaa: int | None = None,
    osamp: int | None = None,
    analytic: bool = False,
):
    for r in records:
        if r.analytic != analytic:
            continue
        if root_pat not in r.root:
            continue
        if interp and r.interpolator != interp:
            continue
        if ssaa is not None and r.ssaa != ssaa:
            continue
        if osamp is not None and r.osamp != osamp:
            continue
        return r
    return None


def figure4_profile_cases() -> tuple[tuple[str, int, str, str], ...]:
    """Return the configured diagonal Figure 4 profile render cases.

    A colour identifies the common ``r_px=r_tex`` level, while the line style
    identifies the texture interpolant.  This keeps the two interpolants easy
    to compare at each requested refinement.
    """
    cases: list[tuple[str, int, str, str]] = []
    for interpolator, linestyle in (("cubic_bspline", "-"), ("cubiccm", "--")):
        for index, level in enumerate(EXP3_FIG4_PROFILE_LEVELS):
            cases.append((interpolator, level, LINE_COLOURS[index], linestyle))
    return tuple(cases)


def get_rmse_vs_ref(rec, ref, is_dic=True) -> list[float]:
    rmses = []
    frames = range(11)
    for frame in frames:
        if is_dic:
            ref_data = load_dic(ref, frame)
            rec_data = load_dic(rec, frame)
            if ref_data is None or rec_data is None:
                rmses.append(np.nan)
                continue
            rx, ry, ru, rv = ref_data
            _, _, cu, cv = rec_data
            if ru.shape != cu.shape:
                rmses.append(np.nan)
                continue
            du, dv = cu - ru, cv - rv
            x_min, x_max = rx.min(), rx.max()
            y_min, y_max = ry.min(), ry.max()
            mask = (
                (rx < x_min + EDGE_EXCLUSION_DIC)
                | (rx > x_max - EDGE_EXCLUSION_DIC)
                | (ry < y_min + EDGE_EXCLUSION_DIC)
                | (ry > y_max - EDGE_EXCLUSION_DIC)
            )
        else:
            ref_data = load_grid(ref, frame)
            rec_data = load_grid(rec, frame)
            if ref_data is None or rec_data is None:
                rmses.append(np.nan)
                continue
            ru, rv = ref_data
            cu, cv = rec_data
            if ru.shape != cu.shape:
                rmses.append(np.nan)
                continue
            du, dv = cu - ru, cv - rv
            mask = np.ones(ru.shape, dtype=bool)
            mask[
                EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
                EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
            ] = False

        du = np.where(mask, np.nan, du)
        dv = np.where(mask, np.nan, dv)
        rmse = float(np.sqrt(np.nanmean(du * du + dv * dv)))
        rmses.append(rmse)
    return rmses


def highest_diagonal_references(records, samplers) -> dict[str, object]:
    """Return each sampler's highest completed Px-SS=Tex-OS record."""
    references: dict[str, object] = {}
    for interpolator, *_ in samplers:
        candidates = [
            record for record in records
            if "riley_render_texf" in record.root
            and record.interpolator == interpolator
            and record.ssaa > 0
            and record.ssaa == record.osamp
        ]
        if candidates:
            references[interpolator] = max(
                candidates, key=lambda record: record.ssaa
            )
    return references


def diagonal_reference_label(references: dict[str, object]) -> str:
    """Display the highest available diagonal reference level for a panel."""
    levels = sorted({int(record.ssaa) for record in references.values()})
    if not levels:
        return "none"
    return str(levels[-1])


def generate_figure1(dic_records) -> None:
    """Figure 1: Subpixel bias hides renderer error."""
    fig, axes = make_figure(
        LAYOUT_LINE_2X2_WIDE_DETACHED, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()

    subset = [
        r for r in dic_records
        if r.case == EXP3_RIGID_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    ref, ref_name = select_dic_reference(subset)

    if ref is None:
        annotate_no_data(axes_flat[0], LABEL_NO_REFERENCE, font_size=FONT_SIZE_PT)
        annotate_no_data(axes_flat[1], LABEL_NO_REFERENCE, font_size=FONT_SIZE_PT)
        annotate_no_data(axes_flat[2], LABEL_NO_REFERENCE, font_size=FONT_SIZE_PT)
        annotate_no_data(axes_flat[3], LABEL_NO_REFERENCE, font_size=FONT_SIZE_PT)
    else:
        translations = [frame * 0.1 for frame in range(11)]

        cases_to_plot = [
            (
                find_rec(subset, "analytic", analytic=True),
                LABEL_ANALYTIC_REFERENCE,
                "black",
                "--",
                None,
            ),
        ]
        for interp, ssaa, osamp, col, lstyle, marker in EXP3_FIG1_CASES:
            rec = find_rec(
                subset, "riley_render_texf", interp, ssaa, osamp
            )
            name = INTERPOLATOR_NAMES.get(interp, interp)
            label = LABEL_FIG1_RILEY_TEMPLATE.format(
                name=name, osamp=osamp, ssaa=ssaa
            )
            cases_to_plot.append((rec, label, col, lstyle, marker))

        # Top Row (All Cases): Row 0
        # Panel a: Bias (all cases)
        max_bias_all = 0.0
        for rec, label, col, lstyle, marker in cases_to_plot:
            if rec is None or label == LABEL_ANALYTIC_REFERENCE:
                continue
            biases = []
            for frame in range(11):
                rec_data = load_dic(rec, frame)
                if rec_data is None:
                    biases.append(np.nan)
                    continue
                rx, ry, ru, rv = rec_data
                biases.append(float(np.nanmean(ru)) - translations[frame])
            axes_flat[0].plot(
                translations,
                biases,
                label=label,
                color=col,
                linestyle=lstyle,
                marker=marker,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
            )
            if biases:
                max_bias_all = max(
                    max_bias_all, np.nanmax(np.abs(biases))
                )

        rec_anal, label_anal, col_anal, lstyle_anal, _ = cases_to_plot[0]
        if rec_anal is not None:
            biases = []
            for frame in range(11):
                rec_data = load_dic(rec_anal, frame)
                if rec_data is None:
                    biases.append(np.nan)
                    continue
                rx, ry, ru, rv = rec_data
                biases.append(float(np.nanmean(ru)) - translations[frame])
            axes_flat[0].plot(
                translations,
                biases,
                label=label_anal,
                color=col_anal,
                linestyle=lstyle_anal,
                marker=None,
                linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
            )

        # Panel b: RMSE (all cases)
        anal_ref = find_rec(subset, "analytic", analytic=True)
        max_rmse_all = 0.0
        if anal_ref:
            for rec, label, _, _, _ in cases_to_plot:
                if rec is None:
                    continue
                rmses = get_rmse_vs_ref(rec, anal_ref, is_dic=True)
                if rmses:
                    valid_rmses = [r for r in rmses if np.isfinite(r)]
                    if valid_rmses:
                        val_max = max(valid_rmses)
                        max_rmse_all = max(max_rmse_all, val_max)

            for rec, label, col, lstyle, marker in cases_to_plot:
                if rec is None or label == LABEL_ANALYTIC_REFERENCE:
                    continue
                rmses = get_rmse_vs_ref(rec, anal_ref, is_dic=True)
                if len(rmses) == 11:
                    axes_flat[1].plot(
                        translations,
                        rmses,
                        label=label,
                        color=col,
                        linestyle=lstyle,
                        marker=marker,
                        linewidth=EXP3_LINE_WIDTH_PT,
                        markersize=EXP3_MARKER_SIZE_PT,
                    )

            if rec_anal is not None:
                rmses = get_rmse_vs_ref(rec_anal, anal_ref, is_dic=True)
                if len(rmses) == 11:
                    axes_flat[1].plot(
                        translations,
                        rmses,
                        label=label_anal,
                        color=col_anal,
                        linestyle=lstyle_anal,
                        marker=None,
                        linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
                    )

        # Bottom row: same cases as the top row, using fixed zoom windows.
        # Panel c: Bias (zoomed)
        for rec, label, col, lstyle, marker in cases_to_plot:
            if rec is None or label == LABEL_ANALYTIC_REFERENCE:
                continue
            biases = []
            for frame in range(11):
                rec_data = load_dic(rec, frame)
                if rec_data is None:
                    biases.append(np.nan)
                    continue
                rx, ry, ru, rv = rec_data
                biases.append(float(np.nanmean(ru)) - translations[frame])
            axes_flat[2].plot(
                translations,
                biases,
                label=label,
                color=col,
                linestyle=lstyle,
                marker=marker,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
            )

        if rec_anal is not None:
            biases = []
            for frame in range(11):
                rec_data = load_dic(rec_anal, frame)
                if rec_data is None:
                    biases.append(np.nan)
                    continue
                rx, ry, ru, rv = rec_data
                biases.append(float(np.nanmean(ru)) - translations[frame])
            axes_flat[2].plot(
                translations,
                biases,
                label=label_anal,
                color=col_anal,
                linestyle=lstyle_anal,
                marker=None,
                linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
            )

        # Panel d: RMSE (zoomed)
        if anal_ref:
            for rec, label, col, lstyle, marker in cases_to_plot:
                if rec is None or label == LABEL_ANALYTIC_REFERENCE:
                    continue
                rmses = get_rmse_vs_ref(rec, anal_ref, is_dic=True)
                if len(rmses) == 11:
                    axes_flat[3].plot(
                        translations,
                        rmses,
                        label=label,
                        color=col,
                        linestyle=lstyle,
                        marker=marker,
                        linewidth=EXP3_LINE_WIDTH_PT,
                        markersize=EXP3_MARKER_SIZE_PT,
                    )

            if rec_anal is not None:
                rmses = get_rmse_vs_ref(rec_anal, anal_ref, is_dic=True)
                if len(rmses) == 11:
                    axes_flat[3].plot(
                        translations,
                        rmses,
                        label=label_anal,
                        color=col_anal,
                        linestyle=lstyle_anal,
                        marker=None,
                        linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
                    )

        # Horizontal dotted zero lines
        axes_flat[0].axhline(
            0.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.55
        )
        axes_flat[2].axhline(
            0.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.55
        )

        # y-limits configuration
        if max_bias_all > 0:
            axes_flat[0].set_ylim(-max_bias_all * 1.15, max_bias_all * 1.15)
        axes_flat[2].set_ylim(*EXP3_FIG1_ZOOM_BIAS_YLIM)
        if max_rmse_all > 0:
            axes_flat[1].set_ylim(bottom=-0.04 * max_rmse_all)
        else:
            axes_flat[1].set_ylim(bottom=-1e-5)
        axes_flat[3].set_ylim(*EXP3_FIG1_ZOOM_RMSE_YLIM)

        # Subplot labeling
        for ax in axes_flat:
            ax.set_xlabel(
                LABEL_IMPOSED_TRANSLATION_PX, fontsize=FONT_SIZE_PT
            )
            ax.grid(True, linestyle=":", alpha=0.6)

        axes_flat[0].set_ylabel(LABEL_MEAN_BIAS_PX, fontsize=FONT_SIZE_PT)
        axes_flat[1].set_ylabel(LABEL_DISP_RMSE_PX, fontsize=FONT_SIZE_PT)
        axes_flat[2].set_ylabel(LABEL_MEAN_BIAS_PX, fontsize=FONT_SIZE_PT)
        axes_flat[3].set_ylabel(LABEL_DISP_RMSE_PX, fontsize=FONT_SIZE_PT)

        axes_flat[0].set_title(TITLE_FIG1_A, fontsize=FONT_SIZE_PT)
        axes_flat[1].set_title(TITLE_FIG1_B, fontsize=FONT_SIZE_PT)
        axes_flat[2].set_title(TITLE_FIG1_C, fontsize=FONT_SIZE_PT)
        axes_flat[3].set_title(TITLE_FIG1_D, fontsize=FONT_SIZE_PT)

        handles = [
            Line2D(
                [0],
                [0],
                color=col,
                linestyle=lstyle,
                marker=marker,
                markersize=EXP3_MARKER_SIZE_PT,
                label=label,
            )
            for _, label, col, lstyle, marker in cases_to_plot
        ]
        add_figure_legend(
            fig,
            handles,
            font_size=LEGEND_FONT_SIZE_PT,
            columns=2,
        )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def generate_figure2(dic_records, grid_records) -> list[Path]:
    """Figure 2: DIC/Grid refinement independence under rigid translation."""
    fig, axes = make_figure(
        LAYOUT_LINE_2X3, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT,
    )
    samplers = [
        ("cubic_bspline", LABEL_RILEY_BSPLINE, COLOR_BSPLINE, "-", "o"),
        ("cubiccm", LABEL_RILEY_CATMULL_ROM, COLOR_CUBICCM, "--", "x"),
        ("line", LABEL_RILEY_TEMPLATE.format(name=INTERPOLATOR_LABELS["line"]),
         LINE_COLOURS[2], ":", "d"),
    ]
    levels = [1, 2, 4, 8, 16, 32, 64, 128]
    panel_specs = (
        (TITLE_FIG2_A_ACTION, TITLE_FIG2_A_CONSTRAINT, "ssaa"),
        (TITLE_FIG2_B_ACTION, TITLE_FIG2_B_CONSTRAINT, "osamp"),
        (TITLE_FIG2_C_ACTION, TITLE_FIG2_C_CONSTRAINT, "diagonal"),
    )
    method_rows = (
        (METHOD_DIC, dic_records, "gausscont", True),
        (METHOD_GRID, grid_records, "eggbox", False),
    )

    for row, (method, records, pattern, is_dic) in enumerate(method_rows):
        subset = [
            record for record in records
            if record.case == EXP3_RIGID_CASE
            and record.pattern == pattern
            and record.bit_depth == EXP3_BIT_DEPTH
        ]
        analytic_reference = find_rec(subset, "analytic", analytic=True)
        for column, (action, constraint, sweep) in enumerate(panel_specs):
            axis = axes[row, column]
            panel = chr(ord("a") + row * len(panel_specs) + column)
            axis.set_title(
                TITLE_FIG2_PANEL_TEMPLATE.format(
                    panel=panel, method=method, action=action,
                    constraint=constraint,
                ),
                fontsize=FONT_SIZE_PT,
            )
            if analytic_reference is None:
                annotate_no_data(axis, LABEL_NO_REFERENCE, font_size=FONT_SIZE_PT)
                continue

            for interpolator, name, colour, linestyle, marker in samplers:
                points: list[tuple[int, float]] = []
                for level in levels:
                    ssaa, osamp = (
                        (level, 1) if sweep == "ssaa" else
                        (1, level) if sweep == "osamp" else
                        (level, level)
                    )
                    record = find_rec(
                        subset, "riley_render_texf", interpolator,
                        ssaa=ssaa, osamp=osamp,
                    )
                    if record is None:
                        continue
                    rmses = get_rmse_vs_ref(
                        record, analytic_reference, is_dic=is_dic,
                    )
                    if len(rmses) > 3 and np.isfinite(rmses[3]):
                        points.append((level, rmses[3]))
                if not points:
                    continue
                x_values, y_values = zip(*points, strict=True)
                axis.plot(
                    x_values, y_values, color=colour, marker=marker,
                    linestyle=linestyle, linewidth=EXP3_LINE_WIDTH_PT,
                    markersize=EXP3_MARKER_SIZE_PT,
                )
                # Show the resolved texture-refinement regime in the fixed
                # pixel-integration and diagonal panels.  The Px-SS-only
                # panels remain uncluttered because their convergence is
                # already clear at full scale.
                if sweep in {"osamp", "diagonal"}:
                    inset_points = [point for point in points if point[0] >= 4]
                    if inset_points:
                        inset = getattr(axis, "inset_ax", None)
                        if inset is None:
                            inset = axis.inset_axes([0.45, 0.45, 0.5, 0.5])
                            axis.inset_ax = inset
                        inset_x, inset_y = zip(*inset_points, strict=True)
                        inset.plot(
                            inset_x, inset_y, color=colour, marker=marker,
                            linestyle=linestyle,
                            linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                            markersize=EXP3_MARKER_SIZE_PT * 0.8,
                        )

            xlabel = (
                LABEL_PX_INTEGRATION if sweep == "ssaa" else
                LABEL_TEX_OVERSAMPLING if sweep == "osamp" else
                LABEL_REF_LEVEL_OS_SS
            )
            set_sample_axis(axis, levels, xlabel, FONT_SIZE_PT)
            axis.grid(True, which="both", linestyle=":", alpha=0.6)
            axis.set_ylim(bottom=0.0)
            if column == 0:
                axis.set_ylabel(LABEL_RMSE_AT_03PX, fontsize=FONT_SIZE_PT)
            inset = getattr(axis, "inset_ax", None)
            if inset is not None:
                set_sample_axis(inset, levels[2:], "", FONT_SIZE_PT - 1)
                inset.grid(True, which="both", linestyle=":", alpha=0.4)
                inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
                if sweep == "osamp":
                    inset.set_ylim(EXP3_FIG2_TEX_INSET_YLIM)
                    inset.set_yticks(EXP3_FIG2_TEX_INSET_YTICKS)
                else:
                    inset.set_ylim(bottom=0.0)

    handles = [
        Line2D(
            [0], [0], color=colour, marker=marker, linestyle=linestyle,
            markersize=EXP3_MARKER_SIZE_PT, label=name,
        )
        for _, name, colour, linestyle, marker in samplers
    ]
    add_figure_legend(
        fig, handles, font_size=LEGEND_FONT_SIZE_PT, columns=3,
    )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def _generate_figure3_with_affine(dic_records, grid_records) -> None:
    """Figure 3: Numerical-reference self-convergence (Rigid & Affine)."""
    fig, axes = make_figure(
        LAYOUT_LINE_2X2_WIDE, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()
    diag_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    samplers = [
        ("cubic_bspline", INTERPOLATOR_LABELS["cubic_bspline"], COLOR_BSPLINE, "-", "o"),
        ("cubiccm", INTERPOLATOR_LABELS["cubiccm"], COLOR_CUBICCM, "--", "x"),
        ("line", INTERPOLATOR_LABELS["line"], LINE_COLOURS[2], ":", "d"),
    ]

    # --- TOP ROW: RIGID TRANSLATION ---
    # 1. Top-Left: DIC Rigid
    dic_subset_rig = [
        r for r in dic_records
        if r.case == EXP3_RIGID_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    dic_refs_rig = highest_diagonal_references(dic_subset_rig, samplers)
    if not dic_refs_rig:
        annotate_no_data(
            axes_flat[0], LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE.format(
                method=METHOD_DIC, deformation="Rigid",
            ), font_size=FONT_SIZE_PT
        )
    else:
        dic_inset = axes_flat[0].inset_axes([0.49, 0.46, 0.47, 0.47])
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for lvl in diag_levels:
                rec = find_rec(
                    dic_subset_rig,
                    "riley_render_texf",
                    interp,
                    ssaa=lvl,
                    osamp=lvl,
                )
                reference = dic_refs_rig.get(interp)
                if rec and reference is not None:
                    rmses = get_rmse_vs_ref(rec, reference, is_dic=True)
                    if rmses and len(rmses) > 3:
                        x_vals.append(lvl)
                        y_vals.append(rmses[3])
            axes_flat[0].plot(
                x_vals,
                y_vals,
                color=col,
                marker=marker,
                linestyle=lstyle,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
                label=LABEL_FIG3_DIC_TEMPLATE.format(name=name),
            )
            selected = [
                (x, y) for x, y in zip(x_vals, y_vals, strict=True) if x >= 4
            ]
            if selected:
                dic_inset.plot(
                    [x for x, _ in selected], [y for _, y in selected],
                    color=col, marker=marker, linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                    markersize=EXP3_MARKER_SIZE_PT * 0.75,
                )

    # 2. Top-Right: Grid Rigid
    grid_subset_rig = [
        r for r in grid_records
        if r.case == EXP3_RIGID_CASE
        and r.pattern == "eggbox"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    grid_refs_rig = highest_diagonal_references(grid_subset_rig, samplers)
    if not grid_refs_rig:
        annotate_no_data(
            axes_flat[1], LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE.format(
                method=METHOD_GRID, deformation="Rigid",
            ), font_size=FONT_SIZE_PT
        )
    else:
        grid_rig_inset = axes_flat[1].inset_axes([0.49, 0.46, 0.47, 0.47])
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for lvl in diag_levels:
                rec = find_rec(
                    grid_subset_rig,
                    "riley_render_texf",
                    interp,
                    ssaa=lvl,
                    osamp=lvl,
                )
                reference = grid_refs_rig.get(interp)
                if rec and reference is not None:
                    rmses = get_rmse_vs_ref(rec, reference, is_dic=False)
                    if rmses and len(rmses) > 3:
                        x_vals.append(lvl)
                        y_vals.append(rmses[3])
            axes_flat[1].plot(
                x_vals,
                y_vals,
                color=col,
                marker=marker,
                linestyle=lstyle,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
                label=LABEL_FIG3_GRID_TEMPLATE.format(name=name),
            )
            selected = [(x, y) for x, y in zip(x_vals, y_vals, strict=True) if x >= 4]
            if selected:
                grid_rig_inset.plot(
                    [x for x, _ in selected], [y for _, y in selected],
                    color=col, marker=marker, linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                    markersize=EXP3_MARKER_SIZE_PT * 0.75,
                )

    # --- BOTTOM ROW: AFFINE DEFORMATION ---
    # 3. Bottom-Left: DIC Affine
    dic_subset_aff = [
        r for r in dic_records
        if r.case == EXP3_AFFINE_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    dic_refs_aff = highest_diagonal_references(dic_subset_aff, samplers)
    if not dic_refs_aff:
        annotate_no_data(
            axes_flat[2], LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE.format(
                method=METHOD_DIC, deformation="Affine",
            ), font_size=FONT_SIZE_PT
        )
    else:
        dic_aff_inset = axes_flat[2].inset_axes([0.49, 0.46, 0.47, 0.47])
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for lvl in diag_levels:
                rec = find_rec(
                    dic_subset_aff,
                    "riley_render_texf",
                    interp,
                    ssaa=lvl,
                    osamp=lvl,
                )
                reference = dic_refs_aff.get(interp)
                if rec and reference is not None:
                    rmses = get_rmse_vs_ref(rec, reference, is_dic=True)
                    if rmses and len(rmses) > 3:
                        x_vals.append(lvl)
                        y_vals.append(rmses[3])
            axes_flat[2].plot(
                x_vals,
                y_vals,
                color=col,
                marker=marker,
                linestyle=lstyle,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
                label=LABEL_FIG3_DIC_TEMPLATE.format(name=name),
            )
            selected = [
                (x, y) for x, y in zip(x_vals, y_vals, strict=True) if x >= 4
            ]
            if selected:
                dic_aff_inset.plot(
                    [x for x, _ in selected], [y for _, y in selected],
                    color=col, marker=marker, linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                    markersize=EXP3_MARKER_SIZE_PT * 0.75,
                )

    # 4. Bottom-Right: Grid Affine
    grid_subset_aff = [
        r for r in grid_records
        if r.case == EXP3_AFFINE_CASE
        and r.pattern == "eggbox"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    grid_refs_aff = highest_diagonal_references(grid_subset_aff, samplers)
    if not grid_refs_aff:
        annotate_no_data(
            axes_flat[3], LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE.format(
                method=METHOD_GRID, deformation="Affine",
            ), font_size=FONT_SIZE_PT
        )
    else:
        grid_aff_inset = axes_flat[3].inset_axes([0.49, 0.46, 0.47, 0.47])
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for lvl in diag_levels:
                rec = find_rec(
                    grid_subset_aff,
                    "riley_render_texf",
                    interp,
                    ssaa=lvl,
                    osamp=lvl,
                )
                reference = grid_refs_aff.get(interp)
                if rec and reference is not None:
                    rmses = get_rmse_vs_ref(rec, reference, is_dic=False)
                    if rmses and len(rmses) > 3:
                        x_vals.append(lvl)
                        y_vals.append(rmses[3])
            axes_flat[3].plot(
                x_vals,
                y_vals,
                color=col,
                marker=marker,
                linestyle=lstyle,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
                label=LABEL_FIG3_GRID_TEMPLATE.format(name=name),
            )
            selected = [(x, y) for x, y in zip(x_vals, y_vals, strict=True) if x >= 4]
            if selected:
                grid_aff_inset.plot(
                    [x for x, _ in selected], [y for _, y in selected],
                    color=col, marker=marker, linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                    markersize=EXP3_MARKER_SIZE_PT * 0.75,
                )

    # Format all axes
    for ax in axes_flat:
        set_sample_axis(
            ax,
            diag_levels,
            LABEL_REF_LEVEL_OS_SS,
            FONT_SIZE_PT,
        )
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(
            LABEL_SELF_CONV_RMSE, fontsize=FONT_SIZE_PT
        )

    # The rigid DIC convergence panel has a useful high-refinement regime that
    # would otherwise be compressed against the lower levels.
    if dic_refs_rig:
        inset_levels = [4, 8, 16, 32, 64, 128]
        set_sample_axis(dic_inset, inset_levels, "", FONT_SIZE_PT - 1)
        dic_inset.grid(True, which="both", linestyle=":", alpha=0.45)
        dic_inset.set_ylim(bottom=0.0)
        dic_inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
    if dic_refs_aff:
        inset_levels = [4, 8, 16, 32, 64, 128]
        set_sample_axis(dic_aff_inset, inset_levels, "", FONT_SIZE_PT - 1)
        dic_aff_inset.grid(True, which="both", linestyle=":", alpha=0.45)
        dic_aff_inset.set_ylim(bottom=0.0)
        dic_aff_inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
    if grid_refs_rig:
        inset_levels = [4, 8, 16, 32, 64, 128]
        set_sample_axis(grid_rig_inset, inset_levels, "", FONT_SIZE_PT - 1)
        grid_rig_inset.grid(True, which="both", linestyle=":", alpha=0.45)
        grid_rig_inset.set_ylim(bottom=0.0)
        grid_rig_inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
    if grid_refs_aff:
        inset_levels = [4, 8, 16, 32, 64, 128]
        set_sample_axis(grid_aff_inset, inset_levels, "", FONT_SIZE_PT - 1)
        grid_aff_inset.grid(True, which="both", linestyle=":", alpha=0.45)
        grid_aff_inset.set_ylim(bottom=0.0)
        grid_aff_inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)

    panel_titles = (
        (axes_flat[0], "a", METHOD_DIC, diagonal_reference_label(dic_refs_rig), "Rigid"),
        (axes_flat[1], "b", METHOD_GRID, diagonal_reference_label(grid_refs_rig), "Rigid"),
        (axes_flat[2], "c", METHOD_DIC, diagonal_reference_label(dic_refs_aff), "Affine"),
        (axes_flat[3], "d", METHOD_GRID, diagonal_reference_label(grid_refs_aff), "Affine"),
    )
    for axis, panel, method, reference, deformation in panel_titles:
        axis.set_title(
            TITLE_FIG3_PANEL_TEMPLATE.format(
                panel=panel, method=method, reference=reference,
                deformation=deformation,
            ),
            fontsize=FONT_SIZE_PT,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=col,
            marker=marker,
            linestyle=lstyle,
            markersize=EXP3_MARKER_SIZE_PT,
            label=LABEL_RILEY_TEMPLATE.format(name=name),
        )
        for _, name, col, lstyle, marker in samplers
    ]
    add_figure_legend(
        fig,
        handles,
        font_size=LEGEND_FONT_SIZE_PT,
        columns=3,
    )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig3_rigid_self_convergence_dic_vs_grid_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def generate_figure3(dic_records, grid_records) -> list[Path]:
    """Figure 3: highest- and sliding-reference self-convergence."""
    figure, axes = make_figure(
        LAYOUT_LINE_2X2_TITLED, rows=2, columns=2,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    diag_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    samplers = [
        ("cubic_bspline", INTERPOLATOR_LABELS["cubic_bspline"], COLOR_BSPLINE, "-", "o"),
        ("cubiccm", INTERPOLATOR_LABELS["cubiccm"], COLOR_CUBICCM, "--", "x"),
        ("line", INTERPOLATOR_LABELS["line"], LINE_COLOURS[2], ":", "d"),
    ]
    panels = (
        (axes[0, 0], dic_records, "gausscont", METHOD_DIC, True, "a", "highest"),
        (axes[0, 1], grid_records, "eggbox", METHOD_GRID, False, "b", "highest"),
        (axes[1, 0], dic_records, "gausscont", METHOD_DIC, True, "c", "sliding"),
        (axes[1, 1], grid_records, "eggbox", METHOD_GRID, False, "d", "sliding"),
    )
    for axis, records, pattern, method, is_dic, panel, reference_mode in panels:
        subset = [
            record for record in records
            if record.case == EXP3_RIGID_CASE
            and record.pattern == pattern
            and record.bit_depth == EXP3_BIT_DEPTH
        ]
        references = highest_diagonal_references(subset, samplers)
        if not references:
            annotate_no_data(
                axis,
                LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE.format(
                    method=method, deformation="Rigid",
                ),
                font_size=FONT_SIZE_PT,
            )
        else:
            inset = axis.inset_axes([0.49, 0.46, 0.47, 0.47])
            for interpolator, name, colour, linestyle, marker in samplers:
                x_values: list[int] = []
                y_values: list[float] = []
                for level in diag_levels:
                    record = find_rec(
                        subset, "riley_render_texf", interpolator,
                        ssaa=level, osamp=level,
                    )
                    if reference_mode == "highest":
                        reference = references.get(interpolator)
                    else:
                        reference = find_rec(
                            subset, "riley_render_texf", interpolator,
                            ssaa=2 * level, osamp=2 * level,
                        )
                    if record is None or reference is None:
                        continue
                    errors = get_rmse_vs_ref(record, reference, is_dic=is_dic)
                    if errors and len(errors) > 3:
                        x_values.append(level)
                        y_values.append(errors[3])
                axis.plot(
                    x_values, y_values, color=colour, marker=marker,
                    linestyle=linestyle, linewidth=EXP3_LINE_WIDTH_PT,
                    markersize=EXP3_MARKER_SIZE_PT,
                    label=(LABEL_FIG3_DIC_TEMPLATE if is_dic else LABEL_FIG3_GRID_TEMPLATE).format(name=name),
                )
                selected = [(x, y) for x, y in zip(x_values, y_values, strict=True) if x >= 4]
                if selected:
                    inset.plot(
                        [x for x, _ in selected], [y for _, y in selected],
                        color=colour, marker=marker, linestyle=linestyle,
                        linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                        markersize=EXP3_MARKER_SIZE_PT * 0.75,
                    )
            inset_levels = [4, 8, 16, 32, 64, 128]
            set_sample_axis(inset, inset_levels, "", FONT_SIZE_PT - 1)
            inset.grid(True, which="both", linestyle=":", alpha=0.45)
            inset.set_ylim(bottom=0.0)
            inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)

        set_sample_axis(axis, diag_levels, LABEL_REF_LEVEL_OS_SS, FONT_SIZE_PT)
        axis.grid(True, which="both", linestyle=":", alpha=0.6)
        axis.set_ylim(bottom=0.0)
        axis.set_ylabel(LABEL_SELF_CONV_RMSE, fontsize=FONT_SIZE_PT)
        if reference_mode == "highest":
            title = TITLE_FIG3_PANEL_TEMPLATE.format(
                panel=panel, method=method,
                reference=diagonal_reference_label(references),
                deformation="Rigid",
            )
        else:
            title = TITLE_FIG3_H2_PANEL_TEMPLATE.format(
                panel=panel, method=method, deformation="Rigid",
            )
        axis.set_title(title, fontsize=FONT_SIZE_PT)

    handles = [
        Line2D(
            [0], [0], color=colour, marker=marker, linestyle=linestyle,
            markersize=EXP3_MARKER_SIZE_PT,
            label=LABEL_RILEY_TEMPLATE.format(name=name),
        )
        for _, name, colour, linestyle, marker in samplers
    ]
    add_figure_legend(
        figure, handles, font_size=LEGEND_FONT_SIZE_PT, columns=3,
    )
    written = save_figure(
        figure,
        Path(PAPER_OUTPUT_DIR) / "exp3_riley_gauss_fig3_rigid_self_convergence_dic_vs_grid_b12",
        PAPER_FORMATS, PAPER_DPI,
    )
    figure.clear()
    return written


def _figure4_data(records, *, is_dic: bool) -> dict[str, object] | None:
    """Load one finite-star method's reference and selected test result."""
    subset = [
        record for record in records
        if record.case == EXP3_CHIRP_CASE
        and record.pattern == ("gausscont" if is_dic else "eggbox")
        and record.bit_depth == EXP3_BIT_DEPTH
    ]
    reference = find_rec(
        subset, "riley_render_texf", "cubic_bspline",
        ssaa=EXP3_FIG4_REFERENCE_SSAA, osamp=EXP3_FIG4_REFERENCE_OSAMP,
    )
    selected = find_rec(
        subset, "riley_render_texf", "cubic_bspline",
        ssaa=EXP3_FIG4_MAP_LEVEL, osamp=EXP3_FIG4_MAP_LEVEL,
    )
    if reference is None or selected is None:
        return None
    ref_data = load_dic(reference, 1) if is_dic else load_grid(reference, 1)
    selected_data = load_dic(selected, 1) if is_dic else load_grid(selected, 1)
    if ref_data is None or selected_data is None:
        return None
    if is_dic:
        x, y, _, ref_v = ref_data
        _, _, _, selected_v = selected_data
        extent = (x.min(), x.max(), y.max(), y.min())
        mask = (
            (x < x.min() + EDGE_EXCLUSION_DIC)
            | (x > x.max() - EDGE_EXCLUSION_DIC)
            | (y < y.min() + EDGE_EXCLUSION_DIC)
            | (y > y.max() - EDGE_EXCLUSION_DIC)
        )
    else:
        _, ref_v = ref_data
        _, selected_v = selected_data
        height, width = ref_v.shape
        x = np.arange(width, dtype=float)
        extent = (0, width, height, 0)
        mask = np.ones(ref_v.shape, dtype=bool)
        mask[
            EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
            EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
        ] = False
    diff_v = selected_v - ref_v
    return {
        "subset": subset, "reference": reference, "selected": selected,
        "ref_v": ref_v, "selected_v": selected_v, "diff_v": diff_v,
        "x": x, "extent": extent, "mask": mask, "is_dic": is_dic,
    }


def generate_figure4(dic_records, grid_records) -> list[Path]:
    """Figure 4: finite-star maps and profiles in one balanced canvas."""
    dic_data = _figure4_data(dic_records, is_dic=True)
    grid_data = _figure4_data(grid_records, is_dic=False)
    fig, axes = make_figure(
        LAYOUT_FIELD_4X2, rows=4, columns=2, tick_font_size=TICK_FONT_SIZE_PT,
    )
    gridspec = axes[0, 0].get_subplotspec().get_gridspec()
    # A finite-star map is roughly 4:1.  Give the profile row enough vertical
    # room for a conventional line plot without leaving map-row whitespace.
    # Taller map rows let the equal-aspect finite-star fields use nearly the
    # same total panel width as the profiles, once the attached colourbar and
    # its left-side tick labels are included.
    gridspec.set_height_ratios((0.95, 0.95, 0.95, 1.55))
    pending_colourbars: list[tuple[object, object]] = []
    method_specs = (
        (dic_data, 0, METHOD_DIC, TITLE_FIG4_A_TEMPLATE,
         TITLE_FIG4_B_TEMPLATE, TITLE_FIG4_C_TEMPLATE),
        (grid_data, 1, METHOD_GRID, TITLE_FIG4_E_TEMPLATE,
         TITLE_FIG4_F_TEMPLATE, TITLE_FIG4_G_TEMPLATE),
    )
    for data, column, method, ref_title, selected_title, diff_title in method_specs:
        if data is None:
            for row in range(3):
                annotate_no_data(
                    axes[row, column],
                    LABEL_MISSING_METHOD_RECORDS_TEMPLATE.format(method=method),
                    font_size=FONT_SIZE_PT,
                )
            continue
        selected = data["selected"]
        name = INTERPOLATOR_NAMES.get(selected.interpolator, selected.interpolator)
        reference = data["reference"]
        reference_name = INTERPOLATOR_NAMES.get(
            reference.interpolator, reference.interpolator,
        )
        values = (
            (data["ref_v"], ref_title.format(
                name=reference_name, method=method,
                ref_level=EXP3_FIG4_REFERENCE_SSAA,
            ),
             EXP3_FIG4_FIELD_LIMIT_PX),
            (data["selected_v"], selected_title.format(
                name=name, osamp=selected.osamp or 1, ssaa=selected.ssaa or 1,
                method=method, ref_level=EXP3_FIG4_REFERENCE_SSAA,
            ), EXP3_FIG4_FIELD_LIMIT_PX),
            (data["diff_v"], diff_title.format(
                name=name, method=method, osamp=selected.osamp or 1,
                ssaa=selected.ssaa or 1,
            ), max(float(np.nanpercentile(
                np.abs(np.where(data["mask"], np.nan, data["diff_v"])), 95)), 1e-5)),
        )
        for row, (field, title, limit) in enumerate(values):
            kwargs = {"cmap": "coolwarm", "aspect": "equal"}
            if limit is not None:
                kwargs.update(vmin=-limit, vmax=limit)
            image = axes[row, column].imshow(field, extent=data["extent"], **kwargs)
            pending_colourbars.append((image, axes[row, column]))
            axes[row, column].set_title(title, fontsize=FONT_SIZE_PT)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    handles: list[Line2D] = []
    for data, column, method, title in (
        (dic_data, 0, METHOD_DIC, TITLE_FIG4_D),
        (grid_data, 1, METHOD_GRID, TITLE_FIG4_H),
    ):
        axis = axes[3, column]
        if data is None:
            annotate_no_data(
                axis, LABEL_MISSING_METHOD_RECORDS_TEMPLATE.format(method=method),
                font_size=FONT_SIZE_PT,
            )
            continue
        max_rmse = 0.0
        for interpolator, level, colour, linestyle in figure4_profile_cases():
            record = find_rec(
                data["subset"], "riley_render_texf", interpolator,
                ssaa=level, osamp=level,
            )
            if record is None:
                continue
            current = load_dic(record, 1) if data["is_dic"] else load_grid(record, 1)
            if current is None:
                continue
            current_v = current[3] if data["is_dic"] else current[1]
            difference = np.where(data["mask"], np.nan, current_v - data["ref_v"])
            if data["is_dic"]:
                x_values = np.unique(data["x"][np.isfinite(data["x"])])
                errors = [
                    float(np.sqrt(np.nanmean(difference[data["x"] == value] ** 2)))
                    if np.any(np.isfinite(difference[data["x"] == value])) else np.nan
                    for value in x_values
                ]
            else:
                x_values = data["x"]
                errors = [
                    float(np.sqrt(np.nanmean(difference[:, index] ** 2)))
                    if np.any(np.isfinite(difference[:, index])) else np.nan
                    for index in range(difference.shape[1])
                ]
            max_rmse = max(max_rmse, max((value for value in errors if np.isfinite(value)), default=0.0))
            name = INTERPOLATOR_NAMES.get(interpolator, interpolator)
            label = LABEL_FIG4_5_PROFILE_TEMPLATE.format(
                name=name, osamp=level, ssaa=level,
            )
            axis.plot(x_values, errors, color=colour, linestyle=linestyle,
                      linewidth=EXP3_LINE_WIDTH_PT, label=label)
            handles.append(Line2D([], [], color=colour, linestyle=linestyle,
                                  linewidth=EXP3_LINE_WIDTH_PT, label=label))
        axis.set_title(title.format(method=method), fontsize=FONT_SIZE_PT)
        axis.set_xlabel(LABEL_HORIZ_COORD_PX, fontsize=FONT_SIZE_PT)
        axis.set_ylabel(LABEL_COLUMN_RMSE_PX, fontsize=FONT_SIZE_PT)
        axis.tick_params(labelsize=TICK_FONT_SIZE_PT)
        axis.set_ylim(bottom=-0.04 * max_rmse if max_rmse else -1e-5)
        axis.grid(True, linestyle=":", alpha=0.6)
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(
        fig, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=3, auto_position=False,
    )

    # Constrained layout now knows the final axes and legend positions.  Add
    # each colourbar in the unused left-hand allowance of its own panel,
    # immediately adjacent to the map.  Automatic ``Figure.colorbar`` left a
    # large layout gap here because it treated the bar as a separate column.
    # Match the export title state before reading any geometry.  Otherwise
    # ``save_figure`` would change two-line title heights after this point.
    prepare_panel_titles(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colourbar_links: list[tuple[object, object, object]] = []
    for image, axis in pending_colourbars:
        # ``get_window_extent`` returns the final active equal-aspect axes
        # box, rather than the taller GridSpec allocation.  This locks the
        # colourbar top and bottom to the rendered map borders exactly.
        position = axis.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        colour_axis = fig.add_axes([
            position.x0 - EXP3_FIG4_COLORBAR_PAD_FIG
            - EXP3_FIG4_COLORBAR_WIDTH_FIG,
            position.y0,
            EXP3_FIG4_COLORBAR_WIDTH_FIG,
            position.height,
        ])
        colour_axis.set_in_layout(False)
        colourbar = fig.colorbar(image, cax=colour_axis)
        colourbar.ax.yaxis.set_ticks_position("left")
        colourbar.ax.yaxis.set_label_position("left")
        colourbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE_PT)
        colourbar_links.append((image, axis, colour_axis))

    # Creating the external axes can trigger one final aspect/layout pass.
    # Re-read the active map boxes afterwards, then snap each colourbar to its
    # final top, bottom, and fixed left-hand separation before export.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for image, axis, colour_axis in colourbar_links:
        # The image bounding box is the authoritative field extent.  An
        # equal-aspect axes can contain a small allocation margin above/below
        # this box, which is why using the axes box alone left bars too high.
        position = image.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        colour_axis.set_position([
            position.x0 - EXP3_FIG4_COLORBAR_PAD_FIG
            - EXP3_FIG4_COLORBAR_WIDTH_FIG,
            position.y0,
            EXP3_FIG4_COLORBAR_WIDTH_FIG,
            position.height,
        ])
    written = save_figure(
        fig, Path(PAPER_OUTPUT_DIR) / "exp3_riley_gauss_fig4_finite_star_combined_b12",
        PAPER_FORMATS, PAPER_DPI,
    )
    fig.clear()
    return written


def figure_stems() -> tuple[str, ...]:
    return (
        "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12",
        "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12",
        "exp3_riley_gauss_fig3_rigid_self_convergence_dic_vs_grid_b12",
        "exp3_riley_gauss_fig4_finite_star_combined_b12",
    )


def generate_figures() -> list[Path]:
    print("Discovering records...")
    dic_records = discover_dic()
    grid_records = discover_grid()
    print("Generating Figure 1 (Rigid Subpixel Bias/RMSE)...")
    written = generate_figure1(dic_records)
    print("Generating Figure 2 (Px-SS vs. Tex-OS Refinement)...")
    written.extend(generate_figure2(dic_records, grid_records))
    print("Generating Figure 3 (Rigid Self-Convergence)...")
    written.extend(generate_figure3(dic_records, grid_records))
    print("Generating Figure 4 (Combined Finite-star Case)...")
    written.extend(generate_figure4(dic_records, grid_records))
    return written


def main() -> None:
    written = generate_figures()
    print("Writing LaTeX previews...")
    write_latex_preview(figure_stems())
    print(
        "All Experiment 3 figures and latex previews generated successfully."
    )


if __name__ == "__main__":
    main()
