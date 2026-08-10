#!/usr/bin/env python3
"""Create the journal-ready Experiment 3 paper figures.

The script computes and plots displacement convergence, subpixel bias,
independent Tex-OS/Px-SS refinement paths, self-convergence, and
finite-star frequency-based diagnostics.
"""

from __future__ import annotations

import gc
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from exp3_analysis_conv_rmse import (
    EDGE_EXCLUSION_DIC,
    EDGE_EXCLUSION_GRID,
    discover_dic,
    discover_grid,
    filter_candidates,
    filter_grid_candidates,
    load_dic,
    load_grid,
    select_dic_reference,
    select_grid_reference,
)
from modules.paperfigs import (
    add_figure_legend,
    annotate_no_data,
    make_figure,
    save_figure,
    set_sample_axis,
    texture_os_style,
    write_latex_preview,
)
from paperfiglabels import (
    LABEL_HORIZ_COORD_PX,
    LABEL_VERT_COORD_PX,
    LABEL_COLUMN_RMSE_PX,
    LABEL_DISP_RMSE_PX,
    LABEL_DISP_BIAS_PX,
    LABEL_SELF_CONV_RMSE,
    TITLE_ANALYTIC_REF,
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
    TITLE_FIG2_A,
    TITLE_FIG2_B,
    TITLE_FIG2_C,
    LABEL_FIG2_A_TEMPLATE,
    LABEL_FIG2_B_TEMPLATE,
    LABEL_FIG2_C_TEMPLATE,
    TITLE_FIG3_A,
    TITLE_FIG3_B,
    TITLE_FIG3_C,
    TITLE_FIG3_D,
    LABEL_FIG3_DIC_TEMPLATE,
    LABEL_FIG3_GRID_TEMPLATE,
    TITLE_FIG4_A,
    TITLE_FIG4_B_TEMPLATE,
    TITLE_FIG4_C_TEMPLATE,
    TITLE_FIG4_D,
    TITLE_FIG4_E,
    TITLE_FIG4_F_TEMPLATE,
    TITLE_FIG4_G_TEMPLATE,
    TITLE_FIG4_H,
    LABEL_FIG4_5_PROFILE_TEMPLATE,
)
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT,
    COLORBAR_FONT_SIZE_PT,
    DIFFERENCE_CMAP,
    EXP3_AFFINE_CASE,
    EXP3_ANALYTIC_LINE_WIDTH_PT,
    EXP3_BIT_DEPTH,
    EXP3_CHIRP_CASE,
    EXP3_FIGURE1_CM,
    EXP3_FIGURE2_CM,
    EXP3_FIGURE3_CM,
    EXP3_FIGURE4_CM,
    EXP3_LINE_WIDTH_PT,
    EXP3_MARKER_SIZE_PT,
    EXP3_RIGID_CASE,
    FIGURE_CAPTIONS,
    FIGURE_LABELS,
    FONT_SIZE_PT,
    LEGEND_FONT_SIZE_PT,
    PAPER_DPI,
    PAPER_FORMATS,
    PAPER_OUTPUT_DIR,
    TICK_FONT_SIZE_PT,
)

# Colors matching the Experiment 1/2 paper convention
COLOR_BSPLINE = "tab:orange"
COLOR_CUBICCM = "tab:blue"
MARKER_BSPLINE = "^"
MARKER_CUBICCM = "s"

INTERPOLATOR_NAMES = {
    "cubic_bspline": "B-spline",
    "cubiccm": "Catmull-Rom",
    "line": "Linear",
    "lanczos3": "Lanczos",
}

# Cases for Figure 1: (interpolator, OS, SS, color, linestyle, marker)
EXP3_FIG1_CASES = (
    ("cubic_bspline", 1, 1, COLOR_BSPLINE, "-", "o"),
    ("cubic_bspline", 8, 8, "tab:purple", "-", "v"),
    ("cubic_bspline", 32, 1, "tab:cyan", "-", "^"),
    ("cubic_bspline", 32, 32, "tab:green", "-", "s"),
    ("cubiccm", 1, 1, COLOR_CUBICCM, "--", "x"),
    ("cubiccm", 8, 8, "tab:brown", "--", "*"),
    ("cubiccm", 32, 32, "tab:red", "--", "+"),
)

# Profile configurations for Figs 4 & 5: (interpolator, OS, SS, color, linestyle)
EXP3_FIG4_5_PROFILES = (
    ("cubic_bspline", 1, 1, COLOR_BSPLINE, "-"),
    ("cubic_bspline", 4, 4, "tab:green", "-"),
    ("cubic_bspline", 32, 32, "tab:purple", "-"),
    ("cubiccm", 1, 1, COLOR_CUBICCM, "--"),
    ("cubiccm", 4, 4, "tab:red", "--"),
    ("cubiccm", 32, 32, "tab:brown", "--"),
)

COLOR_BY_LEVEL = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
    32: "tab:brown",
    64: "tab:pink",
    128: "tab:olive",
}




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


def generate_figure1(dic_records) -> None:
    """Figure 1: Subpixel bias hides renderer error."""
    fig, axes = make_figure(
        EXP3_FIGURE1_CM, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT
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
        annotate_no_data(axes_flat[0], "No Reference", font_size=FONT_SIZE_PT)
        annotate_no_data(axes_flat[1], "No Reference", font_size=FONT_SIZE_PT)
        annotate_no_data(axes_flat[2], "No Reference", font_size=FONT_SIZE_PT)
        annotate_no_data(axes_flat[3], "No Reference", font_size=FONT_SIZE_PT)
    else:
        translations = [frame * 0.1 for frame in range(11)]

        cases_to_plot = [
            (
                find_rec(subset, "analytic", analytic=True),
                "Analytic Reference",
                "black",
                "--",
                None,
            ),
        ]
        for interp, osamp, ssaa, col, lstyle, marker in EXP3_FIG1_CASES:
            rec = find_rec(
                subset, "riley_render_texf", interp, osamp, ssaa
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
            if rec is None or "Analytic Reference" in label:
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

        rec_anal, label_anal, col_anal, _, _ = cases_to_plot[0]
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
                linestyle="-",
                marker=None,
                linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
            )

        # Panel b: RMSE (all cases)
        anal_ref = find_rec(subset, "analytic", analytic=True)
        max_rmse_all = 0.0
        max_rmse_filtered = 0.0
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
                        if "Riley Catmull-Rom (Tex-OS=1, Px-SS=1)" not in label:
                            max_rmse_filtered = max(
                                max_rmse_filtered, val_max
                            )

            for rec, label, col, lstyle, marker in cases_to_plot:
                if rec is None or "Analytic Reference" in label:
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
                        linestyle="-",
                        marker=None,
                        linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
                    )

        # Bottom Row (Excluding Catmull-Rom OS=1, SS=1): Row 1
        # Panel c: Bias (zoomed)
        max_bias_filtered = 0.0
        for rec, label, col, lstyle, marker in cases_to_plot:
            if rec is None or "Analytic Reference" in label:
                continue
            if "Riley Catmull-Rom (Tex-OS=1, Px-SS=1)" in label:
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
            if biases:
                max_bias_filtered = max(
                    max_bias_filtered, np.nanmax(np.abs(biases))
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
                linestyle="-",
                marker=None,
                linewidth=EXP3_ANALYTIC_LINE_WIDTH_PT,
            )

        # Panel d: RMSE (zoomed)
        if anal_ref:
            for rec, label, col, lstyle, marker in cases_to_plot:
                if rec is None or "Analytic Reference" in label:
                    continue
                if "Riley Catmull-Rom (Tex-OS=1, Px-SS=1)" in label:
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
                        linestyle="-",
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
        if max_bias_filtered > 0:
            axes_flat[2].set_ylim(
                -max_bias_filtered * 1.15, max_bias_filtered * 1.15
            )
        if max_rmse_all > 0:
            axes_flat[1].set_ylim(bottom=-0.04 * max_rmse_all)
        else:
            axes_flat[1].set_ylim(bottom=-1e-5)
        if max_rmse_filtered > 0:
            axes_flat[3].set_ylim(bottom=-0.04 * max_rmse_filtered)
        else:
            axes_flat[3].set_ylim(bottom=-1e-5)

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
            y_offset=-0.13,
        )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def generate_figure2(dic_records) -> None:
    """Figure 2: Texture representation and pixel integration independence."""
    fig, axes = make_figure(
        EXP3_FIGURE2_CM, rows=1, columns=3, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()

    subset = [
        r for r in dic_records
        if r.case == EXP3_RIGID_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    anal_ref = find_rec(subset, "analytic", analytic=True)

    if anal_ref is None:
        for ax in axes_flat:
            annotate_no_data(ax, "No Reference", font_size=FONT_SIZE_PT)
    else:
        samplers = [
            ("cubic_bspline", "Riley B-spline", COLOR_BSPLINE, "-", "o"),
            ("cubiccm", "Riley Catmull-Rom", COLOR_CUBICCM, "--", "x"),
        ]

        # Panel a: Fixed Tex-OS=1, Refine Px-SS
        ssaa_levels = [1, 2, 4, 8, 16, 32, 64, 128]
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for ss in ssaa_levels:
                rec = find_rec(
                    subset, "riley_render_texf", interp, ssaa=ss, osamp=1
                )
                if rec:
                    rmses = get_rmse_vs_ref(rec, anal_ref, is_dic=True)
                    if rmses and len(rmses) > 3:
                        x_vals.append(ss)
                        y_vals.append(rmses[3])
            axes_flat[0].plot(
                x_vals,
                y_vals,
                color=col,
                marker=marker,
                linestyle=lstyle,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
                label=LABEL_FIG2_A_TEMPLATE.format(name=name),
            )

        # Panel b: Fixed Px-SS=1, Refine Tex-OS
        os_levels = [1, 2, 4, 8, 16, 32, 64, 128]
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for osamp in os_levels:
                rec = find_rec(
                    subset, "riley_render_texf", interp, ssaa=1, osamp=osamp
                )
                if rec:
                    rmses = get_rmse_vs_ref(rec, anal_ref, is_dic=True)
                    if rmses and len(rmses) > 3:
                        x_vals.append(osamp)
                        y_vals.append(rmses[3])
            axes_flat[1].plot(
                x_vals,
                y_vals,
                color=col,
                marker=marker,
                linestyle=lstyle,
                linewidth=EXP3_LINE_WIDTH_PT,
                markersize=EXP3_MARKER_SIZE_PT,
                label=LABEL_FIG2_B_TEMPLATE.format(name=name),
            )

        # Panel c: Simultaneous Refinement (Tex-OS=Px-SS)
        diag_levels = [1, 2, 4, 8, 16, 32, 64, 128]
        for interp, name, col, lstyle, marker in samplers:
            x_vals = []
            y_vals = []
            for lvl in diag_levels:
                rec = find_rec(
                    subset,
                    "riley_render_texf",
                    interp,
                    ssaa=lvl,
                    osamp=lvl,
                )
                if rec:
                    rmses = get_rmse_vs_ref(rec, anal_ref, is_dic=True)
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
                label=LABEL_FIG2_C_TEMPLATE.format(name=name),
            )
            # Inset on panel c
            x_in = [x for x in x_vals if x >= 4]
            y_in = [y_vals[i] for i, x in enumerate(x_vals) if x >= 4]
            if x_in:
                if not hasattr(axes_flat[2], "inset_ax"):
                    axes_flat[2].inset_ax = axes_flat[2].inset_axes(
                        [0.45, 0.45, 0.5, 0.5]
                    )
                axes_flat[2].inset_ax.plot(
                    x_in,
                    y_in,
                    color=col,
                    marker=marker,
                    linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT * 0.8,
                    markersize=EXP3_MARKER_SIZE_PT * 0.8,
                )

        set_sample_axis(
            axes_flat[0],
            ssaa_levels,
            LABEL_PX_INTEGRATION,
            FONT_SIZE_PT,
        )
        set_sample_axis(
            axes_flat[1],
            os_levels,
            LABEL_TEX_OVERSAMPLING,
            FONT_SIZE_PT,
        )
        set_sample_axis(
            axes_flat[2],
            diag_levels,
            LABEL_REF_LEVEL_OS_SS,
            FONT_SIZE_PT,
        )

        for ax in axes_flat:
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.set_ylim(bottom=0.0)

        # Format inset axis if present on panel c
        if hasattr(axes_flat[2], "inset_ax"):
            inset_levels = [4, 8, 16, 32, 64, 128]
            set_sample_axis(
                axes_flat[2].inset_ax,
                inset_levels,
                "",
                FONT_SIZE_PT - 1,
            )
            axes_flat[2].inset_ax.grid(
                True, which="both", linestyle=":", alpha=0.4
            )
            axes_flat[2].inset_ax.tick_params(
                labelsize=TICK_FONT_SIZE_PT - 1
            )
            axes_flat[2].inset_ax.set_ylim(bottom=0.0)

        axes_flat[0].set_ylabel(
            LABEL_RMSE_AT_03PX, fontsize=FONT_SIZE_PT
        )
        axes_flat[0].set_title(TITLE_FIG2_A, fontsize=FONT_SIZE_PT)
        axes_flat[1].set_title(TITLE_FIG2_B, fontsize=FONT_SIZE_PT)
        axes_flat[2].set_title(TITLE_FIG2_C, fontsize=FONT_SIZE_PT)

        handles = [
            Line2D(
                [0],
                [0],
                color=col,
                marker=marker,
                linestyle=lstyle,
                markersize=EXP3_MARKER_SIZE_PT,
                label=name,
            )
            for _, name, col, lstyle, marker in samplers
        ]
        add_figure_legend(
            fig,
            handles,
            font_size=LEGEND_FONT_SIZE_PT,
            columns=2,
            y_offset=-0.10,
        )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def generate_figure3(dic_records, grid_records) -> None:
    """Figure 3: Numerical-reference self-convergence (Rigid & Affine)."""
    fig, axes = make_figure(
        EXP3_FIGURE3_CM, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()
    diag_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    samplers = [
        ("cubic_bspline", "B-spline", COLOR_BSPLINE, "-", "o"),
        ("cubiccm", "Catmull-Rom", COLOR_CUBICCM, "--", "x"),
        ("line", "Linear", "tab:green", ":", "d"),
    ]

    # --- TOP ROW: RIGID TRANSLATION ---
    # 1. Top-Left: DIC Rigid
    dic_subset_rig = [
        r for r in dic_records
        if r.case == EXP3_RIGID_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    dic_ref_rig = find_rec(
        dic_subset_rig, "riley_render_texf", "cubic_bspline", 128, 16
    )
    if dic_ref_rig is None:
        annotate_no_data(
            axes_flat[0], "No DIC Rigid Reference", font_size=FONT_SIZE_PT
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
                if rec:
                    rmses = get_rmse_vs_ref(rec, dic_ref_rig, is_dic=True)
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
    grid_ref_rig = find_rec(
        grid_subset_rig, "riley_render_texf", "cubic_bspline", 128, 16
    )
    if grid_ref_rig is None:
        annotate_no_data(
            axes_flat[1], "No Grid Rigid Reference", font_size=FONT_SIZE_PT
        )
    else:
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
                if rec:
                    rmses = get_rmse_vs_ref(rec, grid_ref_rig, is_dic=False)
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

    # --- BOTTOM ROW: AFFINE DEFORMATION ---
    # 3. Bottom-Left: DIC Affine
    dic_subset_aff = [
        r for r in dic_records
        if r.case == EXP3_AFFINE_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    dic_ref_aff = find_rec(
        dic_subset_aff, "riley_render_texf", "cubic_bspline", 128, 16
    )
    if dic_ref_aff is None:
        annotate_no_data(
            axes_flat[2], "No DIC Affine Reference", font_size=FONT_SIZE_PT
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
                if rec:
                    rmses = get_rmse_vs_ref(rec, dic_ref_aff, is_dic=True)
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
    grid_ref_aff = find_rec(
        grid_subset_aff, "riley_render_texf", "cubic_bspline", 128, 16
    )
    if grid_ref_aff is None:
        annotate_no_data(
            axes_flat[3], "No Grid Affine Reference", font_size=FONT_SIZE_PT
        )
    else:
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
                if rec:
                    rmses = get_rmse_vs_ref(rec, grid_ref_aff, is_dic=False)
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
    if dic_ref_rig is not None:
        inset_levels = [4, 8, 16, 32, 64, 128]
        set_sample_axis(dic_inset, inset_levels, "", FONT_SIZE_PT - 1)
        dic_inset.grid(True, which="both", linestyle=":", alpha=0.45)
        dic_inset.set_ylim(bottom=0.0)
        dic_inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)
    if dic_ref_aff is not None:
        inset_levels = [4, 8, 16, 32, 64, 128]
        set_sample_axis(dic_aff_inset, inset_levels, "", FONT_SIZE_PT - 1)
        dic_aff_inset.grid(True, which="both", linestyle=":", alpha=0.45)
        dic_aff_inset.set_ylim(bottom=0.0)
        dic_aff_inset.tick_params(labelsize=TICK_FONT_SIZE_PT - 1)

    axes_flat[0].set_title(TITLE_FIG3_A, fontsize=FONT_SIZE_PT)
    axes_flat[1].set_title(TITLE_FIG3_B, fontsize=FONT_SIZE_PT)
    axes_flat[2].set_title(TITLE_FIG3_C, fontsize=FONT_SIZE_PT)
    axes_flat[3].set_title(TITLE_FIG3_D, fontsize=FONT_SIZE_PT)

    handles = [
        Line2D(
            [0],
            [0],
            color=col,
            marker=marker,
            linestyle=lstyle,
            markersize=EXP3_MARKER_SIZE_PT,
            label=f"Riley {name}",
        )
        for _, name, col, lstyle, marker in samplers
    ]
    add_figure_legend(
        fig,
        handles,
        font_size=LEGEND_FONT_SIZE_PT,
        columns=3,
        y_offset=-0.05,
    )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig3_affine_self_convergence_dic_vs_grid_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def generate_figure4(dic_records, grid_records) -> list[Path]:
    """Figure 4: Combined Finite-star diagnostic (DIC & Grid)."""
    fig, axes = make_figure(
        EXP3_FIGURE4_CM, rows=4, columns=2, tick_font_size=TICK_FONT_SIZE_PT
    )
    gridspec = axes[0, 0].get_subplotspec().get_gridspec()
    gridspec.set_height_ratios((1.0, 1.0, 1.0, 1.2))
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.12, wspace=0.12, hspace=0.20)
    profile_handles: list[Line2D] = []

    def add_map_colourbar(image, axis) -> None:
        """Keep each colourbar attached closely to its own shallow map."""
        colourbar = fig.colorbar(
            image, ax=axis, pad=0.02, fraction=0.055, aspect=18,
        )
        colourbar.set_label("px", fontsize=COLORBAR_FONT_SIZE_PT)
        colourbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE_PT)

    # ------------------ COLUMN 0: DIC Method ------------------
    subset_dic = [
        r for r in dic_records
        if r.case == EXP3_CHIRP_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    ref_dic = find_rec(
        subset_dic, "riley_render_texf", "cubic_bspline", 128, 16
    )
    under_dic = find_rec(
        subset_dic, "riley_render_texf", "cubic_bspline", 1, 1
    )

    if ref_dic is None or under_dic is None:
        for row in range(4):
            annotate_no_data(axes[row, 0], "Missing DIC Records", FONT_SIZE_PT)
    else:
        ref_data_dic = load_dic(ref_dic, 1)
        under_data_dic = load_dic(under_dic, 1)
        if ref_data_dic is None or under_data_dic is None:
            for row in range(4):
                annotate_no_data(
                    axes[row, 0], "Error Loading DIC Data", FONT_SIZE_PT
                )
        else:
            x, y, ru_dic, rv_dic = ref_data_dic
            _, _, cu_dic, cv_dic = under_data_dic
            du_dic, dv_dic = cu_dic - ru_dic, cv_dic - rv_dic

            # (a) DIC Reference Field
            im_a = axes[0, 0].imshow(
                rv_dic,
                extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm",
                aspect="auto",
            )
            add_map_colourbar(im_a, axes[0, 0])
            axes[0, 0].set_title(TITLE_FIG4_A, fontsize=FONT_SIZE_PT)

            # (b) DIC Under-resolved Field
            im_b = axes[1, 0].imshow(
                cv_dic,
                extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm",
                aspect="auto",
            )
            add_map_colourbar(im_b, axes[1, 0])
            interp_name_dic = INTERPOLATOR_NAMES.get(
                under_dic.interpolator, under_dic.interpolator
            )
            title_b = TITLE_FIG4_B_TEMPLATE.format(
                name=interp_name_dic,
                osamp=under_dic.osamp or 1,
                ssaa=under_dic.ssaa or 1,
            )
            axes[1, 0].set_title(title_b, fontsize=FONT_SIZE_PT)

            # (c) DIC difference Map
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            mask_dic = (
                (x < x_min + EDGE_EXCLUSION_DIC)
                | (x > x_max - EDGE_EXCLUSION_DIC)
                | (y < y_min + EDGE_EXCLUSION_DIC)
                | (y > y_max - EDGE_EXCLUSION_DIC)
            )
            dv_dic_masked = np.where(mask_dic, np.nan, dv_dic)
            limit_dic = float(np.nanpercentile(np.abs(dv_dic_masked), 95))
            limit_dic = max(limit_dic, 1e-5)
            im_c = axes[2, 0].imshow(
                dv_dic,
                extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm",
                aspect="auto",
                vmin=-limit_dic,
                vmax=limit_dic,
            )
            add_map_colourbar(im_c, axes[2, 0])
            title_c = TITLE_FIG4_C_TEMPLATE.format(
                name=interp_name_dic,
                osamp=under_dic.osamp or 1,
                ssaa=under_dic.ssaa or 1,
            )
            axes[2, 0].set_title(title_c, fontsize=FONT_SIZE_PT)

            # (d) DIC Column RMSE Profiles
            cases_dic = []
            for interp, osamp, ssaa, col, lstyle in EXP3_FIG4_5_PROFILES:
                rec = find_rec(
                    subset_dic, "riley_render_texf", interp, osamp, ssaa
                )
                name = INTERPOLATOR_NAMES.get(interp, interp)
                label = LABEL_FIG4_5_PROFILE_TEMPLATE.format(
                    name=name, osamp=osamp, ssaa=ssaa
                )
                cases_dic.append((rec, label, col, lstyle))

            max_rmse_dic = 0.0
            for rec, label, col, lstyle in cases_dic:
                if rec is None:
                    continue
                c_data = load_dic(rec, 1)
                if c_data is None:
                    continue
                _, _, c_u, c_v = c_data
                d_u, d_v = c_u - ru_dic, c_v - rv_dic
                d_v_masked = np.where(mask_dic, np.nan, d_v)

                unique_x = np.unique(x[np.isfinite(x)])
                col_rmses = []
                for ux_val in unique_x:
                    col_mask = x == ux_val
                    err2 = d_v_masked[col_mask] ** 2
                    if np.any(np.isfinite(err2)):
                        col_rmses.append(float(np.sqrt(np.nanmean(err2))))
                    else:
                        col_rmses.append(np.nan)

                valid_rmses = [r for r in col_rmses if np.isfinite(r)]
                if valid_rmses:
                    max_rmse_dic = max(max_rmse_dic, max(valid_rmses))

                axes[3, 0].plot(
                    unique_x,
                    col_rmses,
                    color=col,
                    linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT,
                    label=label,
                )
                profile_handles.append(Line2D([], [], color=col, linestyle=lstyle,
                                              linewidth=EXP3_LINE_WIDTH_PT, label=label))

            axes[3, 0].set_title(TITLE_FIG4_D, fontsize=FONT_SIZE_PT)
            if max_rmse_dic > 0:
                axes[3, 0].set_ylim(bottom=-0.04 * max_rmse_dic)
            else:
                axes[3, 0].set_ylim(bottom=-1e-5)
            axes[3, 0].grid(True, linestyle=":", alpha=0.6)

    # ------------------ COLUMN 1: Grid Method ------------------
    subset_grid = [
        r for r in grid_records
        if r.case == EXP3_CHIRP_CASE
        and r.pattern == "eggbox"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]
    ref_grid = find_rec(
        subset_grid, "riley_render_texf", "cubic_bspline", 128, 16
    )
    under_grid = find_rec(
        subset_grid, "riley_render_texf", "cubic_bspline", 1, 1
    )

    if ref_grid is None or under_grid is None:
        for row in range(4):
            annotate_no_data(
                axes[row, 1], "Missing Grid Records", FONT_SIZE_PT
            )
    else:
        ref_data_grid = load_grid(ref_grid, 1)
        under_data_grid = load_grid(under_grid, 1)
        if ref_data_grid is None or under_data_grid is None:
            for row in range(4):
                annotate_no_data(
                    axes[row, 1], "Error Loading Grid Data", FONT_SIZE_PT
                )
        else:
            ru_grid, rv_grid = ref_data_grid
            cu_grid, cv_grid = under_data_grid
            du_grid, dv_grid = cu_grid - ru_grid, cv_grid - rv_grid
            H, W = ru_grid.shape

            # (e) Grid Reference Field
            im_e = axes[0, 1].imshow(
                rv_grid,
                extent=(0, W, H, 0),
                cmap="coolwarm",
                aspect="auto",
            )
            add_map_colourbar(im_e, axes[0, 1])
            axes[0, 1].set_title(TITLE_FIG4_E, fontsize=FONT_SIZE_PT)

            # (f) Grid Under-resolved Field
            im_f = axes[1, 1].imshow(
                cv_grid,
                extent=(0, W, H, 0),
                cmap="coolwarm",
                aspect="auto",
            )
            add_map_colourbar(im_f, axes[1, 1])
            interp_name_grid = INTERPOLATOR_NAMES.get(
                under_grid.interpolator, under_grid.interpolator
            )
            title_f = TITLE_FIG4_F_TEMPLATE.format(
                name=interp_name_grid,
                osamp=under_grid.osamp or 1,
                ssaa=under_grid.ssaa or 1,
            )
            axes[1, 1].set_title(title_f, fontsize=FONT_SIZE_PT)

            # (g) Grid difference Map
            mask_grid = np.ones(ru_grid.shape, dtype=bool)
            mask_grid[
                EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
                EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID
            ] = False
            dv_grid_masked = np.where(mask_grid, np.nan, dv_grid)
            limit_grid = float(np.nanpercentile(np.abs(dv_grid_masked), 95))
            limit_grid = max(limit_grid, 1e-5)
            im_g = axes[2, 1].imshow(
                dv_grid,
                extent=(0, W, H, 0),
                cmap="coolwarm",
                aspect="auto",
                vmin=-limit_grid,
                vmax=limit_grid,
            )
            add_map_colourbar(im_g, axes[2, 1])
            title_g = TITLE_FIG4_G_TEMPLATE.format(
                name=interp_name_grid,
                osamp=under_grid.osamp or 1,
                ssaa=under_grid.ssaa or 1,
            )
            axes[2, 1].set_title(title_g, fontsize=FONT_SIZE_PT)

            # (h) Grid Column RMSE Profiles
            cases_grid = []
            for interp, osamp, ssaa, col, lstyle in EXP3_FIG4_5_PROFILES:
                rec = find_rec(
                    subset_grid, "riley_render_texf", interp, osamp, ssaa
                )
                name = INTERPOLATOR_NAMES.get(interp, interp)
                label = LABEL_FIG4_5_PROFILE_TEMPLATE.format(
                    name=name, osamp=osamp, ssaa=ssaa
                )
                cases_grid.append((rec, label, col, lstyle))

            max_rmse_grid = 0.0
            for rec, label, col, lstyle in cases_grid:
                if rec is None:
                    continue
                c_data = load_grid(rec, 1)
                if c_data is None:
                    continue
                _, c_v = c_data
                d_v = c_v - rv_grid
                d_v_masked = np.where(mask_grid, np.nan, d_v)

                col_rmses = []
                for col_idx in range(W):
                    err2 = d_v_masked[:, col_idx] ** 2
                    if np.any(np.isfinite(err2)):
                        col_rmses.append(float(np.sqrt(np.nanmean(err2))))
                    else:
                        col_rmses.append(np.nan)

                valid_rmses = [r for r in col_rmses if np.isfinite(r)]
                if valid_rmses:
                    max_rmse_grid = max(max_rmse_grid, max(valid_rmses))

                axes[3, 1].plot(
                    np.arange(W),
                    col_rmses,
                    color=col,
                    linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT,
                    label=label,
                )
                profile_handles.append(Line2D([], [], color=col, linestyle=lstyle,
                                              linewidth=EXP3_LINE_WIDTH_PT, label=label))

            axes[3, 1].set_title(TITLE_FIG4_H, fontsize=FONT_SIZE_PT)
            if max_rmse_grid > 0:
                axes[3, 1].set_ylim(bottom=-0.04 * max_rmse_grid)
            else:
                axes[3, 1].set_ylim(bottom=-1e-5)
            axes[3, 1].grid(True, linestyle=":", alpha=0.6)

    # Format ticks and labels for all subplots
    for r in range(4):
        for c in range(2):
            ax = axes[r, c]
            ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
            if r == 3:
                ax.set_xlabel(LABEL_HORIZ_COORD_PX, fontsize=FONT_SIZE_PT)
                ax.set_ylabel(LABEL_COLUMN_RMSE_PX, fontsize=FONT_SIZE_PT)
            else:
                ax.set_xlabel(LABEL_HORIZ_COORD_PX, fontsize=FONT_SIZE_PT)
                ax.set_ylabel(LABEL_VERT_COORD_PX, fontsize=FONT_SIZE_PT)

    unique_profiles = {handle.get_label(): handle for handle in profile_handles}
    add_figure_legend(
        fig, list(unique_profiles.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=3, y_offset=-0.035,
    )

    save_path = (
        Path(PAPER_OUTPUT_DIR) / "exp3_riley_gauss_fig4_chirp_combined_b12"
    )
    written = save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()
    return written


def figure_stems() -> tuple[str, ...]:
    return (
        "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12",
        "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12",
        "exp3_riley_gauss_fig3_affine_self_convergence_dic_vs_grid_b12",
        "exp3_riley_gauss_fig4_chirp_combined_b12",
    )


def generate_figures() -> list[Path]:
    print("Discovering records...")
    dic_records = discover_dic()
    grid_records = discover_grid()
    print("Generating Figure 1 (Rigid Subpixel Bias/RMSE)...")
    written = generate_figure1(dic_records)
    print("Generating Figure 2 (Px-SS vs. Tex-OS Refinement)...")
    written.extend(generate_figure2(dic_records))
    print("Generating Figure 3 (Self-Convergence Affine)...")
    written.extend(generate_figure3(dic_records, grid_records))
    print("Generating Figure 4 (Combined Chirp Case)...")
    written.extend(generate_figure4(dic_records, grid_records))
    return written


def main() -> None:
    written = generate_figures()
    print("Writing LaTeX previews...")
    write_latex_preview(figure_stems(), FIGURE_CAPTIONS, FIGURE_LABELS)
    print(
        "All Experiment 3 figures and latex previews generated successfully."
    )


if __name__ == "__main__":
    main()
