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
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT,
    DIFFERENCE_CMAP,
    EXP3_AFFINE_CASE,
    EXP3_ANALYTIC_LINE_WIDTH_PT,
    EXP3_BIT_DEPTH,
    EXP3_CHIRP_CASE,
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

# Layout sizing in centimetres (adjusted to prevent constrained layout collapse)
FIGURE1_CM = (11.3, 10.4)
FIGURE2_CM = (17.0, 5.2)
FIGURE3_CM = (11.3, 10.4)
FIGURE4_CM = (17.0, 16.0)


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
        FIGURE1_CM, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT
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

        bspline_cases = [
            (
                find_rec(subset, "riley_render_texf", "cubic_bspline", 1, 1),
                "Riley B-spline (Tex-OS=1, Px-SS=1)",
                COLOR_BSPLINE,
                "-",
                "o",
            ),
            (
                find_rec(subset, "riley_render_texf", "cubic_bspline", 4, 4),
                "Riley B-spline (Tex-OS=4, Px-SS=4)",
                "tab:purple",
                "-",
                "v",
            ),
            (
                find_rec(subset, "riley_render_texf", "cubic_bspline", 32, 1),
                "Riley B-spline (Tex-OS=1, Px-SS=32)",
                "tab:cyan",
                "-",
                "^",
            ),
            (
                find_rec(
                    subset, "riley_render_texf", "cubic_bspline", 32, 32
                ),
                "Riley B-spline (Tex-OS=32, Px-SS=32)",
                "tab:green",
                "-",
                "s",
            ),
        ]

        cubiccm_cases = [
            (
                find_rec(subset, "riley_render_texf", "cubiccm", 1, 1),
                "Riley Catmull-Rom (Tex-OS=1, Px-SS=1)",
                COLOR_CUBICCM,
                "--",
                "x",
            ),
            (
                find_rec(subset, "riley_render_texf", "cubiccm", 4, 4),
                "Riley Catmull-Rom (Tex-OS=4, Px-SS=4)",
                "tab:brown",
                "--",
                "*",
            ),
            (
                find_rec(
                    subset, "riley_render_texf", "cubiccm", 32, 32
                ),
                "Riley Catmull-Rom (Tex-OS=32, Px-SS=32)",
                "tab:red",
                "--",
                "+",
            ),
        ]

        cases_to_plot = [
            (
                find_rec(subset, "analytic", analytic=True),
                "Analytic Reference",
                "black",
                "--",
                None,
            ),
        ] + bspline_cases + cubiccm_cases

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
            ax.set_xlabel("Imposed translation [px]", fontsize=FONT_SIZE_PT)
            ax.grid(True, linestyle=":", alpha=0.6)

        axes_flat[0].set_ylabel("Mean bias [px]", fontsize=FONT_SIZE_PT)
        axes_flat[1].set_ylabel("Disp. RMSE [px]", fontsize=FONT_SIZE_PT)
        axes_flat[2].set_ylabel("Mean bias [px]", fontsize=FONT_SIZE_PT)
        axes_flat[3].set_ylabel("Disp. RMSE [px]", fontsize=FONT_SIZE_PT)

        axes_flat[0].set_title(
            "(a) Rigid subpixel bias (all cases)", fontsize=FONT_SIZE_PT
        )
        axes_flat[1].set_title(
            "(b) RMSE vs. Analytic Ref (all cases)", fontsize=FONT_SIZE_PT
        )
        axes_flat[2].set_title(
            "(c) Zoomed rigid subpixel bias", fontsize=FONT_SIZE_PT
        )
        axes_flat[3].set_title(
            "(d) Zoomed RMSE vs. Analytic Ref", fontsize=FONT_SIZE_PT
        )

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
        / "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement"
    )
    save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()


def generate_figure2(dic_records) -> None:
    """Figure 2: Texture representation and pixel integration independence."""
    fig, axes = make_figure(
        FIGURE2_CM, rows=1, columns=3, tick_font_size=TICK_FONT_SIZE_PT
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
                label=f"{name} (Tex-OS=1)",
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
                label=f"{name} (Px-SS=1)",
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
                label=f"{name} (Tex-OS=Px-SS)",
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
            "Pixel integration (Px-SS)",
            FONT_SIZE_PT,
        )
        set_sample_axis(
            axes_flat[1],
            os_levels,
            "Texture oversampling (Tex-OS)",
            FONT_SIZE_PT,
        )
        set_sample_axis(
            axes_flat[2],
            diag_levels,
            "Refinement level (Tex-OS=Px-SS)",
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
            "RMSE [px] at 0.3 px translation", fontsize=FONT_SIZE_PT
        )
        axes_flat[0].set_title(
            "(a) Fixed Tex-OS=1, Refine Px-SS", fontsize=FONT_SIZE_PT
        )
        axes_flat[1].set_title(
            "(b) Fixed Px-SS=1, Refine Tex-OS", fontsize=FONT_SIZE_PT
        )
        axes_flat[2].set_title(
            "(c) Simultaneous Refinement", fontsize=FONT_SIZE_PT
        )

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
            y_offset=-0.18,
        )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss"
    )
    save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()


def generate_figure3(dic_records, grid_records) -> None:
    """Figure 3: Numerical-reference self-convergence (Rigid & Affine)."""
    fig, axes = make_figure(
        FIGURE3_CM, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()
    diag_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    samplers = [
        ("cubic_bspline", "B-spline", COLOR_BSPLINE, "-", "o"),
        ("cubiccm", "Catmull-Rom", COLOR_CUBICCM, "--", "x"),
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
                label=f"Riley {name}",
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
                label=f"Grid Method ({name})",
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
                label=f"Riley {name}",
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
                label=f"Grid Method ({name})",
            )

    # Format all axes
    for ax in axes_flat:
        set_sample_axis(
            ax,
            diag_levels,
            "Refinement level (Tex-OS=Px-SS)",
            FONT_SIZE_PT,
        )
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(
            "Self-conv. RMSE [px] at 0.3 px def.", fontsize=FONT_SIZE_PT
        )

    axes_flat[0].set_title("(a) DIC self-conv. (Rigid)", fontsize=FONT_SIZE_PT)
    axes_flat[1].set_title(
        "(b) Grid Method self-conv. (Rigid)", fontsize=FONT_SIZE_PT
    )
    axes_flat[2].set_title("(c) DIC self-conv. (Affine)", fontsize=FONT_SIZE_PT)
    axes_flat[3].set_title(
        "(d) Grid Method self-conv. (Affine)", fontsize=FONT_SIZE_PT
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_BSPLINE,
            marker="o",
            linestyle="-",
            markersize=EXP3_MARKER_SIZE_PT,
            label="B-spline",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_CUBICCM,
            marker="x",
            linestyle="--",
            markersize=EXP3_MARKER_SIZE_PT,
            label="Catmull-Rom",
        ),
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
        / "exp3_riley_gauss_fig3_affine_self_convergence_dic_vs_grid"
    )
    save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()


def generate_figure4(dic_records) -> None:
    """Figure 4: Finite-star frequency-based diagnostic (Chirp Case)."""
    fig, axes = make_figure(
        FIGURE4_CM, rows=4, columns=1, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()

    subset = [
        r for r in dic_records
        if r.case == EXP3_CHIRP_CASE
        and r.pattern == "gausscont"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]

    ref = find_rec(subset, "riley_render_texf", "cubic_bspline", 128, 16)
    rec_under = find_rec(
        subset, "riley_render_texf", "cubic_bspline", 1, 1
    )

    if ref is None or rec_under is None:
        for ax in axes_flat:
            annotate_no_data(
                ax, "Missing Chirp Records", font_size=FONT_SIZE_PT
            )
    else:
        ref_data = load_dic(ref, 1)
        rec_data = load_dic(rec_under, 1)

        if ref_data is None or rec_data is None:
            for ax in axes_flat:
                annotate_no_data(
                    ax, "Error Loading Frame 01", font_size=FONT_SIZE_PT
                )
        else:
            x, y, ru, rv = ref_data
            _, _, cu, cv = rec_data
            du, dv = cu - ru, cv - rv

            # Panel a: Converged Reference field u_y
            im0 = axes_flat[0].imshow(
                rv,
                extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm",
                aspect="auto",
            )
            fig.colorbar(im0, ax=axes_flat[0], label="px")
            axes_flat[0].set_title(
                "(a) Reference displacement $u_y$", fontsize=FONT_SIZE_PT
            )

            # Panel b: Under-resolved field u_y (Tex-OS=1, Px-SS=1)
            im1 = axes_flat[1].imshow(
                cv,
                extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm",
                aspect="auto",
            )
            fig.colorbar(im1, ax=axes_flat[1], label="px")
            axes_flat[1].set_title(
                "(b) Riley B-spline (Tex-OS=1, Px-SS=1) displacement $u_y$",
                fontsize=FONT_SIZE_PT,
            )

            # Panel c: Spatial renderer-error map u_y difference
            # Zero-centered symmetric limit using 95th percentile of masked dev
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            mask_edges = (
                (x < x_min + EDGE_EXCLUSION_DIC)
                | (x > x_max - EDGE_EXCLUSION_DIC)
                | (y < y_min + EDGE_EXCLUSION_DIC)
                | (y > y_max - EDGE_EXCLUSION_DIC)
            )
            dv_masked = np.where(mask_edges, np.nan, dv)
            limit = float(np.nanpercentile(np.abs(dv_masked), 95))
            limit = max(limit, 1e-5)
            im2 = axes_flat[2].imshow(
                dv,
                extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm",
                aspect="auto",
                vmin=-limit,
                vmax=limit,
            )
            fig.colorbar(im2, ax=axes_flat[2], label="px")
            axes_flat[2].set_title(
                "(c) Riley B-spline (Tex-OS=1, Px-SS=1) $u_y$ difference map",
                fontsize=FONT_SIZE_PT,
            )

            # Panel d: 1D Column-wise RMSE of u_y vs Horizontal coordinate x
            bspline_profile = [
                (
                    find_rec(
                        subset, "riley_render_texf", "cubic_bspline", 1, 1
                    ),
                    "B-spline (1, 1)",
                    COLOR_BSPLINE,
                    "-",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubic_bspline", 4, 4
                    ),
                    "B-spline (4, 4)",
                    "tab:green",
                    "-",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubic_bspline", 32, 32
                    ),
                    "B-spline (32, 32)",
                    "tab:purple",
                    "-",
                ),
            ]

            cubiccm_profile = [
                (
                    find_rec(
                        subset, "riley_render_texf", "cubiccm", 1, 1
                    ),
                    "Catmull-Rom (1, 1)",
                    COLOR_CUBICCM,
                    "--",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubiccm", 4, 4
                    ),
                    "Catmull-Rom (4, 4)",
                    "tab:red",
                    "--",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubiccm", 32, 32
                    ),
                    "Catmull-Rom (32, 32)",
                    "tab:brown",
                    "--",
                ),
            ]

            cases_profile = bspline_profile + cubiccm_profile

            for case_rec, label, col, lstyle in cases_profile:
                if case_rec is None:
                    continue
                c_data = load_dic(case_rec, 1)
                if c_data is None:
                    continue
                _, _, c_u, c_v = c_data
                d_u, d_v = c_u - ru, c_v - rv

                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                mask = (
                    (x < x_min + EDGE_EXCLUSION_DIC)
                    | (x > x_max - EDGE_EXCLUSION_DIC)
                    | (y < y_min + EDGE_EXCLUSION_DIC)
                    | (y > y_max - EDGE_EXCLUSION_DIC)
                )
                d_v_masked = np.where(mask, np.nan, d_v)

                unique_x = np.unique(x[np.isfinite(x)])
                col_rmses = []
                for ux_val in unique_x:
                    col_mask = x == ux_val
                    err2 = d_v_masked[col_mask] ** 2
                    if np.any(np.isfinite(err2)):
                        col_rmses.append(float(np.sqrt(np.nanmean(err2))))
                    else:
                        col_rmses.append(np.nan)

                axes_flat[3].plot(
                    unique_x,
                    col_rmses,
                    color=col,
                    linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT,
                    label=label,
                )

            axes_flat[3].set_xlabel(
                "Horizontal coordinate [px]", fontsize=FONT_SIZE_PT
            )
            axes_flat[3].set_ylabel(
                "Column RMSE [px]", fontsize=FONT_SIZE_PT
            )
            axes_flat[3].set_title(
                "(d) $u_y$ RMSE along frequency gradient", fontsize=FONT_SIZE_PT
            )
            axes_flat[3].set_ylim(bottom=0.0)
            axes_flat[3].grid(True, linestyle=":", alpha=0.6)
            leg = axes_flat[3].legend(
                fontsize=LEGEND_FONT_SIZE_PT,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )
            # Retain in layout to adjust Panel (d) width matching (a)-(c)
            leg.set_in_layout(True)

            for ax in axes_flat:
                ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
                if ax != axes_flat[3]:
                    ax.set_xlabel(
                        "Horizontal coordinate [px]", fontsize=FONT_SIZE_PT
                    )
                    ax.set_ylabel(
                        "Vertical coordinate [px]", fontsize=FONT_SIZE_PT
                    )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig4_chirp_spatial_frequency_error_star"
    )
    save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()


def generate_figure5(grid_records) -> None:
    """Figure 5: Finite-star frequency-based diagnostic (Grid Method)."""
    fig, axes = make_figure(
        FIGURE4_CM, rows=4, columns=1, tick_font_size=TICK_FONT_SIZE_PT
    )
    axes_flat = axes.flatten()

    subset = [
        r for r in grid_records
        if r.case == EXP3_CHIRP_CASE
        and r.pattern == "eggbox"
        and r.bit_depth == EXP3_BIT_DEPTH
    ]

    ref = find_rec(subset, "riley_render_texf", "cubic_bspline", 128, 16)
    rec_under = find_rec(
        subset, "riley_render_texf", "cubic_bspline", 1, 1
    )

    if ref is None or rec_under is None:
        for ax in axes_flat:
            annotate_no_data(
                ax, "Missing Chirp Records", font_size=FONT_SIZE_PT
            )
    else:
        ref_data = load_grid(ref, 1)
        rec_data = load_grid(rec_under, 1)

        if ref_data is None or rec_data is None:
            for ax in axes_flat:
                annotate_no_data(
                    ax, "Error Loading Frame 01", font_size=FONT_SIZE_PT
                )
        else:
            ru, rv = ref_data
            cu, cv = rec_data
            du, dv = cu - ru, cv - rv
            H, W = ru.shape

            # Panel a: Converged Reference field u_y
            im0 = axes_flat[0].imshow(
                rv,
                extent=(0, W, H, 0),
                cmap="coolwarm",
                aspect="auto",
            )
            fig.colorbar(im0, ax=axes_flat[0], label="px")
            axes_flat[0].set_title(
                "(a) Reference displacement $u_y$", fontsize=FONT_SIZE_PT
            )

            # Panel b: Under-resolved field u_y (Tex-OS=1, Px-SS=1)
            im1 = axes_flat[1].imshow(
                cv,
                extent=(0, W, H, 0),
                cmap="coolwarm",
                aspect="auto",
            )
            fig.colorbar(im1, ax=axes_flat[1], label="px")
            axes_flat[1].set_title(
                "(b) Riley B-spline (Tex-OS=1, Px-SS=1) displacement $u_y$",
                fontsize=FONT_SIZE_PT,
            )

            # Panel c: Spatial renderer-error map u_y difference
            mask_edges = np.ones(ru.shape, dtype=bool)
            mask_edges[
                EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
                EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID
            ] = False
            dv_masked = np.where(mask_edges, np.nan, dv)
            limit = float(np.nanpercentile(np.abs(dv_masked), 95))
            limit = max(limit, 1e-5)
            im2 = axes_flat[2].imshow(
                dv,
                extent=(0, W, H, 0),
                cmap="coolwarm",
                aspect="auto",
                vmin=-limit,
                vmax=limit,
            )
            fig.colorbar(im2, ax=axes_flat[2], label="px")
            axes_flat[2].set_title(
                "(c) Riley B-spline (Tex-OS=1, Px-SS=1) $u_y$ difference map",
                fontsize=FONT_SIZE_PT,
            )

            # Panel d: 1D Column-wise RMSE of u_y vs Horizontal coordinate x
            bspline_profile = [
                (
                    find_rec(
                        subset, "riley_render_texf", "cubic_bspline", 1, 1
                    ),
                    "B-spline (1, 1)",
                    COLOR_BSPLINE,
                    "-",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubic_bspline", 4, 4
                    ),
                    "B-spline (4, 4)",
                    "tab:green",
                    "-",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubic_bspline", 32, 32
                    ),
                    "B-spline (32, 32)",
                    "tab:purple",
                    "-",
                ),
            ]

            cubiccm_profile = [
                (
                    find_rec(
                        subset, "riley_render_texf", "cubiccm", 1, 1
                    ),
                    "Catmull-Rom (1, 1)",
                    COLOR_CUBICCM,
                    "--",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubiccm", 4, 4
                    ),
                    "Catmull-Rom (4, 4)",
                    "tab:red",
                    "--",
                ),
                (
                    find_rec(
                        subset, "riley_render_texf", "cubiccm", 32, 32
                    ),
                    "Catmull-Rom (32, 32)",
                    "tab:brown",
                    "--",
                ),
            ]

            cases_profile = bspline_profile + cubiccm_profile

            for case_rec, label, col, lstyle in cases_profile:
                if case_rec is None:
                    continue
                c_data = load_grid(case_rec, 1)
                if c_data is None:
                    continue
                _, c_v = c_data
                d_v = c_v - rv
                d_v_masked = np.where(mask_edges, np.nan, d_v)

                col_rmses = []
                for col_idx in range(W):
                    err2 = d_v_masked[:, col_idx] ** 2
                    if np.any(np.isfinite(err2)):
                        col_rmses.append(float(np.sqrt(np.nanmean(err2))))
                    else:
                        col_rmses.append(np.nan)

                axes_flat[3].plot(
                    np.arange(W),
                    col_rmses,
                    color=col,
                    linestyle=lstyle,
                    linewidth=EXP3_LINE_WIDTH_PT,
                    label=label,
                )

            axes_flat[3].set_xlabel(
                "Horizontal coordinate [px]", fontsize=FONT_SIZE_PT
            )
            axes_flat[3].set_ylabel(
                "Column RMSE [px]", fontsize=FONT_SIZE_PT
            )
            axes_flat[3].set_title(
                "(d) $u_y$ RMSE along frequency gradient", fontsize=FONT_SIZE_PT
            )
            axes_flat[3].set_ylim(bottom=0.0)
            axes_flat[3].grid(True, linestyle=":", alpha=0.6)
            leg = axes_flat[3].legend(
                fontsize=LEGEND_FONT_SIZE_PT,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )
            # Retain in layout to adjust Panel (d) width matching (a)-(c)
            leg.set_in_layout(True)

            for ax in axes_flat:
                ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
                if ax != axes_flat[3]:
                    ax.set_xlabel(
                        "Horizontal coordinate [px]", fontsize=FONT_SIZE_PT
                    )
                    ax.set_ylabel(
                        "Vertical coordinate [px]", fontsize=FONT_SIZE_PT
                    )

    save_path = (
        Path(PAPER_OUTPUT_DIR)
        / "exp3_riley_gauss_fig5_chirp_spatial_frequency_error_star_gridmethod"
    )
    save_figure(fig, save_path, PAPER_FORMATS, PAPER_DPI)
    fig.clear()


def main() -> None:
    print("Discovering records...")
    dic_records = discover_dic()
    grid_records = discover_grid()

    print("Generating Figure 1 (Rigid Subpixel Bias/RMSE)...")
    generate_figure1(dic_records)

    print("Generating Figure 2 (Px-SS vs. Tex-OS Refinement)...")
    generate_figure2(dic_records)

    print("Generating Figure 3 (Self-Convergence Affine)...")
    generate_figure3(dic_records, grid_records)

    print("Generating Figure 4 (Finite Star/Chirp Case)...")
    generate_figure4(dic_records)

    print("Generating Figure 5 (Finite Star/Chirp Case - Grid Method)...")
    generate_figure5(grid_records)

    stems = [
        "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement",
        "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss",
        "exp3_riley_gauss_fig3_affine_self_convergence_dic_vs_grid",
        "exp3_riley_gauss_fig4_chirp_spatial_frequency_error_star",
        "exp3_riley_gauss_fig5_chirp_spatial_frequency_error_star_gridmethod",
    ]
    print("Writing LaTeX previews...")
    write_latex_preview(stems, FIGURE_CAPTIONS, FIGURE_LABELS)
    print("All Experiment 3 figures and latex previews generated successfully.")


if __name__ == "__main__":
    main()
