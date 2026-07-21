# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import gc
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.lines import Line2D
from PIL import Image

from exp1common import output_case_name, parse_case_params
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    BIT_DEPTHS,
    CLEAR_DIR,
    DEFORMATION_CASES,
    ACTIVE_FRAMES,
    OUTPUT_DIR,
    SSAA_LEVELS,
    TEX_INTERPOLATORS,
    TEX_OVERSAMPLES,
    exp1_output_dir,
)
from script_timing import ScriptTimer, timed_call

# Defaults make this base entry point usable directly.  The world, UV, and
# texture-only wrappers below override these paths for their specific studies.
OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_world")
RILEY_FUNC_DIR = exp1_output_dir("exp1_riley_render_func_world")
RILEY_TEX_DIR = exp1_output_dir("exp1_riley_render_texuint")

RESULTS_DIR_FUNC = exp1_output_dir("exp1_riley_analysis_func_world")
RESULTS_DIR_TEX = exp1_output_dir("exp1_riley_analysis_texuint")
ANALYSIS_MODE = "both"


def _combine_metric_panels(paths: list[Path], output_path: Path) -> None:
    """Combine four existing metric plots into one 2-by-2 analysis figure."""
    panels = [Image.open(path).convert("RGB") for path in paths]
    try:
        width = max(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        combined = Image.new("RGB", (2 * width, 2 * height), "white")
        for index, panel in enumerate(panels):
            combined.paste(panel, ((index % 2) * width, (index // 2) * height))
        combined.save(output_path)
    finally:
        for panel in panels:
            panel.close()
    for path in paths:
        path.unlink(missing_ok=True)


def _plot_texture_oversample_metrics(
    riley_tex: dict,
    case_name: str,
    frame: int,
    output_dir: Path,
    selected_ssaa: list[int],
    group_name: str,
    bit_depth: int | None = None,
) -> None:
    """Plot texture oversampling convergence with one line per Riley SSAA."""
    group_label = group_name.replace("odd_exp", "odd-exponent").replace(
        "even_exp", "even-exponent"
    )
    float_bit_depth = 16 if 16 in BIT_DEPTHS else max(BIT_DEPTHS)
    available_ssaa = sorted(
        {
            int(round(sample))
            for oversamp in TEX_OVERSAMPLES
            for sample in riley_tex[float_bit_depth][oversamp]["samples"]
        }
    )
    oversamp_values = sorted(
        {
            oversamp
            for oversamp in TEX_OVERSAMPLES
            if any(riley_tex[bit_depth][oversamp]["samples"] for bit_depth in BIT_DEPTHS)
        }
    )
    ssaa_values = [value for value in available_ssaa if value in selected_ssaa]
    if not ssaa_values or not oversamp_values:
        return

    def values_for(bit_depth: int, ssaa: int, metric: str):
        points = []
        for oversamp in oversamp_values:
            values = riley_tex[bit_depth][oversamp]
            for index, sample in enumerate(values["samples"]):
                if int(round(sample)) == ssaa:
                    points.append((oversamp, values[metric][index]))
                    break
        return points

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bit_styles = {8: "-", 12: "--", 16: ":"}
    plotted_bit_depths = [bit_depth] if bit_depth is not None else BIT_DEPTHS
    for index, ssaa in enumerate(ssaa_values):
        color = colors[index % len(colors)]
        for axis, metric, title, ylabel in (
            (axes[0, 0], "e_f64", "Floating-Point RMSE", "RMSE"),
            (axes[0, 1], "e_inf", "Floating-Point Max Error", "Max error"),
        ):
            points = values_for(float_bit_depth, ssaa, metric)
            if points:
                x, y = zip(*points)
                valid = np.asarray(y) > 0.0
                if np.any(valid):
                    axis.loglog(np.asarray(x)[valid], np.asarray(y)[valid], color=color,
                                marker="o", linewidth=1.6)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        for plotted_bit_depth in plotted_bit_depths:
            for axis, metric, title, ylabel in (
                (axes[1, 0], "delta_b", "Digitised Mismatch Fraction", "Fraction of differing pixels"),
                (axes[1, 1], "max_eb", "Maximum Digitised Mismatch", "LSB levels"),
            ):
                points = values_for(plotted_bit_depth, ssaa, metric)
                if points:
                    x, y = zip(*points)
                    if metric == "max_eb":
                        axis.loglog(x, np.maximum(0.2, y), color=color, marker="o",
                                    linestyle=bit_styles.get(plotted_bit_depth, "-"), linewidth=1.4)
                    else:
                        axis.semilogx(x, y, color=color, marker="o",
                                      linestyle=bit_styles.get(plotted_bit_depth, "-"), linewidth=1.4)
                axis.set_title(title)
                axis.set_ylabel(ylabel)

    for plotted_bit_depth in plotted_bit_depths:
        maximum = float(2**plotted_bit_depth - 1)
        style = bit_styles.get(plotted_bit_depth, "-")
        axes[0, 0].axhline(1.0 / maximum, color="black", linestyle=style, alpha=0.35)
        axes[0, 1].axhline(0.5 / maximum, color="red", linestyle=style, alpha=0.35)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[1, 1].axhline(0.2, color="red", linestyle=":", alpha=0.6)
    axes[1, 1].set_ylim(0.16, None)

    for axis in axes.flat:
        axis.set_xlabel("Texture Oversampling Along One Pixel Axis")
        axis.xaxis.set_major_locator(FixedLocator(oversamp_values))
        axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in oversamp_values]))
        axis.set_xlim(0.85 * oversamp_values[0], 1.15 * oversamp_values[-1])
        axis.grid(True, which="both", ls="--", alpha=0.4)
    handles = [
        Line2D([], [], color=colors[index % len(colors)], marker="o", label=f"Riley, Tex, SSAA={ssaa}")
        for index, ssaa in enumerate(ssaa_values)
    ]
    handles.extend(
        Line2D([], [], color="black", linestyle=bit_styles[plotted_bit_depth], label=f"{plotted_bit_depth}-bit")
        for plotted_bit_depth in plotted_bit_depths
    )
    handles.extend(
        [
            Line2D([], [], color="black", linestyle="--", label="1 LSB"),
            Line2D([], [], color="red", linestyle=":", label="0 LSB"),
        ]
    )
    figure.suptitle(
        f"Texture Oversampling Study ({group_label} SSAA){f', {bit_depth}-bit' if bit_depth is not None else ''}: {case_name} (Frame {frame:02d})\n"
        "Reference: analytic renderer",
        fontweight="bold",
    )
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                  ncol=3, fontsize=6, frameon=True, facecolor="white", edgecolor="none")
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    suffix = f"_b{bit_depth:02d}" if bit_depth is not None else ""
    figure.savefig(output_dir / f"{case_name}_tex_oversamp_{group_name}{suffix}_metrics_frame{frame:02d}.png", dpi=150)
    figure.clear()
    plt.close(figure)


def _plot_texture_ssaa_metrics(
    riley_tex: dict,
    case_name: str,
    frame: int,
    output_dir: Path,
    selected_oversamp: list[int],
    group_name: str,
    bit_depth: int | None = None,
) -> None:
    """Plot raster SSAA convergence with a compact selected OS subset."""
    group_label = group_name.replace("odd_exp", "odd-exponent").replace(
        "even_exp", "even-exponent"
    )
    float_bit_depth = 16 if 16 in BIT_DEPTHS else max(BIT_DEPTHS)
    oversamp_values = [value for value in selected_oversamp if value in TEX_OVERSAMPLES]
    ssaa_values = sorted({int(round(sample)) for osamp in oversamp_values for sample in riley_tex[float_bit_depth][osamp]["samples"]})
    if not oversamp_values or not ssaa_values:
        return

    def values_for(bit_depth: int, oversamp: int, metric: str):
        values = riley_tex[bit_depth][oversamp]
        return sorted(zip(values["samples"], values[metric]))

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bit_styles = {8: "-", 12: "--", 16: ":"}
    plotted_bit_depths = [bit_depth] if bit_depth is not None else BIT_DEPTHS
    for index, oversamp in enumerate(oversamp_values):
        color = colors[index % len(colors)]
        for axis, metric, title, ylabel in (
            (axes[0, 0], "e_f64", "Floating-Point RMSE", "RMSE"),
            (axes[0, 1], "e_inf", "Floating-Point Max Error", "Max error"),
        ):
            points = values_for(float_bit_depth, oversamp, metric)
            if points:
                x, y = zip(*points)
                valid = np.asarray(y) > 0.0
                if np.any(valid):
                    axis.loglog(np.asarray(x)[valid], np.asarray(y)[valid], color=color, marker="o", linewidth=1.6)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        for plotted_bit_depth in plotted_bit_depths:
            for axis, metric, title, ylabel in (
                (axes[1, 0], "delta_b", "Digitised Mismatch Fraction", "Fraction of differing pixels"),
                (axes[1, 1], "max_eb", "Maximum Digitised Mismatch", "LSB levels"),
            ):
                points = values_for(plotted_bit_depth, oversamp, metric)
                if points:
                    x, y = zip(*points)
                    if metric == "max_eb":
                        axis.loglog(x, np.maximum(0.2, y), color=color, marker="o", linestyle=bit_styles.get(plotted_bit_depth, "-"), linewidth=1.4)
                    else:
                        axis.semilogx(x, y, color=color, marker="o", linestyle=bit_styles.get(plotted_bit_depth, "-"), linewidth=1.4)
                axis.set_title(title)
                axis.set_ylabel(ylabel)
    for plotted_bit_depth in plotted_bit_depths:
        maximum = float(2**plotted_bit_depth - 1)
        style = bit_styles.get(plotted_bit_depth, "-")
        axes[0, 0].axhline(1.0 / maximum, color="black", linestyle=style, alpha=0.35)
        axes[0, 1].axhline(0.5 / maximum, color="red", linestyle=style, alpha=0.35)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[1, 1].axhline(0.2, color="red", linestyle=":", alpha=0.6)
    axes[1, 1].set_ylim(0.16, None)
    for axis in axes.flat:
        axis.set_xlabel("Riley Samples Along One Pixel Axis")
        axis.xaxis.set_major_locator(FixedLocator(ssaa_values))
        axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in ssaa_values]))
        axis.set_xlim(0.85 * ssaa_values[0], 1.15 * ssaa_values[-1])
        axis.grid(True, which="both", ls="--", alpha=0.4)
    handles = [Line2D([], [], color=colors[index % len(colors)], marker="o", label=f"Riley, Tex, OS={oversamp}") for index, oversamp in enumerate(oversamp_values)]
    handles.extend(Line2D([], [], color="black", linestyle=bit_styles[plotted_bit_depth], label=f"{plotted_bit_depth}-bit") for plotted_bit_depth in plotted_bit_depths)
    handles.extend([Line2D([], [], color="black", linestyle="--", label="1 LSB"), Line2D([], [], color="red", linestyle=":", label="0 LSB")])
    figure.suptitle(f"Texture SSAA Study ({group_label} OS){f', {bit_depth}-bit' if bit_depth is not None else ''}: {case_name} (Frame {frame:02d})\nReference: analytic renderer", fontweight="bold")
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=6, frameon=True, facecolor="white", edgecolor="none")
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    suffix = f"_b{bit_depth:02d}" if bit_depth is not None else ""
    figure.savefig(output_dir / f"{case_name}_tex_ssaa_{group_name}{suffix}_metrics_frame{frame:02d}.png", dpi=150)
    figure.clear()
    plt.close(figure)


def _plot_texture_limit_curves(
    riley_tex: dict, case_name: str, frame: int, output_dir: Path
) -> None:
    """Show only the highest-OS and highest-SSAA texture convergence cuts."""
    float_bit_depth = 16 if 16 in BIT_DEPTHS else max(BIT_DEPTHS)
    oversamp_values = sorted(oversamp for oversamp in TEX_OVERSAMPLES if riley_tex[float_bit_depth][oversamp]["samples"])
    if not oversamp_values:
        return
    highest_os = oversamp_values[-1]
    ssaa_values = sorted(int(round(sample)) for sample in riley_tex[float_bit_depth][highest_os]["samples"])
    if not ssaa_values:
        return
    highest_ssaa = ssaa_values[-1]

    def series(bit_depth: int, oversamp: int, metric: str):
        values = riley_tex[bit_depth][oversamp]
        return sorted(zip(values["samples"], values[metric]))

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, metric, title, ylabel in (
        (axes[0, 0], "e_f64", f"Highest OS={highest_os}: floating RMSE", "RMSE"),
        (axes[0, 1], "e_inf", f"Highest SSAA={highest_ssaa}: floating max error", "Max error"),
    ):
        if axis is axes[0, 0]:
            points = series(float_bit_depth, highest_os, metric)
            x, y = zip(*points)
            axis.loglog(x, np.maximum(np.finfo(float).tiny, y), marker="o", color="#1f77b4", label=f"Riley, Tex, OS={highest_os}")
            axis.set_xlabel("Riley Samples Along One Pixel Axis")
        else:
            points = []
            for oversamp in oversamp_values:
                values = riley_tex[float_bit_depth][oversamp]
                for index, sample in enumerate(values["samples"]):
                    if int(round(sample)) == highest_ssaa:
                        points.append((oversamp, values[metric][index]))
                        break
            if points:
                x, y = zip(*points)
                axis.loglog(x, np.maximum(np.finfo(float).tiny, y), marker="o", color="#ff7f0e", label=f"Riley, Tex, SSAA={highest_ssaa}")
            axis.set_xlabel("Texture Oversampling Along One Pixel Axis")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", ls="--", alpha=0.4)
        axis.legend(loc="lower left", fontsize=7)

    for bit_depth in BIT_DEPTHS:
        style = {8: "-", 12: "--", 16: ":"}.get(bit_depth, "-")
        left = series(bit_depth, highest_os, "max_eb")
        if left:
            x, y = zip(*left)
            axes[1, 0].loglog(x, np.maximum(0.2, y), marker="o", linestyle=style, label=f"Riley, Tex, OS={highest_os}, {bit_depth}-bit")
        right = []
        for oversamp in oversamp_values:
            values = riley_tex[bit_depth][oversamp]
            for index, sample in enumerate(values["samples"]):
                if int(round(sample)) == highest_ssaa:
                    right.append((oversamp, values["max_eb"][index]))
                    break
        if right:
            x, y = zip(*right)
            axes[1, 1].loglog(x, np.maximum(0.2, y), marker="o", linestyle=style, label=f"Riley, Tex, SSAA={highest_ssaa}, {bit_depth}-bit")
    for axis, title, xlabel in (
        (axes[1, 0], f"Highest OS={highest_os}: max digitised mismatch", "Riley Samples Along One Pixel Axis"),
        (axes[1, 1], f"Highest SSAA={highest_ssaa}: max digitised mismatch", "Texture Oversampling Along One Pixel Axis"),
    ):
        axis.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="1 LSB")
        axis.axhline(0.2, color="red", linestyle=":", alpha=0.6, label="0 LSB")
        axis.set_ylim(0.16, None)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("LSB levels")
        axis.grid(True, which="both", ls="--", alpha=0.4)
        axis.legend(loc="lower left", fontsize=6)
    figure.suptitle(f"Texture Limiting Convergence Curves: {case_name} (Frame {frame:02d})\nReference: analytic renderer", fontweight="bold")
    figure.savefig(output_dir / f"{case_name}_tex_limits_metrics_frame{frame:02d}.png", dpi=150)
    figure.clear()
    plt.close(figure)


def _write_texture_analysis_figures(
    riley_tex: dict,
    case_name: str,
    frame: int,
    output_dir: Path,
) -> None:
    """Write only the requested texture-study figures, directly to their outputs.

    The texture analysis used to render four temporary PNGs, read them back
    through Pillow, assemble a fifth PNG, then delete all five.  None of those
    images was a user-facing output, so the final figures are now generated
    directly from the in-memory metric arrays.
    """
    # Clear obsolete output names left by older versions of this analysis.
    for suffix in (
        "tex_metrics", "tex_oversamp_metrics", "tex_float_rmse", "tex_float_max",
        "tex_bits", "tex_max_eb",
    ):
        (output_dir / f"{case_name}_{suffix}_frame{frame:02d}.png").unlink(
            missing_ok=True
        )
    available_os = sorted(
        oversamp
        for oversamp in TEX_OVERSAMPLES
        if any(riley_tex[bit_depth][oversamp]["samples"] for bit_depth in BIT_DEPTHS)
    )
    if not available_os:
        return
    float_bit_depth = 16 if 16 in BIT_DEPTHS else max(BIT_DEPTHS)
    available_ssaa = sorted(
        {
            int(round(sample))
            for oversamp in available_os
            for sample in riley_tex[float_bit_depth][oversamp]["samples"]
        }
    )
    if not available_ssaa:
        return
    odd_os = [value for value in available_os if value.bit_length() % 2 == 0]
    even_os = [value for value in available_os if value.bit_length() % 2 == 1]
    odd_ssaa = [value for value in available_ssaa if value.bit_length() % 2 == 0]
    even_ssaa = [value for value in available_ssaa if value.bit_length() % 2 == 1]
    for group_name, selected_os, selected_ssaa in (
        ("odd_exp", odd_os, odd_ssaa),
        ("even_exp", even_os, even_ssaa),
    ):
        _plot_texture_ssaa_metrics(
            riley_tex, case_name, frame, output_dir, selected_os, group_name
        )
        _plot_texture_oversample_metrics(
            riley_tex, case_name, frame, output_dir, selected_ssaa, group_name
        )
    for bit_depth in BIT_DEPTHS:
        _plot_texture_ssaa_metrics(
            riley_tex, case_name, frame, output_dir, available_os, "all", bit_depth
        )
        _plot_texture_oversample_metrics(
            riley_tex, case_name, frame, output_dir, available_ssaa, "all", bit_depth
        )
    _plot_texture_limit_curves(riley_tex, case_name, frame, output_dir)


def _write_function_analysis_figure(
    custom_data: dict,
    riley_func: dict,
    case_name: str,
    frame: int,
    output_dir: Path,
    float_bit_depth: int,
) -> None:
    """Write the function-shader four-panel figure without temporary PNGs."""
    for suffix in ("func_float_rmse", "func_float_max", "func_bits", "func_max_eb"):
        (output_dir / f"{case_name}_{suffix}_frame{frame:02d}.png").unlink(
            missing_ok=True
        )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    styles = (
        ("rect", "Custom, Rect", "#1f77b4", "o", "-"),
        ("gauss", "Custom, Gauss", "#2ca02c", "s", "-"),
        ("func", "Riley, Func", "black", "x", "--"),
    )

    def series(method: str, bit_depth: int, metric: str):
        values = riley_func[bit_depth] if method == "func" else custom_data[bit_depth][method]
        return sorted(zip(values["samples"], values[metric]))

    for method, label, color, marker, linestyle in styles:
        for axis, metric in ((axes[0, 0], "e_f64"), (axes[0, 1], "e_inf")):
            points = series(method, float_bit_depth, metric)
            if points:
                x, y = zip(*points)
                positive = np.asarray(y) > 0.0
                if np.any(positive):
                    axis.loglog(np.asarray(x)[positive], np.asarray(y)[positive], color=color,
                                marker=marker, linestyle=linestyle, linewidth=1.6,
                                markersize=6, label=label)
    for bit_depth in BIT_DEPTHS:
        maximum = float(2**bit_depth - 1)
        style = {8: "-", 12: "--", 16: ":"}.get(bit_depth, "-")
        for axis in axes[0, :]:
            axis.axhline(1.0 / maximum, color="black", linestyle=style, alpha=0.3)
            axis.axhline(0.5 / maximum, color="red", linestyle=style, alpha=0.3)
        for method, label, color, marker, _ in styles:
            for axis, metric in ((axes[1, 0], "delta_b"), (axes[1, 1], "max_eb")):
                points = series(method, bit_depth, metric)
                if points:
                    x, y = zip(*points)
                    values = np.asarray(y)
                    if metric == "max_eb":
                        axis.loglog(x, np.maximum(0.2, values), color=color, marker=marker,
                                    linestyle=style, linewidth=1.3, markersize=5,
                                    label=f"{label}, {bit_depth}-bit")
                    else:
                        axis.semilogx(x, values, color=color, marker=marker,
                                      linestyle=style, linewidth=1.3, markersize=5,
                                      label=f"{label}, {bit_depth}-bit")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6, label="1 LSB")
    axes[1, 1].axhline(0.2, color="red", linestyle=":", alpha=0.6, label="0 LSB")
    axes[1, 1].set_ylim(0.16, None)
    for axis, title, ylabel in (
        (axes[0, 0], "Floating-Point RMSE", "RMSE"),
        (axes[0, 1], "Floating-Point Max Error", "Max error"),
        (axes[1, 0], "Digitised Mismatch Fraction", "Fraction of differing pixels"),
        (axes[1, 1], "Maximum Digitised Mismatch", "LSB levels"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Samples Along One Pixel Axis")
        axis.grid(True, which="both", ls="--", alpha=0.4)
        axis.set_ylabel(ylabel)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="lower left", fontsize=6, frameon=True,
                        facecolor="white", edgecolor="none")
    figure.suptitle(
        f"Riley, Func: {case_name} (Frame {frame:02d}) | Reference: Analytic",
        fontweight="bold",
    )
    figure.savefig(output_dir / f"{case_name}_func_metrics_frame{frame:02d}.png", dpi=150)
    figure.clear()
    plt.close(figure)


def analyze_riley_case(case_name: str, tex_interp: str) -> None:
    case_name = output_case_name(case_name, TARG_PX_X)
    print(80 * "=")
    print(f"Analyzing Riley vs Custom: {case_name} ({tex_interp})")
    print(80 * "=")

    case_dir = OUTPUT_DIR / case_name
    results_dir_tex = RESULTS_DIR_TEX / tex_interp
    results_dir_tex.mkdir(parents=True, exist_ok=True)
    ssaa_ticks = sorted(SSAA_LEVELS)

    for ff in ACTIVE_FRAMES:
        print(f"\n--- Frame {ff:02d} ---")

        # Load analytic reference images
        ref_float_by_bb = {}
        ref_dig_by_bb = {}

        for bb in BIT_DEPTHS:
            max_val = float(2**bb - 1)
            ref_prefix = (
                f"targ_px{TARG_PX_X}_int_analytic_param_0_b{bb}_frame{ff:02d}"
            )
            ref_npy = case_dir / f"{ref_prefix}.npy"
            ref_tiff = case_dir / f"{ref_prefix}.tiff"

            if ref_npy.exists() and ref_tiff.exists():
                ref_float_by_bb[bb] = np.load(ref_npy) / max_val
                with Image.open(ref_tiff) as img:
                    ref_dig_by_bb[bb] = np.array(img, dtype=np.float64)

        if not ref_float_by_bb:
            print(
                f"Warning: Reference not found for Frame {ff:02d}. Skipping."
            )
            gc.collect()
            continue

        # Data structures for Plotting
        # 1. Custom Renderer (rect & gauss)
        custom_data = {
            bb: {
                "rect": {
                    "samples": [],
                    "e_f64": [],
                    "e_inf": [],
                    "delta_b": [],
                    "max_eb": [],
                },
                "gauss": {
                    "samples": [],
                    "e_f64": [],
                    "e_inf": [],
                    "delta_b": [],
                    "max_eb": [],
                },
            }
            for bb in BIT_DEPTHS
        }

        # Load Custom Renderer data
        # Check params from 1 to 512 for rect, 2 to 128 for gauss
        rect_params = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        gauss_params = [2, 4, 8, 16, 32, 64, 128]

        for method, params in [("rect", rect_params), ("gauss", gauss_params)]:
            if ANALYSIS_MODE not in {"func", "both"}:
                continue
            for param in params:
                samples = param * param
                for bb in BIT_DEPTHS:
                    if bb not in ref_float_by_bb:
                        continue
                    max_val = float(2**bb - 1)
                    prefix = (
                        f"targ_px{TARG_PX_X}_int_{method}_param_{param}"
                        f"_b{bb}_frame{ff:02d}"
                    )
                    npy_path = case_dir / f"{prefix}.npy"
                    tiff_path = case_dir / f"{prefix}.tiff"

                    if npy_path.exists() and tiff_path.exists():
                        # Float metrics
                        img_float = np.load(npy_path) / max_val
                        diff = img_float - ref_float_by_bb[bb]
                        e_f64 = np.sqrt(np.mean(diff**2))
                        e_inf = np.max(np.abs(diff))

                        # Digitised metrics
                        with Image.open(tiff_path) as img:
                            img_dig = np.array(img, dtype=np.float64)
                        diff_dig = img_dig - ref_dig_by_bb[bb]
                        delta_b = np.mean(img_dig != ref_dig_by_bb[bb])
                        max_eb = np.max(np.abs(diff_dig))

                        custom_data[bb][method]["samples"].append(np.sqrt(samples))
                        custom_data[bb][method]["e_f64"].append(e_f64)
                        custom_data[bb][method]["e_inf"].append(e_inf)
                        custom_data[bb][method]["delta_b"].append(delta_b)
                        custom_data[bb][method]["max_eb"].append(max_eb)
                        del img_float, diff, img_dig, diff_dig

        # 2. Riley, Func Data
        riley_func = {
            bb: {
                "samples": [],
                "e_f64": [],
                "e_inf": [],
                "delta_b": [],
                "max_eb": [],
            }
            for bb in BIT_DEPTHS
        }

        # Load Riley, Func data
        func_dir_base = RILEY_FUNC_DIR / case_name
        for ss in SSAA_LEVELS:
            if ANALYSIS_MODE not in {"func", "both"}:
                continue
            samples = ss * ss
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                max_val = float(2**bb - 1)
                case_out = func_dir_base / f"ss{ss}_b{bb}"
                npy_path = case_out / f"image_c00_f{ff:02d}.npy"
                tiff_path = case_out / f"cam0_frame{ff}_field0.tiff"

                if npy_path.exists() and tiff_path.exists():
                    img_float = np.load(npy_path) / max_val
                    diff = img_float - ref_float_by_bb[bb]
                    e_f64 = np.sqrt(np.mean(diff**2))
                    e_inf = np.max(np.abs(diff))

                    with Image.open(tiff_path) as img:
                        img_dig = np.array(img, dtype=np.float64)
                    diff_dig = img_dig - ref_dig_by_bb[bb]
                    delta_b = np.mean(img_dig != ref_dig_by_bb[bb])
                    max_eb = np.max(np.abs(diff_dig))

                    riley_func[bb]["samples"].append(np.sqrt(samples))
                    riley_func[bb]["e_f64"].append(e_f64)
                    riley_func[bb]["e_inf"].append(e_inf)
                    riley_func[bb]["delta_b"].append(delta_b)
                    riley_func[bb]["max_eb"].append(max_eb)
                    del img_float, diff, img_dig, diff_dig

        # 3. Riley Texture Shader Data
        riley_tex = {
            bb: {
                oversamp: {
                    "samples": [],
                    "e_f64": [],
                    "e_inf": [],
                    "delta_b": [],
                    "max_eb": [],
                }
                for oversamp in TEX_OVERSAMPLES
            }
            for bb in BIT_DEPTHS
        }

        # Load Riley Texture Shader data
        riley_texture_sample_count = 0
        for ss in SSAA_LEVELS:
            if ANALYSIS_MODE not in {"tex", "both"}:
                continue
            samples = ss * ss
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                max_val = float(2**bb - 1)
                for oversamp in TEX_OVERSAMPLES:
                    case_out = RILEY_TEX_DIR / f"{case_name}_{tex_interp}" / (
                        f"ss{ss}_b{bb}_oversamp{oversamp}"
                    )
                    npy_path = case_out / f"image_c00_f{ff:02d}.npy"
                    tiff_path = case_out / f"cam0_frame{ff}_field0.tiff"

                    if npy_path.exists() and tiff_path.exists():
                        img_float = np.load(npy_path) / max_val
                        diff = img_float - ref_float_by_bb[bb]
                        e_f64 = np.sqrt(np.mean(diff**2))
                        e_inf = np.max(np.abs(diff))

                        with Image.open(tiff_path) as img:
                            img_dig = np.array(img, dtype=np.float64)
                        diff_dig = img_dig - ref_dig_by_bb[bb]
                        delta_b = np.mean(img_dig != ref_dig_by_bb[bb])
                        max_eb = np.max(np.abs(diff_dig))

                        r_tex = riley_tex[bb][oversamp]
                        r_tex["samples"].append(np.sqrt(samples))
                        r_tex["e_f64"].append(e_f64)
                        r_tex["e_inf"].append(e_inf)
                        r_tex["delta_b"].append(delta_b)
                        r_tex["max_eb"].append(max_eb)
                        riley_texture_sample_count += 1
                        del img_float, diff, img_dig, diff_dig

        if ANALYSIS_MODE == "tex" and riley_texture_sample_count == 0:
            print(
                "Warning: No Riley texture render samples found for "
                f"{case_name} ({tex_interp}); plots contain only the custom baseline."
            )

        # Texture-only analyses have no need for the historical intermediate
        # single-panel PNGs.  Generate their requested four-panel figures
        # directly and move on to the next frame.
        if ANALYSIS_MODE in {"tex", "both"}:
            _write_texture_analysis_figures(riley_tex, case_name, ff, results_dir_tex)

        if ANALYSIS_MODE in {"func", "both"}:
            float_bit_depth = 16 if 16 in ref_float_by_bb else max(ref_float_by_bb)
            _write_function_analysis_figure(
                custom_data, riley_func, case_name, ff, RESULTS_DIR_FUNC,
                float_bit_depth,
            )
        if ANALYSIS_MODE in {"tex", "func", "both"}:
            # Every array loaded for this frame is now represented by scalar
            # metrics in the written figures.  Release its backing storage
            # before loading the next frame.
            del ref_float_by_bb, ref_dig_by_bb, custom_data, riley_func, riley_tex
            plt.close("all")
            gc.collect()
            continue

        # ------------------------------------------------------------------
        # GENERATE PLOTS - GROUP A: Riley, Func vs Custom Renderer
        # ------------------------------------------------------------------
        bb_float = 16 if 16 in ref_float_by_bb else BIT_DEPTHS[-1]
        max_val_float = float(2**bb_float - 1)
        r_info = custom_data[bb_float]["rect"]
        linestyles_ref = {8: "-", 12: "--", 16: ":"}
        bit_colors = {8: "#1f77b4", 12: "#2ca02c", 16: "#ff7f0e"}
        bit_markers = {8: "o", 12: "s", 16: "^"}
        mismatch_floor = 0.2
        y_ticks = [mismatch_floor, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

        if ANALYSIS_MODE == "func":
            # Plot A1: Float RMSE Convergence
            plt.figure(figsize=(11, 7))
            # Custom Rect (blue)
            r_info = custom_data[bb_float]["rect"]
            if r_info["samples"]:
                idx = np.argsort(r_info["samples"])
                plt.loglog(
                    np.array(r_info["samples"])[idx],
                    np.array(r_info["e_f64"])[idx],
                    marker="o",
                    color="#1f77b4",
                    label="Custom, Rect",
                    linewidth=2.0,
                    markersize=8,
                )
            # Custom Gauss (green)
            g_info = custom_data[bb_float]["gauss"]
            if g_info["samples"]:
                idx = np.argsort(g_info["samples"])
                plt.loglog(
                    np.array(g_info["samples"])[idx],
                    np.array(g_info["e_f64"])[idx],
                    marker="s",
                    color="#2ca02c",
                    label="Custom, Gauss",
                    linewidth=2.0,
                    markersize=8,
                )
            # Riley Func (black, plotted last)
            f_info = riley_func[bb_float]
            if f_info["samples"]:
                idx = np.argsort(f_info["samples"])
                plt.loglog(
                    np.array(f_info["samples"])[idx],
                    np.array(f_info["e_f64"])[idx],
                    marker="x",
                    color="black",
                    label="Riley, Func",
                    linewidth=2.0,
                    linestyle="--",
                    markersize=9,
                )
    
            # LSB and 0.5 LSB lines
            linestyles_ref = {8: "-", 12: "--", 16: ":"}
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                mv = float(2**bb - 1)
                plt.axhline(
                    1.0 / mv,
                    color="black",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit LSB Line",
                )
                plt.axhline(
                    0.5 / mv,
                    color="red",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
                )
    
            plt.title(
                f"Riley Func Shader vs Custom Renderer: Floating-Point RMSE\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Floating-Point RMSE ($e_{f64}$)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR_FUNC
                / f"{case_name}_func_float_rmse_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
    
            # Plot A2: Float Max Error Convergence
            plt.figure(figsize=(11, 7))
            if r_info["samples"]:
                idx = np.argsort(r_info["samples"])
                plt.loglog(
                    np.array(r_info["samples"])[idx],
                    np.array(r_info["e_inf"])[idx],
                    marker="o",
                    color="#1f77b4",
                    label="Custom, Rect",
                    linewidth=2.0,
                    markersize=8,
                )
            if g_info["samples"]:
                idx = np.argsort(g_info["samples"])
                plt.loglog(
                    np.array(g_info["samples"])[idx],
                    np.array(g_info["e_inf"])[idx],
                    marker="s",
                    color="#2ca02c",
                    label="Custom, Gauss",
                    linewidth=2.0,
                    markersize=8,
                )
            if f_info["samples"]:
                idx = np.argsort(f_info["samples"])
                plt.loglog(
                    np.array(f_info["samples"])[idx],
                    np.array(f_info["e_inf"])[idx],
                    marker="x",
                    color="black",
                    label="Riley, Func",
                    linewidth=2.0,
                    linestyle="--",
                    markersize=9,
                )
    
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                mv = float(2**bb - 1)
                plt.axhline(
                    1.0 / mv,
                    color="black",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit LSB Line",
                )
                plt.axhline(
                    0.5 / mv,
                    color="red",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
                )
    
            plt.title(
                f"Riley Func Shader vs Custom Renderer: Floating-Point Max Error\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Floating-Point Max Error ($e_{\\infty}$)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR_FUNC
                / f"{case_name}_func_float_max_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
    
            # Plot A3: Digitised Mismatch Fraction (delta_b)
            plt.figure(figsize=(11, 7))
            bit_colors = {8: "#1f77b4", 12: "#2ca02c", 16: "#ff7f0e"}
            bit_markers = {8: "o", 12: "s", 16: "^"}
    
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                # Custom Rect
                cr = custom_data[bb]["rect"]
                if cr["samples"]:
                    idx = np.argsort(cr["samples"])
                    plt.plot(
                        np.array(cr["samples"])[idx],
                        np.array(cr["delta_b"])[idx],
                        marker=bit_markers[bb],
                        color=bit_colors[bb],
                        label=f"Custom, Rect, {bb}-bit",
                        linewidth=1.5,
                        markersize=6,
                    )
                # Riley Func
                rf = riley_func[bb]
                if rf["samples"]:
                    idx = np.argsort(rf["samples"])
                    plt.plot(
                        np.array(rf["samples"])[idx],
                        np.array(rf["delta_b"])[idx],
                        marker="x",
                        color="black",
                        label=f"Riley, Func, {bb}-bit",
                        linewidth=1.5,
                        linestyle="--",
                        markersize=8,
                    )
    
            plt.xscale("log")
            plt.title(
                f"Riley Func Shader vs Custom: Digitised Mismatch Fraction\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Fraction of Differing Pixels ($\\delta_b$)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.ylim(-0.05, 1.05)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR_FUNC
                / f"{case_name}_func_bits_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
    
            # Plot A4: Max Digitised Mismatch (max_eb)
            plt.figure(figsize=(11, 7))
            mismatch_floor = 0.2
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                cr = custom_data[bb]["rect"]
                if cr["samples"]:
                    idx = np.argsort(cr["samples"])
                    s_s = np.array(cr["samples"])[idx]
                    m_s = np.array(cr["max_eb"])[idx]
                    plt.loglog(
                        s_s,
                        np.where(m_s == 0.0, mismatch_floor, m_s),
                        marker=bit_markers[bb],
                        color=bit_colors[bb],
                        label=f"Custom, Rect, {bb}-bit",
                        linewidth=1.5,
                        markersize=6,
                    )
                rf = riley_func[bb]
                if rf["samples"]:
                    idx = np.argsort(rf["samples"])
                    s_s = np.array(rf["samples"])[idx]
                    m_s = np.array(rf["max_eb"])[idx]
                    plt.loglog(
                        s_s,
                        np.where(m_s == 0.0, mismatch_floor, m_s),
                        marker="x",
                        color="black",
                        label=f"Riley, Func, {bb}-bit",
                        linewidth=1.5,
                        linestyle="--",
                        markersize=8,
                    )
    
            # Reference horizontal threshold lines
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                plt.axhline(
                    1.0,
                    color="black",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit LSB Limit",
                )
            plt.axhline(
                mismatch_floor,
                color="red",
                linestyle=":",
                alpha=0.6,
                linewidth=1.2,
                label="No Mismatch (0 LSB)",
            )
    
            plt.title(
                f"Riley Func Shader vs Custom: Max Digitised Mismatch\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Maximum Digitised Mismatch (LSB levels)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            # Explicit integer y-ticks
            y_ticks = [mismatch_floor, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            plt.yticks(y_ticks, ["0", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"])
            plt.ylim(mismatch_floor * 0.8, 4096.0)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR_FUNC
                / f"{case_name}_func_max_eb_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
            func_paths = [
                RESULTS_DIR_FUNC / f"{case_name}_func_float_rmse_frame{ff:02d}.png",
                RESULTS_DIR_FUNC / f"{case_name}_func_float_max_frame{ff:02d}.png",
                RESULTS_DIR_FUNC / f"{case_name}_func_bits_frame{ff:02d}.png",
                RESULTS_DIR_FUNC / f"{case_name}_func_max_eb_frame{ff:02d}.png",
            ]
            _combine_metric_panels(
                func_paths,
                RESULTS_DIR_FUNC / f"{case_name}_func_metrics_frame{ff:02d}.png",
            )

        if ANALYSIS_MODE == "tex":
            # ------------------------------------------------------------------
            # GENERATE PLOTS - GROUP B: Riley Texture Shader vs Custom Baseline
            # ------------------------------------------------------------------
            # Plot B1: Float RMSE Convergence (Texture Shader)
            plt.figure(figsize=(11, 7))
            # Custom Rect Baseline
            if r_info["samples"]:
                idx = np.argsort(r_info["samples"])
                plt.loglog(
                    np.array(r_info["samples"])[idx],
                    np.array(r_info["e_f64"])[idx],
                    marker="o",
                    color="#1f77b4",
                    label="Custom, Rect",
                    linewidth=2.0,
                    markersize=8,
                )
    
            tex_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            tex_markers = ("^", "v", "<", ">", "p", "h", "D", "X", "*")
            tex_styles = {
                oversamp: (
                    tex_colors[index % len(tex_colors)],
                    tex_markers[index % len(tex_markers)],
                )
                for index, oversamp in enumerate(TEX_OVERSAMPLES)
            }
            diagnostic_oversamp = max(
                oversamp
                for oversamp in TEX_OVERSAMPLES
                if any(riley_tex[bit_depth][oversamp]["samples"] for bit_depth in BIT_DEPTHS)
            )
    
            for oversamp in TEX_OVERSAMPLES:
                rt_info = riley_tex[bb_float][oversamp]
                if rt_info["samples"]:
                    idx = np.argsort(rt_info["samples"])
                    color, marker = tex_styles[oversamp]
                    plt.loglog(
                        np.array(rt_info["samples"])[idx],
                        np.array(rt_info["e_f64"])[idx],
                        marker=marker,
                        color=color,
                        label=f"Riley, Tex, OS={oversamp}",
                        linewidth=1.5,
                        markersize=7,
                    )
    
            # LSB and 0.5 LSB lines
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                mv = float(2**bb - 1)
                plt.axhline(
                    1.0 / mv,
                    color="black",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit LSB Line",
                )
                plt.axhline(
                    0.5 / mv,
                    color="red",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
                )
    
            plt.title(
                f"Riley Tex Shader vs Custom: Floating-Point RMSE\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Floating-Point RMSE ($e_{f64}$)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                results_dir_tex
                / f"{case_name}_tex_float_rmse_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
    
            # Plot B2: Float Max Error Convergence (Texture Shader)
            plt.figure(figsize=(11, 7))
            if r_info["samples"]:
                idx = np.argsort(r_info["samples"])
                plt.loglog(
                    np.array(r_info["samples"])[idx],
                    np.array(r_info["e_inf"])[idx],
                    marker="o",
                    color="#1f77b4",
                    label="Custom, Rect",
                    linewidth=2.0,
                    markersize=8,
                )
    
            for oversamp in TEX_OVERSAMPLES:
                rt_info = riley_tex[bb_float][oversamp]
                if rt_info["samples"]:
                    idx = np.argsort(rt_info["samples"])
                    color, marker = tex_styles[oversamp]
                    plt.loglog(
                        np.array(rt_info["samples"])[idx],
                        np.array(rt_info["e_inf"])[idx],
                        marker=marker,
                        color=color,
                        label=f"Riley, Tex, OS={oversamp}",
                        linewidth=1.5,
                        markersize=7,
                    )
    
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                mv = float(2**bb - 1)
                plt.axhline(
                    1.0 / mv,
                    color="black",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit LSB Line",
                )
                plt.axhline(
                    0.5 / mv,
                    color="red",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
                )
    
            plt.title(
                f"Riley Tex Shader vs Custom: Floating-Point Max Error\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Floating-Point Max Error ($e_{\\infty}$)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                results_dir_tex
                / f"{case_name}_tex_float_max_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
    
            # Plot B3: Digitised Mismatch Fraction (Texture Shader)
            plt.figure(figsize=(11, 7))
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                cr = custom_data[bb]["rect"]
                if cr["samples"]:
                    idx = np.argsort(cr["samples"])
                    plt.plot(
                        np.array(cr["samples"])[idx],
                        np.array(cr["delta_b"])[idx],
                        marker=bit_markers[bb],
                        color=bit_colors[bb],
                        label=f"Custom, Rect, {bb}-bit",
                        linewidth=1.5,
                        markersize=6,
                    )
    
                rt = riley_tex[bb][diagnostic_oversamp]
                if rt["samples"]:
                    idx = np.argsort(rt["samples"])
                    plt.plot(
                        np.array(rt["samples"])[idx],
                        np.array(rt["delta_b"])[idx],
                        marker="x",
                        color=bit_colors[bb],
                        label=f"Riley, Tex, OS={diagnostic_oversamp}, {bb}-bit",
                        linewidth=1.5,
                        linestyle="--",
                        markersize=8,
                    )
    
            plt.xscale("log")
            plt.title(
                f"Riley Tex Shader vs Custom: Digitised Mismatch Fraction\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Fraction of Differing Pixels ($\\delta_b$)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.ylim(-0.05, 1.05)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                results_dir_tex
                / f"{case_name}_tex_bits_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
    
            # Plot B4: Max Digitised Mismatch (Texture Shader)
            plt.figure(figsize=(11, 7))
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                cr = custom_data[bb]["rect"]
                if cr["samples"]:
                    idx = np.argsort(cr["samples"])
                    s_s = np.array(cr["samples"])[idx]
                    m_s = np.array(cr["max_eb"])[idx]
                    plt.loglog(
                        s_s,
                        np.where(m_s == 0.0, mismatch_floor, m_s),
                        marker=bit_markers[bb],
                        color=bit_colors[bb],
                        label=f"Custom, Rect, {bb}-bit",
                        linewidth=1.5,
                        markersize=6,
                    )
    
                rt = riley_tex[bb][diagnostic_oversamp]
                if rt["samples"]:
                    idx = np.argsort(rt["samples"])
                    s_s = np.array(rt["samples"])[idx]
                    m_s = np.array(rt["max_eb"])[idx]
                    plt.loglog(
                        s_s,
                        np.where(m_s == 0.0, mismatch_floor, m_s),
                        marker="x",
                        color=bit_colors[bb],
                        label=f"Riley, Tex, OS={diagnostic_oversamp}, {bb}-bit",
                        linewidth=1.5,
                        linestyle="--",
                        markersize=8,
                    )
    
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                plt.axhline(
                    1.0,
                    color="black",
                    linestyle=linestyles_ref[bb],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{bb}-bit LSB Limit",
                )
            plt.axhline(
                mismatch_floor,
                color="red",
                linestyle=":",
                alpha=0.6,
                linewidth=1.2,
                label="No Mismatch (0 LSB)",
            )
    
            plt.title(
                f"Riley Tex Shader vs Custom: Max Digitised Mismatch\n"
                f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Samples Along One Pixel Axis", fontsize=10)
            plt.ylabel("Maximum Digitised Mismatch (LSB levels)", fontsize=10)
            plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
            plt.yticks(y_ticks, [str(yt) for yt in y_ticks])
            plt.ylim(mismatch_floor * 0.8, 4096.0)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="lower left",
                fontsize=9,
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(
                results_dir_tex
                / f"{case_name}_tex_max_eb_frame{ff:02d}.png",
                dpi=150,
            )
            plt.close()
            tex_paths = [
                results_dir_tex / f"{case_name}_tex_float_rmse_frame{ff:02d}.png",
                results_dir_tex / f"{case_name}_tex_float_max_frame{ff:02d}.png",
                results_dir_tex / f"{case_name}_tex_bits_frame{ff:02d}.png",
                results_dir_tex / f"{case_name}_tex_max_eb_frame{ff:02d}.png",
            ]
            _combine_metric_panels(
                tex_paths,
                results_dir_tex / f"{case_name}_tex_metrics_frame{ff:02d}.png",
            )
            (results_dir_tex / f"{case_name}_tex_metrics_frame{ff:02d}.png").unlink(
                missing_ok=True
            )
            # Superseded by the two exponent-parity OS figures below.
            (results_dir_tex / f"{case_name}_tex_oversamp_metrics_frame{ff:02d}.png").unlink(
                missing_ok=True
            )
            available_os = sorted(
                oversamp for oversamp in TEX_OVERSAMPLES
                if any(riley_tex[bit_depth][oversamp]["samples"] for bit_depth in BIT_DEPTHS)
            )
            available_ssaa = sorted(
                {
                    int(round(sample))
                    for oversamp in available_os
                    for sample in riley_tex[16 if 16 in BIT_DEPTHS else max(BIT_DEPTHS)][oversamp]["samples"]
                }
            )
            odd_os = [value for value in available_os if value.bit_length() % 2 == 0]
            even_os = [value for value in available_os if value.bit_length() % 2 == 1]
            odd_ssaa = [value for value in available_ssaa if value.bit_length() % 2 == 0]
            even_ssaa = [value for value in available_ssaa if value.bit_length() % 2 == 1]
            for group_name, selected_os, selected_ssaa in (
                ("odd_exp", odd_os, odd_ssaa),
                ("even_exp", even_os, even_ssaa),
            ):
                _plot_texture_ssaa_metrics(
                    riley_tex, case_name, ff, results_dir_tex, selected_os, group_name
                )
                _plot_texture_oversample_metrics(
                    riley_tex, case_name, ff, results_dir_tex, selected_ssaa, group_name
                )
            # Keep complete per-bit studies alongside the odd/even exponent
            # figures.  These make within-bit trends legible without removing
            # the across-bit comparison in the split figures.
            for bit_depth in BIT_DEPTHS:
                _plot_texture_ssaa_metrics(
                    riley_tex, case_name, ff, results_dir_tex, available_os,
                    "all", bit_depth,
                )
                _plot_texture_oversample_metrics(
                    riley_tex, case_name, ff, results_dir_tex, available_ssaa,
                    "all", bit_depth,
                )
            _plot_texture_limit_curves(riley_tex, case_name, ff, results_dir_tex)


def _load_riley_pair(npy_path: Path, tiff_path: Path, bit_depth: int):
    """Load one Riley render in normalised float and digitised form."""
    if not npy_path.exists() or not tiff_path.exists():
        return None
    with Image.open(tiff_path) as image:
        digitised = np.asarray(image, dtype=np.float64)
    return np.load(npy_path) / float(2**bit_depth - 1), digitised


def analyse_riley_self_convergence(
    case_name: str,
    tex_interp: str,
    output_dir: Path,
) -> None:
    """Compare each Riley SSAA result with its highest available SSAA result."""
    case_name = output_case_name(case_name, TARG_PX_X)
    mode = ANALYSIS_MODE
    if mode not in {"func", "tex"}:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in ACTIVE_FRAMES:
        sources = (
            [("Riley, Func", None)]
            if mode == "func"
            else [(f"Riley, Tex, OS={oversamp}", oversamp) for oversamp in TEX_OVERSAMPLES]
        )
        labels = [label for label, _ in sources]
        records = []
        for label, oversamp in sources:
            for bit_depth in BIT_DEPTHS:
                def pair_paths(ssaa: int) -> tuple[Path, Path]:
                    if oversamp is None:
                        directory = RILEY_FUNC_DIR / case_name / f"ss{ssaa}_b{bit_depth}"
                    else:
                        directory = (
                            RILEY_TEX_DIR / f"{case_name}_{tex_interp}"
                            / f"ss{ssaa}_b{bit_depth}_oversamp{oversamp}"
                        )
                    return (
                        directory / f"image_c00_f{frame:02d}.npy",
                        directory / f"cam0_frame{frame}_field0.tiff",
                    )

                available_ssaa = [
                    ssaa for ssaa in SSAA_LEVELS
                    if all(path.exists() for path in pair_paths(ssaa))
                ]
                if len(available_ssaa) < 2:
                    continue
                ref_ssaa = max(available_ssaa)
                reference = _load_riley_pair(*pair_paths(ref_ssaa), bit_depth)
                if reference is None:
                    continue
                ref_float, ref_digitised = reference
                for ssaa in available_ssaa:
                    if ssaa == ref_ssaa:
                        continue
                    image = _load_riley_pair(*pair_paths(ssaa), bit_depth)
                    if image is None:
                        continue
                    image_float, image_digitised = image
                    float_diff = image_float - ref_float
                    digitised_diff = image_digitised - ref_digitised
                    records.append(
                        {
                            "label": label,
                            "bit_depth": bit_depth,
                            "ssaa": ssaa,
                            "ref_ssaa": ref_ssaa,
                            "e_f64": float(np.sqrt(np.mean(float_diff**2))),
                            "e_inf": float(np.max(np.abs(float_diff))),
                            "delta_b": float(np.mean(image_digitised != ref_digitised)),
                            "max_eb": float(np.max(np.abs(digitised_diff))),
                        }
                    )
                    del image_float, image_digitised, float_diff, digitised_diff
                del ref_float, ref_digitised
        if not records:
            print(f"Warning: {case_name}, frame {frame:02d}: insufficient Riley SSAA data for self convergence.")
            del records, labels
            gc.collect()
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        preferred_bit_depth = 16 if 16 in BIT_DEPTHS else max(BIT_DEPTHS)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for index, label in enumerate(labels):
            color = colors[index % len(colors)]
            float_records = [
                record for record in records
                if record["label"] == label and record["bit_depth"] == preferred_bit_depth
            ]
            if float_records:
                float_records.sort(key=lambda record: record["ssaa"])
                ref_ssaa = float_records[0]["ref_ssaa"]
                samples = [record["ssaa"] for record in float_records]
                axes[0, 0].loglog(samples, [record["e_f64"] for record in float_records], marker="o", color=color, label=f"{label}, ref SSAA={ref_ssaa}")
                axes[0, 1].loglog(samples, [record["e_inf"] for record in float_records], marker="o", color=color, label=f"{label}, ref SSAA={ref_ssaa}")
            for bit_depth in BIT_DEPTHS:
                digitised_records = [
                    record for record in records
                    if record["label"] == label and record["bit_depth"] == bit_depth
                ]
                if not digitised_records:
                    continue
                digitised_records.sort(key=lambda record: record["ssaa"])
                samples = [record["ssaa"] for record in digitised_records]
                axes[1, 0].semilogx(samples, [record["delta_b"] for record in digitised_records], marker="o", color=color, linestyle={8: "-", 12: "--", 16: ":"}.get(bit_depth, "-"), label=f"{label}, {bit_depth}-bit")
                axes[1, 1].loglog(samples, np.maximum(0.2, [record["max_eb"] for record in digitised_records]), marker="o", color=color, linestyle={8: "-", 12: "--", 16: ":"}.get(bit_depth, "-"), label=f"{label}, {bit_depth}-bit")

        for bit_depth in BIT_DEPTHS:
            max_value = float(2**bit_depth - 1)
            axes[0, 0].axhline(1.0 / max_value, color="black", linestyle={8: "-", 12: "--", 16: ":"}.get(bit_depth, "-"), alpha=0.35)
            axes[0, 1].axhline(0.5 / max_value, color="red", linestyle={8: "-", 12: "--", 16: ":"}.get(bit_depth, "-"), alpha=0.35)
        axes[1, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6, label="1 LSB")
        axes[1, 1].axhline(0.2, color="red", linestyle=":", alpha=0.6, label="0 LSB")
        titles = ("Floating-Point RMSE", "Floating-Point Max Error", "Digitised Mismatch Fraction", "Maximum Digitised Mismatch")
        ylabels = ("RMSE", "Max error", "Fraction of differing pixels", "LSB levels")
        sample_ticks = sorted(
            {
                record["ssaa"]
                for record in records
            }
            | {
                record["ref_ssaa"]
                for record in records
            }
        )
        for axis, title, ylabel in zip(axes.flat, titles, ylabels):
            axis.set_title(title)
            axis.set_xlabel("Samples Along One Pixel Axis")
            axis.set_ylabel(ylabel)
            axis.xaxis.set_major_locator(FixedLocator(sample_ticks))
            axis.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in sample_ticks]))
            axis.set_xlim(
                0.85,
                1.15 * max(
                    max(record["ssaa"], record["ref_ssaa"]) for record in records
                ),
            )
            axis.grid(True, which="both", ls="--", alpha=0.4)
        handles = []
        for index, label in enumerate(labels):
            reference_ssaa = max(
                record["ref_ssaa"] for record in records if record["label"] == label
            )
            handles.append(
                Line2D(
                    [], [], color=colors[index % len(colors)], marker="o",
                    label=f"{label}, ref SSAA={reference_ssaa}",
                )
            )
        handles.extend(
            Line2D([], [], color="black", linestyle=style, label=f"{bit_depth}-bit")
            for bit_depth, style in ((8, "-"), (12, "--"), (16, ":"))
            if bit_depth in BIT_DEPTHS
        )
        handles.extend(
            [
                Line2D([], [], color="black", linestyle="--", label="1 LSB"),
                Line2D([], [], color="red", linestyle=":", label="0 LSB"),
            ]
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
            fontsize=6,
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )
        axes[1, 1].set_ylim(0.16, None)
        fig.suptitle(f"Riley Self-Convergence: {case_name} (Frame {frame:02d})\nReference: highest available SSAA per series", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        prefix = "func" if mode == "func" else "tex"
        fig.savefig(output_dir / f"{case_name}_{prefix}_self_convergence_frame{frame:02d}.png", dpi=150)
        fig.clear()
        plt.close(fig)
        del records, labels
        gc.collect()


def main() -> None:
    global ACTIVE_FRAMES
    timer = ScriptTimer(__file__)

    frames_str = os.environ.get("EXP1_ACTIVE_FRAMES")
    cases_str = os.environ.get("EXP1_CASES")
    interps_str = os.environ.get("EXP1_TEX_INTERPOLATORS")
    is_subset_analysis = bool(frames_str or interps_str)
    if frames_str:
        ACTIVE_FRAMES = [
            int(val.strip()) for val in frames_str.split(",") if val.strip()
        ]

    results_dir = RESULTS_DIR_FUNC if ANALYSIS_MODE == "func" else RESULTS_DIR_TEX
    rectconv_dir = Path(f"{results_dir}_rectconv")
    if CLEAR_DIR and not is_subset_analysis:
        shutil.rmtree(results_dir, ignore_errors=True)
        shutil.rmtree(rectconv_dir, ignore_errors=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rectconv_dir.mkdir(parents=True, exist_ok=True)

    cases = (
        [case.strip() for case in cases_str.split(",") if case.strip()]
        if cases_str
        else DEFORMATION_CASES
    )
    interps = (
        TEX_INTERPOLATORS
        if not interps_str
        else tuple(
            interp.strip()
            for interp in interps_str.split(",")
            if interp.strip()
        )
    )
    invalid = set(interps).difference(TEX_INTERPOLATORS)
    if invalid:
        raise ValueError(
            f"Unsupported texture interpolator(s): {', '.join(sorted(invalid))}"
        )
    for tex_interp in interps:
        for case_name in cases:
            timed_call(timer, f"{case_name}_{tex_interp}", analyze_riley_case, case_name, tex_interp)
            self_convergence_dir = rectconv_dir if ANALYSIS_MODE == "func" else rectconv_dir / tex_interp
            timed_call(
                timer,
                f"{case_name}_{tex_interp}_rectconv",
                analyse_riley_self_convergence,
                case_name,
                tex_interp,
                self_convergence_dir,
            )
            plt.close("all")
            gc.collect()

    print("\nRiley analysis completed successfully.")


if __name__ == "__main__":
    main()
