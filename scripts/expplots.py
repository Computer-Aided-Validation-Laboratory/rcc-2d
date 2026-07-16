# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Shared plotting utilities for bespoke two-dimensional renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "rect": ("#1f77b4", "o", "Rectangular / SSAA"),
    "gauss": ("#2ca02c", "s", "Gauss Quadrature"),
    "mc": ("#ff7f0e", "^", "Monte Carlo"),
}
BIT_LINESTYLES = {8: "-", 12: "--", 16: ":"}
SAMPLE_TICKS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]


def samples_for_method(method: str, param: int) -> int:
    """Return the total sample count represented by one integration setting."""
    return param * param if method in {"rect", "gauss"} else param


def _style(method: str) -> tuple[str, str, str]:
    return METHOD_STYLES.get(method, ("#7f7f7f", "D", method.title()))


def _plot_float_metric(
    metric: str,
    ylabel: str,
    filename_suffix: str,
    title_metric: str,
    case_name: str,
    frame: int,
    reference_name: str,
    output_dir: Path,
    float_data: Mapping[str, Mapping[str, Sequence[float]]],
    bit_depths: Sequence[int],
) -> Path:
    plt.figure(figsize=(11, 7))
    for method, values in float_data.items():
        if not values["samples"]:
            continue
        order = np.argsort(values["samples"])
        samples = np.asarray(values["samples"])[order]
        errors = np.asarray(values[metric])[order]
        valid = errors > 0.0
        if not np.any(valid):
            continue
        color, marker, label = _style(method)
        plt.loglog(samples[valid], errors[valid], marker=marker, color=color,
                   label=label, linewidth=2.0, markersize=8)

    for bit_depth in bit_depths:
        max_value = float(2**bit_depth - 1)
        linestyle = BIT_LINESTYLES.get(bit_depth, "-.")
        plt.axhline(1.0 / max_value, color="black", linestyle=linestyle,
                    alpha=0.6, linewidth=1.2,
                    label=f"{bit_depth}-bit LSB Line")
        plt.axhline(0.5 / max_value, color="red", linestyle=linestyle,
                    alpha=0.6, linewidth=1.2,
                    label=f"{bit_depth}-bit No Pixels Diff (0.5 LSB)")

    plt.title(f"{title_metric} Convergence:\n{case_name} (Frame {frame:02d}) "
              f"| Reference: {reference_name}", fontsize=12,
              fontweight="bold", pad=15)
    plt.xlabel("Total Samples per Pixel", fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(SAMPLE_TICKS, [str(tick) for tick in SAMPLE_TICKS])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none",
               loc="lower left", fontsize=9, ncol=2)
    plt.tight_layout()
    path = output_dir / f"convergence_{case_name}_{filename_suffix}_frame{frame:02d}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_bespoke_convergence(
    case_name: str,
    frame: int,
    reference_name: str,
    output_dir: Path,
    float_data: Mapping[str, Mapping[str, Sequence[float]]],
    digitised_data: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    bit_depths: Sequence[int],
) -> list[Path]:
    """Write the four standard convergence plots for bespoke renderers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_float_metric("e_f64", "Floating-Point RMSE ($e_{f64}$)",
                           "float_rmse", "Continuous Floating-Point RMSE ($e_{f64}$)",
                           case_name, frame, reference_name, output_dir,
                           float_data, bit_depths),
        _plot_float_metric("e_inf", "Floating-Point Max Error ($e_{\\infty}$)",
                           "float_max", "Continuous Floating-Point Max Error ($e_{\\infty}$)",
                           case_name, frame, reference_name, output_dir,
                           float_data, bit_depths),
    ]

    plt.figure(figsize=(11, 7))
    floor = 0.2
    for bit_depth in bit_depths:
        for method, values in digitised_data.get(bit_depth, {}).items():
            if not values["samples"]:
                continue
            order = np.argsort(values["samples"])
            color, marker, label = _style(method)
            mismatch = np.asarray(values["max_eb"])[order]
            plt.loglog(np.asarray(values["samples"])[order],
                       np.where(mismatch == 0.0, floor, mismatch),
                       linestyle=BIT_LINESTYLES.get(bit_depth, "-."),
                       marker=marker, color=color, label=f"{label} ({bit_depth}-bit)",
                       linewidth=1.8, markersize=7)
    plt.axhline(1.0, color="black", linestyle="-", alpha=0.6, linewidth=1.2,
                label="1 LSB Mismatch Line")
    plt.axhline(floor, color="red", linestyle=":", alpha=0.6, linewidth=1.2,
                label="No Mismatch (0 LSB)")
    plt.title(f"Digitised Maximum Error Convergence (LSB Mismatch):\n{case_name} "
              f"(Frame {frame:02d}) | Reference: {reference_name}", fontsize=12,
              fontweight="bold", pad=15)
    plt.xlabel("Total Samples per Pixel")
    plt.ylabel("Maximum Digitised Mismatch (LSB levels)")
    plt.xticks(SAMPLE_TICKS, [str(tick) for tick in SAMPLE_TICKS])
    plt.yticks([0.2, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
               ["0", "1", "2", "5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000"])
    plt.ylim(floor * 0.8, 10000.0)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower left",
               fontsize=9, ncol=2)
    plt.tight_layout()
    max_path = output_dir / f"convergence_{case_name}_max_eb_frame{frame:02d}.png"
    plt.savefig(max_path, dpi=150)
    plt.close()
    paths.append(max_path)

    plt.figure(figsize=(11, 7))
    for bit_depth in bit_depths:
        for method, values in digitised_data.get(bit_depth, {}).items():
            if not values["samples"]:
                continue
            order = np.argsort(values["samples"])
            color, marker, label = _style(method)
            plt.plot(np.asarray(values["samples"])[order],
                     np.asarray(values["delta_b"])[order],
                     linestyle=BIT_LINESTYLES.get(bit_depth, "-."), marker=marker,
                     color=color, label=f"{label} ({bit_depth}-bit)",
                     linewidth=1.8, markersize=7)
    plt.axhline(0.0, color="red", linestyle=":", alpha=0.6, linewidth=1.2,
                label="No Pixels Different (0 pixels)")
    plt.xscale("log")
    plt.title(f"Fraction of Differing Pixels ($\\delta_b$):\n{case_name} "
              f"(Frame {frame:02d}) | Reference: {reference_name}", fontsize=12,
              fontweight="bold", pad=15)
    plt.xlabel("Total Samples per Pixel")
    plt.ylabel("Fraction of Differing Pixels ($\\delta_b$)")
    plt.xticks(SAMPLE_TICKS, [str(tick) for tick in SAMPLE_TICKS])
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower left",
               fontsize=9, ncol=2)
    plt.tight_layout()
    fraction_path = output_dir / f"convergence_{case_name}_bits_frame{frame:02d}.png"
    plt.savefig(fraction_path, dpi=150)
    plt.close()
    paths.append(fraction_path)
    return paths
