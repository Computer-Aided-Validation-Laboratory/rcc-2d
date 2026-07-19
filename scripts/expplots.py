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
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np


METHOD_STYLES = {
    "rect": ("#1f77b4", "o", "Custom, Rect"),
    "gauss": ("#2ca02c", "s", "Custom, Gauss"),
    "mc": ("#ff7f0e", "^", "Custom, MC"),
}
BIT_LINESTYLES = {8: "-", 12: "--", 16: ":"}
# The input data records total samples per pixel.  Figures show the equivalent
# number of samples along one pixel axis, so a two-dimensional N-by-N grid is
# plotted at N rather than N².
SAMPLE_TICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def samples_for_method(method: str, param: int) -> int:
    """Return the total sample count represented by one integration setting."""
    return param * param if method in {"rect", "gauss"} else param


def samples_per_pixel_axis(samples: Sequence[float]) -> np.ndarray:
    """Convert total per-pixel sample counts to their one-axis equivalent."""
    return np.sqrt(np.asarray(samples, dtype=float))


def _style(method: str) -> tuple[str, str, str]:
    return METHOD_STYLES.get(method, ("#7f7f7f", "D", method.title()))


def _set_sample_axis(axis, samples: Sequence[float]) -> None:
    """Label a logarithmic x axis from the sample levels actually plotted."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        return
    ticks = np.unique(np.sort(values))
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in ticks]))
    axis.set_xlim(float(ticks[0]) * 0.85, float(ticks[-1]) * 1.15)


def plot_bespoke_four_panel(
    case_name: str,
    frame: int,
    reference_name: str,
    output_dir: Path,
    float_data: Mapping[str, Mapping[str, Sequence[float]]],
    digitised_data: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    bit_depths: Sequence[int],
) -> Path:
    """Write the standard metrics as one four-panel convergence figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("float_rmse", "float_max", "bits", "max_eb"):
        (output_dir / f"{case_name}_{suffix}_frame{frame:02d}.png").unlink(
            missing_ok=True
        )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    all_samples: list[float] = []
    for values in float_data.values():
        all_samples.extend(samples_per_pixel_axis(values["samples"]))
    for methods in digitised_data.values():
        for values in methods.values():
            all_samples.extend(samples_per_pixel_axis(values["samples"]))

    for metric, axis, title, ylabel in (
        ("e_f64", axes[0, 0], "Floating-Point RMSE", "RMSE ($e_{f64}$)"),
        ("e_inf", axes[0, 1], "Floating-Point Max Error", "Max error ($e_{∞}$)"),
    ):
        for method, values in float_data.items():
            if not values["samples"]:
                continue
            order = np.argsort(values["samples"])
            samples = samples_per_pixel_axis(np.asarray(values["samples"])[order])
            errors = np.asarray(values[metric])[order]
            valid = errors > 0.0
            if np.any(valid):
                color, marker, label = _style(method)
                axis.loglog(samples[valid], errors[valid], color=color, marker=marker,
                            label=label, linewidth=1.8, markersize=7)
        for bit_depth in bit_depths:
            maximum = float(2**bit_depth - 1)
            style = BIT_LINESTYLES.get(bit_depth, "-.")
            axis.axhline(1.0 / maximum, color="black", linestyle=style, alpha=0.4)
            axis.axhline(0.5 / maximum, color="red", linestyle=style, alpha=0.35)
        axis.set_title(title)
        axis.set_ylabel(ylabel)

    floor = 0.2
    for bit_depth in bit_depths:
        for method, values in digitised_data.get(bit_depth, {}).items():
            if not values["samples"]:
                continue
            order = np.argsort(values["samples"])
            samples = samples_per_pixel_axis(np.asarray(values["samples"])[order])
            color, marker, label = _style(method)
            style = BIT_LINESTYLES.get(bit_depth, "-.")
            axes[1, 0].semilogx(samples, np.asarray(values["delta_b"])[order], color=color,
                                marker=marker, linestyle=style, label=f"{label}, {bit_depth}-bit", linewidth=1.6)
            axes[1, 1].loglog(samples, np.maximum(floor, np.asarray(values["max_eb"])[order]),
                              color=color, marker=marker, linestyle=style,
                              label=f"{label}, {bit_depth}-bit", linewidth=1.6)
    axes[1, 0].axhline(0.0, color="red", linestyle=":", alpha=0.6, label="0 differing pixels")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title("Digitised Mismatch Fraction")
    axes[1, 0].set_ylabel("Fraction of differing pixels")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6, label="1 LSB")
    axes[1, 1].axhline(floor, color="red", linestyle=":", alpha=0.6, label="0 LSB")
    axes[1, 1].set_ylim(floor * 0.8, None)
    axes[1, 1].set_title("Maximum Digitised Mismatch")
    axes[1, 1].set_ylabel("LSB levels")
    for axis in axes.flat:
        axis.set_xlabel("Samples Along One Pixel Axis")
        _set_sample_axis(axis, all_samples)
        axis.grid(True, which="both", ls="--", alpha=0.4)
    for axis in axes.flat:
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                loc="lower left",
                fontsize=6,
                frameon=True,
                facecolor="white",
                edgecolor="none",
            )
    figure.suptitle(f"{case_name} (Frame {frame:02d}) | Reference: {reference_name}", fontweight="bold")
    path = output_dir / f"{case_name}_metrics_frame{frame:02d}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
