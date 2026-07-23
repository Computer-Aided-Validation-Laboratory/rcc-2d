# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Analyse clamped Exp2 Riley floating-texture renders against references."""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

from matplotlib import rcParams
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from exp2params import (
    ACTIVE_FRAMES,
    ANALYTIC_SPECKLE_TYPES,
    BLACK_AREA_FRACTIONS,
    BIT_DEPTHS,
    DEFORMATION_CASES,
    additive_jitter_for,
    RANDOM_SEED,
    TEX_INTERPOLATORS,
    TARG_PX_X,
    exp2_output_dir,
)
from exp1common import output_case_name
from analysis_memory import make_agg_figure, release_batch, release_figure
from script_timing import ScriptTimer, timed_call


RILEY_OUTPUT_DIR = exp2_output_dir("exp2_riley_render_texfloat")
REFERENCE_OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs")
RESULTS_DIR = exp2_output_dir("exp2_riley_analysis_texfloat")
REFERENCE_SUFFIX = ""
RUN_RE = re.compile(r"^ss(?P<ssaa>\d+)_oversamp(?P<oversamp>\d+)$")
INTERPOLATOR_COLORS = rcParams["axes.prop_cycle"].by_key()["color"]
OVERSAMP_MARKERS = ("o", "s", "^", "v", "<", ">", "D", "P", "X")


def _selected_frames() -> list[int]:
    value = os.environ.get("EXP2_ACTIVE_FRAMES")
    if not value:
        return list(ACTIVE_FRAMES)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _selected_cases() -> list[str]:
    value = os.environ.get("EXP2_CASES")
    if not value:
        return list(DEFORMATION_CASES)
    return [item.strip() for item in value.split(",") if item.strip()]


def _selected_interpolators() -> set[str]:
    value = os.environ.get("EXP2_TEX_INTERPOLATORS")
    if not value:
        return set(TEX_INTERPOLATORS)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    invalid = selected.difference(TEX_INTERPOLATORS)
    if invalid:
        raise ValueError(
            "Unsupported texture interpolator(s): "
            f"{', '.join(sorted(invalid))}. Choose from: "
            f"{', '.join(TEX_INTERPOLATORS)}"
        )
    return selected


def pattern_tag(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
) -> str:
    return (
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_seed{RANDOM_SEED}"
    )


def _reference_path(directory: Path, method: str, param: int, frame: int) -> Path:
    return directory / (
        f"targ_px{TARG_PX_X}_int_{method}_param_{param}{REFERENCE_SUFFIX}_frame{frame:02d}.npy"
    )


def _reference_for_frame(
    case_name: str,
    tag: str,
    pattern_type: str,
    frame: int,
) -> tuple[np.ndarray, str, str, int] | None:
    """Load analytic, or the pattern-appropriate highest-rule reference."""
    base = f"{case_name}_{tag}_int_"
    analytic_dir = REFERENCE_OUTPUT_DIR / f"{base}analytic_param_0"
    analytic_path = _reference_path(analytic_dir, "analytic", 0, frame)
    if analytic_path.exists():
        return np.load(analytic_path), "Analytic", "analytic", 0

    method = "rect" if pattern_type == "diskaddsat" else "gauss"
    candidates: list[tuple[int, Path]] = []
    for directory in REFERENCE_OUTPUT_DIR.glob(f"{base}{method}_param_*"):
        try:
            param = int(directory.name.rsplit("_param_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((param, directory))
    for param, directory in sorted(candidates, reverse=True):
        path = _reference_path(directory, method, param, frame)
        if path.exists():
            kind = "Rectangular SSAA" if method == "rect" else "Gauss Quadrature"
            return np.load(path), f"{kind} ({param}x{param})", method, param
    return None


def _discover_riley_runs(
    case_name: str,
    tag: str,
    allowed_interpolators: set[str] | None,
) -> list[tuple[str, int, int, Path]]:
    """Discover every completed Riley run for one case/pattern tag."""
    prefix = f"{case_name}_{tag}_"
    runs: list[tuple[str, int, int, Path]] = []
    for interpolator_dir in RILEY_OUTPUT_DIR.glob(f"{prefix}*"):
        if not interpolator_dir.is_dir() or not interpolator_dir.name.startswith(prefix):
            continue
        interpolator = interpolator_dir.name[len(prefix):]
        if allowed_interpolators is not None and interpolator not in allowed_interpolators:
            continue
        for run_dir in interpolator_dir.iterdir():
            if not run_dir.is_dir():
                continue
            match = RUN_RE.match(run_dir.name)
            if match is None:
                continue
            runs.append(
                (
                    interpolator,
                    int(match.group("ssaa")),
                    int(match.group("oversamp")),
                    run_dir,
                )
            )
    return sorted(runs, key=lambda item: (item[0], item[2], item[1]))


def _clear_old_metric_images(output_dir: Path, frame: int) -> None:
    """Remove only obsolete pre-direct-plot output names for one frame."""
    for stem in (
        "metrics", "float_rmse", "float_max", "bits_delta", "bits_max",
        "limits_metrics", "limit_ssaa", "limit_oversamp",
    ):
        (output_dir / f"{stem}_frame{frame:02d}.png").unlink(missing_ok=True)


def _axis_samples(axis, values: list[int], label: str) -> None:
    """Apply the actual completed SSAA or OS levels to a log x axis."""
    ticks = sorted(set(values))
    if not ticks:
        return
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in ticks]))
    axis.set_xlim(0.85 * ticks[0], 1.15 * ticks[-1])
    axis.set_xlabel(label)


def _nonnegative_max(rows: list[dict[str, object]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if float(row[metric]) >= 0.0]
    return max(values, default=0.0)


def _set_float_axis(axis, rows: list[dict[str, object]], bit_depths: list[int]) -> None:
    """Use a zero-inclusive floating-error scale derived from image precision."""
    finest_half_lsb = 0.5 / float(2 ** max(bit_depths) - 1)
    coarsest_lsb = 1.0 / float(2 ** min(bit_depths) - 1)
    axis.set_yscale("symlog", linthresh=finest_half_lsb, linscale=0.8)
    for bit_depth in bit_depths:
        maximum = float(2**bit_depth - 1)
        axis.axhline(1.0 / maximum, color="black", linestyle="--", alpha=0.35)
        axis.axhline(0.5 / maximum, color="red", linestyle=":", alpha=0.35)
    axis.set_ylim(0.0, 1.15 * max(_nonnegative_max(rows, "e_f64"), _nonnegative_max(rows, "e_inf"), coarsest_lsb))


def _set_max_lsb_axis(axis, rows: list[dict[str, object]]) -> None:
    axis.set_yscale("symlog", linthresh=1.0, linscale=0.8)
    axis.axhline(1.0, color="black", linestyle="--", alpha=0.55, label="1 LSB")
    axis.axhline(0.0, color="red", linestyle=":", alpha=0.6, label="0 LSB")
    axis.set_ylim(0.0, 1.15 * max(_nonnegative_max(rows, "max_eb"), 1.0))


def _plot_four_panel(
    float_rows: list[dict[str, object]],
    digitised_rows: list[dict[str, object]],
    bit_depth: int,
    x_key: str,
    line_key: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot one bit depth, varying either SSAA or texture OS."""
    selected_digitised = [row for row in digitised_rows if int(row["BitDepth"]) == bit_depth]
    figure, axes = make_agg_figure(
        2, 2, figsize=(15, 10), constrained_layout=True
    )
    grouped_float: dict[int, list[dict[str, object]]] = defaultdict(list)
    grouped_digitised: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in float_rows:
        grouped_float[int(row[line_key])].append(row)
    for row in selected_digitised:
        grouped_digitised[int(row[line_key])].append(row)
    line_values = sorted(set(grouped_float) | set(grouped_digitised))
    colors = rcParams["axes.prop_cycle"].by_key()["color"]
    markers = OVERSAMP_MARKERS
    for index, line_value in enumerate(line_values):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        float_group = sorted(grouped_float.get(line_value, []), key=lambda row: int(row[x_key]))
        digitised_group = sorted(grouped_digitised.get(line_value, []), key=lambda row: int(row[x_key]))
        label = f"Riley, Tex, {'OS' if line_key == 'Oversamp' else 'SSAA'}={line_value}"
        for axis, metric in ((axes[0, 0], "e_f64"), (axes[0, 1], "e_inf")):
            if float_group:
                axis.plot(
                    [int(row[x_key]) for row in float_group],
                    [float(row[metric]) for row in float_group],
                    color=color, marker=marker, linewidth=1.6, markersize=6, label=label,
                )
        for axis, metric in ((axes[1, 0], "delta_b"), (axes[1, 1], "max_eb")):
            if digitised_group:
                axis.plot(
                    [int(row[x_key]) for row in digitised_group],
                    [float(row[metric]) for row in digitised_group],
                    color=color, marker=marker, linewidth=1.6, markersize=6, label=label,
                )
    x_values = [int(row[x_key]) for row in float_rows]
    _set_float_axis(axes[0, 0], float_rows, [bit_depth])
    _set_float_axis(axes[0, 1], float_rows, [bit_depth])
    _set_max_lsb_axis(axes[1, 1], selected_digitised)
    axes[1, 0].set_ylim(0.0, 1.0)
    for axis, panel_title, ylabel in (
        (axes[0, 0], "Floating-Point RMSE", "RMSE"),
        (axes[0, 1], "Floating-Point Max Error", "Max error"),
        (axes[1, 0], "Digitised Mismatch Fraction", "Fraction of differing pixels"),
        (axes[1, 1], "Maximum Digitised Mismatch", "LSB levels"),
    ):
        _axis_samples(axis, x_values, x_label)
        axis.set_title(panel_title)
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", ls="--", alpha=0.4)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="lower left", fontsize=6, frameon=True, facecolor="white", edgecolor="none")
    figure.suptitle(title, fontweight="bold")
    figure.savefig(output_path, dpi=150)
    release_figure(figure)


def _plot_limit_cuts(
    float_rows: list[dict[str, object]],
    digitised_rows: list[dict[str, object]],
    title: str,
    output_dir: Path,
    frame: int,
) -> None:
    """Write separate limiting cuts for texture OS and raster SSAA."""
    highest_os = max(int(row["Oversamp"]) for row in float_rows)
    lowest_os = min(int(row["Oversamp"]) for row in float_rows)
    highest_ssaa = max(int(row["SSAA"]) for row in float_rows)
    lowest_ssaa = min(int(row["SSAA"]) for row in float_rows)
    high_os_float = sorted((row for row in float_rows if int(row["Oversamp"]) == highest_os), key=lambda row: int(row["SSAA"]))
    high_ssaa_float = sorted((row for row in float_rows if int(row["SSAA"]) == highest_ssaa), key=lambda row: int(row["Oversamp"]))
    for suffix, rows_float, x_key, fixed_key, fixed_value, fixed_name, x_label in (
        ("max_ssaa", high_os_float, "SSAA", "Oversamp", highest_os, "OS", "Riley Samples Along One Pixel Axis"),
        ("max_oversamp", high_ssaa_float, "Oversamp", "SSAA", highest_ssaa, "SSAA", "Texture Oversampling Along One Pixel Axis"),
        ("min_ssaa", sorted((row for row in float_rows if int(row["Oversamp"]) == lowest_os), key=lambda row: int(row["SSAA"])), "SSAA", "Oversamp", lowest_os, "OS", "Riley Samples Along One Pixel Axis"),
        ("min_oversamp", sorted((row for row in float_rows if int(row["SSAA"]) == lowest_ssaa), key=lambda row: int(row["Oversamp"])), "Oversamp", "SSAA", lowest_ssaa, "SSAA", "Texture Oversampling Along One Pixel Axis"),
    ):
        figure, axes = make_agg_figure(1, 2, figsize=(12, 6), constrained_layout=True)
        axes[0].plot(
            [int(row[x_key]) for row in rows_float],
            [float(row["e_inf"]) for row in rows_float], marker="o",
            color="#1f77b4", label=f"Riley, Tex, {fixed_name}={fixed_value}",
        )
        _set_float_axis(axes[0], rows_float, list(BIT_DEPTHS))
        _axis_samples(axes[0], [int(row[x_key]) for row in rows_float], x_label)
        axes[0].set_title("Floating-Point Maximum Error")
        axes[0].set_ylabel("Max error")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)
        axes[0].legend(loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor="none")
        rows = [row for row in digitised_rows if int(row[fixed_key]) == fixed_value]
        for index, bit_depth in enumerate(BIT_DEPTHS):
            series = sorted((row for row in rows if int(row["BitDepth"]) == bit_depth), key=lambda row: int(row[x_key]))
            if series:
                axes[1].plot(
                    [int(row[x_key]) for row in series], [float(row["max_eb"]) for row in series],
                    marker="o", linestyle={8: "-", 12: "--", 16: ":"}.get(bit_depth, "-"),
                    color="#1f77b4", label=f"Riley, Tex, {fixed_name}={fixed_value}, {bit_depth}-bit",
                )
        _set_max_lsb_axis(axes[1], rows)
        _axis_samples(axes[1], [int(row[x_key]) for row in rows], x_label)
        axes[1].set_title("Maximum Digitised Mismatch")
        axes[1].set_ylabel("LSB levels")
        axes[1].grid(True, which="both", ls="--", alpha=0.4)
        axes[1].legend(loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor="none")
        figure.suptitle(f"{title}\nLimit: {fixed_name}={fixed_value}", fontweight="bold")
        figure.savefig(output_dir / f"limit_{suffix}_frame{frame:02d}.png", dpi=150)
        release_figure(figure)


def _write_analysis_figures(
    output_dir: Path,
    frame: int,
    float_rows: list[dict[str, object]],
    digitised_rows: list[dict[str, object]],
    title: str,
) -> None:
    """Write per-bit SSAA/OS studies plus a compact all-bit limits figure."""
    for bit_depth in BIT_DEPTHS:
        bit_title = f"{title} | {bit_depth}-bit"
        _plot_four_panel(
            float_rows, digitised_rows, bit_depth,
            "SSAA", "Oversamp", "Riley Samples Along One Pixel Axis",
            bit_title,
            output_dir / f"metrics_b{bit_depth:02d}_frame{frame:02d}.png",
        )
        _plot_four_panel(
            float_rows, digitised_rows, bit_depth,
            "Oversamp", "SSAA", "Texture Oversampling Along One Pixel Axis",
            bit_title,
            output_dir / f"os_metrics_b{bit_depth:02d}_frame{frame:02d}.png",
        )
    _plot_limit_cuts(
        float_rows, digitised_rows, title, output_dir, frame,
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "Case", "Pattern", "Interpolator", "Frame", "SSAA", "Oversamp",
        "Samples", "Reference", "ReferenceMethod", "ReferenceParam",
        "e_f64", "e_inf",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_digitised_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "Case", "Pattern", "Interpolator", "Frame", "SSAA", "Oversamp",
        "Samples", "BitDepth", "Reference", "ReferenceMethod",
        "ReferenceParam", "delta_b", "max_eb",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantize(image: np.ndarray, bit_depth: int) -> np.ndarray:
    max_value = float(2**bit_depth - 1)
    return np.clip(np.rint(image * max_value), 0.0, max_value)


def _analyse_riley_self_convergence_frame(
    runs: list[tuple[str, int, int, Path]],
    frame: int,
    group_name: str,
) -> None:
    """Plot each Riley run against the highest available SSAA of itself."""
    # Keep paths, rather than every rendered array.  A complete SSAA/OS sweep
    # can otherwise retain dozens of full-resolution textures at once.
    paths: dict[tuple[str, int], dict[int, Path]] = defaultdict(dict)
    for interpolator, ssaa, oversamp, run_dir in runs:
        image_path = run_dir / f"image_c00_f{frame:02d}_clamped.npy"
        if image_path.exists():
            paths[(interpolator, oversamp)][ssaa] = image_path

    rows_by_interpolator: dict[str, list[dict[str, object]]] = defaultdict(list)
    digitised_by_interpolator: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (interpolator, oversamp), by_ssaa in paths.items():
        if len(by_ssaa) < 2:
            continue
        reference_ssaa = max(by_ssaa)
        reference = np.load(by_ssaa[reference_ssaa])
        for ssaa, image_path in by_ssaa.items():
            if ssaa == reference_ssaa:
                continue
            image = np.load(image_path)
            if image.shape != reference.shape:
                print(f"Warning: {image_path} shape {image.shape} does not match self reference.")
                del image
                continue
            difference = image - reference
            base_row = {
                "Case": group_name,
                "Pattern": group_name,
                "Interpolator": interpolator,
                "Frame": frame,
                "SSAA": ssaa,
                "Oversamp": oversamp,
                "Samples": ssaa * ssaa,
                "Reference": f"Riley SSAA {reference_ssaa}x{reference_ssaa}",
                "ReferenceMethod": "riley_ssaa",
                "ReferenceParam": reference_ssaa,
            }
            rows_by_interpolator[interpolator].append({
                **base_row,
                "e_f64": float(np.sqrt(np.mean(difference**2))),
                "e_inf": float(np.max(np.abs(difference))),
            })
            for bit_depth in BIT_DEPTHS:
                digitised_difference = (
                    _quantize(image, bit_depth) - _quantize(reference, bit_depth)
                )
                digitised_by_interpolator[interpolator].append({
                    **base_row,
                    "BitDepth": bit_depth,
                    "delta_b": float(np.mean(digitised_difference != 0.0)),
                    "max_eb": float(np.max(np.abs(digitised_difference))),
                })
                del digitised_difference
            del image, difference
        del reference

    rectconv_root = Path(f"{RESULTS_DIR}_rectconv") / group_name
    for interpolator, rows in rows_by_interpolator.items():
        output_dir = rectconv_root / interpolator
        output_dir.mkdir(parents=True, exist_ok=True)
        _clear_old_metric_images(output_dir, frame)
        title = (
            f"Riley, Tex, {interpolator}: self-convergence | frame {frame:02d}\n"
            "Reference: highest SSAA at each OS"
        )
        digitised_rows = digitised_by_interpolator[interpolator]
        _write_analysis_figures(output_dir, frame, rows, digitised_rows, title)
    del paths, rows_by_interpolator, digitised_by_interpolator
    release_batch()


def analyse_pattern(
    case_name: str,
    pattern_type: str,
    tag: str,
    frames: list[int],
    allowed_interpolators: set[str] | None,
) -> list[dict[str, object]]:
    runs = _discover_riley_runs(case_name, tag, allowed_interpolators)
    group_name = f"{case_name}_{tag}"
    if not runs:
        print(f"  Warning: no Riley renders for {group_name}.")
        return []
    print(f"\nAnalysing {group_name}: {len(runs)} Riley run configurations")
    rows: list[dict[str, object]] = []
    digitised_rows: list[dict[str, object]] = []
    rows_by_interpolator: dict[str, list[dict[str, object]]] = defaultdict(list)
    digitised_by_interpolator: dict[str, list[dict[str, object]]] = defaultdict(list)
    for frame in frames:
        reference = _reference_for_frame(case_name, tag, pattern_type, frame)
        if reference is None:
            print(f"  Frame {frame:02d}: no reference; skipping.")
            continue
        reference_image, reference_name, method, param = reference
        print(
            f"  Frame {frame:02d}: {reference_name} "
            f"({method}:{param}); comparing {len(runs)} runs"
        )
        frame_rows: list[dict[str, object]] = []
        frame_digitised_rows: list[dict[str, object]] = []
        for interpolator, ssaa, oversamp, run_dir in runs:
            image_path = run_dir / f"image_c00_f{frame:02d}_clamped.npy"
            if not image_path.exists():
                continue
            image = np.load(image_path)
            if image.shape != reference_image.shape:
                print(f"Warning: {image_path} shape {image.shape} does not match reference.")
                del image
                continue
            difference = image - reference_image
            base_row = {
                "Case": case_name,
                "Pattern": tag,
                "Interpolator": interpolator,
                "Frame": frame,
                "SSAA": ssaa,
                "Oversamp": oversamp,
                "Samples": ssaa * ssaa,
                "Reference": reference_name,
                "ReferenceMethod": method,
                "ReferenceParam": param,
            }
            frame_rows.append({
                **base_row,
                "e_f64": float(np.sqrt(np.mean(difference**2))),
                "e_inf": float(np.max(np.abs(difference))),
            })
            for bit_depth in BIT_DEPTHS:
                digitised_difference = (
                    _quantize(image, bit_depth)
                    - _quantize(reference_image, bit_depth)
                )
                frame_digitised_rows.append({
                    **base_row,
                    "BitDepth": bit_depth,
                    "delta_b": float(np.mean(digitised_difference != 0.0)),
                    "max_eb": float(np.max(np.abs(digitised_difference))),
                })
                del digitised_difference
            del difference, image
        if not frame_rows:
            print(f"    No completed clamped Riley outputs; skipping plots.")
            del reference_image, reference
            release_batch()
            continue
        for interpolator in sorted({str(row["Interpolator"]) for row in frame_rows}):
            output_dir = RESULTS_DIR / group_name / interpolator
            output_dir.mkdir(parents=True, exist_ok=True)
            _clear_old_metric_images(output_dir, frame)
            interp_rows = [row for row in frame_rows if row["Interpolator"] == interpolator]
            title_prefix = (
                f"Riley, Tex, {interpolator}: analytic convergence | frame {frame:02d}\n"
                f"Reference: {reference_name}"
            )
            interpolator_digitised_rows = [
                row for row in frame_digitised_rows
                if row["Interpolator"] == interpolator
            ]
            _write_analysis_figures(
                output_dir, frame, interp_rows, interpolator_digitised_rows,
                title_prefix,
            )
            rows_by_interpolator[interpolator].extend(interp_rows)
            digitised_by_interpolator[interpolator].extend(
                interpolator_digitised_rows
            )
            print(f"    Saved {len(interp_rows)} comparisons and figures to {output_dir}")
        rows.extend(frame_rows)
        digitised_rows.extend(frame_digitised_rows)
        _analyse_riley_self_convergence_frame(
            runs, frame, group_name
        )
        del frame_rows, frame_digitised_rows
        del reference_image, reference
        release_batch()
    group_dir = RESULTS_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(group_dir / "summary.csv", rows)
    _write_digitised_rows(group_dir / "summary_digitised.csv", digitised_rows)
    for interpolator, interpolator_rows in rows_by_interpolator.items():
        output_dir = group_dir / interpolator
        _write_rows(output_dir / "summary.csv", interpolator_rows)
        _write_digitised_rows(
            output_dir / "summary_digitised.csv",
            digitised_by_interpolator[interpolator],
        )
    print(f"  Saved group summaries: {group_dir}")
    return rows


def main() -> None:
    timer = ScriptTimer(__file__)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = _selected_frames()
    allowed_interpolators = _selected_interpolators()
    cases = _selected_cases()
    print("Experiment 2 Riley floating-texture analysis")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Cases: {', '.join(cases)}")
    print(f"  Frames: {', '.join(str(frame) for frame in frames)}")
    print(
        "  Interpolators: "
        + (", ".join(sorted(allowed_interpolators)) if allowed_interpolators else "all discovered")
    )
    all_rows: list[dict[str, object]] = []
    for case_name in cases:
        case_name = output_case_name(case_name, TARG_PX_X)
        for pattern_type in ANALYTIC_SPECKLE_TYPES:
            for black_fraction in BLACK_AREA_FRACTIONS:
                for distribution, fraction in (additive_jitter_for(pattern_type),):
                        tag = pattern_tag(
                            pattern_type, black_fraction, distribution, fraction
                        )
                        all_rows.extend(timed_call(
                            timer, f"{case_name}_{tag}", analyse_pattern,
                            case_name, pattern_type, tag, frames, allowed_interpolators,
                        ))
                        release_batch()
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    print(f"\nSaved overall summary: {RESULTS_DIR / 'summary.csv'}")
    print("Experiment 2 Riley floating-texture analysis completed.")


if __name__ == "__main__":
    main()
