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

import matplotlib.pyplot as plt
import numpy as np

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
from script_timing import ScriptTimer, timed_call


RILEY_OUTPUT_DIR = exp2_output_dir("exp2_riley_render_texf")
REFERENCE_OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs")
RESULTS_DIR = exp2_output_dir("exp2_riley_analysis_texf")
RUN_RE = re.compile(r"^ss(?P<ssaa>\d+)_oversamp(?P<oversamp>\d+)$")
INTERPOLATOR_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
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
        f"targ_px{TARG_PX_X}_int_{method}_param_{param}_frame{frame:02d}.npy"
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


def _plot_metric(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["Oversamp"])].append(row)
    for index, (oversamp, group) in enumerate(sorted(grouped.items())):
        group.sort(key=lambda row: int(row["Samples"]))
        samples = np.sqrt(np.asarray([int(row["Samples"]) for row in group]))
        values = np.asarray([float(row[metric]) for row in group])
        color = INTERPOLATOR_COLORS[index % len(INTERPOLATOR_COLORS)]
        marker = OVERSAMP_MARKERS[index % len(OVERSAMP_MARKERS)]
        axes.loglog(
            samples,
            np.maximum(values, np.finfo(np.float64).tiny),
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=7,
            label=f"oversamp={oversamp}",
        )
    axes.set_title(title, fontweight="bold")
    axes.set_xlabel("Riley Samples Along One Pixel Axis")
    axes.set_ylabel(ylabel)
    axes.grid(True, which="both", ls="--", alpha=0.5)
    if grouped:
        axes.legend(frameon=True, facecolor="white", edgecolor="none")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


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
        if not frame_rows:
            print(f"    No completed clamped Riley outputs; skipping plots.")
            continue
        for interpolator in sorted({str(row["Interpolator"]) for row in frame_rows}):
            output_dir = RESULTS_DIR / group_name / interpolator
            output_dir.mkdir(parents=True, exist_ok=True)
            interp_rows = [row for row in frame_rows if row["Interpolator"] == interpolator]
            title_prefix = (
                f"Riley {interpolator} texture vs {reference_name}\n"
                f"{group_name}, frame {frame:02d}"
            )
            _plot_metric(
                interp_rows,
                "e_f64",
                "Clamped-intensity RMSE ($e_{f64}$)",
                output_dir / f"float_rmse_frame{frame:02d}.png",
                title_prefix,
            )
            _plot_metric(
                interp_rows,
                "e_inf",
                "Clamped-intensity max error ($e_{∞}$)",
                output_dir / f"float_max_frame{frame:02d}.png",
                title_prefix,
            )
            interpolator_digitised_rows = [
                row for row in frame_digitised_rows
                if row["Interpolator"] == interpolator
            ]
            for bit_depth in BIT_DEPTHS:
                interp_digitised = [
                    row for row in interpolator_digitised_rows
                    if row["BitDepth"] == bit_depth
                ]
                _plot_metric(
                    interp_digitised,
                    "delta_b",
                    f"Digitised mismatch fraction ({bit_depth}-bit)",
                    output_dir / f"bits_delta_b{bit_depth}_frame{frame:02d}.png",
                    title_prefix,
                )
                _plot_metric(
                    interp_digitised,
                    "max_eb",
                    f"Maximum digitised error (LSB, {bit_depth}-bit)",
                    output_dir / f"bits_max_eb{bit_depth}_frame{frame:02d}.png",
                    title_prefix,
                )
            rows_by_interpolator[interpolator].extend(interp_rows)
            digitised_by_interpolator[interpolator].extend(
                interpolator_digitised_rows
            )
            print(f"    Saved {len(interp_rows)} comparisons and figures to {output_dir}")
        rows.extend(frame_rows)
        digitised_rows.extend(frame_digitised_rows)
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
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    print(f"\nSaved overall summary: {RESULTS_DIR / 'summary.csv'}")
    print("Experiment 2 Riley floating-texture analysis completed.")


if __name__ == "__main__":
    main()
