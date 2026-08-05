#!/usr/bin/env python3
"""Exp3 render convergence in the same figure/layout convention as Exp2.

Custom and Riley-function studies use an analytic image whenever it exists.
Otherwise their highest completed SSAA image is the reference.  A Riley
texture is a reconstructed signal rather than the procedural continuum, so
each interpolator/OS series instead uses its own highest completed SSAA image.
This is the same reference convention used by the Exp2 texture analysis.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

from exp3params import BIT_DEPTHS
from modules.analysis_memory import make_agg_figure, release_batch, release_figure
from modules.analysis_parallel import run_analysis_jobs
from modules.analysis_selection import analysis_should_run, mark_analysis_complete
from modules.exp_common_analysis import image_error_metrics
from modules.exp3_analysis_common import Render, discover_renders, image_frames, load_image
from modules.render_outputs import quantise_camera


RESULTS = Path("out/exp3_analysis_conv")
RECT_RESULTS = Path("out/exp3_analysis_conv_rectconv")
BIT_LINESTYLES = {8: "-", 10: "--", 12: "-.", 16: ":"}
MARKERS = ("o", "s", "^", "v", "<", ">", "D", "P", "X")


def renderer_family(item: Render) -> str:
    if "grid2d" in item.root:
        return "custom_grid"
    if "speck2d" in item.root:
        return "custom_speck"
    if "riley_render_func" in item.root:
        return "riley_func"
    return "riley_texuint" if "texuint" in item.root else "riley_texfloat"


def is_texture(item: Render) -> bool:
    return renderer_family(item).startswith("riley_tex")


def is_psf(item: Render) -> bool:
    return "_psf" in item.root or "_psf" in item.config


def render_case(family: str, psf: bool) -> str:
    """Name the Exp3 renderer root like Exp1/2's separate analysis suites."""
    if family == "custom_grid":
        return "grid2d_psf" if psf else "grid2d"
    if family == "custom_speck":
        return "speck2d_psf" if psf else "speck2d"
    if family == "riley_func":
        return "riley_func_psf" if psf else "riley_func"
    storage = "texu" if family == "riley_texuint" else "texf"
    return f"riley_{storage}{'_psf' if psf else ''}"


def output_group(case: str, pattern: str, psf: bool, family: str, interpolator: str) -> Path:
    """Place every study below its own renderer root, like Exp1/Exp2."""
    group = RESULTS / render_case(family, psf) / f"{case}_{pattern}"
    if family.startswith("riley_tex"):
        return group / interpolator
    return group


def rectconv_group(case: str, pattern: str, psf: bool, family: str, interpolator: str) -> Path:
    relative = output_group(case, pattern, psf, family, interpolator).relative_to(RESULTS)
    return RECT_RESULTS / relative


def load_frame(item: Render, frame: int) -> np.ndarray | None:
    path = image_frames(item.directory).get(frame)
    return load_image(path) if path is not None else None


def reference_label(reference: Render, *, texture: bool) -> str:
    if reference.analytic:
        return "Analytic Reference"
    if texture:
        return f"Highest SSAA Reference at OS={reference.oversamp} ({reference.ssaa}x{reference.ssaa})"
    return f"Highest SSAA Reference ({reference.ssaa}x{reference.ssaa})"


def reference_for(items: list[Render], candidates: list[Render], *, texture: bool) -> Render | None:
    analytic = [value for value in candidates if value.analytic]
    if analytic:
        return sorted(analytic, key=lambda value: (value.root, value.config))[0]
    # An analytic image may be shared by equivalent renderer families.  Once
    # that option is absent, do not silently substitute another renderer's
    # model: converge the current study to its own highest completed SSAA.
    return max(items, key=lambda value: (value.ssaa, value.oversamp)) if items else None


def metric_row(image: np.ndarray, reference: np.ndarray, bit_depth: int) -> dict[str, float]:
    return image_error_metrics(image, reference, bit_depth, quantise_camera)


def set_samples_axis(axis, values: list[int], label: str) -> None:
    ticks = sorted({int(value) for value in values if value > 0})
    if not ticks:
        return
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in ticks]))
    axis.set_xlim(0.85 * ticks[0], 1.15 * ticks[-1])
    axis.set_xlabel(label)


def set_float_axis(axis, rows: list[dict[str, object]], metric: str, bit_depths: list[int]) -> None:
    values = [float(row[metric]) for row in rows if np.isfinite(float(row[metric]))]
    finest_half_lsb = 0.5 / float(2 ** max(bit_depths) - 1)
    coarsest_lsb = 1.0 / float(2 ** min(bit_depths) - 1)
    axis.set_yscale("symlog", linthresh=finest_half_lsb, linscale=0.8)
    for bits in bit_depths:
        maximum = float(2 ** bits - 1)
        axis.axhline(1.0 / maximum, color="black", linestyle=BIT_LINESTYLES.get(bits, "-"), alpha=0.35)
        axis.axhline(0.5 / maximum, color="red", linestyle=BIT_LINESTYLES.get(bits, "-"), alpha=0.35)
    ceiling = 1.15 * max([abs(value) for value in values] + [coarsest_lsb])
    axis.set_ylim((-ceiling, ceiling) if metric.startswith("mean_") else (0.0, ceiling))
    if metric.startswith("mean_"):
        axis.axhline(0.0, color="black", linestyle=":", alpha=0.55)


def set_max_lsb_axis(axis, rows: list[dict[str, object]], metric: str = "max_eb") -> None:
    values = [float(row[metric]) for row in rows]
    maximum = max([abs(value) for value in values] + [1.0])
    axis.set_yscale("symlog", linthresh=0.25, linscale=0.8)
    axis.axhline(1.0, color="black", linestyle="--", alpha=0.55, label="1 LSB")
    axis.axhline(0.0, color="red", linestyle=":", alpha=0.6, label="0 LSB")
    axis.set_ylim((-1.15 * maximum, 1.15 * maximum) if metric == "mean_eb" else (0.0, 1.15 * maximum))


def line_label(family: str, line_key: str, value: int) -> str:
    if family.startswith("riley_tex"):
        return f"Riley, Tex, {'OS' if line_key == 'OS' else 'SSAA'}={value}"
    if family == "riley_func":
        return f"Riley, Func, SSAA={value}"
    return f"Custom, SSAA={value}"


def plot_four_panel(
    rows: list[dict[str, object]], bit_depth: int, x_key: str, line_key: str,
    x_label: str, title: str, output_path: Path, family: str,
) -> None:
    selected = [row for row in rows if int(row["BitDepth"]) == bit_depth]
    if not selected:
        return
    figure, axes = make_agg_figure(3, 2, figsize=(12, 15), constrained_layout=True)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[int(row[line_key])].append(row)
    line_values = sorted(grouped)
    colors = rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {value: (colors[index % len(colors)], MARKERS[index % len(MARKERS)]) for index, value in enumerate(line_values)}
    # Exp2 convention: plot the highest line-control value first so lower
    # curves remain readable where the converged curves overlap.
    for value in reversed(line_values):
        series = sorted(grouped[value], key=lambda row: int(row[x_key]))
        color, marker = styles[value]
        label = line_label(family, line_key, value)
        for axis, metric in ((axes[0, 0], "e_f64"), (axes[1, 0], "mean_f64"), (axes[2, 0], "e_inf"), (axes[0, 1], "e_b"), (axes[1, 1], "mean_eb"), (axes[2, 1], "max_eb")):
            axis.plot([int(row[x_key]) for row in series], [float(row[metric]) for row in series], color=color, marker=marker, linewidth=1.6, markersize=6, label=label)
    set_float_axis(axes[0, 0], selected, "e_f64", [bit_depth])
    set_float_axis(axes[0, 1], selected, "e_inf", [bit_depth])
    set_float_axis(axes[1, 0], selected, "mean_f64", [bit_depth])
    set_max_lsb_axis(axes[0, 1], selected, "e_b")
    set_max_lsb_axis(axes[1, 1], selected, "mean_eb")
    set_max_lsb_axis(axes[2, 1], selected, "max_eb")
    for axis, panel_title, ylabel in (
        (axes[0, 0], "Floating-Point RMSE", "RMSE"),
        (axes[1, 0], "Floating-Point Signed Mean Error", "Mean error"),
        (axes[2, 0], "Floating-Point Maximum Error", "Max error"),
        (axes[0, 1], "Digitised RMSE", "LSB levels"),
        (axes[1, 1], "Digitised Signed Mean Error", "LSB levels"),
        (axes[2, 1], "Maximum Digitised Error", "LSB levels"),
    ):
        set_samples_axis(axis, [int(row[x_key]) for row in selected], x_label)
        axis.set_title(panel_title); axis.set_ylabel(ylabel); axis.grid(True, which="both", ls="--", alpha=0.4)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), loc="lower left", fontsize=6, frameon=True, facecolor="white", edgecolor="none")
    figure.suptitle(f"{title} | {bit_depth}-bit", fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    release_figure(figure)


def plot_mismatch_panel(rows, bit_depth, x_key, line_key, x_label, title, output_path, family):
    """Four complementary digitised-error distribution measures."""
    selected = [row for row in rows if int(row["BitDepth"]) == bit_depth]
    if not selected:
        return
    figure, axes = make_agg_figure(2, 2, figsize=(12, 9), constrained_layout=True)
    grouped = defaultdict(list)
    for row in selected:
        grouped[int(row[line_key])].append(row)
    colors = rcParams["axes.prop_cycle"].by_key()["color"]
    for index, value in enumerate(reversed(sorted(grouped))):
        series = sorted(grouped[value], key=lambda row: int(row[x_key]))
        for axis, metric in zip(axes.flat, ("delta_b", "severe_b", "p95_eb", "p99_eb")):
            axis.plot([int(row[x_key]) for row in series], [float(row[metric]) for row in series], color=colors[index % len(colors)], marker=MARKERS[index % len(MARKERS)], linewidth=1.6, markersize=6, label=line_label(family, line_key, value))
    for axis, name, ylabel in zip(axes.flat, ("Mismatch Fraction (≥1 LSB)", "Severe Mismatch Fraction (≥2 LSB)", "95th Percentile Absolute Digitised Error", "99th Percentile Absolute Digitised Error"), ("Fraction of pixels", "Fraction of pixels", "LSB levels", "LSB levels")):
        set_samples_axis(axis, [int(row[x_key]) for row in selected], x_label)
        axis.set_title(name); axis.set_ylabel(ylabel); axis.grid(True, which="both", ls="--", alpha=0.4)
        axis.legend(loc="lower left", fontsize=6, frameon=True, facecolor="white", edgecolor="none")
    figure.suptitle(f"{title} | {bit_depth}-bit", fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    release_figure(figure)


def plot_limit_cuts(rows: list[dict[str, object]], title: str, output_dir: Path, frame: int, family: str) -> None:
    """Match Exp2's four min/max SSAA/OS texture-study cuts."""
    ssaa = sorted({int(row["SSAA"]) for row in rows})
    oversamp = sorted({int(row["OS"]) for row in rows})
    if len(ssaa) < 2 or len(oversamp) < 2:
        return
    cuts = (
        ("max_ssaa", "SSAA", "OS", max(oversamp), "OS", "Riley Samples Along One Pixel Axis"),
        ("max_oversamp", "OS", "SSAA", max(ssaa), "SSAA", "Texture Oversampling Along One Pixel Axis"),
        ("min_ssaa", "SSAA", "OS", min(oversamp), "OS", "Riley Samples Along One Pixel Axis"),
        ("min_oversamp", "OS", "SSAA", min(ssaa), "SSAA", "Texture Oversampling Along One Pixel Axis"),
    )
    colors = rcParams["axes.prop_cycle"].by_key()["color"]
    for suffix, x_key, fixed_key, fixed_value, fixed_name, x_label in cuts:
        fixed = [row for row in rows if int(row[fixed_key]) == fixed_value]
        if not fixed:
            continue
        figure, axes = make_agg_figure(1, 2, figsize=(12, 6), constrained_layout=True)
        # Float data are numerically identical for each digitisation depth;
        # use the finest selected depth to avoid duplicate overlapping lines.
        float_depth = max(int(row["BitDepth"]) for row in fixed)
        float_rows = sorted((row for row in fixed if int(row["BitDepth"]) == float_depth), key=lambda row: int(row[x_key]))
        axes[0].plot([int(row[x_key]) for row in float_rows], [float(row["e_inf"]) for row in float_rows], marker="o", color="#1f77b4", label=line_label(family, fixed_name, fixed_value))
        set_float_axis(axes[0], float_rows, "e_inf", sorted({int(row["BitDepth"]) for row in fixed}))
        set_samples_axis(axes[0], [int(row[x_key]) for row in float_rows], x_label)
        axes[0].set_title("Floating-Point Maximum Error"); axes[0].set_ylabel("Max error"); axes[0].grid(True, which="both", ls="--", alpha=0.4); axes[0].legend(loc="lower left", fontsize=7)
        for index, bits in enumerate(sorted({int(row["BitDepth"]) for row in fixed}, reverse=True)):
            series = sorted((row for row in fixed if int(row["BitDepth"]) == bits), key=lambda row: int(row[x_key]))
            axes[1].plot([int(row[x_key]) for row in series], [float(row["max_eb"]) for row in series], marker="o", linestyle=BIT_LINESTYLES.get(bits, "-"), color=colors[index % len(colors)], label=f"Riley, Tex, {fixed_name}={fixed_value}, {bits}-bit")
        set_max_lsb_axis(axes[1], fixed)
        set_samples_axis(axes[1], [int(row[x_key]) for row in fixed], x_label)
        axes[1].set_title("Maximum Digitised Mismatch"); axes[1].set_ylabel("LSB levels"); axes[1].grid(True, which="both", ls="--", alpha=0.4); axes[1].legend(loc="lower left", fontsize=7)
        figure.suptitle(f"{title}\nLimit: {fixed_name}={fixed_value}", fontweight="bold")
        figure.savefig(output_dir / f"limit_{suffix}_frame{frame:02d}.png", dpi=150)
        release_figure(figure)


def write_figures(rows: list[dict[str, object]], output_dir: Path, frame: int, title: str, family: str, *, texture: bool) -> None:
    frame_rows = [row for row in rows if int(row["Frame"]) == frame]
    # The current texture layout always uses the complete levels; remove stale
    # exponent-parity figures from prior analysis versions.
    for stale in output_dir.glob(f"*odd_exp*frame{frame:02d}.png"):
        stale.unlink(missing_ok=True)
    for stale in output_dir.glob(f"*even_exp*frame{frame:02d}.png"):
        stale.unlink(missing_ok=True)
    for bit_depth in sorted({int(row["BitDepth"]) for row in frame_rows}):
        plot_four_panel(frame_rows, bit_depth, "SSAA", "OS", "Riley Samples Along One Pixel Axis" if texture else "Samples Along One Pixel Axis", title, output_dir / f"metrics_b{bit_depth:02d}_frame{frame:02d}.png", family)
        plot_mismatch_panel(frame_rows, bit_depth, "SSAA", "OS", "Riley Samples Along One Pixel Axis" if texture else "Samples Along One Pixel Axis", title, output_dir / f"mismatch_b{bit_depth:02d}_frame{frame:02d}.png", family)
        if texture:
            plot_four_panel(frame_rows, bit_depth, "OS", "SSAA", "Texture Oversampling Along One Pixel Axis", title, output_dir / f"os_metrics_b{bit_depth:02d}_frame{frame:02d}.png", family)
            plot_mismatch_panel(frame_rows, bit_depth, "OS", "SSAA", "Texture Oversampling Along One Pixel Axis", title, output_dir / f"os_mismatch_b{bit_depth:02d}_frame{frame:02d}.png", family)
    if texture:
        plot_limit_cuts(frame_rows, title, output_dir, frame, family)


def make_rows(
    case: str, family: str, pattern: str, psf: bool, interpolator: str,
    items: list[Render], candidates: list[Render], bit_depths: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    texture = family.startswith("riley_tex")
    by_os: dict[int, list[Render]] = defaultdict(list)
    for item in items:
        by_os[item.oversamp if texture else 1].append(item)
    primary: list[dict[str, object]] = []
    self_rows: list[dict[str, object]] = []
    for osamp, series in sorted(by_os.items()):
        reference = reference_for(series, candidates, texture=texture)
        if reference is None:
            continue
        self_reference = max(series, key=lambda value: (value.ssaa, value.oversamp))
        primary_label, self_label = reference_label(reference, texture=texture), reference_label(self_reference, texture=True)
        for frame in sorted(image_frames(reference.directory)):
            ref_image = load_frame(reference, frame)
            self_image = load_frame(self_reference, frame)
            if ref_image is None or self_image is None:
                continue
            for item in series:
                image = load_frame(item, frame)
                if image is None:
                    continue
                for bit_depth in bit_depths:
                    base = {"Case": case, "Family": family, "Pattern": pattern, "PSF": psf, "Interpolator": interpolator, "Config": item.config, "Frame": frame, "BitDepth": bit_depth, "SSAA": item.ssaa or 1, "OS": item.oversamp or 1}
                    # The analytic image is a reference, not an SSAA=1
                    # sample.  Its parsed parameter is zero (displayed as
                    # one otherwise), which previously made a spurious zero
                    # point appear beside the real SSAA=1 render.
                    if item != reference and image.shape == ref_image.shape:
                        primary.append({**base, "Reference": primary_label, **metric_row(image, ref_image, bit_depth)})
                    # Exp1/2 _rectconv convention: always compare against
                    # the highest available SSAA (at each OS for textures),
                    # independently of the primary analytic-reference study.
                    if not item.analytic and image.shape == self_image.shape:
                        self_rows.append({**base, "Reference": self_label, **metric_row(image, self_image, bit_depth)})
                del image
            del ref_image, self_image
            release_batch()
    return primary, self_rows


def analyse_task(task: tuple[str, str, str, bool, str, list[Render], list[Render], list[int]]) -> tuple[Path, Path, Path, Path, list[dict[str, object]], list[dict[str, object]]]:
    case, family, pattern, psf, interpolator, items, candidates, bit_depths = task
    output_dir = output_group(case, pattern, psf, family, interpolator)
    rect_dir = rectconv_group(case, pattern, psf, family, interpolator)
    primary, self_rows = make_rows(case, family, pattern, psf, interpolator, items, candidates, bit_depths)
    if primary:
        references = {str(row["Reference"]) for row in primary}
        reference_title = "Highest SSAA Reference at each OS" if family.startswith("riley_tex") and len(references) > 1 else str(primary[0]["Reference"])
        renderer_title = "Riley, Tex" if family.startswith("riley_tex") else ("Riley, Func" if family == "riley_func" else "Custom")
        title = f"{renderer_title}: {case}, {pattern} | Reference: {reference_title}"
    else:
        title = f"{case}, {pattern}"
    for frame in sorted({int(row["Frame"]) for row in primary}):
        write_figures(primary, output_dir, frame, title, family, texture=family.startswith("riley_tex"))
    if self_rows:
        self_title = f"{case}, {pattern}: self convergence | Reference: {self_rows[0]['Reference']}"
        for frame in sorted({int(row["Frame"]) for row in self_rows}):
            write_figures(self_rows, rect_dir, frame, self_title, family, texture=family.startswith("riley_tex"))
    release_batch()
    return output_dir, rect_dir, RESULTS / render_case(family, psf), RECT_RESULTS / render_case(family, psf), primary, self_rows


def write_summary(directory: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["Case", "Family", "Pattern", "PSF", "Interpolator", "Config", "Frame", "BitDepth", "SSAA", "OS", "Reference", "e_f64", "mean_f64", "e_inf", "e_b", "mean_eb", "delta_b", "severe_b", "p95_eb", "p99_eb", "max_eb"]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if not analysis_should_run(RESULTS, "Experiment 3 convergence analysis"):
        return
    renders = [item for item in discover_renders() if "_oldver" not in item.root]
    all_by_case_pattern: dict[tuple[str, str, bool], list[Render]] = defaultdict(list)
    grouped: dict[tuple[str, str, str, bool, str], list[Render]] = defaultdict(list)
    for item in renders:
        family, psf = renderer_family(item), is_psf(item)
        all_by_case_pattern[(item.case, item.pattern, psf)].append(item)
        grouped[(item.case, family, item.pattern, psf, item.interpolator if is_texture(item) else "")].append(item)
    tasks = []
    for (case, family, pattern, psf, interpolator), items in sorted(grouped.items()):
        # Match Exp1/2: every renderer uses the shared analytic image whenever
        # that image exists for this deformation/pattern/PSF case.  Texture
        # studies without an analytic image retain their per-OS highest-SSAA
        # fallback because each reconstructed texture is a distinct signal.
        candidates = all_by_case_pattern[(case, pattern, psf)]
        tasks.append((case, family, pattern, psf, interpolator, items, candidates, list(BIT_DEPTHS)))
    limit = int(os.environ.get("EXP3_ANALYSIS_LIMIT", "0"))
    if limit:
        tasks = tasks[:limit]
    primary_rows: list[dict[str, object]] = []
    self_rows: list[dict[str, object]] = []
    per_output: dict[Path, list[dict[str, object]]] = defaultdict(list)
    per_rect: dict[Path, list[dict[str, object]]] = defaultdict(list)
    per_renderer: dict[Path, list[dict[str, object]]] = defaultdict(list)
    per_rect_renderer: dict[Path, list[dict[str, object]]] = defaultdict(list)
    print(f"Experiment 3 convergence analysis: {len(tasks)} studies; bit depths={BIT_DEPTHS}")
    for output_dir, rect_dir, renderer_dir, rect_renderer_dir, primary, self_data in run_analysis_jobs("Experiment 3 convergence analysis", tasks, analyse_task):
        primary_rows.extend(primary); self_rows.extend(self_data)
        per_output[output_dir].extend(primary); per_rect[rect_dir].extend(self_data)
        per_renderer[renderer_dir].extend(primary); per_rect_renderer[rect_renderer_dir].extend(self_data)
    for directory, rows in per_output.items():
        write_summary(directory, rows)
    for directory, rows in per_rect.items():
        write_summary(directory, rows)
    for directory, rows in per_renderer.items():
        write_summary(directory, rows)
    for directory, rows in per_rect_renderer.items():
        write_summary(directory, rows)
    mark_analysis_complete(RESULTS)
    print(f"Wrote {len(primary_rows)} primary and {len(self_rows)} self-convergence rows.")


if __name__ == "__main__":
    main()
