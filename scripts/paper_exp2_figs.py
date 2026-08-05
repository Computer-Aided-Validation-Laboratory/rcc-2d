#!/usr/bin/env python3
"""Journal-ready Experiment 2 additive-speckle convergence figures."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from modules.exp_common_analysis import image_error_metrics
from modules.paperfigs import add_figure_legend, annotate_no_data, finish_axis, make_figure, save_figure
from modules.render_outputs import quantise_camera
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT, FIGURE_2X3_CM, FIGURE_4X4_CM, FONT_SIZE_PT,
    LEGEND_FONT_SIZE_PT, PAPER_DPI, PAPER_EXP2_BIT_DEPTHS,
    PAPER_EXP2_TEX_METRIC, PAPER_EXP2_TEX_METRIC_LABEL, PAPER_FORMATS,
    PAPER_OUTPUT_DIR, PAPER_TEXFLOAT_BIT_DEPTH, PAPER_TEXTURE_INTERPOLATOR,
    RILEY_LINE_WIDTH_PT, RILEY_MARKER_SIZE_PT, TICK_FONT_SIZE_PT,
)

OUT = Path("out")
SPECK_RENDER = OUT / "exp2_speck2d_render_uvs"
RILEY_TEXF_RENDER = OUT / "exp2_riley_render_texf"
CASES = (
    ("pt42_cam32_q9_rig", 0, "Undeformed"),
    ("pt42_cam32_q9_rig", 3, "Rigid 0.3px"),
    ("pt42_cam32_q9_aff", 3, "Affine 0.3px"),
)
METRICS = (("e_b", "Digitised RMSE [bits]"), ("max_eb", "Max. digitised err. [bits]"))
METHOD_STYLE = {"gauss": ("#1b9e77", "s"), "rect": ("#377eb8", "o")}
OS_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")


@dataclass(frozen=True)
class Series:
    label: str
    samples: tuple[int, ...]
    values: tuple[float, ...]
    bit_depth: int
    colour: str
    marker: str
    linestyle: str = "-"


def panel_prefix(index: int) -> str:
    return f"({chr(ord('a') + index)})"


def _image_path(directory: Path, method: str, parameter: int, frame: int) -> Path:
    return directory / f"targ_px32_int_{method}_param_{parameter}_frame{frame:02d}.npy"


def _load(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float64)


def _runs(case: str, pattern: str) -> list[tuple[str, int, Path]]:
    expression = re.compile(rf"^{re.escape(case)}_{pattern}_seed3_(analytic|rect|gauss)_(\d+)$")
    found: list[tuple[str, int, Path]] = []
    if not SPECK_RENDER.is_dir():
        return found
    for directory in SPECK_RENDER.iterdir():
        match = expression.fullmatch(directory.name)
        if match:
            found.append((match.group(1), int(match.group(2)), directory))
    return found


def reference(case: str, pattern: str, frame: int) -> tuple[np.ndarray, str]:
    """Use analytic when present; otherwise the customary highest rule."""
    runs = _runs(case, pattern)
    for method, parameter, directory in runs:
        if method == "analytic":
            path = _image_path(directory, method, parameter, frame)
            if path.is_file():
                return _load(path), "Analytic"
    preferred = "rect" if pattern == "diskadd" else "gauss"
    candidates = [item for item in runs if item[0] == preferred and _image_path(item[2], item[0], item[1], frame).is_file()]
    if not candidates and preferred != "rect":
        candidates = [item for item in runs if item[0] == "rect" and _image_path(item[2], item[0], item[1], frame).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No reference found for {case}, {pattern}, frame {frame:02d}")
    method, parameter, directory = max(candidates, key=lambda item: item[1])
    return _load(_image_path(directory, method, parameter, frame)), f"{'Rect' if method == 'rect' else 'Gauss'} {parameter}"


def speck_series(case: str, pattern: str, frame: int, metric: str) -> tuple[list[Series], str]:
    ref, ref_label = reference(case, pattern, frame)
    groups: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for method, parameter, directory in _runs(case, pattern):
        if method == "analytic":
            continue
        path = _image_path(directory, method, parameter, frame)
        if not path.is_file():
            continue
        image = _load(path)
        if image.shape != ref.shape:
            continue
        for bits in PAPER_EXP2_BIT_DEPTHS:
            groups[(method, bits)].append((parameter, image_error_metrics(image, ref, bits, quantise_camera)[metric]))
    rows = []
    for (method, bits), points in sorted(groups.items()):
        colour, marker = METHOD_STYLE[method]
        ordered = sorted(points)
        rows.append(Series(f"Speck2D {method.title()}, {bits}-bit", tuple(x for x, _ in ordered), tuple(y for _, y in ordered), bits, colour, marker, "-" if bits <= 8 else "--"))
    return rows, ref_label


def texf_series(case: str, pattern: str, frame: int, metric: str) -> tuple[list[Series], str]:
    ref, ref_label = reference(case, pattern, frame)
    root = RILEY_TEXF_RENDER / f"{case}_{pattern}_seed3_{PAPER_TEXTURE_INTERPOLATOR}"
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if root.is_dir():
        expression = re.compile(r"^ss(\d+)_os(\d+)$")
        for directory in root.iterdir():
            match = expression.fullmatch(directory.name)
            path = directory / f"image_c00_f{frame:02d}_clamped.npy"
            if not match or not path.is_file():
                continue
            image = _load(path)
            if image.shape == ref.shape:
                grouped[int(match.group(2))].append((int(match.group(1)), image_error_metrics(image, ref, PAPER_TEXFLOAT_BIT_DEPTH, quantise_camera)[metric]))
    rows = []
    for index, (oversamp, points) in enumerate(sorted(grouped.items())):
        ordered = sorted(points)
        rows.append(Series(f"Tex-OS={oversamp}", tuple(x for x, _ in ordered), tuple(y for _, y in ordered), PAPER_TEXFLOAT_BIT_DEPTH, OS_COLOURS[index % len(OS_COLOURS)], "o"))
    return rows, ref_label


def draw(axis, data: list[Series], ylabel: str, title: str) -> list[Line2D]:
    handles: list[Line2D] = []
    samples: list[int] = []
    values: list[float] = []
    for row in data:
        axis.plot(row.samples, row.values, color=row.colour, marker=row.marker, linestyle=row.linestyle, linewidth=RILEY_LINE_WIDTH_PT, markersize=RILEY_MARKER_SIZE_PT)
        handles.append(Line2D([], [], color=row.colour, marker=row.marker, linestyle=row.linestyle, linewidth=RILEY_LINE_WIDTH_PT, markersize=RILEY_MARKER_SIZE_PT, label=row.label))
        samples.extend(row.samples); values.extend(row.values)
    if not data:
        annotate_no_data(axis, "No completed render data", font_size=FONT_SIZE_PT)
        return handles
    finish_axis(axis, title=title, samples=samples, bit_depth=max(row.bit_depth for row in data), values=values, ylabel=ylabel, title_font_size=FONT_SIZE_PT, axis_label_font_size=AXIS_LABEL_FONT_SIZE_PT)
    return handles


def figure_speck2d(pattern: str, number: int) -> list[Path]:
    figure, axes = make_figure(FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    handles: list[Line2D] = []
    name = "Disk" if pattern == "diskadd" else "Gauss"
    for column, (case, frame, deformation) in enumerate(CASES):
        for row, (metric, ylabel) in enumerate(METRICS):
            data, ref_label = speck_series(case, pattern, frame, metric)
            handles.extend(draw(axes[row, column], data, ylabel, f"{panel_prefix(row * 3 + column)} {deformation}, Ref: {ref_label}"))
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT, columns=3)
    return save_figure(figure, PAPER_OUTPUT_DIR / f"exp2_fig{number}_speck2d_{name.lower()}", PAPER_FORMATS, PAPER_DPI)


def figure_riley_texf() -> list[Path]:
    figure, axes = make_figure(FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    handles: list[Line2D] = []
    for row, (pattern, name) in enumerate((("diskadd", "Disk"), ("gaussadd", "Gauss"))):
        for column, (case, frame, deformation) in enumerate(CASES):
            data, ref_label = texf_series(case, pattern, frame, PAPER_EXP2_TEX_METRIC)
            handles.extend(draw(axes[row, column], data, PAPER_EXP2_TEX_METRIC_LABEL, f"{panel_prefix(row * 3 + column)} {name}, {deformation}, Ref: {ref_label}"))
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT, columns=4)
    return save_figure(figure, PAPER_OUTPUT_DIR / "exp2_fig3_riley_texf", PAPER_FORMATS, PAPER_DPI)


def figure_texf_difference_maps(pattern: str, stem: str) -> list[Path]:
    """4×4 signed 8-bit difference maps for the rigid 0.3 px texture case."""
    case, frame = "pt42_cam32_q9_rig", 3
    reference_image, _ = reference(case, pattern, frame)
    ssaa_levels = (1, 4, 8, 32)
    oversamples = (1, 4, 8, 32)
    root = RILEY_TEXF_RENDER / f"{case}_{pattern}_seed3_{PAPER_TEXTURE_INTERPOLATOR}"
    figure, axes = make_figure(FIGURE_4X4_CM, rows=4, columns=4, tick_font_size=TICK_FONT_SIZE_PT)
    differences: dict[tuple[int, int], np.ndarray | None] = {}
    for ssaa in ssaa_levels:
        for oversamp in oversamples:
            path = root / f"ss{ssaa}_os{oversamp}" / f"image_c00_f{frame:02d}_clamped.npy"
            if path.is_file():
                image = _load(path)
                if image.shape == reference_image.shape:
                    differences[(ssaa, oversamp)] = (
                        quantise_camera(image, 8).astype(np.float64)
                        - quantise_camera(reference_image, 8).astype(np.float64)
                    )
                    continue
            differences[(ssaa, oversamp)] = None
    scale = max(
        (float(np.max(np.abs(value))) for value in differences.values() if value is not None),
        default=1.0,
    )
    images = []
    for row, ssaa in enumerate(ssaa_levels):
        for column, oversamp in enumerate(oversamples):
            axis = axes[row, column]
            difference = differences[(ssaa, oversamp)]
            axis.set_title(
                f"{panel_prefix(row * len(oversamples) + column)} SSAA={ssaa}, Tex-OS={oversamp}",
                fontsize=FONT_SIZE_PT,
            )
            if difference is None:
                annotate_no_data(axis, "No completed render data", font_size=FONT_SIZE_PT)
                continue
            images.append(axis.imshow(
                difference, cmap="gray", vmin=-scale, vmax=scale,
                interpolation="nearest", origin="upper",
            ))
            if row == len(ssaa_levels) - 1:
                axis.set_xlabel("Pixel x", fontsize=AXIS_LABEL_FONT_SIZE_PT)
            if column == 0:
                axis.set_ylabel("Pixel y", fontsize=AXIS_LABEL_FONT_SIZE_PT)
    if images:
        colourbar = figure.colorbar(images[0], ax=list(axes.flat), shrink=0.9, pad=0.015)
        colourbar.set_label("Digitised difference [bits]", fontsize=AXIS_LABEL_FONT_SIZE_PT)
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(figure, PAPER_OUTPUT_DIR / stem, PAPER_FORMATS, PAPER_DPI)


def figure_stems() -> tuple[str, ...]:
    return (
        "exp2_fig1_speck2d_disk", "exp2_fig2_speck2d_gauss", "exp2_fig3_riley_texf",
        "exp2_fig4_riley_texf_disk_difference_maps",
        "exp2_fig5_riley_texf_gauss_difference_maps",
    )


def generate_figures() -> list[Path]:
    written = figure_speck2d("diskadd", 1)
    written.extend(figure_speck2d("gaussadd", 2))
    written.extend(figure_riley_texf())
    written.extend(figure_texf_difference_maps("diskadd", "exp2_fig4_riley_texf_disk_difference_maps"))
    written.extend(figure_texf_difference_maps("gaussadd", "exp2_fig5_riley_texf_gauss_difference_maps"))
    return written


def main() -> None:
    for path in generate_figures():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
