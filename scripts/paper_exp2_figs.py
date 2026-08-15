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
from modules.paperfigs import (
    add_figure_legend, annotate_no_data, finish_axis,
    make_figure, paper_output_directories, save_figure, texture_os_style,
)
from modules.render_outputs import quantise_camera
from paperfiglabels import (
    LABEL_DIGITISED_RMSE, LABEL_DIGITISED_DIFF,
    TITLE_UNDEFORMED, TITLE_RIGID_03PX,
    LABEL_PIXEL_X, LABEL_PIXEL_Y, LABEL_NO_DATA, PANEL_PREFIX_TEMPLATE,
    LABEL_TEX_OS_TEMPLATE, LABEL_REFERENCE_PX_SS_TEMPLATE,
    TITLE_EXP2_SPECK2D_PANEL_TEMPLATE,
    TITLE_TEXTURE_CONVERGENCE_PANEL_TEMPLATE,
    TITLE_PANEL_PX_SS_TEX_OS_TEMPLATE,
    LABEL_SPECK2D_METHOD_TEMPLATE,
    TITLE_EXP2_TEXF_GAUSS, TITLE_EXP2_TEXF_DISK,
    INTERPOLATOR_LABELS, TITLE_EXP2_DIAGONAL_PANEL_TEMPLATE,
    LABEL_DIAGONAL_ANALYTIC_TEMPLATE, LABEL_DIAGONAL_H2_TEMPLATE,
)
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT,
    DIFFERENCE_CMAP,
    LAYOUT_LINE_2X2_WIDE,
    LAYOUT_IMAGE_3X3,
    FONT_SIZE_PT,
    LEGEND_FONT_SIZE_PT,
    PAPER_DPI,
    PAPER_EXP2_BIT_DEPTHS,
    PAPER_FORMATS,
    PAPER_OUTPUT_DIR,
    PAPER_TEXTURE_INTERPOLATOR,
    RILEY_LINE_WIDTH_PT,
    RILEY_MARKER_SIZE_PT,
    TICK_FONT_SIZE_PT,
    EXP2_DIFF_SSAA_LEVELS,
    EXP2_DIFF_OVERSAMPLES,
    EXP2_DIFF_TEX_CASE,
    EXP2_DIFF_TEX_FRAME,
    EXP2_FIG2_STEM,
    EXP2_FIG3_STEM,
    EXP2_LEGACY_DIFF_DISK_STEM, EXP2_LEGACY_DIFF_GAUSS_STEM,
    EXP2_FIG7_DIFF_LIMIT_BITS,
    EXP2_FIG8_DIFF_LIMIT_BITS,
    DIFFERENCE_MATRIX_COLORBAR_FRACTION,
    DIFFERENCE_MATRIX_COLORBAR_ASPECT,
    DIFFERENCE_MATRIX_COLORBAR_SHRINK,
    DIFFERENCE_MATRIX_COLORBAR_PAD,
    LINE_COLOURS, PAPER_MAIN_TEXTURE_INTERPOLATORS,
    PAPER_DIAGONAL_ANALYTIC_COLOUR, PAPER_DIAGONAL_H2_COLOUR,
    PAPER_DIAGONAL_INTERPOLATOR_MARKERS,
)

OUT = Path("out")
SPECK_RENDER = OUT / "exp2_speck2d_render_uvs"
RILEY_TEXF_RENDER = OUT / "exp2_riley_render_texf"
CASES = (
    ("pt42_cam32_q9_rig", 0, TITLE_UNDEFORMED),
    ("pt42_cam32_q9_rig", 3, TITLE_RIGID_03PX),
)
METHOD_STYLE = {"gauss": (LINE_COLOURS[0], "s"), "rect": (LINE_COLOURS[1], "o")}


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
    return PANEL_PREFIX_TEMPLATE.format(letter=chr(ord("a") + index))


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
        rows.append(Series(LABEL_SPECK2D_METHOD_TEMPLATE.format(method=method.title(), bit_depth=bits), tuple(x for x, _ in ordered), tuple(y for _, y in ordered), bits, colour, marker, "-" if bits <= 8 else "--"))
    return rows, ref_label


def texf_series(
    case: str, pattern: str, frame: int, metric: str, camera_bits: int = 8,
    interpolator: str | None = None,
) -> tuple[list[Series], str]:
    ref, ref_label = reference(case, pattern, frame)
    interpolator = interpolator or PAPER_TEXTURE_INTERPOLATOR
    root = (
        RILEY_TEXF_RENDER
        / f"{case}_{pattern}_seed3_{interpolator}"
    )
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
                grouped[int(match.group(2))].append((
                    int(match.group(1)),
                    image_error_metrics(
                        image, ref, camera_bits, quantise_camera,
                    )[metric],
                ))
    rows = []
    for oversamp, points in sorted(grouped.items()):
        ordered = sorted(points)
        rows.append(Series(
            LABEL_TEX_OS_TEMPLATE.format(osamp=oversamp), tuple(x for x, _ in ordered),
            tuple(y for _, y in ordered), camera_bits,
            *texture_os_style(oversamp),
        ))
    return rows, ref_label


def diagonal_reference_series(
    case: str, pattern: str, frame: int, *, camera_bits: int,
) -> list[Series]:
    """Compare diagonal f64 texture renders with analytic and 2x references."""
    analytic, _ = reference(case, pattern, frame)
    result: list[Series] = []
    expression = re.compile(r"ss(\d+)_os(\d+)")
    for interpolator in PAPER_MAIN_TEXTURE_INTERPOLATORS:
        root = RILEY_TEXF_RENDER / f"{case}_{pattern}_seed3_{interpolator}"
        images: dict[int, np.ndarray] = {}
        if root.is_dir():
            for directory in root.iterdir():
                match = expression.fullmatch(directory.name)
                path = directory / f"image_c00_f{frame:02d}_clamped.npy"
                if not match or not path.is_file():
                    continue
                ssaa, osamp = int(match.group(1)), int(match.group(2))
                if ssaa == osamp:
                    images[ssaa] = _load(path)
        if not images:
            continue
        marker = PAPER_DIAGONAL_INTERPOLATOR_MARKERS[interpolator]
        name = INTERPOLATOR_LABELS[interpolator]
        analytic_points = [
            (level, image_error_metrics(
                image, analytic, camera_bits, quantise_camera,
            )["e_b"])
            for level, image in sorted(images.items())
            if image.shape == analytic.shape
        ]
        if analytic_points:
            result.append(Series(
                LABEL_DIAGONAL_ANALYTIC_TEMPLATE.format(interpolator=name),
                tuple(level for level, _ in analytic_points),
                tuple(value for _, value in analytic_points), camera_bits,
                PAPER_DIAGONAL_ANALYTIC_COLOUR, marker, "-",
            ))
        h2_points = [
            (level, image_error_metrics(
                image, images[2 * level], camera_bits, quantise_camera,
            )["e_b"])
            for level, image in sorted(images.items())
            if 2 * level in images and images[2 * level].shape == image.shape
        ]
        if h2_points:
            result.append(Series(
                LABEL_DIAGONAL_H2_TEMPLATE.format(interpolator=name),
                tuple(level for level, _ in h2_points),
                tuple(value for _, value in h2_points), camera_bits,
                PAPER_DIAGONAL_H2_COLOUR, marker, "--",
            ))
    return result


def draw(axis, data: list[Series], ylabel: str, title: str) -> list[Line2D]:
    handles: list[Line2D] = []
    samples: list[int] = []
    values: list[float] = []
    for row in data:
        axis.plot(row.samples, row.values, color=row.colour, marker=row.marker, linestyle=row.linestyle, linewidth=RILEY_LINE_WIDTH_PT, markersize=RILEY_MARKER_SIZE_PT)
        handles.append(Line2D([], [], color=row.colour, marker=row.marker, linestyle=row.linestyle, linewidth=RILEY_LINE_WIDTH_PT, markersize=RILEY_MARKER_SIZE_PT, label=row.label))
        samples.extend(row.samples); values.extend(row.values)
    if not data:
        annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
        return handles
    finish_axis(axis, title=title, samples=samples, bit_depth=max(row.bit_depth for row in data), values=values, ylabel=ylabel, title_font_size=FONT_SIZE_PT, axis_label_font_size=AXIS_LABEL_FONT_SIZE_PT)
    return handles





def display_speck_reference(reference: str) -> str:
    """Use the paper's pixel-integration shorthand in Speck2D subtitles."""
    return reference if reference == "Analytic" else LABEL_REFERENCE_PX_SS_TEMPLATE.format(reference=reference)


def figure_speck2d_combined() -> list[Path]:
    """Fig. 1: Gaussian and disk Speck2D convergence in a common 2x3 grid."""
    figure, axes = make_figure(
        LAYOUT_LINE_2X2_WIDE, rows=2, columns=2, tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles: list[Line2D] = []
    # Gaussian first, then the sharper disk texture as requested.
    for row, (pattern, name) in enumerate((("gaussadd", "Gauss"), ("diskadd", "Disk"))):
        for column, (case, frame, deformation) in enumerate(CASES):
            data, reference = speck_series(case, pattern, frame, "e_b")
            index = row * len(CASES) + column
            handles.extend(draw(
                axes[row, column], data, LABEL_DIGITISED_RMSE,
                TITLE_EXP2_SPECK2D_PANEL_TEMPLATE.format(
                    panel=panel_prefix(index), pattern=name,
                    deformation=deformation,
                    reference=display_speck_reference(reference),
                ),
            ))
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(
        figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=3,
    )
    return save_figure(
        figure, PAPER_OUTPUT_DIR / "exp2_fig1_speck2d_gauss_disk_rmse",
        PAPER_FORMATS, PAPER_DPI,
    )


def figure_riley_texf() -> list[Path]:
    """Fig. 2: f64 textures at u12 camera output, Gaussian above disk."""
    figure, axes = make_figure(
        LAYOUT_LINE_2X2_WIDE, rows=2, columns=2,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles: list[Line2D] = []
    for row, (pattern, texture_label) in enumerate((
        ("gaussadd", TITLE_EXP2_TEXF_GAUSS),
        ("diskadd", TITLE_EXP2_TEXF_DISK),
    )):
        for column, (case, frame, deformation) in enumerate(CASES):
            data, reference = texf_series(case, pattern, frame, "e_b", 12)
            handles.extend(draw(
                axes[row, column], data, LABEL_DIGITISED_RMSE,
                TITLE_TEXTURE_CONVERGENCE_PANEL_TEMPLATE.format(
                    panel=panel_prefix(row * len(CASES) + column),
                    texture=texture_label, deformation=deformation,
                    reference=reference,
                ),
            ))
    unique = {handle.get_label(): handle for handle in handles}
    add_figure_legend(
        figure, list(unique.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=4,
    )
    return save_figure(
        figure, PAPER_OUTPUT_DIR / EXP2_FIG2_STEM, PAPER_FORMATS, PAPER_DPI,
    )


def figure_diagonal_refinement() -> list[Path]:
    """Figure 3: f64 diagonal refinement for all selected interpolants."""
    figure, axes = make_figure(
        LAYOUT_LINE_2X2_WIDE, rows=2, columns=2,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles: list[Line2D] = []
    for row, (pattern, texture) in enumerate((
        ("gaussadd", TITLE_EXP2_TEXF_GAUSS),
        ("diskadd", TITLE_EXP2_TEXF_DISK),
    )):
        for column, (case, frame, deformation) in enumerate(CASES):
            data = diagonal_reference_series(
                case, pattern, frame, camera_bits=12,
            )
            handles.extend(draw(
                axes[row, column], data, LABEL_DIGITISED_RMSE, "",
            ))
            axes[row, column].set_title(
                TITLE_EXP2_DIAGONAL_PANEL_TEMPLATE.format(
                    panel=panel_prefix(row * len(CASES) + column),
                    texture=texture, deformation=deformation,
                ),
                fontsize=FONT_SIZE_PT,
            )
    add_figure_legend(
        figure, list({item.get_label(): item for item in handles}.values()),
        font_size=LEGEND_FONT_SIZE_PT, columns=3,
    )
    return save_figure(
        figure, PAPER_OUTPUT_DIR / EXP2_FIG3_STEM, PAPER_FORMATS, PAPER_DPI,
    )


def figure_stems() -> tuple[str, ...]:
    return (
        "exp2_fig1_speck2d_gauss_disk_rmse",
        EXP2_FIG2_STEM,
        EXP2_FIG3_STEM,
    )


def generate_figures() -> list[Path]:
    remove_superseded_figures()
    written = figure_speck2d_combined()
    written.extend(figure_riley_texf())
    written.extend(figure_diagonal_refinement())
    return written


def remove_superseded_figures() -> None:
    """Remove superseded four-figure texture-convergence paper products."""
    stems = (
        # Difference maps are supplementary-only as of the current paper
        # layout.  Remove stale main-paper copies when figures are rebuilt.
        EXP2_LEGACY_DIFF_DISK_STEM,
        EXP2_LEGACY_DIFF_GAUSS_STEM,
        "exp2_fig1_speck2d_disk_rmse",
        "exp2_fig2_speck2d_gauss_rmse",
        "exp2_fig3_riley_textures_disk_b8_rmse",
        "exp2_fig4_riley_textures_disk_b12_rmse",
        "exp2_fig5_riley_textures_gauss_b8_rmse",
        "exp2_fig6_riley_textures_gauss_b12_rmse",
        "exp2_fig7_riley_texf_disk_difference_maps",
        "exp2_fig8_riley_texf_gauss_difference_maps",
        f"exp2_fig3_riley_texf_disk_u8_u12_{PAPER_TEXTURE_INTERPOLATOR}_rmse",
        f"exp2_fig5_riley_texf_gauss_u8_u12_{PAPER_TEXTURE_INTERPOLATOR}_rmse",
        f"exp2_fig7_riley_texf_disk_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps",
        f"exp2_fig8_riley_texf_gauss_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps",
    )
    for output_dir in paper_output_directories():
        for stem in stems:
            for extension in (*PAPER_FORMATS, "tex"):
                (output_dir / stem).with_suffix(f".{extension}").unlink(
                    missing_ok=True
                )
        # The immediately preceding figures included the interpolant token in
        # their filenames.  These are the four one-output-depth predecessors
        # of the two combined f64 figures generated above.
        for path in output_dir.glob("exp2_fig[3456]_riley_textures_*_rmse.*"):
            if path.suffix.lstrip(".") in (*PAPER_FORMATS, "tex"):
                path.unlink()


def figure_texf_difference_maps(
    pattern: str, stem: str, colour_limit_bits: float,
    *, output_dir: Path = PAPER_OUTPUT_DIR,
) -> list[Path]:
    """4×4 signed 8-bit difference maps for the rigid 0.3 px texture case."""
    case, frame = EXP2_DIFF_TEX_CASE, EXP2_DIFF_TEX_FRAME
    reference_image, _ = reference(case, pattern, frame)
    ssaa_levels = EXP2_DIFF_SSAA_LEVELS
    oversamples = EXP2_DIFF_OVERSAMPLES
    root = RILEY_TEXF_RENDER / f"{case}_{pattern}_seed3_{PAPER_TEXTURE_INTERPOLATOR}"
    rows = len(ssaa_levels)
    cols = len(oversamples)
    figure, all_axes = make_figure(
        LAYOUT_IMAGE_3X3,
        rows=rows,
        # Reserve a narrow, dedicated GridSpec column for the colourbar.
        # Letting ``figure.colorbar(..., ax=...)`` carve this from the image
        # grid caused it to overlap the long third-column panel headings.
        columns=cols + 1,
        tick_font_size=TICK_FONT_SIZE_PT,
    )
    axes = all_axes[:, :cols]
    gridspec = all_axes[0, 0].get_subplotspec().get_gridspec()
    gridspec.set_width_ratios([1.0] * cols + [0.075])
    for axis in all_axes[:, -1]:
        axis.remove()
    colourbar_axis = figure.add_subplot(gridspec[:, cols])
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
    scale = colour_limit_bits
    images = []
    for row, ssaa in enumerate(ssaa_levels):
        for column, oversamp in enumerate(oversamples):
            axis = axes[row, column]
            difference = differences[(ssaa, oversamp)]
            axis.set_title(
                TITLE_PANEL_PX_SS_TEX_OS_TEMPLATE.format(
                    panel=panel_prefix(row * len(oversamples) + column),
                    ssaa=ssaa, osamp=oversamp,
                ),
                fontsize=FONT_SIZE_PT,
            )
            if difference is None:
                annotate_no_data(axis, LABEL_NO_DATA, font_size=FONT_SIZE_PT)
                continue
            images.append(axis.imshow(
                difference, cmap=DIFFERENCE_CMAP, vmin=-scale, vmax=scale,
                interpolation="nearest", origin="upper",
            ))
            if row == len(ssaa_levels) - 1:
                axis.set_xlabel(
                    LABEL_PIXEL_X, fontsize=AXIS_LABEL_FONT_SIZE_PT
                )
            if column == 0:
                axis.set_ylabel(
                    LABEL_PIXEL_Y, fontsize=AXIS_LABEL_FONT_SIZE_PT
                )
    if images:
        colourbar = figure.colorbar(
            images[0], cax=colourbar_axis,
            aspect=DIFFERENCE_MATRIX_COLORBAR_ASPECT,
        )
        colourbar.set_label(
            LABEL_DIGITISED_DIFF, fontsize=AXIS_LABEL_FONT_SIZE_PT
        )
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(
        figure, output_dir / stem, PAPER_FORMATS, PAPER_DPI
    )





def main() -> None:
    for path in generate_figures():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
