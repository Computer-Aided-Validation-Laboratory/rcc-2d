#!/usr/bin/env python3
"""Create the four journal-ready Experiment 1 convergence figures.

The script computes digitised metrics directly from canonical floating-point
renders.  TIFF previews are deliberately not inputs, so each figure uses the
same digitisation definition as the analysis scripts.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from modules.exp_common_analysis import image_error_metrics
from modules.paperfigs import (
    add_figure_legend, annotate_no_data, finish_axis, finish_signed_axis,
    make_figure, paper_output_directories, save_figure, texture_os_style,
    write_latex_preview,
)
from modules.render_outputs import quantise_camera
from paperparams import (
    AXIS_LABEL_FONT_SIZE_PT, DIFFERENCE_CMAP, FIGURE_1X3_CM,
    FIGURE_2X3_CM, FIGURE_3X3_CM, FIGURE_4X4_CM, FONT_SIZE_PT,
    LEGEND_FONT_SIZE_PT, GRID_LINE_WIDTH_PT, GRID_MARKER_SIZE_PT,
    RILEY_LINE_WIDTH_PT, RILEY_MARKER_SIZE_PT, PAPER_DPI, PAPER_FORMATS,
    PAPER_FRAME, PAPER_OUTPUT_DIR, PAPER_TEXFLOAT_BIT_DEPTH,
    PAPER_TEXTURE_INTERPOLATOR, TICK_FONT_SIZE_PT, FIGURE_CAPTIONS,
    FIGURE_LABELS, EXP1_DIFF_SSAA_LEVELS, EXP2_DIFF_SSAA_LEVELS,
    EXP2_DIFF_OVERSAMPLES, EXP1_DIFF_FUNC_CASE, EXP1_DIFF_FUNC_FRAME,
    EXP1_DIFF_TEX_CASE, EXP1_DIFF_TEX_FRAME,
)

OUT = Path("out")
GRID_SUMMARY = OUT / "exp1_analysis" / "grid2d_uvs" / "summary.csv"
GRID_RENDER = OUT / "exp1_grid2d_render_uvs"
FUNC_RENDER = OUT / "exp1_riley_render_func_uvs"
TEXFLOAT_RENDER = OUT / "exp1_riley_render_texf"
TEXUINT_RENDER = OUT / "exp1_riley_render_texu"
TEXTURE_STUDIES = (
    ("pt42_cam32_q9_rig", 0, "Undeformed"),
    ("pt42_cam32_q9_rig", 3, "Rigid 0.3px"),
    ("pt42_cam32_q9_aff", 3, "Affine 0.3px"),
)
METRICS = (("e_b", "Digitised RMSE [bits]"), ("max_eb", "Max. digitised err. [bits]"))


@dataclass(frozen=True)
class Series:
    label: str
    samples: tuple[int, ...]
    values: tuple[float, ...]
    bit_depth: int
    colour: str
    marker: str
    linestyle: str


def panel_prefix(index: int) -> str:
    """Return the conventional journal sub-panel prefix, e.g. ``(a)``."""
    return f"({chr(ord('a') + index)})"


def display_reference(value: str) -> str:
    """Turn analysis-internal reference tokens into article-facing labels."""
    if value.strip().lower() in {"analytic", "analytic:0", "analytic reference"}:
        return "Analytic"
    return value.replace("_", " ")


def load_normalised(path: Path, bit_depth: int | None = None) -> np.ndarray:
    image = np.asarray(np.load(path), dtype=np.float64)
    if image.size and np.nanmax(np.abs(image)) > 1.0 + 1e-12:
        scale = float(2**bit_depth - 1) if bit_depth is not None else 255.0
        image /= scale
    return image


def analytic_reference(case: str, frame: int = PAPER_FRAME) -> tuple[np.ndarray, str]:
    direct = sorted((GRID_RENDER / case).glob(f"targ_px*_int_analytic_param_0_frame{frame:02d}.npy"))
    if direct:
        return load_normalised(direct[0]), "Analytic"
    candidates = []
    for path in (GRID_RENDER / case).glob(f"targ_px*_int_gauss_param_*_frame{frame:02d}.npy"):
        match = re.search(r"param_(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No analytic or Gauss reference for {case} frame {frame:02d}")
    order, path = max(candidates)
    return load_normalised(path), f"Gauss {order}"


def grid_metric_series(case: str, metric: str, frame: int = PAPER_FRAME) -> tuple[list[Series], str]:
    if not GRID_SUMMARY.is_file():
        return [], "No grid summary"
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    ref = "Analytic"
    with GRID_SUMMARY.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Case"] == case and int(row["Frame"]) == frame and row["Method"] != "analytic":
                groups[(row["Method"], int(row["BitDepth"]))].append(row); ref = row["Reference"]
    colours = {"gauss": "#1b9e77", "rect": "#377eb8"}; markers = {"gauss": "s", "rect": "o"}
    return [Series(f"Grid2D {method.title()}, {bits}-bit", tuple(int(round(float(r["Samples"]) ** .5)) for r in sorted(rows, key=lambda r: float(r["Samples"]))), tuple(float(r[metric]) for r in sorted(rows, key=lambda r: float(r["Samples"]))), bits, colours[method], markers[method], "-" if bits <= 8 else "--") for (method, bits), rows in sorted(groups.items())], ref


def function_series(case: str, reference: np.ndarray, metric: str, bit_depths: set[int], frame: int = PAPER_FRAME) -> list[Series]:
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for directory in (FUNC_RENDER / case).glob("ss*_f"):
        match = re.fullmatch(r"ss(\d+)_f", directory.name)
        path = directory / f"image_c00_f{frame:02d}.npy"
        if not match or not path.is_file():
            continue
        image = load_normalised(path)
        if image.shape != reference.shape:
            continue
        for bits in bit_depths:
            groups[bits].append((int(match.group(1)), image_error_metrics(image, reference, bits, quantise_camera)[metric]))
    return [Series(f"Riley Rect, {bits}-bit", tuple(x for x, _ in sorted(points)), tuple(y for _, y in sorted(points)), bits, "#d95f02", "^", "-" if bits <= 8 else "--") for bits, points in sorted(groups.items())]


def texture_series(case: str, root: Path, *, source_bits: int | None, camera_bits: int, metric: str, frame: int) -> list[Series]:
    reference, _ = analytic_reference(case, frame)
    pattern = re.compile(r"ss(\d+)_(?:b(\d+)_)?os(\d+)(?:_f)?")
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for case_dir in root.glob(f"{case}_{PAPER_TEXTURE_INTERPOLATOR}"):
        for directory in case_dir.iterdir():
            match = pattern.fullmatch(directory.name)
            path = directory / f"image_c00_f{frame:02d}.npy"
            if not match or not path.is_file():
                continue
            texture_bits = int(match.group(2)) if match.group(2) else None
            if source_bits is not None and texture_bits != source_bits:
                continue
            image = load_normalised(path, texture_bits)
            if image.shape == reference.shape:
                grouped[int(match.group(3))].append((int(match.group(1)), image_error_metrics(image, reference, camera_bits, quantise_camera)[metric]))
    return [
        Series(
            f"Tex-OS={osamp}", tuple(x for x, _ in sorted(points)),
            tuple(y for _, y in sorted(points)), camera_bits, *texture_os_style(osamp),
        )
        for osamp, points in sorted(grouped.items())
    ]


def draw_series(axis, series: list[Series], metric_label: str, reference: str) -> list[Line2D]:
    handles = []
    all_samples: list[int] = []; all_values: list[float] = []
    for item in series:
        is_grid = item.label.startswith("Grid2D")
        # Riley is drawn after Grid2D.  The heavier Grid2D trace remains
        # visible around the narrower Riley trace where parity makes them
        # coincide.
        linewidth, markersize = (
            (GRID_LINE_WIDTH_PT, GRID_MARKER_SIZE_PT)
            if is_grid
            else (RILEY_LINE_WIDTH_PT, RILEY_MARKER_SIZE_PT)
        )
        axis.plot(item.samples, item.values, color=item.colour, marker=item.marker, linestyle=item.linestyle, linewidth=linewidth, markersize=markersize)
        handles.append(Line2D([], [], color=item.colour, marker=item.marker, linestyle=item.linestyle, linewidth=linewidth, markersize=markersize, label=item.label))
        all_samples.extend(item.samples); all_values.extend(item.values)
    if not series:
        annotate_no_data(axis, "No completed render data", font_size=FONT_SIZE_PT)
        return handles
    finish_axis(axis, title=f"Reference: {reference}", samples=all_samples, bit_depth=max(item.bit_depth for item in series), values=all_values, ylabel=metric_label, title_font_size=FONT_SIZE_PT, axis_label_font_size=AXIS_LABEL_FONT_SIZE_PT)
    return handles





def figure_function_shaders() -> list[Path]:
    """Figure 1: undeformed and 0.3 px rigid/affine comparisons."""
    studies = (
        ("pt42_cam32_q9_rig", 0, "Undeformed"),
        ("pt42_cam32_q9_rig", 3, "Rigid 0.3px"),
        ("pt42_cam32_q9_aff", 3, "Affine 0.3px"),
    )
    written = []

    # 1. RMSE Figure
    fig_rmse, axes_rmse = make_figure(
        FIGURE_1X3_CM, rows=1, columns=3, tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles_rmse = []
    for col, (case, frame, subtitle) in enumerate(studies):
        reference, ref_label = analytic_reference(case, frame)
        grid, summary_ref = grid_metric_series(case, "e_b", frame)
        bits = {item.bit_depth for item in grid}
        data = grid + function_series(case, reference, "e_b", bits, frame)
        label = display_reference(summary_ref or ref_label)
        handles_rmse.extend(draw_series(
            axes_rmse[0, col], data, "Digitised RMSE [bits]", label,
        ))
        axes_rmse[0, col].set_title(
            f"{panel_prefix(col)} {subtitle}, Ref: {label}",
            fontsize=FONT_SIZE_PT,
        )
    unique_rmse = {h.get_label(): h for h in handles_rmse}
    add_figure_legend(
        fig_rmse, list(unique_rmse.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=3, y_offset=-0.18,
    )
    written.extend(save_figure(
        fig_rmse,
        PAPER_OUTPUT_DIR / "exp1_fig1_eggbox_function_shaders_rmse",
        PAPER_FORMATS, PAPER_DPI,
    ))

    # 2. Max Error Figure
    fig_max, axes_max = make_figure(
        FIGURE_1X3_CM, rows=1, columns=3, tick_font_size=TICK_FONT_SIZE_PT,
    )
    handles_max = []
    for col, (case, frame, subtitle) in enumerate(studies):
        reference, ref_label = analytic_reference(case, frame)
        grid, summary_ref = grid_metric_series(case, "max_eb", frame)
        bits = {item.bit_depth for item in grid}
        data = grid + function_series(case, reference, "max_eb", bits, frame)
        label = display_reference(summary_ref or ref_label)
        handles_max.extend(draw_series(
            axes_max[0, col], data, "Max. digitised err. [bits]", label,
        ))
        axes_max[0, col].set_title(
            f"{panel_prefix(col)} {subtitle}, Ref: {label}",
            fontsize=FONT_SIZE_PT,
        )
    unique_max = {h.get_label(): h for h in handles_max}
    add_figure_legend(
        fig_max, list(unique_max.values()), font_size=LEGEND_FONT_SIZE_PT,
        columns=3, y_offset=-0.18,
    )
    written.extend(save_figure(
        fig_max,
        PAPER_OUTPUT_DIR / "exp1_fig1_eggbox_function_shaders_max_eb",
        PAPER_FORMATS, PAPER_DPI,
    ))

    return written


def figure_texture_convergence() -> list[Path]:
    """Figure 2 & 3: texture convergence with Riley renders."""
    figs_config = (
        ("b8", (
            ("Riley, In: Tex f64, Out: u8", TEXFLOAT_RENDER, None, 8),
            ("Riley, In: Tex u8, Out: u8", TEXUINT_RENDER, 8, 8),
        )),
        ("b12", (
            ("Riley, In: Tex f64, Out: u12", TEXFLOAT_RENDER, None, 12),
            ("Riley, In: Tex u12, Out: u12", TEXUINT_RENDER, 12, 12),
        )),
    )
    written = []

    for suffix, rows_config in figs_config:
        stem = f"exp1_fig2_riley_textures_{suffix}" if suffix == "b8" else f"exp1_fig3_riley_textures_{suffix}"

        # 1. RMSE Figure
        fig_rmse, axes_rmse = make_figure(
            FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT,
        )
        handles_rmse = []
        for row, (row_label, root, src_bits, cam_bits) in enumerate(rows_config):
            for col, (case, frame, deformation) in enumerate(TEXTURE_STUDIES):
                _, ref_label = analytic_reference(case, frame)
                data = texture_series(
                    case, root, source_bits=src_bits, camera_bits=cam_bits,
                    metric="e_b", frame=frame,
                )
                handles_rmse.extend(draw_series(
                    axes_rmse[row, col], data, "Digitised RMSE [bits]",
                    ref_label,
                ))
                axes_rmse[row, col].set_title(
                    f"{panel_prefix(row * 3 + col)} {row_label}\n"
                    f"{deformation}, Ref: {ref_label}",
                    fontsize=FONT_SIZE_PT,
                )
        unique_rmse = {h.get_label(): h for h in handles_rmse}
        add_figure_legend(
            fig_rmse, list(unique_rmse.values()), font_size=LEGEND_FONT_SIZE_PT,
            columns=4, y_offset=-0.13,
        )
        written.extend(save_figure(
            fig_rmse, PAPER_OUTPUT_DIR / f"{stem}_rmse",
            PAPER_FORMATS, PAPER_DPI,
        ))

        # 2. Max Error Figure
        fig_max, axes_max = make_figure(
            FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT,
        )
        handles_max = []
        for row, (row_label, root, src_bits, cam_bits) in enumerate(rows_config):
            for col, (case, frame, deformation) in enumerate(TEXTURE_STUDIES):
                _, ref_label = analytic_reference(case, frame)
                data = texture_series(
                    case, root, source_bits=src_bits, camera_bits=cam_bits,
                    metric="max_eb", frame=frame,
                )
                handles_max.extend(draw_series(
                    axes_max[row, col], data, "Max. digitised err. [bits]",
                    ref_label,
                ))
                axes_max[row, col].set_title(
                    f"{panel_prefix(row * 3 + col)} {row_label}\n"
                    f"{deformation}, Ref: {ref_label}",
                    fontsize=FONT_SIZE_PT,
                )
        unique_max = {h.get_label(): h for h in handles_max}
        add_figure_legend(
            fig_max, list(unique_max.values()), font_size=LEGEND_FONT_SIZE_PT,
            columns=4, y_offset=-0.13,
        )
        written.extend(save_figure(
            fig_max, PAPER_OUTPUT_DIR / f"{stem}_max_eb",
            PAPER_FORMATS, PAPER_DPI,
        ))

    return written


def exp1_figure_stems() -> tuple[str, ...]:
    return (
        "exp1_fig1_eggbox_function_shaders_rmse",
        "exp1_fig1_eggbox_function_shaders_max_eb",
        "exp1_fig2_riley_textures_b8_rmse",
        "exp1_fig2_riley_textures_b8_max_eb",
        "exp1_fig3_riley_textures_b12_rmse",
        "exp1_fig3_riley_textures_b12_max_eb",
        "exp1_fig4_affine_eggbox_difference_maps",
        "exp1_fig5_riley_texf_difference_maps",
    )


def generate_texture_figures() -> list[Path]:
    """Generate the combined texture convergence and difference figures."""
    written = figure_texture_convergence()
    written.extend(figure_texture_difference_maps())
    return written


def figure_affine_function_difference_maps() -> list[Path]:
    """Fig. 4: signed 8-bit affine Eggbox errors for selected Riley SSAA."""
    case, frame = EXP1_DIFF_FUNC_CASE, EXP1_DIFF_FUNC_FRAME
    reference, _ = analytic_reference(case, frame)
    levels = EXP1_DIFF_SSAA_LEVELS
    figure, axes = make_figure(FIGURE_2X3_CM, rows=2, columns=3, tick_font_size=TICK_FONT_SIZE_PT)
    differences: list[np.ndarray | None] = []
    for samples in levels:
        path = FUNC_RENDER / case / f"ss{samples}_f" / f"image_c00_f{frame:02d}.npy"
        if path.is_file():
            image = load_normalised(path)
            if image.shape == reference.shape:
                differences.append(
                    quantise_camera(image, 8).astype(np.float64)
                    - quantise_camera(reference, 8).astype(np.float64)
                )
                continue
        differences.append(None)
    scale = max((float(np.max(np.abs(value))) for value in differences if value is not None), default=1.0)
    images = []
    for index, (samples, difference) in enumerate(zip(levels, differences, strict=True)):
        axis = axes.flat[index]
        axis.set_title(f"{panel_prefix(index)} Px-SS={samples}", fontsize=FONT_SIZE_PT)
        if difference is None:
            annotate_no_data(axis, "No completed render data", font_size=FONT_SIZE_PT)
            continue
        images.append(axis.imshow(
            difference, cmap=DIFFERENCE_CMAP, vmin=-scale, vmax=scale,
            interpolation="nearest", origin="upper"
        ))
        axis.set_xlabel("Pixel x", fontsize=AXIS_LABEL_FONT_SIZE_PT)
        axis.set_ylabel("Pixel y", fontsize=AXIS_LABEL_FONT_SIZE_PT)
    if images:
        colourbar = figure.colorbar(images[0], ax=list(axes.flat), shrink=0.86, pad=0.02)
        colourbar.set_label("Digitised difference [bits]", fontsize=AXIS_LABEL_FONT_SIZE_PT)
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(figure, PAPER_OUTPUT_DIR / "exp1_fig4_affine_eggbox_difference_maps", PAPER_FORMATS, PAPER_DPI)


def figure_texture_difference_maps() -> list[Path]:
    """Fig. 5: 4×4 signed 8-bit difference maps for the rigid 0.3 px case."""
    case, frame = EXP1_DIFF_TEX_CASE, EXP1_DIFF_TEX_FRAME
    reference_image, _ = analytic_reference(case, frame)
    ssaa_levels = EXP2_DIFF_SSAA_LEVELS
    oversamples = EXP2_DIFF_OVERSAMPLES
    root = TEXFLOAT_RENDER / f"{case}_{PAPER_TEXTURE_INTERPOLATOR}"
    figure, axes = make_figure(
        FIGURE_4X4_CM, rows=4, columns=4, tick_font_size=TICK_FONT_SIZE_PT,
    )
    differences: dict[tuple[int, int], np.ndarray | None] = {}
    for ssaa in ssaa_levels:
        for oversamp in oversamples:
            found_path = None
            for suffix in ("", "_f"):
                path = (
                    root
                    / f"ss{ssaa}_os{oversamp}{suffix}"
                    / f"image_c00_f{frame:02d}.npy"
                )
                if path.is_file():
                    found_path = path
                    break
            if found_path is not None:
                image = load_normalised(found_path)
                if image.shape == reference_image.shape:
                    differences[(ssaa, oversamp)] = (
                        quantise_camera(image, 8).astype(np.float64)
                        - quantise_camera(reference_image, 8).astype(np.float64)
                    )
                    continue
            differences[(ssaa, oversamp)] = None
    scale = max(
        (float(np.max(np.abs(value))) for value in differences.values()
         if value is not None),
        default=1.0,
    )
    images = []
    for row, ssaa in enumerate(ssaa_levels):
        for column, oversamp in enumerate(oversamples):
            axis = axes[row, column]
            difference = differences[(ssaa, oversamp)]
            axis.set_title(
                f"{panel_prefix(row * len(oversamples) + column)} "
                f"Px-SS={ssaa}, Tex-OS={oversamp}",
                fontsize=FONT_SIZE_PT,
            )
            if difference is None:
                annotate_no_data(
                    axis, "No completed render data",
                    font_size=FONT_SIZE_PT,
                )
                continue
            images.append(axis.imshow(
                difference, cmap=DIFFERENCE_CMAP, vmin=-scale, vmax=scale,
                interpolation="nearest", origin="upper",
            ))
            if row == len(ssaa_levels) - 1:
                axis.set_xlabel("Pixel x", fontsize=AXIS_LABEL_FONT_SIZE_PT)
            if column == 0:
                axis.set_ylabel("Pixel y", fontsize=AXIS_LABEL_FONT_SIZE_PT)
    if images:
        colourbar = figure.colorbar(
            images[0], ax=list(axes.flat), shrink=0.9, pad=0.015,
        )
        colourbar.set_label(
            "Digitised difference [bits]", fontsize=AXIS_LABEL_FONT_SIZE_PT,
        )
        colourbar.ax.tick_params(labelsize=TICK_FONT_SIZE_PT)
    return save_figure(
        figure,
        PAPER_OUTPUT_DIR / "exp1_fig5_riley_texf_difference_maps",
        PAPER_FORMATS, PAPER_DPI,
    )


def remove_superseded_figures() -> None:
    """Remove only previously generated Exp1 paper figures no longer used."""
    stems = (
        "exp1_fig1_1_rigid_eggbox_function_shaders",
        "exp1_fig1_2_affine_eggbox_function_shaders",
        "exp1_fig2_riley_textures_pt42_cam32_q9_rig",
        "exp1_fig3_riley_textures_pt42_cam32_q9_aff",
        "exp1_fig4_riley_textures_pt42_cam32_q9_qsadd",
        "exp1_fig2_riley_texf_b8_rmse",
        "exp1_fig2_riley_texf_b8_max_eb",
        "exp1_fig3_riley_texu8_b8_rmse",
        "exp1_fig3_riley_texu8_b8_max_eb",
        "exp1_fig4_riley_texu12_b12_rmse",
        "exp1_fig4_riley_texu12_b12_max_eb",
        "exp1_fig5_riley_texf_b12_rmse",
        "exp1_fig5_riley_texf_b12_max_eb",
        "exp2_fig3_riley_texf_rmse",
        "exp2_fig3_riley_texf_max_eb",
        "exp1_fig2_riley_textures_rmse",
        "exp1_fig2_riley_textures_max_eb",
        "exp2_fig3_riley_textures_disk_rmse",
        "exp2_fig3_riley_textures_disk_max_eb",
        "exp2_fig3_riley_textures_gauss_rmse",
        "exp2_fig3_riley_textures_gauss_max_eb",
    )
    for output_dir in paper_output_directories():
        for stem in stems:
            for extension in (*PAPER_FORMATS, "tex"):
                (output_dir / stem).with_suffix(f".{extension}").unlink(missing_ok=True)


def write_tex_preview() -> list[Path]:
    """Write editable figure blocks and compile a minimal A4 preview article."""
    return write_latex_preview(exp1_figure_stems(), FIGURE_CAPTIONS, FIGURE_LABELS)


def main() -> None:
    remove_superseded_figures()
    written = figure_function_shaders()
    written.extend(generate_texture_figures())
    written.extend(figure_affine_function_difference_maps())
    written.extend(write_tex_preview())
    print("Wrote paper figures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
