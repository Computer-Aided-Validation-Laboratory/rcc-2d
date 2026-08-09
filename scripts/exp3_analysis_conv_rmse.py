#!/usr/bin/env python3
"""Plot convergence of RMSE vs Reference for DIC and Grid Method."""

from __future__ import annotations

import csv
import gc
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Ensure local imports work correctly
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp3_analysis_conv import is_psf
from exp3params import FORCE_INTERP_BIAS_OVERWRITE
from modules.analysis_parallel import run_analysis_jobs
from modules.analysis_selection import (
    analysis_should_run,
    mark_analysis_complete,
)
from exp0params_common import DIAGNOSTIC_FIGURE_DPI
from modules.exp3_analysis_common import (
    OUT,
    OS_RE,
    SS_RE,
    interpolator_of,
    numeric_y_axis,
    parameter,
    pattern_of,
    release,
    title_lines,
)

# Reference exclusions
EDGE_EXCLUSION_DIC = 15
EDGE_EXCLUSION_GRID = 12

from modules.render_outputs import quantise_camera


def load_render_image(render_dir: Path, frame: int) -> np.ndarray | None:
    p1 = render_dir / f"frame{frame:02d}.npy"
    if p1.is_file():
        return np.load(p1)
    p2 = render_dir / f"image_c00_f{frame:02d}.npy"
    if p2.is_file():
        return np.load(p2)
    return None

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


@dataclass(frozen=True)
class DicRecord:
    case: str
    root: str
    config: str
    directory: Path
    bit_depth: int
    pattern: str
    ssaa: int
    osamp: int
    interpolator: str
    analytic: bool


@dataclass(frozen=True)
class GridRecord:
    case: str
    root: str
    config: str
    directory: Path
    bit_depth: int
    pattern: str
    ssaa: int
    osamp: int
    interpolator: str
    analytic: bool


def discover_dic() -> list[DicRecord]:
    from modules.exp3_dic_data import reconstruct_config_name
    rows = []
    for directory in (OUT / "exp3_dic").glob("*/*/*"):
        if not directory.is_dir():
            continue
        case = directory.parent.parent.name
        root = directory.parent.name
        base_config = directory.name
        npzs = list(directory.glob("dic_*_b*_frame*.npz"))
        if not npzs:
            continue
        seen_combinations = set()
        for npz in npzs:
            match = re.match(r"^dic_(.+)_b(\d+)_frame\d+\.npz$", npz.name)
            if not match:
                continue
            suffix = match.group(1)
            bit_depth = int(match.group(2))
            combo = (suffix, bit_depth)
            if combo in seen_combinations:
                continue
            seen_combinations.add(combo)
            config = reconstruct_config_name(base_config, suffix)
            pattern = pattern_of(config)
            rows.append(DicRecord(
                case=case,
                root=root,
                config=config,
                directory=directory,
                bit_depth=bit_depth,
                pattern=pattern,
                ssaa=parameter(config, SS_RE),
                osamp=parameter(config, OS_RE),
                interpolator=interpolator_of(config),
                analytic=("_analytic" in config),
            ))
    return rows


def discover_grid() -> list[GridRecord]:
    values = []
    for directory in (OUT / "exp3_gridmethod").glob("*/*/*/b*"):
        if not directory.is_dir():
            continue
        if not list(directory.glob("displacement_frame*.npz")):
            continue
        try:
            bit_depth = int(directory.name.removeprefix("b"))
        except ValueError:
            continue
        case = directory.parent.parent.parent.name
        root = directory.parent.parent.name
        config = directory.parent.name
        values.append(GridRecord(
            case=case,
            root=root,
            config=config,
            directory=directory,
            bit_depth=bit_depth,
            pattern=pattern_of(config),
            ssaa=parameter(config, SS_RE),
            osamp=parameter(config, OS_RE),
            interpolator=interpolator_of(config),
            analytic=("_analytic" in config or "_analytic" in root),
        ))
    return values


def load_dic(record: DicRecord, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    from modules.exp3_dic_data import load_result, parse_config, result_path
    _, suffix = parse_config(record.config)
    path = result_path(record.directory, suffix, record.bit_depth, frame)
    if not path.is_file():
        return None
    try:
        data = load_result(path)
        return data["ss_x"], data["ss_y"], data["u_px"][0], -data["v_px"][0]
    except Exception:
        return None


def load_grid(record: GridRecord, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = record.directory / f"displacement_frame{frame:02d}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return (
                np.asarray(data["ux"], dtype=np.float64),
                np.asarray(data["uy"], dtype=np.float64),
            )
    except Exception:
        return None


def select_dic_reference(records: list[DicRecord]) -> tuple[DicRecord | None, str]:
    analytic = [r for r in records if r.analytic]
    if analytic:
        return analytic[0], "Analytic Baseline"
    if not records:
        return None, "No reference"
    bespoke = [r for r in records if "grid2d" in r.root or "speck2d" in r.root]
    if bespoke:
        ref = max(bespoke, key=lambda r: r.ssaa)
        return ref, f"Highest Bespoke SSAA Ref (SSAA={ref.ssaa})"
    ref = max(records, key=lambda r: (r.ssaa, r.osamp))
    return ref, f"Highest SSAA/OS Ref (SSAA={ref.ssaa}, OS={ref.osamp})"


def select_grid_reference(records: list[GridRecord]) -> tuple[GridRecord | None, str]:
    analytic = [r for r in records if r.analytic]
    if analytic:
        return analytic[0], "Analytic Baseline"
    if not records:
        return None, "No reference"
    ref = max(records, key=lambda r: (r.ssaa, r.osamp))
    return ref, f"Highest SSAA/OS Ref (SSAA={ref.ssaa}, OS={ref.osamp})"


def get_series_label(record: DicRecord | GridRecord) -> str:
    psf = "_psf" in record.root or "_psf" in record.config
    if "riley_render_tex" in record.root:
        return f"riley_texf_{record.interpolator}{'_psf' if psf else ''}"
    suffix = "_psf" if psf and "_psf" not in record.root else ""
    return f"{record.root.replace('_render_ssaa', '')}{suffix}"


def analyse_dic_job(
    payload: tuple[DicRecord, list[DicRecord]]
) -> list[dict]:
    rec, candidates = payload
    ref, ref_name = select_dic_reference(candidates)
    if ref is None or rec == ref:
        return []
    rows = []
    for frame in range(1, 11):
        ref_data = load_dic(ref, frame)
        rec_data = load_dic(rec, frame)
        if ref_data is None or rec_data is None:
            continue
        rx, ry, ru, rv = ref_data
        _, _, cu, cv = rec_data
        if ru.shape != cu.shape:
            continue

        du, dv = cu - ru, cv - rv
        # Exclude boundary region
        x_min, x_max = rx.min(), rx.max()
        y_min, y_max = ry.min(), ry.max()
        mask = (
            (rx < x_min + EDGE_EXCLUSION_DIC)
            | (rx > x_max - EDGE_EXCLUSION_DIC)
            | (ry < y_min + EDGE_EXCLUSION_DIC)
            | (ry > y_max - EDGE_EXCLUSION_DIC)
        )
        du = np.where(mask, np.nan, du)
        dv = np.where(mask, np.nan, dv)

        rmse = float(np.sqrt(np.nanmean(du * du + dv * dv)))
        stacked = np.stack((du, dv))
        disp_max = (
            float(np.nanmax(np.abs(stacked)))
            if np.any(np.isfinite(stacked))
            else np.nan
        )

        rec_render_dir = OUT / rec.root / rec.case / rec.config
        ref_render_dir = OUT / ref.root / ref.case / ref.config
        rec_img = load_render_image(rec_render_dir, frame)
        ref_img = load_render_image(ref_render_dir, frame)
        if rec_img is not None and ref_img is not None:
            rec_codes = quantise_camera(
                rec_img, rec.bit_depth
            ).astype(np.int64)
            ref_codes = quantise_camera(
                ref_img, rec.bit_depth
            ).astype(np.int64)
            code_diff = rec_codes - ref_codes
            digitised_rmse = float(np.sqrt(np.mean(code_diff ** 2)))
            digitised_max_err = float(np.max(np.abs(code_diff)))
        else:
            digitised_rmse = np.nan
            digitised_max_err = np.nan

        rows.append({
            "Case": rec.case,
            "Pattern": rec.pattern,
            "Series": get_series_label(rec),
            "Config": rec.config,
            "BitDepth": rec.bit_depth,
            "Frame": frame,
            "SSAA": rec.ssaa or 1,
            "OS": rec.osamp or 1,
            "Reference": ref_name,
            "DispErrRMSEToRef(px)": rmse,
            "DispErrMaxToRef(px)": disp_max,
            "DigitisedRMSE(bits)": digitised_rmse,
            "DigitisedMaxErr(bits)": digitised_max_err,
        })
    return rows


def analyse_grid_job(
    payload: tuple[GridRecord, list[GridRecord]]
) -> list[dict]:
    rec, candidates = payload
    ref, ref_name = select_grid_reference(candidates)
    if ref is None or rec == ref:
        return []
    rows = []
    for frame in range(1, 11):
        ref_data = load_grid(ref, frame)
        rec_data = load_grid(rec, frame)
        if ref_data is None or rec_data is None:
            continue
        ru, rv = ref_data
        cu, cv = rec_data
        if ru.shape != cu.shape:
            continue

        du, dv = cu - ru, cv - rv
        # Exclude boundary region
        mask = np.ones(ru.shape, dtype=bool)
        mask[
            EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID,
            EDGE_EXCLUSION_GRID:-EDGE_EXCLUSION_GRID
        ] = False
        du = np.where(mask, np.nan, du)
        dv = np.where(mask, np.nan, dv)

        rmse = float(np.sqrt(np.nanmean(du * du + dv * dv)))
        stacked = np.stack((du, dv))
        disp_max = (
            float(np.nanmax(np.abs(stacked)))
            if np.any(np.isfinite(stacked))
            else np.nan
        )

        rec_render_dir = OUT / rec.root / rec.case / rec.config
        ref_render_dir = OUT / ref.root / ref.case / ref.config
        rec_img = load_render_image(rec_render_dir, frame)
        ref_img = load_render_image(ref_render_dir, frame)
        if rec_img is not None and ref_img is not None:
            rec_codes = quantise_camera(
                rec_img, rec.bit_depth
            ).astype(np.int64)
            ref_codes = quantise_camera(
                ref_img, rec.bit_depth
            ).astype(np.int64)
            code_diff = rec_codes - ref_codes
            digitised_rmse = float(np.sqrt(np.mean(code_diff ** 2)))
            digitised_max_err = float(np.max(np.abs(code_diff)))
        else:
            digitised_rmse = np.nan
            digitised_max_err = np.nan

        rows.append({
            "Case": rec.case,
            "Pattern": rec.pattern,
            "Series": get_series_label(rec),
            "Config": rec.config,
            "BitDepth": rec.bit_depth,
            "Frame": frame,
            "SSAA": rec.ssaa or 1,
            "OS": rec.osamp or 1,
            "Reference": ref_name,
            "DispErrRMSEToRef(px)": rmse,
            "DispErrMaxToRef(px)": disp_max,
            "DigitisedRMSE(bits)": digitised_rmse,
            "DigitisedMaxErr(bits)": digitised_max_err,
        })
    return rows


def plot_convergence_grid(
    fig_dir: Path,
    title_prefix: str,
    case: str,
    pattern: str,
    series_name: str,
    bit_depth: int,
    values: list[dict],
) -> None:
    available_frames = sorted({int(row["Frame"]) for row in values})
    psf_str = " (with PSF)" if "_psf" in series_name else ""

    if len(available_frames) == 1:
        # Single panel for single deformed frame
        fig = Figure(figsize=(7, 6), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        frame = available_frames[0]
        frame_vals = [row for row in values if int(row["Frame"]) == frame]

        by_os = defaultdict(list)
        for r in frame_vals:
            by_os[int(r["OS"])].append(r)

        plotted = []
        for osamp, series in sorted(by_os.items(), reverse=True):
            series.sort(key=lambda r: int(r["SSAA"]))
            ssaa_vals = [int(r["SSAA"]) for r in series]
            rmse_errs = [float(r["DispErrRMSEToRef(px)"]) for r in series]
            label = f"OS={osamp}" if len(by_os) > 1 else "SSAA Series"
            ax.plot(
                ssaa_vals, rmse_errs, "o-",
                color=COLOR_BY_LEVEL.get(osamp), label=label
            )
            plotted.extend(rmse_errs)

        ssaa_ticks = sorted({int(r["SSAA"]) for r in frame_vals})
        ax.set_xscale("log", base=2)
        numeric_y_axis(ax, plotted)
        ax.axhline(
            0.01, color="red", linestyle="--", alpha=0.5, label="0.01 px"
        )
        ax.set_xticks(ssaa_ticks)
        ax.set_xticklabels([str(t) for t in ssaa_ticks])
        ax.set_xlabel("Axis integration samples")
        ax.set_ylabel("Disp. RMSE [px]")
        ax.set_title(f"Frame {frame}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ref_desc = str(values[0]["Reference"])
        fig.suptitle(
            f"{title_prefix}\n"
            f"Case: {case} | Pattern: {clean_pattern_desc(pattern)} "
            f"| {bit_depth}-bit{psf_str}\n"
            f"Reference: {ref_desc}",
            fontsize=11,
            fontweight="bold",
        )
    else:
        # Multi-panel (3x2 grid) for multi-frame studies
        fig = Figure(figsize=(10, 8), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 2)
        flat_axes = axes.flatten()

        target_frames = [1, 3, 5, 7, 10]
        val_by_frame = defaultdict(list)
        for row in values:
            val_by_frame[int(row["Frame"])].append(row)

        # Plot 5 frames
        for i, frame in enumerate(target_frames):
            ax = flat_axes[i]
            frame_vals = val_by_frame[frame]

            if not frame_vals:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                continue

            by_os = defaultdict(list)
            for r in frame_vals:
                by_os[int(r["OS"])].append(r)
            plotted = []
            for osamp, series in sorted(by_os.items(), reverse=True):
                series.sort(key=lambda r: int(r["SSAA"]))
                ssaa_vals = [int(r["SSAA"]) for r in series]
                rmse_errs = [float(r["DispErrRMSEToRef(px)"]) for r in series]
                label = f"OS={osamp}" if len(by_os) > 1 else "SSAA Series"
                ax.plot(
                    ssaa_vals, rmse_errs, "o-",
                    color=COLOR_BY_LEVEL.get(osamp), label=label
                )
                plotted.extend(rmse_errs)

            ssaa_ticks = sorted({int(r["SSAA"]) for r in frame_vals})
            ax.set_xscale("log", base=2)
            numeric_y_axis(ax, plotted)
            ax.axhline(
                0.01, color="red", linestyle="--", alpha=0.5, label="0.01 px"
            )
            ax.set_xticks(ssaa_ticks)
            ax.set_xticklabels([str(t) for t in ssaa_ticks])
            ax.set_xlabel("Axis integration samples")
            ax.set_ylabel("Disp. RMSE [px]")
            ax.set_title(f"Frame {frame}", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        # Empty 6th subplot for shared legend/info
        ax_info = flat_axes[5]
        ax_info.axis("off")
        ref_desc = str(values[0]["Reference"])
        ax_info.text(
            0.1, 0.5,
            f"Case: {case}\n"
            f"Pattern: {clean_pattern_desc(pattern)}\n"
            f"Series: {series_name}\n"
            f"Reference: {ref_desc}",
            fontsize=10,
            ha="left", va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        fig.suptitle(
            f"{title_prefix}\n"
            f"Case: {case} | Pattern: {clean_pattern_desc(pattern)} "
            f"| {bit_depth}-bit{psf_str}",
            fontsize=12,
            fontweight="bold",
        )

    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / f"convergence_rmse_{pattern}_b{bit_depth:02d}.png"
    fig.savefig(save_path, dpi=DIAGNOSTIC_FIGURE_DPI)
    fig.clear()
    gc.collect()
    release()


def clean_pattern_desc(name: str) -> str:
    mapping = {
        "diskaddsat": "Disk-addition Speckle",
        "gausscont": "Gaussian-continuous Speckle",
        "eggb": "Eggbox Grid",
    }
    return mapping.get(name, name)


def filter_candidates(record: DicRecord, records: list[DicRecord]) -> list[DicRecord]:
    rec_psf = is_psf(record)
    candidates = []
    for item in records:
        if (
            item.case == record.case
            and item.bit_depth == record.bit_depth
            and item.pattern == record.pattern
        ):
            if is_psf(item) == rec_psf:
                candidates.append(item)
    return candidates


def filter_grid_candidates(record: GridRecord, records: list[GridRecord]) -> list[GridRecord]:
    rec_psf = is_psf(record)
    candidates = []
    for item in records:
        if (
            item.case == record.case
            and item.bit_depth == record.bit_depth
            and item.pattern == record.pattern
        ):
            if is_psf(item) == rec_psf:
                candidates.append(item)
    return candidates


def main() -> None:
    dic_out_dir = OUT / "exp3_analysis_dic/dic_disp_err_conv"
    grid_out_dir = OUT / "exp3_analysis_gridmethod/grid_disp_err_conv"

    # We will overwrite whenever FORCE_INTERP_BIAS_OVERWRITE is True
    force_overwrite = FORCE_INTERP_BIAS_OVERWRITE

    # ----------------------------------------------------
    # Part 1: DIC Displacement Field Convergence (Affine & Chirp/Star)
    # ----------------------------------------------------
    if analysis_should_run(
        dic_out_dir,
        "DIC Displacement Convergence (RMSE vs Ref)",
        force_overwrite=force_overwrite,
    ):
        dic_records = discover_dic()
        # Keep only affine and chirp cases
        target_dic = [
            r for r in dic_records
            if r.case in ("pt516_cam512_q9_aff", "pt260x65_cam256_q9_chirp")
        ]

        if target_dic:
            print("Processing DIC convergence records...")
            jobs = [(r, filter_candidates(r, dic_records)) for r in target_dic]
            results = run_analysis_jobs(
                "Analyzing DIC displacement RMSE",
                jobs,
                analyse_dic_job,
                mp_context=get_context("spawn"),
            )

            csv_rows = []
            for res in results:
                csv_rows.extend(res)

            # Group and plot DIC convergence
            groups = defaultdict(list)
            for row in csv_rows:
                key = (row["Case"], row["Pattern"], row["Series"], row["BitDepth"])
                groups[key].append(row)

            print("Plotting DIC convergence figures...")
            for key, values in groups.items():
                case, pattern, series_name, bit_depth = key
                fig_dir = dic_out_dir / case / f"{series_name}_rmse_conv"
                plot_convergence_grid(
                    fig_dir,
                    "DIC Displacement Field Convergence (RMSE vs Reference)",
                    case, pattern, series_name, bit_depth, values
                )
            # Write DIC summary CSV
            csv_path = dic_out_dir.parent / "summary.csv"
            dic_out_dir.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(csv_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(csv_rows)
            mark_analysis_complete(dic_out_dir)

    # ----------------------------------------------------
    # Part 2: Grid Method Displacement Field Convergence
    # ----------------------------------------------------
    if analysis_should_run(
        grid_out_dir,
        "Grid Method Displacement Convergence (RMSE vs Ref)",
        force_overwrite=force_overwrite,
    ):
        grid_records = discover_grid()
        target_grid = [
            r for r in grid_records
            if r.case in ("pt516_cam512_q9_aff", "pt260x65_cam256_q9_chirp")
        ]
        if target_grid:
            print("Processing Grid Method convergence records...")
            jobs = [
                (r, filter_grid_candidates(r, grid_records))
                for r in target_grid
            ]
            results = run_analysis_jobs(
                "Analyzing Grid Method displacement RMSE",
                jobs,
                analyse_grid_job,
                mp_context=get_context("spawn"),
            )

            csv_rows = []
            for res in results:
                csv_rows.extend(res)

            # Group and plot Grid Method convergence
            groups = defaultdict(list)
            for row in csv_rows:
                key = (row["Case"], row["Pattern"], row["Series"], row["BitDepth"])
                groups[key].append(row)

            print("Plotting Grid Method convergence figures...")
            for key, values in groups.items():
                case, pattern, series_name, bit_depth = key
                fig_dir = grid_out_dir / case / f"{series_name}_rmse_conv"
                plot_convergence_grid(
                    fig_dir,
                    "Grid Method Displacement Field Convergence (RMSE vs Reference)",
                    case, pattern, series_name, bit_depth, values
                )
            # Write Grid Method summary CSV
            csv_path = grid_out_dir.parent / "summary.csv"
            grid_out_dir.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(csv_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(csv_rows)
            mark_analysis_complete(grid_out_dir)


if __name__ == "__main__":
    main()
