#!/usr/bin/env python3
"""Compare and plot rigid body interpolation bias in grid method."""

from __future__ import annotations

import csv
import gc
import os
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Ensure local imports work correctly
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp3_analysis_gridmethod import Record, discover
from exp3params import FORCE_INTERP_BIAS_OVERWRITE
from modules.analysis_parallel import run_analysis_jobs
from modules.analysis_selection import (
    analysis_should_run,
    mark_analysis_complete,
)
from modules.exp3_analysis_common import (
    OUT,
    OS_RE,
    SS_RE,
    interpolator_of,
    parameter,
    pattern_of,
    release,
)
from modules.output_naming import is_rigid_case
from exp0params_common import DIAGNOSTIC_FIGURE_DPI

EDGE_EXCLUSION_PX = 12

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


def is_psf(record: Record) -> bool:
    return "_psf" in record.root or "_psf" in record.config


def clean_pattern_name(pattern: str) -> str:
    mapping = {
        "diskaddsat": "Disk-addition Speckle",
        "gausscont": "Gaussian-continuous Speckle",
        "eggb": "Eggbox Grid",
    }
    return mapping.get(pattern, pattern)


def physical_expected_rigid(
    case: str,
    frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    from modules.output_naming import data_case_name
    data_dir = Path("data") / data_case_name(case)
    ux_path = data_dir / "field_disp_x.csv"
    uy_path = data_dir / "field_disp_y.csv"
    if not ux_path.is_file() or not uy_path.is_file():
        return None
    ux = np.loadtxt(ux_path, delimiter=",")
    uy = np.loadtxt(uy_path, delimiter=",")
    if ux.ndim == 1:
        ux, uy = ux[None, :], uy[None, :]
    return (
        ux[:, :frames].mean(axis=0),
        uy[:, :frames].mean(axis=0),
    )


def clean_interpolator_name(name: str) -> str:
    mapping = {
        "cubiccm": "Catmull-Rom",
        "cubic_catmull_rom": "Catmull-Rom",
        "cubic_bspline": "B-spline",
        "cubic_mitchell_netravali": "Mitchell-Netravali",
        "nearest": "Nearest",
        "near": "Nearest",
        "linear": "Linear",
        "line": "Linear",
    }
    return mapping.get(name, name.replace("_", " ").title())


def process_record_data(
    record: Record,
    expected_ux_arr: np.ndarray,
    expected_uy_arr: np.ndarray,
    reference_fields: dict[int, tuple[np.ndarray, np.ndarray]] | None,
) -> list[dict] | None:
    rows = []
    psf_val = is_psf(record)
    interpolator = interpolator_of(record.config)

    if (
        "speck2d_render_analytic" in record.root
        or "grid2d_render_analytic" in record.root
    ):
        renderer = "Analytic"
        interpolant = "None"
        method = "None"
        samples = "1"
    elif (
        "speck2d_render_ssaa" in record.root
        or "grid2d_render_ssaa" in record.root
    ):
        renderer = "Bespoke Shader"
        interpolant = "None"
        method = "SSAA"
        samples = str(record.ssaa)
    elif "riley_render_texf" in record.root:
        renderer = "Riley Texture"
        interpolant = clean_interpolator_name(interpolator)
        method = "SSAA+OS"
        samples = f"{record.ssaa}x{record.osamp}"
    else:
        renderer = record.root
        interpolant = interpolator
        method = "Unknown"
        samples = "Unknown"

    rows.append({
        "Case": record.case,
        "Pattern": record.config.split("_")[0],
        "BitDepth": record.bit_depth,
        "PSF": psf_val,
        "Renderer": renderer,
        "Interpolant": interpolant,
        "PixelIntegrationMethod": method,
        "PixelIntegrationSamples": samples,
        "Frame": 0,
        "ExpectedShiftX": 0.0,
        "ExpectedShiftY": 0.0,
        "MeanUxCase": 0.0,
        "MeanUyCase": 0.0,
        "MeanUxDiffAnal": 0.0,
        "MeanUyDiffAnal": 0.0,
        "StdUxCase": 0.0,
        "StdUyCase": 0.0,
        "StdUxDiffAnal": 0.0,
        "StdUyDiffAnal": 0.0,
        "RmseAnalRefUx": 0.0,
        "RmseAnalRefUy": 0.0,
        "RmseExactDispUx": 0.0,
        "RmseExactDispUy": 0.0,
        "MaxAbsErrUx": 0.0,
        "MaxAbsErrUy": 0.0,
    })

    for frame in range(1, 11):
        path = record.directory / f"displacement_frame{frame:02d}.npz"
        if not path.is_file():
            return None
        try:
            with np.load(path) as value:
                ux = np.asarray(value["ux"], dtype=np.float64)
                uy = np.asarray(value["uy"], dtype=np.float64)

            # Exclude boundary region
            mask = np.ones(ux.shape, dtype=bool)
            mask[
                EDGE_EXCLUSION_PX:-EDGE_EXCLUSION_PX,
                EDGE_EXCLUSION_PX:-EDGE_EXCLUSION_PX
            ] = False
            u_masked = np.where(mask, np.nan, ux)
            v_masked = np.where(mask, np.nan, uy)

            mean_ux_case = float(np.nanmean(u_masked))
            mean_uy_case = float(np.nanmean(v_masked))
            std_case_ux = float(np.nanstd(u_masked))
            std_case_uy = float(np.nanstd(v_masked))

            # Calculate metrics relative to reference
            rmse_anal_ref_ux = 0.0
            rmse_anal_ref_uy = 0.0
            mean_diff_anal_ux = 0.0
            mean_diff_anal_uy = 0.0
            std_diff_anal_ux = 0.0
            std_diff_anal_uy = 0.0
            max_abs_err_ux = 0.0
            max_abs_err_uy = 0.0
            if (
                reference_fields is not None
                and frame in reference_fields
            ):
                ref_u, ref_v = reference_fields[frame]
                diff_u = u_masked - ref_u
                diff_v = v_masked - ref_v
                rmse_anal_ref_ux = float(np.sqrt(np.nanmean(diff_u * diff_u)))
                rmse_anal_ref_uy = float(np.sqrt(np.nanmean(diff_v * diff_v)))
                mean_diff_anal_ux = float(np.nanmean(diff_u))
                mean_diff_anal_uy = float(np.nanmean(diff_v))
                std_diff_anal_ux = float(np.nanstd(diff_u))
                std_diff_anal_uy = float(np.nanstd(diff_v))
                max_abs_err_ux = float(np.nanmax(np.abs(diff_u)))
                max_abs_err_uy = float(np.nanmax(np.abs(diff_v)))

            # Calculate RMSE(ExactDisp) relative to prescribed shift
            exp_ux = float(expected_ux_arr[frame])
            exp_uy = float(expected_uy_arr[frame])

            diff_exact_u = u_masked - exp_ux
            diff_exact_v = v_masked - exp_uy
            rmse_exact_disp_ux = float(
                np.sqrt(np.nanmean(diff_exact_u * diff_exact_u))
            )
            rmse_exact_disp_uy = float(
                np.sqrt(np.nanmean(diff_exact_v * diff_exact_v))
            )

            rows.append({
                "Case": record.case,
                "Pattern": record.config.split("_")[0],
                "BitDepth": record.bit_depth,
                "PSF": psf_val,
                "Renderer": renderer,
                "Interpolant": interpolant,
                "PixelIntegrationMethod": method,
                "PixelIntegrationSamples": samples,
                "Frame": frame,
                "ExpectedShiftX": exp_ux,
                "ExpectedShiftY": exp_uy,
                "MeanUxCase": mean_ux_case,
                "MeanUyCase": mean_uy_case,
                "MeanUxDiffAnal": mean_diff_anal_ux,
                "MeanUyDiffAnal": mean_diff_anal_uy,
                "StdUxCase": std_case_ux,
                "StdUyCase": std_case_uy,
                "StdUxDiffAnal": std_diff_anal_ux,
                "StdUyDiffAnal": std_diff_anal_uy,
                "RmseAnalRefUx": rmse_anal_ref_ux,
                "RmseAnalRefUy": rmse_anal_ref_uy,
                "RmseExactDispUx": rmse_exact_disp_ux,
                "RmseExactDispUy": rmse_exact_disp_uy,
                "MaxAbsErrUx": max_abs_err_ux,
                "MaxAbsErrUy": max_abs_err_uy,
            })
        except Exception:
            return None

    return rows


def process_record_job(
    task: tuple[Record, dict[int, tuple[np.ndarray, np.ndarray]] | None]
) -> tuple[Record, list[dict] | None]:
    record, analytic_ref_fields = task
    expected = physical_expected_rigid(record.case, 11)
    if expected is None:
        return record, None
    expected_ux_arr, expected_uy_arr = expected
    res = process_record_data(
        record,
        expected_ux_arr,
        expected_uy_arr,
        analytic_ref_fields,
    )
    return record, res


def find_record(
    records: list[Record],
    render_root_prefix: str,
    interpolator: str | None = None,
    ssaa: int | None = None,
    osamp: int | None = None,
    analytic: bool | None = None,
    highest_quality: bool = False,
) -> Record | None:
    candidates = []
    for r in records:
        if not r.root.startswith(render_root_prefix):
            continue
        r_interp = interpolator_of(r.config)
        if interpolator is not None and r_interp != interpolator:
            continue
        if ssaa is not None and r.ssaa != ssaa:
            continue
        if osamp is not None and r.osamp != osamp:
            continue
        if analytic is not None and r.analytic != analytic:
            continue
        candidates.append(r)

    if not candidates:
        return None

    if highest_quality:
        candidates.sort(key=lambda r: (r.ssaa, r.osamp), reverse=True)
        return candidates[0]

    return candidates[0]


def plot_bias_lines(
    dir_path: Path,
    filename: str,
    title_prefix: str,
    case: str,
    pattern: str,
    bit_depth: int,
    psf: bool,
    lines_to_plot: list[dict],
    mode: str,
    ref_label: str,
) -> None:
    valid_lines = [line for line in lines_to_plot if "data" in line]

    if not valid_lines:
        return

    analytic_line = None
    for line in valid_lines:
        if line["record"].analytic:
            analytic_line = line
            break

    max_err_ux = 0.0
    max_err_uy = 0.0
    if analytic_line is not None:
        if mode == "std_case":
            max_err_ux = np.max(analytic_line["data"]["std_case_ux"])
            max_err_uy = np.max(analytic_line["data"]["std_case_uy"])
        elif mode == "std_diff_anal_ref":
            max_err_ux = np.max([
                np.max(line["data"]["std_diff_anal_ux"])
                for line in valid_lines
            ])
            max_err_uy = np.max([
                np.max(line["data"]["std_diff_anal_uy"])
                for line in valid_lines
            ])
        elif mode == "rmse_anal_ref":
            max_err_ux = np.max([
                np.max(line["data"]["rmse_anal_ref_ux"]) for line in valid_lines
            ])
            max_err_uy = np.max([
                np.max(line["data"]["rmse_anal_ref_uy"]) for line in valid_lines
            ])
        elif mode == "rmse_exact_disp":
            max_err_ux = np.max([
                np.max(line["data"]["rmse_exact_disp_ux"])
                for line in valid_lines
            ])
            max_err_uy = np.max([
                np.max(line["data"]["rmse_exact_disp_uy"])
                for line in valid_lines
            ])
        else:
            max_err_ux = np.max(np.abs(analytic_line["data"]["bias_ux"]))
            max_err_uy = np.max(np.abs(analytic_line["data"]["bias_uy"]))

    fig = Figure(figsize=(10, 4.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)

    components = [
        ("ux", "$u_x$ component", max_err_ux),
        ("uy", "$u_y$ component", max_err_uy),
    ]

    for col_idx, (comp_key, comp_title, max_err) in enumerate(components):
        ax = axes[col_idx]
        if (
            mode != "std_case"
            and mode != "std_diff_anal_ref"
            and mode != "rmse_anal_ref"
            and mode != "rmse_exact_disp"
        ):
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)

        # Plot non-analytic lines first
        for line in valid_lines:
            if line["record"].analytic:
                continue
            x = line["data"]["expected_ux"]

            if mode == "std_case":
                y = line["data"][f"std_case_{comp_key}"]
            elif mode == "std_diff_anal_ref":
                y = line["data"][f"std_diff_anal_{comp_key}"]
            elif mode == "rmse_anal_ref":
                y = line["data"][f"rmse_anal_ref_{comp_key}"]
            elif mode == "rmse_exact_disp":
                y = line["data"][f"rmse_exact_disp_{comp_key}"]
            else:
                y = line["data"][f"bias_{comp_key}"]

            ax.plot(
                x,
                y,
                label=line["label"],
                color=line.get("color"),
                linestyle=line.get("linestyle", "-"),
                marker=line.get("marker"),
                linewidth=line.get("linewidth", 1.5),
                markersize=5,
                markevery=1,
            )

        # Plot analytic line last
        for line in valid_lines:
            if line["record"].analytic:
                x = line["data"]["expected_ux"]

                if mode == "std_case":
                    y = line["data"][f"std_case_{comp_key}"]
                elif mode == "std_diff_anal_ref":
                    y = line["data"][f"std_diff_anal_{comp_key}"]
                elif mode == "rmse_anal_ref":
                    y = line["data"][f"rmse_anal_ref_{comp_key}"]
                elif mode == "rmse_exact_disp":
                    y = line["data"][f"rmse_exact_disp_{comp_key}"]
                else:
                    y = line["data"][f"bias_{comp_key}"]

                ax.plot(
                    x,
                    y,
                    label=line["label"],
                    color=line.get("color", "black"),
                    linestyle=line.get("linestyle", "--"),
                    marker=line.get("marker"),
                    linewidth=line.get("linewidth", 1.5),
                    markersize=5,
                    markevery=1,
                )

        ax.set_xlabel("Prescribed rigid shift [px]")
        if mode == "std_case":
            ax.set_ylabel(
                f"Disp. Std. Dev. (Case) ({comp_key}) [px]"
            )
            title_suffix = f"\n(max std dev: {max_err:.5f} px)"
        elif mode == "std_diff_anal_ref":
            ax.set_ylabel(
                f"Disp. Diff. Std. Dev. vs Ref ({comp_key}) [px]"
            )
            title_suffix = f"\n(max std dev: {max_err:.5f} px)"
        elif mode == "rmse_anal_ref":
            ax.set_ylabel(
                f"Disp. Field RMSE vs Ref ({comp_key}) [px]"
            )
            title_suffix = f"\n(max RMSE vs ref: {max_err:.5f} px)"
        elif mode == "rmse_exact_disp":
            ax.set_ylabel(
                f"Disp. Field RMSE vs Exact ({comp_key}) [px]"
            )
            title_suffix = f"\n(max RMSE vs exact: {max_err:.5f} px)"
        else:
            ax.set_ylabel(
                f"Disp. Bias ({comp_key}) [px]"
            )
            title_suffix = f"\n(max error vs nominal: {max_err:.5f} px)"
        ax.set_title(comp_title + title_suffix, fontsize=10)
        ax.grid(alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    psf_str = " (with PSF)" if psf else ""
    pattern_desc = clean_pattern_name(pattern)
    fig.suptitle(
        f"{title_prefix}\n"
        f"{pattern_desc}, {bit_depth}-bit{psf_str}",
        fontsize=12,
        fontweight="bold",
    )

    dir_path.mkdir(parents=True, exist_ok=True)
    save_path = dir_path / filename
    fig.savefig(save_path, dpi=DIAGNOSTIC_FIGURE_DPI)
    fig.clear()
    gc.collect()
    release()


def plot_all_modes_data(
    dir_path: Path,
    filename_base: str,
    title_prefix_base: str,
    case: str,
    pattern: str,
    bit_depth: int,
    psf: bool,
    lines: list[dict],
    ref_label: str,
) -> None:
    # Mode 1: bias as is
    plot_bias_lines(
        dir_path,
        f"{filename_base}.png",
        title_prefix_base,
        case, pattern, bit_depth, psf,
        lines,
        "bias",
        ref_label,
    )
    # Mode 2: std dev of case displacement
    title_prefix_std_case = (
        title_prefix_base
        .replace("Bias", "Std Dev of Case")
        .replace("bias", "std dev of case")
    )
    plot_bias_lines(
        dir_path,
        f"{filename_base}_std_case.png",
        title_prefix_std_case,
        case, pattern, bit_depth, psf,
        lines,
        "std_case",
        ref_label,
    )
    # Mode 3: std dev of diff vs analytic ref
    title_prefix_std_diff = (
        title_prefix_base
        .replace("Bias", f"Std Dev of Diff vs {ref_label}")
        .replace("bias", f"std dev of diff vs {ref_label.lower()}")
    )
    plot_bias_lines(
        dir_path,
        f"{filename_base}_std_diff_anal_ref.png",
        title_prefix_std_diff,
        case, pattern, bit_depth, psf,
        lines,
        "std_diff_anal_ref",
        ref_label,
    )
    # Mode 4: RMSE vs analytic reference DIC
    title_prefix_rmse_ref = (
        title_prefix_base
        .replace("Bias", f"RMSE vs {ref_label}")
        .replace("bias", f"rmse vs {ref_label.lower()}")
    )
    plot_bias_lines(
        dir_path,
        f"{filename_base}_rmse_anal_ref.png",
        title_prefix_rmse_ref,
        case, pattern, bit_depth, psf,
        lines,
        "rmse_anal_ref",
        ref_label,
    )
    # Mode 5: RMSE vs exact displacement
    title_prefix_rmse_exact = (
        title_prefix_base
        .replace("Bias", "RMSE vs Exact Disp")
        .replace("bias", "rmse vs exact disp")
    )
    plot_bias_lines(
        dir_path,
        f"{filename_base}_rmse_exact_disp.png",
        title_prefix_rmse_exact,
        case, pattern, bit_depth, psf,
        lines,
        "rmse_exact_disp",
        ref_label,
    )


def plot_job(task: tuple) -> None:
    (
        out_dir,
        filename_base,
        title_prefix,
        case,
        pattern,
        bit_depth,
        psf,
        lines,
        ref_label,
    ) = task
    plot_all_modes_data(
        out_dir,
        filename_base,
        title_prefix,
        case,
        pattern,
        bit_depth,
        psf,
        lines,
        ref_label,
    )


def load_reference_fields(
    records: list[Record],
    groups: dict[tuple[str, str, int, bool], list[Record]],
) -> tuple[
    dict[tuple[str, str, int, bool, int], tuple[np.ndarray, np.ndarray]],
    dict[tuple[str, str, int, bool], str],
]:
    reference_fields = {}
    reference_labels = {}

    from modules.exp3_dic_data import load_result

    for group_key, records_group in groups.items():
        case, pattern, bit_depth, psf = group_key

        # Try to find matching analytic record first
        ref_rec = None
        for r in records:
            r_interp = interpolator_of(r.config)
            if (
                r.analytic
                and pattern_of(r.config) == pattern
                and r.bit_depth == bit_depth
                and is_psf(r) == psf
            ):
                ref_rec = r
                break

        if ref_rec is not None:
            reference_labels[group_key] = "Analytic Ref"
        else:
            # Fallback to highest quality non-analytic record in the group
            group_candidates = [r for r in records_group if not r.analytic]
            if group_candidates:
                group_candidates.sort(
                    key=lambda r: (r.ssaa, r.osamp), reverse=True
                )
                ref_rec = group_candidates[0]
                if "grid2d_render_ssaa" in ref_rec.root:
                    label = f"SSAA={ref_rec.ssaa}"
                else:
                    label = f"SSAA={ref_rec.ssaa}, OS={ref_rec.osamp}"
                reference_labels[group_key] = f"Highest Quality Ref ({label})"

        if ref_rec is not None:
            for frame in range(1, 11):
                path = ref_rec.directory / f"displacement_frame{frame:02d}.npz"
                if path.is_file():
                    try:
                        with np.load(path) as data:
                            ux = np.asarray(data["ux"], dtype=np.float64)
                            uy = np.asarray(data["uy"], dtype=np.float64)

                        # Mask boundary
                        mask = np.ones(ux.shape, dtype=bool)
                        mask[
                            EDGE_EXCLUSION_PX:-EDGE_EXCLUSION_PX,
                            EDGE_EXCLUSION_PX:-EDGE_EXCLUSION_PX
                        ] = False
                        u_masked = np.where(mask, np.nan, ux)
                        v_masked = np.where(mask, np.nan, uy)
                        key = (case, pattern, bit_depth, psf, frame)
                        reference_fields[key] = (u_masked, v_masked)
                    except Exception:
                        pass
    return reference_fields, reference_labels


def main() -> None:
    out_dir_base = Path("out/exp3_analysis_gridmethod/grid_rigid_interp_bias")
    if not analysis_should_run(
        out_dir_base,
        "Grid Method Rigid Interpolation Bias analysis",
        force_overwrite=FORCE_INTERP_BIAS_OVERWRITE,
    ):
        return

    records = discover()
    rigid_records = [r for r in records if is_rigid_case(r.case)]
    if not rigid_records:
        print("No rigid gridmethod records discovered.")
        return

    # Group records by (case, pattern, bit_depth, psf) for plotting
    groups = defaultdict(list)
    for r in rigid_records:
        psf_val = is_psf(r)
        key = (r.case, pattern_of(r.config), r.bit_depth, psf_val)
        groups[key].append(r)

    # Load reference fields (analytic or highest quality fallback)
    print("Loading reference fields...")
    reference_fields_db, reference_labels_db = load_reference_fields(
        records, groups
    )

    # Process all record data in parallel (Case IO Multiprocessing)
    print("Processing rigid records data in parallel...")
    jobs = []
    for r in rigid_records:
        case_fields = {}
        psf_val = is_psf(r)
        key = (r.case, pattern_of(r.config), r.bit_depth, psf_val)
        for frame in range(1, 11):
            ref_key = (r.case, key[1], r.bit_depth, psf_val, frame)
            if ref_key in reference_fields_db:
                case_fields[frame] = reference_fields_db[ref_key]
        jobs.append((r, case_fields))

    results = run_analysis_jobs(
        "Loading and processing rigid records",
        jobs,
        process_record_job,
        mp_context=get_context("spawn"),
    )

    csv_rows = []
    data_cache = {}

    for r, res in results:
        if res is not None:
            csv_rows.extend(res)
            data_cache[r] = {
                "expected_ux": np.array(
                    [row["ExpectedShiftX"] for row in res]
                ),
                "expected_uy": np.array(
                    [row["ExpectedShiftY"] for row in res]
                ),
                "bias_ux": np.array(
                    [
                        row["MeanUxCase"] - row["ExpectedShiftX"]
                        for row in res
                    ]
                ),
                "bias_uy": np.array(
                    [
                        row["MeanUyCase"] - row["ExpectedShiftY"]
                        for row in res
                    ]
                ),
                "std_case_ux": np.array(
                    [row["StdUxCase"] for row in res]
                ),
                "std_case_uy": np.array(
                    [row["StdUyCase"] for row in res]
                ),
                "std_diff_anal_ux": np.array(
                    [row["StdUxDiffAnal"] for row in res]
                ),
                "std_diff_anal_uy": np.array(
                    [row["StdUyDiffAnal"] for row in res]
                ),
                "rmse_anal_ref_ux": np.array(
                    [row["RmseAnalRefUx"] for row in res]
                ),
                "rmse_anal_ref_uy": np.array(
                    [row["RmseAnalRefUy"] for row in res]
                ),
                "rmse_exact_disp_ux": np.array(
                    [row["RmseExactDispUx"] for row in res]
                ),
                "rmse_exact_disp_uy": np.array(
                    [row["RmseExactDispUy"] for row in res]
                ),
            }

    out_dir_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir_base / "grid_interpolation_bias_results.csv"
    print(f"Writing CSV of results to {csv_path}...")
    if csv_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    plotting_tasks = []

    print("Preparing plotting tasks...")
    for key, records_group in sorted(groups.items()):
        case, pattern, bit_depth, psf = key
        out_dir = out_dir_base / f"b{bit_depth:02d}"
        ref_label = reference_labels_db.get(key, "Analytic Ref")

        analytic_rec = find_record(
            records,
            "exp3_grid2d_render_analytic",
            analytic=True,
        )

        def make_line(rec_val, label, color=None, linestyle="-", marker=None, lw=1.5):
            if rec_val is None:
                return None
            line_data = data_cache.get(rec_val)
            if line_data is None:
                return None
            return {
                "record": rec_val,
                "label": label,
                "color": color,
                "linestyle": linestyle,
                "marker": marker,
                "linewidth": lw,
                "data": line_data,
            }

        # ----------------------------------------------------
        # Figure 1: Renderer & Interpolant Overview Comparison
        # ----------------------------------------------------
        lines_f1 = []
        l_analytic = make_line(analytic_rec, "Analytic Baseline", "black", "--", None, 1.5)
        if l_analytic:
            lines_f1.append(l_analytic)

        ssaa_conv = find_record(records_group, "exp3_grid2d_render_ssaa", highest_quality=True)
        l_ssaa_conv = make_line(ssaa_conv, f"Bespoke SSAA (ss={ssaa_conv.ssaa if ssaa_conv else 0})", "tab:green", "-", "o", 1.5)
        if l_ssaa_conv:
            lines_f1.append(l_ssaa_conv)

        ssaa_alias = find_record(records_group, "exp3_grid2d_render_ssaa", ssaa=1)
        l_ssaa_alias = make_line(ssaa_alias, "Bespoke SSAA (ss=1)", "tab:green", "--", "x", 1.0)
        if l_ssaa_alias:
            lines_f1.append(l_ssaa_alias)

        for interp, name, col, mc, ma in [
            ("cubiccm", "Catmull-Rom", "tab:blue", "s", "+"),
            ("cubic_bspline", "B-spline", "tab:orange", "^", "2"),
        ]:
            tex_conv = find_record(records_group, "exp3_riley_render_texf", interpolator=interp, highest_quality=True)
            l_tex_conv = make_line(tex_conv, f"Riley {name} (os={tex_conv.osamp if tex_conv else 0}, ss={tex_conv.ssaa if tex_conv else 0})", col, "-", mc, 1.5)
            if l_tex_conv:
                lines_f1.append(l_tex_conv)

            tex_alias = find_record(records_group, "exp3_riley_render_texf", interpolator=interp, ssaa=1, osamp=1)
            l_tex_alias = make_line(tex_alias, f"Riley {name} (os=1, ss=1)", col, "--", ma, 1.0)
            if l_tex_alias:
                lines_f1.append(l_tex_alias)

        f1_name = (
            f"dicbias_compare_{case}_"
            f"{pattern}_b{bit_depth:02d}"
            f"{'_psf' if psf else ''}"
        )
        plotting_tasks.append((
            out_dir, f1_name, "Grid Method Rigid Interpolation Bias",
            case, pattern, bit_depth, psf, lines_f1, ref_label
        ))

        # ----------------------------------------------------
        # Figure 2: OS=1, Sweeping SSAA
        # ----------------------------------------------------
        lines_f2 = []
        l_analytic = make_line(analytic_rec, "Analytic Reference", "black", "--", None, 1.5)
        if l_analytic:
            lines_f2.append(l_analytic)

        # Get unique levels present in this records group
        group_ssaa_levels = sorted({
            r.ssaa for r in records_group
            if not r.analytic and r.ssaa > 0
        })
        group_osamp_levels = sorted({
            r.osamp for r in records_group
            if not r.analytic and r.osamp > 0
        })
        group_diag_levels = sorted({
            r.ssaa for r in records_group
            if not r.analytic and r.ssaa == r.osamp and r.ssaa > 0
        })

        # ----------------------------------------------------
        # Figure 2: OS=1, Sweeping SSAA
        # ----------------------------------------------------
        lines_f2 = []
        l_analytic = make_line(analytic_rec, "Analytic Reference", "black", "--", None, 1.5)
        if l_analytic:
            lines_f2.append(l_analytic)

        for interp, name, lstyle in [
            ("cubic_bspline", "B-spline", "-"),
            ("cubiccm", "Catmull-Rom", "-."),
        ]:
            for ss in group_ssaa_levels:
                rec = find_record(records_group, "exp3_riley_render_texf", interpolator=interp, ssaa=ss, osamp=1)
                l_rec = make_line(rec, f"Riley {name} (ss={ss})", COLOR_BY_LEVEL.get(ss), lstyle, None, 1.5)
                if l_rec:
                    lines_f2.append(l_rec)

        f2_name = (
            f"dicbias_sweep_ssaa_os1_{case}_"
            f"{pattern}_b{bit_depth:02d}"
            f"{'_psf' if psf else ''}"
        )
        plotting_tasks.append((
            out_dir, f2_name, "Grid Method Interpolation Bias (OS=1, Sweeping SSAA)",
            case, pattern, bit_depth, psf, lines_f2, ref_label
        ))

        # ----------------------------------------------------
        # Figure 3 & 4: Diagonal Refinement (OS=SSAA)
        # ----------------------------------------------------
        for interp, name, f_key in [
            ("cubic_bspline", "B-spline", "bspline"),
            ("cubiccm", "Catmull-Rom", "cubiccm"),
        ]:
            lines_diag = []
            l_analytic = make_line(analytic_rec, "Analytic Reference", "black", "--", None, 1.5)
            if l_analytic:
                lines_diag.append(l_analytic)

            for level in group_diag_levels:
                rec = find_record(records_group, "exp3_riley_render_texf", interpolator=interp, ssaa=level, osamp=level)
                l_rec = make_line(rec, f"Riley {name} (ss,os={level})", COLOR_BY_LEVEL.get(level), "-", None, 1.5)
                if l_rec:
                    lines_diag.append(l_rec)

            diag_name = (
                f"dicbias_diagonal_{f_key}_{case}_"
                f"{pattern}_b{bit_depth:02d}"
                f"{'_psf' if psf else ''}"
            )
            plotting_tasks.append((
                out_dir, diag_name, f"Grid Method Interpolation Bias ({name} Diagonal Refinement)",
                case, pattern, bit_depth, psf, lines_diag, ref_label
            ))

        # ----------------------------------------------------
        # Figure 5 & 6: Fixed OS=max, Sweeping SSAA
        # ----------------------------------------------------
        for interp, name, f_key in [
            ("cubiccm", "Catmull-Rom", "cubiccm"),
            ("cubic_bspline", "B-spline", "bspline"),
        ]:
            riley_interp_recs = [
                r for r in records_group
                if "riley_render_texf" in r.root and interpolator_of(r.config) == interp
            ]
            max_os = (
                max(r.osamp for r in riley_interp_recs)
                if riley_interp_recs else 1
            )

            lines_ssaa = []
            l_analytic = make_line(analytic_rec, "Analytic Reference", "black", "--", None, 1.5)
            if l_analytic:
                lines_ssaa.append(l_analytic)

            for ss in group_ssaa_levels:
                rec = find_record(records_group, "exp3_riley_render_texf", interpolator=interp, ssaa=ss, osamp=max_os)
                l_rec = make_line(rec, f"Riley {name} (ss={ss}, os={max_os})", COLOR_BY_LEVEL.get(ss), "-", None, 1.5)
                if l_rec:
                    lines_ssaa.append(l_rec)

            ssaa_name = (
                f"dicbias_sweep_ssaa_osmax_{f_key}_{case}_"
                f"{pattern}_b{bit_depth:02d}"
                f"{'_psf' if psf else ''}"
            )
            plotting_tasks.append((
                out_dir, ssaa_name, f"Grid Method Interpolation Bias ({name}, OS={max_os}, Sweeping SSAA)",
                case, pattern, bit_depth, psf, lines_ssaa, ref_label
            ))

        # ----------------------------------------------------
        # Figure 7 & 8: Fixed SSAA=max, Sweeping OS
        # ----------------------------------------------------
        for interp, name, f_key in [
            ("cubiccm", "Catmull-Rom", "cubiccm"),
            ("cubic_bspline", "B-spline", "bspline"),
        ]:
            riley_interp_recs = [
                r for r in records_group
                if "riley_render_texf" in r.root and interpolator_of(r.config) == interp
            ]
            max_ss = (
                max(r.ssaa for r in riley_interp_recs)
                if riley_interp_recs else 1
            )

            lines_os = []
            l_analytic = make_line(analytic_rec, "Analytic Reference", "black", "--", None, 1.5)
            if l_analytic:
                lines_os.append(l_analytic)

            for osamp_val in group_osamp_levels:
                rec = find_record(records_group, "exp3_riley_render_texf", interpolator=interp, ssaa=max_ss, osamp=osamp_val)
                l_rec = make_line(rec, f"Riley {name} (ss={max_ss}, os={osamp_val})", COLOR_BY_LEVEL.get(osamp_val), "-", None, 1.5)
                if l_rec:
                    lines_os.append(l_rec)

            os_name = (
                f"dicbias_sweep_osamp_ssmax_{f_key}_{case}_"
                f"{pattern}_b{bit_depth:02d}"
                f"{'_psf' if psf else ''}"
            )
            plotting_tasks.append((
                out_dir, os_name, f"Grid Method Interpolation Bias ({name}, SSAA={max_ss}, Sweeping OS)",
                case, pattern, bit_depth, psf, lines_os, ref_label
            ))

    # Parallel Figure Rendering
    print("Generating figures in parallel...")
    run_analysis_jobs(
        "Plotting figures",
        plotting_tasks,
        plot_job,
        mp_context=get_context("spawn"),
    )

    print("Grid method interpolation bias plots complete.")
    mark_analysis_complete(out_dir_base)


if __name__ == "__main__":
    main()
