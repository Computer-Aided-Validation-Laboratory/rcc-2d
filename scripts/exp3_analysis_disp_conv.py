#!/usr/bin/env python3
"""Compare and plot standard and self-convergence convergence dashboards."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS"
):
    os.environ.setdefault(name, "1")

from multiprocessing import get_context
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Ensure local imports work correctly
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modules.exp3_analysis_common import (
    OUT, numeric_y_axis, release, title_lines
)
from modules.analysis_parallel import run_analysis_jobs

# Import records and discovery from DIC and Grid Method analysis scripts
from exp3_analysis_dic import (
    Record as DicRecord,
    discover as discover_dic,
    load as load_dic,
    reference as reference_dic,
    reference_candidates as reference_candidates_dic,
    series_label as series_label_dic,
)
from exp3_analysis_gridmethod import (
    Record as GridRecord,
    discover as discover_grid,
    load as load_grid,
    reference as reference_grid,
)

RESULTS_DIC = OUT / "exp3_analysis_dic"
RESULTS_GRID = OUT / "exp3_analysis_gridmethod"

EDGE_EXCLUSION_PX_DIC = 15
EDGE_EXCLUSION_PX_GRID = 12


def reference_candidates_dic_self(
    record: DicRecord, records: list[DicRecord]
) -> list[DicRecord]:
    rec_psf = "_psf" in record.root or "_psf" in record.config
    candidates = []
    for item in records:
        if (
            item.case == record.case
            and item.bit_depth == record.bit_depth
            and item.pattern == record.pattern
        ):
            if series_label_dic(item) != series_label_dic(record):
                continue
            if item.analytic:
                continue
            item_psf = "_psf" in item.root or "_psf" in item.config
            if item_psf == rec_psf:
                candidates.append(item)
    return candidates


def reference_candidates_grid_self(
    record: GridRecord, records: list[GridRecord]
) -> list[GridRecord]:
    candidates = []
    for item in records:
        if (
            item.case == record.case
            and item.bit_depth == record.bit_depth
            and item.root == record.root
            and not item.analytic
        ):
            candidates.append(item)
    return candidates


def analyse_dic_item(
    payload: tuple[DicRecord, list[DicRecord], list[DicRecord]]
) -> list[dict[str, object]]:
    rec, std_candidates, self_candidates = payload
    std_ref, std_label = reference_dic(std_candidates)
    self_ref, self_label = reference_dic(self_candidates)

    rows = []
    target_frames = [1, 3, 5, 7, 10]

    for frame in target_frames:
        curr_data = load_dic(rec, frame)
        if curr_data is None:
            continue
        cx, cy, cu, cv = curr_data

        # 1. Standard Convergence
        if std_ref is not None and rec != std_ref:
            std_data = load_dic(std_ref, frame)
            if std_data is not None:
                sx, sy, su, sv = std_data
                if su.shape == cu.shape:
                    du, dv = cu - su, cv - sv
                    x_min, x_max = cx.min(), cx.max()
                    y_min, y_max = cy.min(), cy.max()
                    mask = (
                        (cx < x_min + EDGE_EXCLUSION_PX_DIC)
                        | (cx > x_max - EDGE_EXCLUSION_PX_DIC)
                        | (cy < y_min + EDGE_EXCLUSION_PX_DIC)
                        | (cy > y_max - EDGE_EXCLUSION_PX_DIC)
                    )
                    du = np.where(mask, np.nan, du)
                    dv = np.where(mask, np.nan, dv)

                    maximum = float(max(np.nanmax(abs(du)), np.nanmax(abs(dv))))
                    rms = float(np.sqrt(np.nanmean(du * du + dv * dv)))

                    rows.append({
                        "Case": rec.case,
                        "Root": rec.root,
                        "Series": series_label_dic(rec),
                        "Config": rec.config,
                        "BitDepth": rec.bit_depth,
                        "Pattern": rec.pattern,
                        "Frame": frame,
                        "SSAA": rec.ssaa or 1,
                        "OS": rec.osamp or 1,
                        "Reference": std_label,
                        "max_difference_px": maximum,
                        "rms_difference_px": rms,
                        "Type": "std",
                    })

        # 2. Self Convergence
        if self_ref is not None:
            if rec == self_ref:
                rows.append({
                    "Case": rec.case,
                    "Root": rec.root,
                    "Series": series_label_dic(rec),
                    "Config": rec.config,
                    "BitDepth": rec.bit_depth,
                    "Pattern": rec.pattern,
                    "Frame": frame,
                    "SSAA": rec.ssaa or 1,
                    "OS": rec.osamp or 1,
                    "Reference": self_label,
                    "max_difference_px": 0.0,
                    "rms_difference_px": 0.0,
                    "Type": "selfconv",
                })
            else:
                self_data = load_dic(self_ref, frame)
                if self_data is not None:
                    sx, sy, su, sv = self_data
                    if su.shape == cu.shape:
                        du, dv = cu - su, cv - sv
                        x_min, x_max = cx.min(), cx.max()
                        y_min, y_max = cy.min(), cy.max()
                        mask = (
                            (cx < x_min + EDGE_EXCLUSION_PX_DIC)
                            | (cx > x_max - EDGE_EXCLUSION_PX_DIC)
                            | (cy < y_min + EDGE_EXCLUSION_PX_DIC)
                            | (cy > y_max - EDGE_EXCLUSION_PX_DIC)
                        )
                        du = np.where(mask, np.nan, du)
                        dv = np.where(mask, np.nan, dv)

                        maximum = float(max(np.nanmax(abs(du)), np.nanmax(abs(dv))))
                        rms = float(np.sqrt(np.nanmean(du * du + dv * dv)))

                        rows.append({
                            "Case": rec.case,
                            "Root": rec.root,
                            "Series": series_label_dic(rec),
                            "Config": rec.config,
                            "BitDepth": rec.bit_depth,
                            "Pattern": rec.pattern,
                            "Frame": frame,
                            "SSAA": rec.ssaa or 1,
                            "OS": rec.osamp or 1,
                            "Reference": self_label,
                            "max_difference_px": maximum,
                            "rms_difference_px": rms,
                            "Type": "selfconv",
                        })

    release()
    return rows


def analyse_grid_item(
    payload: tuple[GridRecord, list[GridRecord], list[GridRecord]]
) -> list[dict[str, object]]:
    rec, std_candidates, self_candidates = payload
    std_ref, std_label = reference_grid(std_candidates)
    self_ref, self_label = reference_grid(self_candidates)

    rows = []
    target_frames = [1, 3, 5, 7, 10]

    for frame in target_frames:
        curr_data = load_grid(rec, frame)
        if curr_data is None:
            continue
        cu, cv = curr_data

        # 1. Standard Convergence
        if std_ref is not None and rec != std_ref:
            std_data = load_grid(std_ref, frame)
            if std_data is not None:
                su, sv = std_data
                if su.shape == cu.shape:
                    du, dv = cu - su, cv - sv
                    mask = np.ones(su.shape, dtype=bool)
                    mask[
                        EDGE_EXCLUSION_PX_GRID:-EDGE_EXCLUSION_PX_GRID,
                        EDGE_EXCLUSION_PX_GRID:-EDGE_EXCLUSION_PX_GRID
                    ] = False
                    du = np.where(mask, np.nan, du)
                    dv = np.where(mask, np.nan, dv)

                    maximum = float(max(np.nanmax(abs(du)), np.nanmax(abs(dv))))
                    rms = float(np.sqrt(np.nanmean(du * du + dv * dv)))

                    rows.append({
                        "Case": rec.case,
                        "Root": rec.root,
                        "Config": rec.config,
                        "BitDepth": rec.bit_depth,
                        "Frame": frame,
                        "SSAA": rec.ssaa or 1,
                        "OS": rec.osamp or 1,
                        "Reference": std_label,
                        "max_difference_px": maximum,
                        "rms_difference_px": rms,
                        "Type": "std",
                    })

        # 2. Self Convergence
        if self_ref is not None:
            if rec == self_ref:
                rows.append({
                    "Case": rec.case,
                    "Root": rec.root,
                    "Config": rec.config,
                    "BitDepth": rec.bit_depth,
                    "Frame": frame,
                    "SSAA": rec.ssaa or 1,
                    "OS": rec.osamp or 1,
                    "Reference": self_label,
                    "max_difference_px": 0.0,
                    "rms_difference_px": 0.0,
                    "Type": "selfconv",
                })
            else:
                self_data = load_grid(self_ref, frame)
                if self_data is not None:
                    su, sv = self_data
                    if su.shape == cu.shape:
                        du, dv = cu - su, cv - sv
                        mask = np.ones(su.shape, dtype=bool)
                        mask[
                            EDGE_EXCLUSION_PX_GRID:-EDGE_EXCLUSION_PX_GRID,
                            EDGE_EXCLUSION_PX_GRID:-EDGE_EXCLUSION_PX_GRID
                        ] = False
                        du = np.where(mask, np.nan, du)
                        dv = np.where(mask, np.nan, dv)

                        maximum = float(max(np.nanmax(abs(du)), np.nanmax(abs(dv))))
                        rms = float(np.sqrt(np.nanmean(du * du + dv * dv)))

                        rows.append({
                            "Case": rec.case,
                            "Root": rec.root,
                            "Config": rec.config,
                            "BitDepth": rec.bit_depth,
                            "Frame": frame,
                            "SSAA": rec.ssaa or 1,
                            "OS": rec.osamp or 1,
                            "Reference": self_label,
                            "max_difference_px": maximum,
                            "rms_difference_px": rms,
                            "Type": "selfconv",
                        })

    release()
    return rows


def plot_dic_convergence(rows: list[dict[str, object]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["Case"], row["Pattern"], row["Series"], row["BitDepth"],
            row["Type"]
        )].append(row)
    target_frames = [1, 3, 5, 7, 10]
    for (case, pattern, series_name, bit_depth, type_name), values in groups.items():
        fig = Figure(figsize=(10, 15), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(5, 2)

        val_by_frame = defaultdict(list)
        for row in values:
            f = int(row["Frame"])
            if f in target_frames:
                val_by_frame[f].append(row)

        for row_idx, frame in enumerate(target_frames):
            frame_vals = val_by_frame[frame]
            ax_rms = axes[row_idx, 0]
            ax_max = axes[row_idx, 1]

            if not frame_vals:
                ax_rms.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax_max.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            by_os = defaultdict(list)
            for row in frame_vals:
                by_os[int(row["OS"])].append(row)

            plotted_rms = []
            plotted_max = []
            for osamp, series in sorted(by_os.items(), reverse=True):
                series.sort(key=lambda r: int(r["SSAA"]))
                ssaa_vals = [int(r["SSAA"]) for r in series]
                rms_errs = [float(r["rms_difference_px"]) for r in series]
                max_errs = [float(r["max_difference_px"]) for r in series]

                # Floor 0.0 values to 1e-7 for log-scale plotting
                rms_errs_plot = [val if val > 0.0 else 1e-7 for val in rms_errs]
                max_errs_plot = [val if val > 0.0 else 1e-7 for val in max_errs]

                label = f"OS={osamp}" if len(by_os) > 1 else "SSAA series"
                ax_rms.plot(ssaa_vals, rms_errs_plot, "o-", label=label)
                ax_max.plot(ssaa_vals, max_errs_plot, "o-", label=label)

                plotted_rms.extend(rms_errs_plot)
                plotted_max.extend(max_errs_plot)

            ssaa_ticks = sorted({int(r["SSAA"]) for r in frame_vals})
            if not ssaa_ticks:
                ssaa_ticks = [1, 2, 4, 8, 16]

            for ax, plotted, title in zip(
                (ax_rms, ax_max),
                (plotted_rms, plotted_max),
                (f"Frame {frame} - RMSE", f"Frame {frame} - Max. Abs. Error")
            ):
                ax.set_xscale("log", base=2)
                numeric_y_axis(ax, plotted)

                ylim = ax.get_ylim()
                if ax.get_yscale() == "log":
                    ymin = min(ylim[0], 0.005)
                    ymax = max(ylim[1], 0.02)
                    ax.set_ylim(ymin, ymax)
                else:
                    ymin = min(ylim[0], -0.001)
                    ymax = max(ylim[1], 0.015)
                    ax.set_ylim(ymin, ymax)

                ax.axhline(0.01, color="red", linestyle="--", alpha=0.5, label="0.01 px")
                ax.set_xticks(ssaa_ticks)
                ax.set_xticklabels([str(t) for t in ssaa_ticks])
                ax.set_xlabel("Axis integration samples")
                ax.set_ylabel("Disp. Err. [px]")
                ax.set_title(title, fontsize=9)
                ax.grid(alpha=.3)
                ax.legend(fontsize=8)

        conv_desc = "self-convergence" if type_name == "selfconv" else "standard convergence"
        fig.suptitle(
            f"{title_lines(case + ': ' + pattern + ' DIC ' + conv_desc)} | {bit_depth}-bit\n"
            f"Render series: {title_lines(series_name)}\n"
            f"Reference: {title_lines(str(values[0]['Reference']))}",
            fontsize=11, fontweight="bold"
        )

        dir_path = RESULTS_DIC / case / f"{series_name}_disp_err_conv"
        dir_path.mkdir(parents=True, exist_ok=True)
        suffix = "_selfconv" if type_name == "selfconv" else ""
        path = dir_path / f"{pattern}_convergence{suffix}_b{int(bit_depth):02d}.png"
        fig.savefig(path, dpi=150)
        fig.clear()
        release()


def plot_grid_convergence(rows: list[dict[str, object]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["Case"], row["Root"], row["BitDepth"], row["Type"])].append(row)
    target_frames = [1, 3, 5, 7, 10]
    for (case, root, bit_depth, type_name), values in groups.items():
        fig = Figure(figsize=(10, 15), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(5, 2)

        val_by_frame = defaultdict(list)
        for row in values:
            f = int(row["Frame"])
            if f in target_frames:
                val_by_frame[f].append(row)

        for row_idx, frame in enumerate(target_frames):
            frame_vals = val_by_frame[frame]
            ax_rms = axes[row_idx, 0]
            ax_max = axes[row_idx, 1]

            if not frame_vals:
                ax_rms.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax_max.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            by_os = defaultdict(list)
            for row in frame_vals:
                by_os[int(row["OS"])].append(row)

            plotted_rms = []
            plotted_max = []
            for osamp, series in sorted(by_os.items(), reverse=True):
                series.sort(key=lambda r: int(r["SSAA"]))
                ssaa_vals = [int(r["SSAA"]) for r in series]
                rms_errs = [float(r["rms_difference_px"]) for r in series]
                max_errs = [float(r["max_difference_px"]) for r in series]

                # Floor 0.0 values to 1e-7 for log-scale plotting
                rms_errs_plot = [val if val > 0.0 else 1e-7 for val in rms_errs]
                max_errs_plot = [val if val > 0.0 else 1e-7 for val in max_errs]

                label = f"OS={osamp}" if len(by_os) > 1 else "SSAA series"
                ax_rms.plot(ssaa_vals, rms_errs_plot, "o-", label=label)
                ax_max.plot(ssaa_vals, max_errs_plot, "o-", label=label)

                plotted_rms.extend(rms_errs_plot)
                plotted_max.extend(max_errs_plot)

            ssaa_ticks = sorted({int(r["SSAA"]) for r in frame_vals})
            if not ssaa_ticks:
                ssaa_ticks = [1, 2, 4, 8, 16]

            for ax, plotted, title in zip(
                (ax_rms, ax_max),
                (plotted_rms, plotted_max),
                (f"Frame {frame} - RMSE", f"Frame {frame} - Max. Abs. Error")
            ):
                ax.set_xscale("log", base=2)
                numeric_y_axis(ax, plotted)

                ylim = ax.get_ylim()
                if ax.get_yscale() == "log":
                    ymin = min(ylim[0], 0.005)
                    ymax = max(ylim[1], 0.02)
                    ax.set_ylim(ymin, ymax)
                else:
                    ymin = min(ylim[0], -0.001)
                    ymax = max(ylim[1], 0.015)
                    ax.set_ylim(ymin, ymax)

                ax.axhline(0.01, color="red", linestyle="--", alpha=0.5, label="0.01 px")
                ax.set_xticks(ssaa_ticks)
                ax.set_xticklabels([str(t) for t in ssaa_ticks])
                ax.set_xlabel("Axis integration samples")
                ax.set_ylabel("Disp. Err. [px]")
                ax.set_title(title, fontsize=9)
                ax.grid(alpha=.3)
                ax.legend(fontsize=8)

        conv_desc = "self-convergence" if type_name == "selfconv" else "standard convergence"
        fig.suptitle(
            f"{title_lines(case + ': ' + root + ' grid-method ' + conv_desc)} | {bit_depth}-bit\n"
            f"Reference: {title_lines(str(values[0]['Reference']))}",
            fontsize=11, fontweight="bold"
        )

        dir_path = RESULTS_GRID / case / f"{root}_disp_err_conv"
        dir_path.mkdir(parents=True, exist_ok=True)
        suffix = "_selfconv" if type_name == "selfconv" else ""
        path = dir_path / f"convergence{suffix}_b{int(bit_depth):02d}.png"
        fig.savefig(path, dpi=150)
        fig.clear()
        release()


def run_dic_analysis() -> None:
    records = discover_dic()
    if not records:
        print("No DIC records found; skipping.")
        return
    print(f"Running DIC convergence analysis on {len(records)} configurations...")
    jobs = []
    for rec in records:
        std_cand = reference_candidates_dic(rec, records)
        self_cand = reference_candidates_dic_self(rec, records)
        jobs.append((rec, std_cand, self_cand))

    rows = []
    for result in run_analysis_jobs(
        "Experiment 3 DIC Convergence", jobs, analyse_dic_item,
        mp_context=get_context("spawn")
    ):
        rows.extend(result)
    if rows:
        plot_dic_convergence(rows)
    print("DIC convergence figures complete.")


def run_grid_analysis() -> None:
    records = discover_grid()
    if not records:
        print("No Grid Method records found; skipping.")
        return
    print(f"Running Grid Method convergence analysis on {len(records)} configurations...")
    groups = defaultdict(list)
    for record in records:
        groups[(record.case, record.bit_depth)].append(record)

    jobs = []
    for rec in records:
        std_cand = groups[(rec.case, rec.bit_depth)]
        self_cand = reference_candidates_grid_self(rec, records)
        jobs.append((rec, std_cand, self_cand))

    rows = []
    for result in run_analysis_jobs(
        "Experiment 3 Grid Method Convergence", jobs, analyse_grid_item,
        mp_context=get_context("spawn")
    ):
        rows.extend(result)
    if rows:
        plot_grid_convergence(rows)
    print("Grid Method convergence figures complete.")


def main() -> None:
    run_dic_analysis()
    run_grid_analysis()
    print("All standard and self-convergence figures updated successfully.")


if __name__ == "__main__":
    main()
