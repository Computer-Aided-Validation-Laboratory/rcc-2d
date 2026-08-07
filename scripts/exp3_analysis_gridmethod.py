#!/usr/bin/env python3
"""Compare Exp3 grid-method displacement fields across render references."""
from __future__ import annotations

import csv
import os
import re

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from exp0params_common import CORES
from exp3params import FORCE_GRIDMETHOD_OVERWRITE
from modules.exp3_analysis_common import OUT, OS_RE, SS_RE, numeric_y_axis, parameter, pattern_of, release, title_lines
from modules.analysis_selection import analysis_should_run, mark_analysis_complete
from modules.analysis_parallel import run_analysis_jobs

RESULTS=OUT/"exp3_analysis_gridmethod"
EDGE_EXCLUSION_PX = 12

@dataclass(frozen=True)
class Record:
    case:str;root:str;config:str;directory:Path;bit_depth:int;ssaa:int;osamp:int;analytic:bool

def discover()->list[Record]:
    values=[]
    for directory in (OUT/"exp3_gridmethod").glob("*/*/*/b*"):
        if not directory.is_dir() or not list(directory.glob("displacement_frame*.npz")):continue
        try:bit_depth=int(directory.name.removeprefix("b"))
        except ValueError:continue
        case,root,config=directory.parent.parent.parent.name,directory.parent.parent.name,directory.parent.name
        values.append(Record(case,root,config,directory,bit_depth,parameter(config,SS_RE),parameter(config,OS_RE),"_analytic_" in config))
    return values

_reference_cache = {}


def load(record: Record, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = record.directory / f"displacement_frame{frame:02d}.npz"
    if not path.exists():
        return None
    if record.analytic:
        cache_key = (path, frame)
        if cache_key in _reference_cache:
            return _reference_cache[cache_key]
        with np.load(path) as value:
            val = (
                np.asarray(value["ux"], dtype=np.float64),
                np.asarray(value["uy"], dtype=np.float64),
            )
        _reference_cache[cache_key] = val
        return val
    with np.load(path) as value:
        return (
            np.asarray(value["ux"], dtype=np.float64),
            np.asarray(value["uy"], dtype=np.float64),
        )


def reference(records: list[Record]) -> tuple[Record | None, str]:
    analytic = [r for r in records if r.analytic]
    if analytic:
        return analytic[0], "Analytic render grid-method reference"
    if not records:
        return None, "No reference"
    ref = max(records, key=lambda r: (r.ssaa, r.osamp))
    return ref, f"Highest SSAA/OS render grid-method reference: SSAA={ref.ssaa}, OS={ref.osamp or 1}"


def plot(path: Path, rec: Record, ref: Record, label: str, frame: int, ru: np.ndarray, rv: np.ndarray, cu: np.ndarray, cv: np.ndarray) -> tuple[float, float]:
    du, dv = cu - ru, cv - rv

    # Exclude boundary region
    mask = np.ones(ru.shape, dtype=bool)
    mask[EDGE_EXCLUSION_PX:-EDGE_EXCLUSION_PX, EDGE_EXCLUSION_PX:-EDGE_EXCLUSION_PX] = False
    ru = np.where(mask, np.nan, ru)
    rv = np.where(mask, np.nan, rv)
    cu = np.where(mask, np.nan, cu)
    cv = np.where(mask, np.nan, cv)
    du = np.where(mask, np.nan, du)
    dv = np.where(mask, np.nan, dv)

    maximum = float(max(np.nanmax(abs(du)), np.nanmax(abs(dv))))
    rms = float(np.sqrt(np.nanmean(du * du + dv * dv)))

    fig = Figure(figsize=(12, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    for row, (r, c, d, name) in enumerate(((ru, cu, du, "$u_x$"), (rv, cv, dv, "$u_y$"))):
        center_target = float(np.nanmean(r)) if np.any(np.isfinite(r)) else 0.0
        for axis, field, part, center in zip(
            axes[row], (r, c, d),
            ("reference", "current", "difference"),
            (center_target, center_target, 0.0)
        ):
            if np.any(np.isfinite(field)):
                max_dev = float(np.nanmax(np.abs(field - center)))
                radius = max(max_dev, 0.05)
            else:
                radius = 0.05
            im = axis.imshow(
                field, cmap="coolwarm", origin="upper",
                vmin=center - radius, vmax=center + radius
            )
            axis.set_title(f"{name}: {part}", fontsize=9)
            axis.set_xlabel("column [px]")
            axis.set_ylabel("row [px]")
            fig.colorbar(im, ax=axis, label="px")
    fig.suptitle(f"{title_lines(rec.case + ': ' + rec.config)} | {rec.bit_depth}-bit, frame {frame:02d}\nReference: {title_lines(ref.config)} ({label}); max difference={maximum:.4g} px", fontsize=10, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.clear()
    release()
    return maximum, rms


def analyse(payload: tuple[Record, list[Record]]) -> list[dict[str, object]]:
    rec, candidates = payload
    ref, label = reference(candidates)
    if ref is None or rec == ref:
        return []
    rows = []
    for frame in range(11):
        a, b = load(ref, frame), load(rec, frame)
        if a is None or b is None or a[0].shape != b[0].shape:
            continue
        maximum, rms = plot(RESULTS / rec.case / rec.root / rec.config / f"b{rec.bit_depth:02d}" / f"frame{frame:02d}_difference.png", rec, ref, label, frame, *a, *b)
        rows.append({
            "Case": rec.case,
            "Root": rec.root,
            "Config": rec.config,
            "BitDepth": rec.bit_depth,
            "Frame": frame,
            "SSAA": rec.ssaa or 1,
            "OS": rec.osamp or 1,
            "Reference": label,
            "max_difference_px": maximum,
            "rms_difference_px": rms,
        })
        del a, b
        release()
    return rows


def convergence(rows: list[dict[str, object]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["Case"], row["Root"], row["BitDepth"])].append(row)
    target_frames = [1, 3, 5, 7, 10]
    for (case, root, bit_depth), values in groups.items():
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

                label = f"OS={osamp}" if len(by_os) > 1 else "SSAA series"

                ax_rms.plot(ssaa_vals, rms_errs, "o-", label=label)
                ax_max.plot(ssaa_vals, max_errs, "o-", label=label)

                plotted_rms.extend(rms_errs)
                plotted_max.extend(max_errs)

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

        fig.suptitle(
            f"{title_lines(case + ': ' + root + ' grid-method convergence')} | {bit_depth}-bit\n"
            f"Reference: {title_lines(str(values[0]['Reference']))}",
            fontsize=11, fontweight="bold"
        )

        dir_path = RESULTS / case / f"{root}_disp_err_conv"
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"convergence_b{int(bit_depth):02d}.png"
        fig.savefig(path, dpi=150)
        fig.clear()
        release()

def main()->None:
    if not analysis_should_run(
        RESULTS,
        "Experiment 3 Grid Method analysis",
        force_overwrite=FORCE_GRIDMETHOD_OVERWRITE,
    ):
        return
    records=discover();groups=defaultdict(list)
    for record in records:groups[(record.case,record.bit_depth)].append(record)
    rows=[]
    limit=int(os.environ.get("EXP3_ANALYSIS_LIMIT", "0"))
    if limit: records=records[:limit]
    jobs=[(record,groups[(record.case,record.bit_depth)]) for record in records]
    for result in run_analysis_jobs(
        "Experiment 3 Grid Method analysis", jobs, analyse,
        mp_context=get_context("spawn")
    ):
        rows.extend(result)
    if rows:
        RESULTS.mkdir(parents=True,exist_ok=True)
        with (RESULTS/"summary.csv").open("w",newline="") as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        convergence(rows)
    mark_analysis_complete(RESULTS)
    print(f"Wrote {len(rows)} grid-method displacement comparisons.")

if __name__=="__main__":main()
