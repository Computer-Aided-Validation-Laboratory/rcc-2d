#!/usr/bin/env python3
"""Compare Exp3 PyVale DIC displacement fields to render-reference DIC fields."""
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
from modules.exp3_analysis_common import OUT, OS_RE, SS_RE, interpolator_of, numeric_y_axis, parameter, pattern_of, release, title_lines
from modules.exp3_dic_data import (
    load_result, result_path, parse_config, reconstruct_config_name,
)
from modules.analysis_selection import analysis_should_run, mark_analysis_complete
from modules.analysis_parallel import run_analysis_jobs

RESULTS = OUT / "exp3_analysis_dic"
EDGE_EXCLUSION_PX = 15


@dataclass(frozen=True)
class Record:
    case: str; root: str; config: str; directory: Path; bit_depth: int; pattern: str; ssaa: int; osamp: int; interpolator: str; analytic: bool


def discover() -> list[Record]:
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
            rows.append(Record(
                case=case,
                root=root,
                config=config,
                directory=directory,
                bit_depth=bit_depth,
                pattern=pattern,
                ssaa=parameter(config, SS_RE),
                osamp=parameter(config, OS_RE),
                interpolator=interpolator_of(config),
                analytic=("_analytic_" in config),
            ))
    return rows


_reference_cache = {}


def load(record: Record, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    _, suffix = parse_config(record.config)
    path = result_path(record.directory, suffix, record.bit_depth, frame)
    if not path.is_file():
        return None
    if record.analytic:
        cache_key = (path, frame)
        if cache_key in _reference_cache:
            return _reference_cache[cache_key]
        data = load_result(path)
        val = (data["ss_x"], data["ss_y"], data["u_px"][0], -data["v_px"][0])
        _reference_cache[cache_key] = val
        return val

    data = load_result(path)
    return data["ss_x"], data["ss_y"], data["u_px"][0], -data["v_px"][0]


def reference(records: list[Record]) -> tuple[Record | None,str]:
    analytic=[r for r in records if r.analytic]
    if analytic: return analytic[0], "Analytic render DIC reference"
    if not records: return None,"No reference"
    bespoke=[r for r in records if "grid2d" in r.root or "speck2d" in r.root]
    if bespoke:
        ref=max(bespoke,key=lambda r:r.ssaa)
        return ref,f"Highest bespoke SSAA render DIC reference: SSAA={ref.ssaa}"
    ref=max(records,key=lambda r:(r.ssaa,r.osamp)); return ref,f"Highest SSAA/OS render DIC reference: SSAA={ref.ssaa}, OS={ref.osamp or 1}"


def reference_candidates(record: Record, records: list[Record]) -> list[Record]:
    """Find candidates for the reference baseline."""
    rec_psf = "_psf" in record.root or "_psf" in record.config
    candidates = []
    for item in records:
        if (
            item.case == record.case
            and item.bit_depth == record.bit_depth
            and item.pattern == record.pattern
        ):
            item_psf = "_psf" in item.root or "_psf" in item.config
            if item_psf == rec_psf:
                candidates.append(item)
    return candidates


def series_label(record: Record) -> str:
    """A homogeneous plotting series: renderer/storage, sampler and PSF mode."""
    psf = "_psf" in record.root or "_psf" in record.config
    if "riley_render_tex" in record.root:
        storage = "texuint" if "texuint" in record.root else "texfloat"
        return f"riley_{storage}_{record.interpolator}{'_psf' if psf else ''}"
    return f"{record.root.replace('_render_ssaa', '')}{'_psf' if psf and '_psf' not in record.root else ''}"


def field_plot(
    path: Path, rec: Record, frame: int, ref: Record, ref_name: str,
    arrays: tuple[np.ndarray, ...]
) -> tuple[float, float]:
    x, y, ru, rv, cu, cv = arrays
    du, dv = cu - ru, cv - rv

    # Exclude boundary region
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    mask = (
        (x < x_min + EDGE_EXCLUSION_PX)
        | (x > x_max - EDGE_EXCLUSION_PX)
        | (y < y_min + EDGE_EXCLUSION_PX)
        | (y > y_max - EDGE_EXCLUSION_PX)
    )
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
    for row, (ref_field, current, diff, name) in enumerate(
        ((ru, cu, du, "$u_x$"), (rv, cv, dv, "$u_y$"))
    ):
        center_target = (
            float(np.nanmean(ref_field))
            if np.any(np.isfinite(ref_field))
            else 0.0
        )
        for axis, field, part, center in zip(
            axes[row], (ref_field, current, diff),
            ("reference", "current", "difference"),
            (center_target, center_target, 0.0)
        ):
            if np.any(np.isfinite(field)):
                max_dev = float(np.nanmax(np.abs(field - center)))
                r = max(max_dev, 0.05)
            else:
                r = 0.05
            im = axis.imshow(
                field, extent=(x.min(), x.max(), y.max(), y.min()),
                cmap="coolwarm", aspect="auto",
                vmin=center - r, vmax=center + r
            )
            axis.set_title(f"{name}: {part}", fontsize=9)
            axis.set_xlabel("column [px]")
            axis.set_ylabel("row [px]")
            fig.colorbar(im, ax=axis, label="px")
    fig.suptitle(
        f"{title_lines(rec.case + ': ' + rec.config)} | "
        f"{rec.bit_depth}-bit, frame {frame:02d}\n"
        f"Reference: {title_lines(ref.config)} ({ref_name}); "
        f"max difference={maximum:.4g} px",
        fontsize=10, fontweight="bold"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.clear()
    release()
    return maximum, rms


def analyse(payload: tuple[Record, list[Record]]) -> list[dict[str, object]]:
    rec, candidates = payload
    ref, ref_name = reference(candidates)
    if ref is None or rec == ref:
        return []
    rows = []
    base_config, rec_suffix = parse_config(rec.config)
    for frame in range(1, 11):
        a, b = load(ref, frame), load(rec, frame)
        if a is None or b is None:
            continue
        x, y, ru, rv = a
        _, _, cu, cv = b
        if ru.shape != cu.shape:
            continue
        image_path = (
            RESULTS
            / rec.case
            / rec.root
            / base_config
            / f"difference_{rec_suffix}_b{rec.bit_depth:02d}_frame{frame:02d}.png"
        )
        maximum, rms = field_plot(
            image_path, rec, frame, ref, ref_name, (x, y, ru, rv, cu, cv)
        )
        rows.append({
            "Case": rec.case,
            "Root": rec.root,
            "Series": series_label(rec),
            "Config": rec.config,
            "BitDepth": rec.bit_depth,
            "Pattern": rec.pattern,
            "Frame": frame,
            "SSAA": rec.ssaa or 1,
            "OS": rec.osamp or 1,
            "Reference": ref_name,
            "max_difference_px": maximum,
            "rms_difference_px": rms,
        })
        del a, b, x, y, ru, rv, cu, cv
        release()
    return rows


def convergence(rows: list[dict[str, object]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["Case"], row["Pattern"], row["Series"], row["BitDepth"]
        )].append(row)
    target_frames = [1, 3, 5, 7, 10]
    for (case, pattern, series_name, bit_depth), values in groups.items():
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
            f"{title_lines(case + ': ' + pattern + ' DIC convergence')} | {bit_depth}-bit\n"
            f"Render series: {title_lines(series_name)}\n"
            f"Reference: {title_lines(str(values[0]['Reference']))}",
            fontsize=11, fontweight="bold"
        )

        dir_path = RESULTS / case / f"{series_name}_disp_err_conv"
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{pattern}_convergence_b{int(bit_depth):02d}.png"
        fig.savefig(path, dpi=150)
        fig.clear()
        release()


def main()->None:
    if not analysis_should_run(RESULTS, "Experiment 3 DIC analysis"):
        return
    records=discover()
    rows=[]
    limit=int(os.environ.get("EXP3_ANALYSIS_LIMIT", "0"))
    if limit: records=records[:limit]
    jobs=[(rec,reference_candidates(rec, records)) for rec in records]
    for result in run_analysis_jobs(
        "Experiment 3 DIC analysis", jobs, analyse,
        mp_context=get_context("spawn")
    ):
        rows.extend(result)
    if rows:
        RESULTS.mkdir(parents=True,exist_ok=True)
        with (RESULTS/"summary.csv").open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        convergence(rows)
    mark_analysis_complete(RESULTS)
    print(f"Wrote {len(rows)} DIC displacement comparisons.")

if __name__=="__main__":main()
