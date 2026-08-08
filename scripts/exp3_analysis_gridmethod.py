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

from exp0params_common import CORES, DIAGNOSTIC_FIGURE_DPI
from exp3params import FORCE_GRIDMETHOD_OVERWRITE
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
from modules.analysis_selection import (
    analysis_should_run,
    mark_analysis_complete,
)
from modules.analysis_parallel import run_analysis_jobs

RESULTS = OUT / "exp3_analysis_gridmethod"
EDGE_EXCLUSION_PX = 12

from modules.render_outputs import quantise_camera


def load_render_image(render_dir: Path, frame: int) -> np.ndarray | None:
    p1 = render_dir / f"frame{frame:02d}.npy"
    if p1.is_file():
        return np.load(p1)
    p2 = render_dir / f"image_c00_f{frame:02d}.npy"
    if p2.is_file():
        return np.load(p2)
    return None


@dataclass(frozen=True)
class Record:
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


def discover() -> list[Record]:
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
        values.append(
            Record(
                case=case,
                root=root,
                config=config,
                directory=directory,
                bit_depth=bit_depth,
                pattern=pattern_of(config),
                ssaa=parameter(config, SS_RE),
                osamp=parameter(config, OS_RE),
                interpolator=interpolator_of(config),
                analytic=("_analytic" in config),
            )
        )
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

    is_chirp = "chirp" in rec.case
    if is_chirp:
        # Create freq err folder
        freq_dir = Path("out/exp3_analysis_gridmethod/grid_disp_freq_err")
        freq_dir.mkdir(parents=True, exist_ok=True)

        # Compute column-wise RMSE
        H, W = du.shape
        unique_x = np.arange(W)
        col_rmses = []
        for col_idx in range(W):
            err2 = du[:, col_idx]**2 + dv[:, col_idx]**2
            col_rmses.append(float(np.sqrt(np.nanmean(err2))))

        fig_freq = Figure(figsize=(6, 4.5), constrained_layout=True)
        FigureCanvasAgg(fig_freq)
        ax_freq = fig_freq.subplots()
        ax_freq.plot(unique_x, col_rmses, color="tab:red", label="Local RMSE")
        ax_freq.set_xlabel("Horizontal coordinate [px]")
        ax_freq.set_ylabel("Displacement RMSE [px]")
        ax_freq.set_title(
            f"Local RMSE along spatial frequency gradient\n"
            f"{rec.config} | Frame {frame:02d}",
            fontsize=10, fontweight="bold"
        )
        ax_freq.grid(alpha=0.3)
        ax_freq.legend(fontsize=8)

        series = rec.root.replace("_render_ssaa", "")
        if "riley_render_tex" in rec.root:
            series = f"riley_texf_{rec.interpolator}"

        freq_path = (
            freq_dir /
            f"freq_err_{rec.case}_{rec.pattern}_{series}_"
            f"b{rec.bit_depth:02d}_frame{frame:02d}.png"
        )
        fig_freq.savefig(freq_path, dpi=DIAGNOSTIC_FIGURE_DPI)
        fig_freq.clear()

        fig = Figure(figsize=(11, 5), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 2)
    else:
        fig = Figure(figsize=(12, 7), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3)

    for field_idx, (r, c, d, name) in enumerate(
        ((ru, cu, du, "$u_x$"), (rv, cv, dv, "$u_y$"))
    ):
        center_target = (
            float(np.nanmean(r)) if np.any(np.isfinite(r)) else 0.0
        )
        if is_chirp:
            field_axes = [
                axes[0, field_idx],
                axes[1, field_idx],
                axes[2, field_idx],
            ]
        else:
            field_axes = list(axes[field_idx])

        for axis, field, part, center in zip(
            field_axes, (r, c, d),
            ("reference", "current", "difference"),
            (center_target, center_target, 0.0)
        ):
            if np.any(np.isfinite(field)):
                max_dev = float(np.nanmax(np.abs(field - center)))
                if part == "difference":
                    radius = max(max_dev, 1e-6)
                else:
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
    fig.suptitle(
        f"{title_lines(rec.case + ': ' + rec.config)} | "
        f"{rec.bit_depth}-bit, frame {frame:02d}\n"
        f"Reference: {title_lines(ref.config)} ({label}); "
        f"max difference={maximum:.4g} px",
        fontsize=10, fontweight="bold"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DIAGNOSTIC_FIGURE_DPI)
    fig.clear()
    release()
    return maximum, rms


def get_series_label(record: Record) -> str:
    psf = "_psf" in record.root or "_psf" in record.config
    if "riley_render_tex" in record.root:
        return f"riley_texf_{record.interpolator}{'_psf' if psf else ''}"
    suffix = "_psf" if psf and "_psf" not in record.root else ""
    return f"{record.root.replace('_render_ssaa', '')}{suffix}"


def analyse(
    payload: tuple[Record, list[Record]]
) -> list[dict[str, object]]:
    rec, candidates = payload
    ref, label = reference(candidates)
    if ref is None or rec == ref:
        return []
    rows = []
    for frame in range(11):
        a, b = load(ref, frame), load(rec, frame)
        if a is None or b is None or a[0].shape != b[0].shape:
            continue
        diff_path = (
            RESULTS
            / rec.case
            / rec.root
            / rec.config
            / f"b{rec.bit_depth:02d}"
            / f"frame{frame:02d}_difference.png"
        )
        maximum, rms = plot(diff_path, rec, ref, label, frame, *a, *b)

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
            "Reference": label,
            "Rmse": rms,
            "DigitisedRMSE(bits)": digitised_rmse,
            "DigitisedMaxErr(bits)": digitised_max_err,
        })
        del a, b
        release()
    return rows


def convergence(rows: list[dict[str, object]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["Case"], row["Series"], row["BitDepth"])].append(row)
    target_frames = [1, 3, 5, 7, 10]
    for (case, series, bit_depth), values in groups.items():
        fig = Figure(figsize=(6, 12), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(5, 1)

        val_by_frame = defaultdict(list)
        for row in values:
            f = int(row["Frame"])
            if f in target_frames:
                val_by_frame[f].append(row)

        for row_idx, frame in enumerate(target_frames):
            frame_vals = val_by_frame[frame]
            ax = axes[row_idx]

            if not frame_vals:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            by_os = defaultdict(list)
            for row in frame_vals:
                by_os[int(row["OS"])].append(row)

            plotted = []
            for osamp, s_list in sorted(by_os.items(), reverse=True):
                s_list.sort(key=lambda r: int(r["SSAA"]))

                ssaa_vals = [int(r["SSAA"]) for r in s_list]
                rms_errs = [float(r["Rmse"]) for r in s_list]

                label = f"OS={osamp}" if len(by_os) > 1 else "SSAA series"
                ax.plot(ssaa_vals, rms_errs, "o-", label=label)
                plotted.extend(rms_errs)

            ssaa_ticks = sorted({int(r["SSAA"]) for r in frame_vals})
            if not ssaa_ticks:
                ssaa_ticks = [1, 2, 4, 8, 16]

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

            ax.axhline(
                0.01, color="red", linestyle="--", alpha=0.5,
                label="0.01 px"
            )
            ax.set_xticks(ssaa_ticks)
            ax.set_xticklabels([str(t) for t in ssaa_ticks])
            ax.set_xlabel("Axis integration samples")
            ax.set_ylabel("Disp. Err. [px]")
            ax.set_title(f"Frame {frame} - RMSE", fontsize=9)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)

        title_str = (
            f"{case}: {series} grid-method convergence\n"
            f"{bit_depth}-bit\n"
        )
        fig.suptitle(
            f"{title_lines(title_str)}"
            f"Reference: {title_lines(str(values[0]['Reference']))}",
            fontsize=11, fontweight="bold"
        )

        dir_path = RESULTS / case / f"{series}_disp_err_conv"
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"convergence_b{int(bit_depth):02d}.png"
        fig.savefig(path, dpi=DIAGNOSTIC_FIGURE_DPI)
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
