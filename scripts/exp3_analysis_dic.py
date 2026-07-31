#!/usr/bin/env python3
"""Compare Exp3 PyVale DIC displacement fields to render-reference DIC fields."""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from exp0params_common import CORES
from modules.exp3_analysis_common import OUT, OS_RE, SS_RE, interpolator_of, numeric_y_axis, parameter, pattern_of, release, title_lines
from modules.exp3_dic_data import load_result, result_path

RESULTS = OUT / "exp3_analysis_dic"


@dataclass(frozen=True)
class Record:
    case: str; root: str; config: str; directory: Path; pattern: str; ssaa: int; osamp: int; interpolator: str; analytic: bool


def discover() -> list[Record]:
    rows=[]
    for directory in (OUT / "exp3_dic").glob("*/*/*"):
        if not directory.is_dir() or not list(directory.glob("dic_frame*.npz")): continue
        case, root, config = directory.parent.parent.name, directory.parent.name, directory.name
        rows.append(Record(case, root, config, directory, pattern_of(config), parameter(config, SS_RE), parameter(config, OS_RE), interpolator_of(config), "_analytic_" in config))
    return rows


def load(record: Record, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    path = result_path(record.directory, frame)
    if not path.is_file(): return None
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


def series_label(record: Record) -> str:
    """A homogeneous plotting series: renderer/storage, sampler and PSF mode."""
    psf = "_psf" in record.root or "_psf" in record.config
    if "riley_render_tex" in record.root:
        storage = "texuint" if "texuint" in record.root else "texfloat"
        return f"riley_{storage}_{record.interpolator}{'_psf' if psf else ''}"
    return f"{record.root.replace('_render_ssaa', '')}{'_psf' if psf and '_psf' not in record.root else ''}"


def field_plot(path: Path, rec: Record, frame: int, ref: Record, ref_name: str, arrays: tuple[np.ndarray,...]) -> tuple[float,float]:
    x,y,ru,rv,cu,cv=arrays; du, dv=cu-ru, cv-rv; maximum=float(max(np.max(abs(du)),np.max(abs(dv)))); rms=float(np.sqrt(np.mean(du*du+dv*dv)))
    fig=Figure(figsize=(12,7),constrained_layout=True); FigureCanvasAgg(fig); axes=fig.subplots(2,3)
    for row,(ref_field,current,diff,name) in enumerate(((ru,cu,du,"$u_x$"),(rv,cv,dv,"$u_y$"))):
        for axis,field,part in zip(axes[row],(ref_field,current,diff),("reference","current","difference")):
            im=axis.imshow(field,extent=(x.min(),x.max(),y.max(),y.min()),cmap="coolwarm",aspect="auto")
            axis.set_title(f"{name}: {part}",fontsize=9); axis.set_xlabel("column [px]");axis.set_ylabel("row [px]");fig.colorbar(im,ax=axis,label="px")
    fig.suptitle(f"{title_lines(rec.case + ': ' + rec.config)} | frame {frame:02d}\nReference: {title_lines(ref.config)} ({ref_name}); max difference={maximum:.4g} px",fontsize=10,fontweight="bold")
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);fig.clear();release();return maximum,rms


def analyse(payload: tuple[Record,list[Record]]) -> list[dict[str,object]]:
    rec,candidates=payload; ref,ref_name=reference(candidates)
    if ref is None or rec==ref:return []
    rows=[]
    for frame in range(1,11):
        a,b=load(ref,frame),load(rec,frame)
        if a is None or b is None: continue
        x,y,ru,rv=a; _,_,cu,cv=b
        if ru.shape!=cu.shape: continue
        maximum,rms=field_plot(RESULTS/rec.case/rec.root/rec.config/f"frame{frame:02d}_difference.png",rec,frame,ref,ref_name,(x,y,ru,rv,cu,cv))
        rows.append({"Case":rec.case,"Root":rec.root,"Series":series_label(rec),"Config":rec.config,"Pattern":rec.pattern,"Frame":frame,"SSAA":rec.ssaa or 1,"OS":rec.osamp or 1,"Reference":ref_name,"max_difference_px":maximum,"rms_difference_px":rms})
        del a,b,x,y,ru,rv,cu,cv;release()
    return rows


def convergence(rows:list[dict[str,object]])->None:
    groups=defaultdict(list)
    for row in rows: groups[(row["Case"],row["Pattern"],row["Series"],row["Frame"])].append(row)
    for (case,pattern,series_name,frame),values in groups.items():
        fig=Figure(figsize=(7,4.5),constrained_layout=True);FigureCanvasAgg(fig);axis=fig.subplots()
        by_os=defaultdict(list)
        for row in values:by_os[int(row["OS"])].append(row)
        plotted=[]
        for osamp,series in sorted(by_os.items()):
            series.sort(key=lambda r:int(r["SSAA"])); y=[r["max_difference_px"] for r in series];axis.plot([r["SSAA"] for r in series],y,"o-",label=f"OS={osamp}" if any(int(r["OS"]) > 1 for r in values) else "SSAA series");plotted.extend(y)
        axis.set_xscale("log",base=2);numeric_y_axis(axis,plotted);axis.set_xticks(sorted({int(r["SSAA"]) for r in values}));axis.set_xlabel("SSAA samples along one pixel axis");axis.set_ylabel("max displacement difference [px]");axis.grid(alpha=.3);axis.legend(fontsize=8);axis.set_title(f"{title_lines(case+': '+pattern+' DIC convergence')} | frame {frame:02d}\nRender series: {title_lines(series_name)}\nReference: {title_lines(str(values[0]['Reference']))}",fontsize=9)
        path=RESULTS/case/series_name/f"{pattern}_frame{frame:02d}_convergence.png";path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);fig.clear();release()


def main()->None:
    records=discover();by_group=defaultdict(list)
    for rec in records:by_group[(rec.case,rec.pattern,"_psf" in rec.root or "_psf" in rec.config)].append(rec)
    rows=[]
    limit=int(os.environ.get("EXP3_ANALYSIS_LIMIT", "0"))
    if limit: records=records[:limit]
    with ProcessPoolExecutor(max_workers=CORES) as pool:
        futures=[pool.submit(analyse,(rec,by_group[(rec.case,rec.pattern,"_psf" in rec.root or "_psf" in rec.config)])) for rec in records]
        for future in as_completed(futures):rows.extend(future.result())
    if rows:
        RESULTS.mkdir(parents=True,exist_ok=True)
        with (RESULTS/"summary.csv").open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        convergence(rows)
    print(f"Wrote {len(rows)} DIC displacement comparisons.")

if __name__=="__main__":main()
