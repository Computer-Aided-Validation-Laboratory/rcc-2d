#!/usr/bin/env python3
"""Compare Exp3 grid-method displacement fields across render references."""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from exp0params_common import CORES
from modules.exp3_analysis_common import OUT, OS_RE, SS_RE, numeric_y_axis, parameter, pattern_of, release, title_lines
from modules.analysis_selection import analysis_should_run, mark_analysis_complete
from modules.analysis_parallel import run_analysis_jobs

RESULTS=OUT/"exp3_analysis_gridmethod"

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

def load(record:Record,frame:int)->tuple[np.ndarray,np.ndarray]|None:
    path=record.directory/f"displacement_frame{frame:02d}.npz"
    if not path.exists():return None
    with np.load(path) as value:return np.asarray(value["ux"],dtype=np.float64),np.asarray(value["uy"],dtype=np.float64)

def reference(records:list[Record])->tuple[Record|None,str]:
    analytic=[r for r in records if r.analytic]
    if analytic:return analytic[0],"Analytic render grid-method reference"
    if not records:return None,"No reference"
    ref=max(records,key=lambda r:(r.ssaa,r.osamp));return ref,f"Highest SSAA/OS render grid-method reference: SSAA={ref.ssaa}, OS={ref.osamp or 1}"

def plot(path:Path,rec:Record,ref:Record,label:str,frame:int,ru:np.ndarray,rv:np.ndarray,cu:np.ndarray,cv:np.ndarray)->tuple[float,float]:
    du,dv=cu-ru,cv-rv;maximum=float(max(np.nanmax(abs(du)),np.nanmax(abs(dv))));rms=float(np.sqrt(np.nanmean(du*du+dv*dv)))
    fig=Figure(figsize=(12,7),constrained_layout=True);FigureCanvasAgg(fig);axes=fig.subplots(2,3)
    for row,(r,c,d,name) in enumerate(((ru,cu,du,"$u_x$"),(rv,cv,dv,"$u_y$"))):
        for axis,field,part in zip(axes[row],(r,c,d),("reference","current","difference")):
            im=axis.imshow(field,cmap="coolwarm",origin="upper");axis.set_title(f"{name}: {part}",fontsize=9);axis.set_xlabel("column [px]");axis.set_ylabel("row [px]");fig.colorbar(im,ax=axis,label="px")
    fig.suptitle(f"{title_lines(rec.case+': '+rec.config)} | {rec.bit_depth}-bit, frame {frame:02d}\nReference: {title_lines(ref.config)} ({label}); max difference={maximum:.4g} px",fontsize=10,fontweight="bold")
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);fig.clear();release();return maximum,rms

def analyse(payload:tuple[Record,list[Record]])->list[dict[str,object]]:
    rec,candidates=payload;ref,label=reference(candidates)
    if ref is None or rec==ref:return []
    rows=[]
    for frame in range(11):
        a,b=load(ref,frame),load(rec,frame)
        if a is None or b is None or a[0].shape!=b[0].shape:continue
        maximum,rms=plot(RESULTS/rec.case/rec.root/rec.config/f"b{rec.bit_depth:02d}"/f"frame{frame:02d}_difference.png",rec,ref,label,frame,*a,*b)
        rows.append({"Case":rec.case,"Root":rec.root,"Config":rec.config,"BitDepth":rec.bit_depth,"Frame":frame,"SSAA":rec.ssaa or 1,"OS":rec.osamp or 1,"Reference":label,"max_difference_px":maximum,"rms_difference_px":rms})
        del a,b;release()
    return rows

def convergence(rows:list[dict[str,object]])->None:
    groups=defaultdict(list)
    for row in rows:groups[(row["Case"],row["Root"],row["BitDepth"],row["Frame"])].append(row)
    for (case,root,bit_depth,frame),values in groups.items():
        fig=Figure(figsize=(7,4.5),constrained_layout=True);FigureCanvasAgg(fig);axis=fig.subplots();by_os=defaultdict(list)
        for row in values:by_os[int(row["OS"])].append(row)
        plotted=[]
        for osamp,series in sorted(by_os.items(), reverse=True):
            series.sort(key=lambda r:int(r["SSAA"])); y=[r["max_difference_px"] for r in series];axis.plot([r["SSAA"] for r in series],y,"o-",label=f"OS={osamp}" if any(int(r["OS"]) > 1 for r in values) else "SSAA series");plotted.extend(y)
        axis.set_xscale("log",base=2);numeric_y_axis(axis,plotted);axis.set_xlabel("SSAA samples along one pixel axis");axis.set_ylabel("max displacement difference [px]");axis.grid(alpha=.3);axis.legend(fontsize=8);axis.set_title(f"{title_lines(case+': '+root+' grid-method convergence')} | {bit_depth}-bit, frame {frame:02d}\nReference: {title_lines(str(values[0]['Reference']))}",fontsize=9)
        path=RESULTS/case/root/f"b{int(bit_depth):02d}"/f"frame{frame:02d}_convergence.png";path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);fig.clear();release()

def main()->None:
    if not analysis_should_run(RESULTS, "Experiment 3 Grid Method analysis"):
        return
    records=discover();groups=defaultdict(list)
    for record in records:groups[(record.case,record.bit_depth)].append(record)
    rows=[]
    limit=int(os.environ.get("EXP3_ANALYSIS_LIMIT", "0"))
    if limit: records=records[:limit]
    jobs=[(record,groups[(record.case,record.bit_depth)]) for record in records]
    for result in run_analysis_jobs("Experiment 3 Grid Method analysis", jobs, analyse): rows.extend(result)
    if rows:
        RESULTS.mkdir(parents=True,exist_ok=True)
        with (RESULTS/"summary.csv").open("w",newline="") as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        convergence(rows)
    mark_analysis_complete(RESULTS)
    print(f"Wrote {len(rows)} grid-method displacement comparisons.")

if __name__=="__main__":main()
