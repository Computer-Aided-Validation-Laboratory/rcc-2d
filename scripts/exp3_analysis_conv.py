#!/usr/bin/env python3
"""Convergence analysis for all completed Exp3 renderer outputs.

Analytic bespoke images are the primary reference wherever available.  A
highest completed SSAA/OS image is used otherwise.  The companion ``_rectconv``
tree always uses the highest SSAA/OS of the same renderer series.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from exp0params_common import CORES
from exp3_analysis_common import Render, best_reference, discover_renders, image_frames, load_image, numeric_y_axis, release, title_lines

RESULTS = Path("out/exp3_analysis_conv")
RECT_RESULTS = Path("out/exp3_analysis_conv_rectconv")


def family(item: Render) -> str:
    if "gridint2d" in item.root:
        return "gridint2d"
    if "speckint2d" in item.root:
        return "speckint2d"
    if "riley_render_func" in item.root:
        return "riley_func"
    storage = "texuint" if "texuint" in item.root else "texfloat"
    return f"riley_{storage}_{item.interpolator}"


def metrics(image: np.ndarray, reference: np.ndarray) -> tuple[float, float, float, float]:
    delta = image - reference
    digitised = np.rint(np.clip(image, 0, 1) * 255) - np.rint(np.clip(reference, 0, 1) * 255)
    values = (float(np.sqrt(np.mean(delta * delta))), float(np.max(np.abs(delta))), float(np.max(np.abs(digitised))), float(np.mean(digitised != 0)))
    del delta, digitised
    return values


def plot(rows: list[dict[str, object]], path: Path, heading: str, reference_name: str) -> None:
    figure = Figure(figsize=(11, 7), constrained_layout=True); FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    fields = (("e_rms", "Float RMS error"), ("e_max", "Float max error"), ("max_lsb", "Max 8-bit code error [LSB]"), ("fraction_changed", "Changed-pixel fraction"))
    by_os: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows: by_os[int(row["OS"])].append(row)
    texture_series = any("_os" in str(row["Config"]) for row in rows)
    for axis, (field, ylabel) in zip(axes, fields):
        plotted: list[float] = []
        for osamp, values in sorted(by_os.items()):
            values.sort(key=lambda row: int(row["SSAA"]))
            x = [int(row["SSAA"]) for row in values]; y = [float(row[field]) for row in values]
            axis.plot(x, y, "o-", label=f"OS={osamp}" if texture_series else "SSAA series")
            plotted.extend(y)
        axis.set_xscale("log", base=2); numeric_y_axis(axis, plotted)
        axis.set_xticks(sorted({int(row["SSAA"]) for row in rows})); axis.set_xticklabels(sorted({int(row["SSAA"]) for row in rows}))
        axis.set_xlabel("SSAA samples along one pixel axis"); axis.set_ylabel(ylabel); axis.grid(alpha=.3); axis.legend(fontsize=8)
    figure.suptitle(f"{title_lines(heading)}\nReference: {reference_name}", fontsize=11, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=160); figure.clear(); release()


def analyse_group(payload: tuple[str, str, str, str, list[Render], list[Render]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case, fam, pattern, output_key, items, reference_candidates = payload
    reference, ref_label = best_reference(reference_candidates)
    if reference is None: return [], []
    frames = image_frames(reference.directory)
    primary: list[dict[str, object]] = []; self_rows: list[dict[str, object]] = []
    self_reference = max(items, key=lambda item: (item.ssaa, item.oversamp)) if items else None
    self_label = (
        (f"Highest SSAA/OS: SSAA={self_reference.ssaa}, OS={self_reference.oversamp}" if self_reference.oversamp else f"Highest SSAA: SSAA={self_reference.ssaa}")
        if self_reference else "No self reference"
    )
    for frame, ref_path in frames.items():
        ref_image = load_image(ref_path)
        self_path = image_frames(self_reference.directory).get(frame) if self_reference else None
        self_image = load_image(self_path) if self_path else None
        for item in items:
            if item == reference:
                continue
            path = image_frames(item.directory).get(frame)
            if path is None: continue
            image = load_image(path)
            if image.shape == ref_image.shape:
                rms, maximum, lsb, changed = metrics(image, ref_image)
                primary.append({"Case": case, "Family": fam, "Pattern": pattern, "Config": item.config, "Frame": frame, "SSAA": item.ssaa or 1, "OS": item.oversamp or 1, "Reference": ref_label, "e_rms": rms, "e_max": maximum, "max_lsb": lsb, "fraction_changed": changed})
            if self_image is not None and image.shape == self_image.shape and item != self_reference:
                rms, maximum, lsb, changed = metrics(image, self_image)
                self_rows.append({"Case": case, "Family": fam, "Pattern": pattern, "Config": item.config, "Frame": frame, "SSAA": item.ssaa or 1, "OS": item.oversamp or 1, "Reference": self_label, "e_rms": rms, "e_max": maximum, "max_lsb": lsb, "fraction_changed": changed})
            del image
        if primary:
            plot([row for row in primary if int(row["Frame"]) == frame], RESULTS / case / fam / f"{pattern}_frame{frame:02d}_conv.png", f"{case}: {fam}, {pattern}", ref_label)
        if reference.analytic and self_rows:
            plot([row for row in self_rows if int(row["Frame"]) == frame], RECT_RESULTS / case / fam / f"{pattern}_frame{frame:02d}_rectconv.png", f"{case}: {fam}, {pattern} self convergence", self_label)
        del ref_image, self_image
        release()
    return primary, self_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def comparison_overlays(rows: list[dict[str, object]]) -> None:
    """Overlay like-for-like bespoke Exp3 and Exp1/2 SSAA convergence."""
    comparison_dir = RESULTS / "exp1_exp2_comparisons"; comparison_dir.mkdir(parents=True, exist_ok=True)
    for pattern, previous in (("eggbox", Path("out/exp1_gridint2d_analysis_uvs_im32/summary.csv")), ("diskaddsat", Path("out/exp2_speckint2d_analysis_im32/summary.csv")), ("gausscont", Path("out/exp2_speckint2d_analysis_im32/summary.csv"))):
        if not previous.exists(): continue
        old = list(csv.DictReader(previous.open()))
        for case_kind in ("rigid", "affine"):
            # These are intentionally bespoke-only comparisons.  Texture and
            # function-shader rows have independent OS/shader controls and
            # must never be concatenated into an SSAA curve.
            exp3_family = "gridint2d" if pattern == "eggbox" else "speckint2d"
            new = [
                row for row in rows
                if row["Pattern"] == pattern
                and row["Family"] == exp3_family
                and case_kind in str(row["Case"])
                and int(row["Frame"]) == 0
                and int(row["OS"]) == 1
            ]
            old_rows = [
                row for row in old
                if case_kind in row.get("Case", row.get("Group", ""))
                and (pattern == "eggbox" or pattern in row.get("Group", row.get("Pattern", "")))
                and int(row.get("Frame", 0)) == 0
                and row.get("Method", "rect") == "rect"
            ]
            if not new or not old_rows: continue
            figure = Figure(figsize=(7, 4.5), constrained_layout=True); FigureCanvasAgg(figure); axis = figure.subplots()
            plotted: list[float] = []
            for label, data, xkey, ykey in ((f"Exp3 {exp3_family}, SSAA", new, "SSAA", "e_max"), ("Exp1 rectangular SSAA" if pattern == "eggbox" else "Exp2 rectangular SSAA", old_rows, "Samples", "e_inf")):
                values = sorted(data, key=lambda row: float(row[xkey])); axis.plot([float(row[xkey]) for row in values], [float(row[ykey]) for row in values], "o-", label=label)
                plotted.extend(float(row[ykey]) for row in values)
            axis.set_xscale("log", base=2); numeric_y_axis(axis, plotted); axis.set_xlabel("SSAA samples along one pixel axis"); axis.set_ylabel("max floating-point error"); axis.grid(alpha=.3); axis.legend(fontsize=8)
            previous_name = "Exp1" if pattern == "eggbox" else "Exp2"
            axis.set_title(
                f"{case_kind.title()}, {pattern}: bespoke Exp3 vs {previous_name}\n"
                "Frame 00; analytic-reference max error",
                fontsize=10,
            )
            figure.savefig(comparison_dir / f"{pattern}_{case_kind}_bespoke_ssaa_overlay.png", dpi=160); figure.clear(); release()


def main() -> None:
    renders = discover_renders(); groups: dict[tuple[str, str, str], list[Render]] = defaultdict(list)
    by_case_pattern: dict[tuple[str, str, bool], list[Render]] = defaultdict(list)
    for item in renders:
        psf = "_psf" in item.root or "_psf" in item.config
        groups[(item.case, family(item), item.pattern, psf)].append(item); by_case_pattern[(item.case, item.pattern, psf)].append(item)
    tasks = [
        (case, f"{fam}_psf" if psf else fam, pattern, f"{case}/{fam}/{pattern}", items, by_case_pattern[(case, pattern, psf)])
        for (case, fam, pattern, psf), items in groups.items()
    ]
    limit = int(os.environ.get("EXP3_ANALYSIS_LIMIT", "0"))
    if limit: tasks = tasks[:limit]
    primary: list[dict[str, object]] = []; self_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=CORES) as pool:
        for future in as_completed([pool.submit(analyse_group, task) for task in tasks]):
            a, b = future.result(); primary.extend(a); self_rows.extend(b)
    write_csv(RESULTS, primary); write_csv(RECT_RESULTS, self_rows); comparison_overlays(primary)
    print(f"Wrote {len(primary)} primary and {len(self_rows)} self-convergence rows.")


if __name__ == "__main__": main()
