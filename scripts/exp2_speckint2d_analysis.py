# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Analyse Experiment 2 bespoke-renderer convergence against best references."""

from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from exp2params import (
    ACTIVE_FRAMES,
    BIT_DEPTHS,
    DEFORMATION_CASES,
    OUTPUT_DIR,
    TARG_PX_X,
    exp2_output_dir,
)
from modules.exp1common import output_case_name
from modules.analysis_memory import release_batch
from modules.expplots import plot_bespoke_four_panel, samples_for_method
from modules.script_timing import ScriptTimer, timed_call
from modules.render_selection import custom_enabled
from modules.analysis_selection import analysis_should_run, mark_analysis_complete
from modules.analysis_parallel import run_analysis_jobs
from modules.render_outputs import quantise_camera


RESULTS_DIR = exp2_output_dir("exp2_speckint2d_analysis")
RENDER_SUFFIX = ""
WRITE_RECTCONV = True
JOB_RE = re.compile(
    r"^(?P<pattern>.+)_(?P<method>analytic|rect|gauss|mc)_(?P<param>\d+)(?P<suffix>_psf)?$"
)


def _image_pair(directory: Path, method: str, param: int, bit_depth: int, frame: int):
    prefix = f"targ_px{TARG_PX_X}_int_{method}_param_{param}{RENDER_SUFFIX}_frame{frame:02d}"
    npy_path = directory / f"{prefix}.npy"
    if npy_path.exists():
        floating = np.asarray(np.load(npy_path), dtype=np.float64)
        return floating, quantise_camera(floating, bit_depth).astype(np.float64)

    # Support output produced before the f64 texture convention, whose NumPy
    # files held digitised code values separately for every bit depth.
    return None


def _discover_jobs() -> dict[str, dict[tuple[str, int], Path]]:
    """Group rendered jobs by ``<case>_<pattern-tag>``."""
    groups: dict[str, dict[tuple[str, int], Path]] = defaultdict(dict)
    if not OUTPUT_DIR.exists():
        return groups
    for directory in OUTPUT_DIR.iterdir():
        if not directory.is_dir():
            continue
        for case_name in DEFORMATION_CASES:
            case_name = output_case_name(case_name, TARG_PX_X)
            prefix = f"{case_name}_"
            if not directory.name.startswith(prefix):
                continue
            match = JOB_RE.match(directory.name[len(prefix):])
            if match is None:
                continue
            # ``re`` returns None for an unmatched optional suffix, while
            # the normal renderer uses the empty-string suffix.
            if (match.group("suffix") or "") != RENDER_SUFFIX:
                continue
            group_name = f"{case_name}_{match.group('pattern')}"
            groups[group_name][(match.group("method"), int(match.group("param")))] = directory
            break
    return groups


def _reference_job(
    group_name: str,
    jobs: dict[tuple[str, int], Path],
    frame: int,
):
    """Prefer analytic; disks fall back to rect, other patterns to Gauss."""
    analytic = jobs.get(("analytic", 0))
    if analytic is not None and any(
        _image_pair(analytic, "analytic", 0, bit_depth, frame) is not None
        for bit_depth in BIT_DEPTHS
    ):
        return ("analytic", 0), analytic, "Analytic Reference"
    preferred = "rect" if "_diskadd_" in group_name else "gauss"
    # Smooth fields prefer Gauss; sharp disks prefer rectangular SSAA.  Always
    # try the highest rectangular SSAA afterwards as a universal fallback.
    methods = [preferred] + ([] if preferred == "rect" else ["rect"])
    for fallback_method in methods:
        fallback_jobs = sorted(
            ((param, directory) for (method, param), directory in jobs.items() if method == fallback_method),
            reverse=True,
        )
        for param, directory in fallback_jobs:
            if any(_image_pair(directory, fallback_method, param, bit_depth, frame) is not None for bit_depth in BIT_DEPTHS):
                label = "Rectangular SSAA" if fallback_method == "rect" else "Gauss Quadrature"
                return ((fallback_method, param), directory, f"{label} Reference ({param}x{param})")
    return None


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = ["Group", "Frame", "BitDepth", "Method", "Param", "Samples", "Reference", "e_f64", "e_inf", "e_b", "delta_b", "max_eb"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyse_group(group_name: str, jobs: dict[tuple[str, int], Path]) -> list[dict[str, object]]:
    output_dir = RESULTS_DIR / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({method for method, _ in jobs if method != "analytic"})
    rows: list[dict[str, object]] = []
    for frame in ACTIVE_FRAMES:
        selected = _reference_job(group_name, jobs, frame)
        if selected is None:
            print(f"No analytic or Gaussian reference: {group_name}, frame {frame:02d}.")
            continue
        (ref_method, ref_param), ref_directory, ref_name = selected
        references = {bit_depth: _image_pair(ref_directory, ref_method, ref_param, bit_depth, frame) for bit_depth in BIT_DEPTHS}
        references = {bit_depth: value for bit_depth, value in references.items() if value is not None}
        if not references:
            continue
        float_data = {method: {"samples": [], "e_f64": [], "e_inf": []} for method in methods}
        digitised_data = {bit_depth: {method: {"samples": [], "max_eb": [], "delta_b": []} for method in methods} for bit_depth in BIT_DEPTHS}
        preferred_bit_depth = 16 if 16 in references else max(references)
        for (method, param), directory in sorted(jobs.items()):
            if (method, param) == (ref_method, ref_param) or method == "analytic":
                continue
            samples = samples_for_method(method, param)
            for bit_depth, (ref_float, ref_digitised) in references.items():
                image = _image_pair(directory, method, param, bit_depth, frame)
                if image is None:
                    continue
                image_float, image_digitised = image
                float_diff = image_float - ref_float
                digitised_diff = image_digitised - ref_digitised
                e_f64 = float(np.sqrt(np.mean(float_diff**2)))
                e_inf = float(np.max(np.abs(float_diff)))
                e_b = float(np.sqrt(np.mean(digitised_diff**2)))
                delta_b = float(np.mean(image_digitised != ref_digitised))
                max_eb = float(np.max(np.abs(digitised_diff)))
                digitised_data[bit_depth][method]["samples"].append(samples)
                digitised_data[bit_depth][method]["max_eb"].append(max_eb)
                digitised_data[bit_depth][method]["delta_b"].append(delta_b)
                if bit_depth == preferred_bit_depth:
                    float_data[method]["samples"].append(samples)
                    float_data[method]["e_f64"].append(e_f64)
                    float_data[method]["e_inf"].append(e_inf)
                rows.append({"Group": group_name, "Frame": frame, "BitDepth": bit_depth, "Method": method, "Param": param, "Samples": samples, "Reference": f"{ref_method}:{ref_param}", "e_f64": e_f64, "e_inf": e_inf, "e_b": e_b, "delta_b": delta_b, "max_eb": max_eb})
                del image, image_float, image_digitised, float_diff, digitised_diff
        path = plot_bespoke_four_panel(group_name, frame, ref_name, output_dir, float_data, digitised_data, sorted(references))
        print(f"Saved {path}")
        del references, float_data, digitised_data
        release_batch()
    _write_rows(output_dir / "summary.csv", rows)
    return rows


def analyse_rectangular_self_convergence(
    group_name: str,
    jobs: dict[tuple[str, int], Path],
) -> list[dict[str, object]]:
    """Compare each rectangular rule to the highest available rule itself."""
    output_dir = Path(f"{RESULTS_DIR}_rectconv") / group_name
    rect_params = sorted(param for method, param in jobs if method == "rect")
    rows: list[dict[str, object]] = []
    for frame in ACTIVE_FRAMES:
        ref_param = next(
            (
                param
                for param in reversed(rect_params)
                if any(
                    _image_pair(jobs[("rect", param)], "rect", param, bit_depth, frame)
                    is not None
                    for bit_depth in BIT_DEPTHS
                )
            ),
            None,
        )
        if ref_param is None:
            print(f"No rectangular reference: {group_name}, frame {frame:02d}.")
            continue
        ref_directory = jobs[("rect", ref_param)]
        references = {
            bit_depth: _image_pair(ref_directory, "rect", ref_param, bit_depth, frame)
            for bit_depth in BIT_DEPTHS
        }
        references = {
            bit_depth: value for bit_depth, value in references.items()
            if value is not None
        }
        float_data = {"rect": {"samples": [], "e_f64": [], "e_inf": []}}
        digitised_data = {
            bit_depth: {"rect": {"samples": [], "max_eb": [], "delta_b": []}}
            for bit_depth in BIT_DEPTHS
        }
        preferred_bit_depth = 16 if 16 in references else max(references)
        frame_rows = []
        for param in rect_params:
            directory = jobs[("rect", param)]
            samples = samples_for_method("rect", param)
            for bit_depth, (ref_float, ref_digitised) in references.items():
                if param == ref_param:
                    e_f64 = e_inf = e_b = delta_b = max_eb = 0.0
                    digitised_data[bit_depth]["rect"]["samples"].append(samples)
                    digitised_data[bit_depth]["rect"]["max_eb"].append(max_eb)
                    digitised_data[bit_depth]["rect"]["delta_b"].append(delta_b)
                    if bit_depth == preferred_bit_depth:
                        float_data["rect"]["samples"].append(samples)
                        float_data["rect"]["e_f64"].append(e_f64)
                        float_data["rect"]["e_inf"].append(e_inf)
                    frame_rows.append({"Group": group_name, "Frame": frame, "BitDepth": bit_depth, "Method": "rect", "Param": param, "Samples": samples, "Reference": f"rect:{ref_param}", "e_f64": e_f64, "e_inf": e_inf, "e_b": e_b, "delta_b": delta_b, "max_eb": max_eb})
                    continue
                image = _image_pair(directory, "rect", param, bit_depth, frame)
                if image is None:
                    continue
                image_float, image_digitised = image
                float_diff = image_float - ref_float
                digitised_diff = image_digitised - ref_digitised
                e_f64 = float(np.sqrt(np.mean(float_diff**2)))
                e_inf = float(np.max(np.abs(float_diff)))
                e_b = float(np.sqrt(np.mean(digitised_diff**2)))
                delta_b = float(np.mean(image_digitised != ref_digitised))
                max_eb = float(np.max(np.abs(digitised_diff)))
                digitised_data[bit_depth]["rect"]["samples"].append(samples)
                digitised_data[bit_depth]["rect"]["max_eb"].append(max_eb)
                digitised_data[bit_depth]["rect"]["delta_b"].append(delta_b)
                if bit_depth == preferred_bit_depth:
                    float_data["rect"]["samples"].append(samples)
                    float_data["rect"]["e_f64"].append(e_f64)
                    float_data["rect"]["e_inf"].append(e_inf)
                frame_rows.append({"Group": group_name, "Frame": frame, "BitDepth": bit_depth, "Method": "rect", "Param": param, "Samples": samples, "Reference": f"rect:{ref_param}", "e_f64": e_f64, "e_inf": e_inf, "e_b": e_b, "delta_b": delta_b, "max_eb": max_eb})
                del image, image_float, image_digitised, float_diff, digitised_diff
        if frame_rows:
            plot_bespoke_four_panel(
                group_name,
                frame,
                f"Rectangular SSAA Reference ({ref_param}x{ref_param})",
                output_dir,
                float_data,
                digitised_data,
                sorted(references),
            )
            rows.extend(frame_rows)
        del references, float_data, digitised_data, frame_rows
        release_batch()
    _write_rows(output_dir / "summary.csv", rows)
    return rows


def _analyse_group_job(item: tuple[str, dict[tuple[str, int], Path]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    group_name, jobs = item
    rows = analyse_group(group_name, jobs)
    rect_rows = (
        analyse_rectangular_self_convergence(group_name, jobs)
        if WRITE_RECTCONV else []
    )
    release_batch()
    return rows, rect_rows


def main() -> None:
    if not any(custom_enabled(case) for case in ("disk", "gauss")):
        print("Experiment 2 speckle analysis disabled by CUSTOM_RENDER_CASES; skipping.")
        return
    if not analysis_should_run(RESULTS_DIR, "Experiment 2 speckle analysis"):
        return
    timer = ScriptTimer(__file__)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rectconv_dir = Path(f"{RESULTS_DIR}_rectconv")
    if not WRITE_RECTCONV:
        shutil.rmtree(rectconv_dir, ignore_errors=True)
    else:
        rectconv_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    rectconv_rows: list[dict[str, object]] = []
    groups = sorted(_discover_jobs().items())
    for rows, rect_rows in run_analysis_jobs("Experiment 2 speckle analysis", groups, _analyse_group_job):
        all_rows.extend(rows)
        rectconv_rows.extend(rect_rows)
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    if WRITE_RECTCONV:
        _write_rows(rectconv_dir / "summary.csv", rectconv_rows)
    mark_analysis_complete(RESULTS_DIR)
    print("Experiment 2 grid analysis completed.")


if __name__ == "__main__":
    main()
