# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Convergence analysis for Experiment 1 bespoke grid renderers."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from exp1params import (
    ACTIVE_FRAMES,
    BIT_DEPTHS,
    CLEAR_DIR,
    DEFORMATION_CASES,
    INTEGRATION_METHODS,
    OUTPUT_DIR,
    TARG_PX_X,
    exp1_output_dir,
)
from modules.exp1common import output_case_name
from modules.analysis_memory import release_batch
from modules.expplots import plot_bespoke_four_panel, samples_for_method
from modules.script_timing import ScriptTimer, timed_call
from modules.render_selection import custom_enabled
from modules.analysis_selection import analysis_should_run, mark_analysis_complete
from modules.analysis_parallel import run_analysis_jobs
from modules.render_outputs import quantise_camera
from modules.exp_common_analysis import image_error_metrics


def _include_completed_integration_levels() -> None:
    """Extend parameter defaults with every completed rule found on disk."""
    global INTEGRATION_METHODS
    found = set(INTEGRATION_METHODS)
    pattern = re.compile(r"_int_(analytic|rect|gauss)_param_(\d+)(?:_psf)?_frame\d+\.npy$")
    for image in OUTPUT_DIR.glob("*/*.npy"):
        match = pattern.search(image.name)
        if match:
            found.add((match.group(1), int(match.group(2))))
    INTEGRATION_METHODS = tuple(sorted(found, key=lambda item: (item[0], item[1])))

RESULTS_DIR = exp1_output_dir("exp1_gridint2d_analysis")
RENDER_SUFFIX = ""
WRITE_RECTCONV = True


def _paths(directory: Path, method: str, param: int, bit_depth: int, frame: int) -> tuple[Path, Path]:
    canonical = directory / f"targ_px{TARG_PX_X}_int_{method}_param_{param}{RENDER_SUFFIX}_frame{frame:02d}.npy"
    canonical_tiff = canonical.with_name(f"{canonical.stem}_b{bit_depth}.tiff")
    if canonical.exists() and canonical_tiff.exists():
        return canonical, canonical_tiff
    return canonical, canonical_tiff


def _load_pair(directory: Path, method: str, param: int, bit_depth: int, frame: int):
    npy_path, tiff_path = _paths(directory, method, param, bit_depth, frame)
    if not npy_path.exists():
        return None
    floating = np.asarray(np.load(npy_path), dtype=np.float64)
    if floating.size and np.nanmax(np.abs(floating)) > 1.0 + 1e-12:
        floating /= float(2**bit_depth - 1)
    return floating, quantise_camera(floating, bit_depth).astype(np.float64)


def _empty_float(methods: list[str]) -> dict[str, dict[str, list[float]]]:
    return {method: {"samples": [], "e_f64": [], "mean_f64": [], "e_inf": []} for method in methods}


def _empty_digitised(methods: list[str]) -> dict[int, dict[str, dict[str, list[float]]]]:
    keys = ("samples", "e_b", "mean_eb", "max_eb", "delta_b", "severe_b", "p95_eb", "p99_eb")
    return {bit_depth: {method: {key: [] for key in keys} for method in methods} for bit_depth in BIT_DEPTHS}


def _reference_for_frame(case_dir: Path, frame: int):
    """Prefer an analytic image, otherwise use the highest completed Gauss rule."""
    if RENDER_SUFFIX:
        candidates = [
            ("rect", param, f"Rectangular SSAA Reference ({param}x{param})")
            for param in sorted((p for method, p in INTEGRATION_METHODS if method == "rect"), reverse=True)
        ]
    else:
        candidates = [("analytic", 0, "Analytic Reference")]
    candidates.extend(
        ("gauss", param, f"Gauss Quadrature Reference ({param}x{param})")
        for param in sorted(
            (param for method, param in INTEGRATION_METHODS if method == "gauss"),
            reverse=True,
        )
    )
    # A highest rectangular SSAA image is the universal last resort: it is
    # required for sharp fields and remains available when Gauss is absent.
    if not RENDER_SUFFIX:
        candidates.extend(
            ("rect", param, f"Rectangular SSAA Reference ({param}x{param})")
            for param in sorted((p for method, p in INTEGRATION_METHODS if method == "rect"), reverse=True)
        )
    for method, param, label in candidates:
        references = {
            bit_depth: _load_pair(case_dir, method, param, bit_depth, frame)
            for bit_depth in BIT_DEPTHS
        }
        references = {
            bit_depth: value for bit_depth, value in references.items()
            if value is not None
        }
        if references:
            return (method, param), references, label
    return None


def analyse_case(case_dir: Path) -> list[dict[str, object]]:
    methods = sorted({method for method, _ in INTEGRATION_METHODS if method != "analytic"})
    rows: list[dict[str, object]] = []
    for frame in ACTIVE_FRAMES:
        selected = _reference_for_frame(case_dir, frame)
        if selected is None:
            print(f"No analytic or Gauss reference: {case_dir.name}, frame {frame:02d}.")
            continue
        (ref_method, ref_param), references, ref_label = selected
        float_data = _empty_float(methods)
        digitised_data = _empty_digitised(methods)
        preferred_bit_depth = 16 if 16 in references else max(references)
        for method, param in INTEGRATION_METHODS:
            if method == "analytic" or (method, param) == (ref_method, ref_param):
                continue
            samples = samples_for_method(method, param)
            for bit_depth, (ref_float, ref_digitised) in references.items():
                image = _load_pair(case_dir, method, param, bit_depth, frame)
                if image is None:
                    continue
                image_float, image_digitised = image
                metrics = image_error_metrics(image_float, ref_float, bit_depth, quantise_camera)
                digitised_data[bit_depth][method]["samples"].append(samples)
                for key in ("e_b", "mean_eb", "max_eb", "delta_b", "severe_b", "p95_eb", "p99_eb"):
                    digitised_data[bit_depth][method][key].append(metrics[key])
                if bit_depth == preferred_bit_depth:
                    float_data[method]["samples"].append(samples)
                    for key in ("e_f64", "mean_f64", "e_inf"): float_data[method][key].append(metrics[key])
                rows.append({"Case": case_dir.name, "Frame": frame, "BitDepth": bit_depth, "Method": method, "Param": param, "Samples": samples, "Reference": f"{ref_method}:{ref_param}", **metrics})
                del image, image_float, image_digitised
        path = plot_bespoke_four_panel(case_dir.name, frame, ref_label, RESULTS_DIR, float_data, digitised_data, sorted(references))
        print(f"Saved {path}")
        del references, float_data, digitised_data
        release_batch()
    return rows


def analyse_rectangular_self_convergence(
    case_dir: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Compare rectangular rules to the highest available rectangular rule."""
    rect_params = sorted(
        param for method, param in INTEGRATION_METHODS if method == "rect"
    )
    rows: list[dict[str, object]] = []
    for frame in ACTIVE_FRAMES:
        ref_param = next(
            (
                param
                for param in reversed(rect_params)
                if any(
                    _load_pair(case_dir, "rect", param, bit_depth, frame)
                    is not None
                    for bit_depth in BIT_DEPTHS
                )
            ),
            None,
        )
        if ref_param is None:
            print(f"No rectangular reference: {case_dir.name}, frame {frame:02d}.")
            continue
        references = {
            bit_depth: _load_pair(case_dir, "rect", ref_param, bit_depth, frame)
            for bit_depth in BIT_DEPTHS
        }
        references = {
            bit_depth: value for bit_depth, value in references.items()
            if value is not None
        }
        float_data = _empty_float(["rect"])
        digitised_data = _empty_digitised(["rect"])
        preferred_bit_depth = 16 if 16 in references else max(references)
        frame_rows = []
        for param in rect_params:
            samples = samples_for_method("rect", param)
            for bit_depth, (ref_float, ref_digitised) in references.items():
                if param == ref_param:
                    metrics = {key: 0.0 for key in ("e_f64", "mean_f64", "e_inf", "e_b", "mean_eb", "delta_b", "severe_b", "p95_eb", "p99_eb", "max_eb")}
                    digitised_data[bit_depth]["rect"]["samples"].append(samples)
                    for key in ("e_b", "mean_eb", "max_eb", "delta_b", "severe_b", "p95_eb", "p99_eb"): digitised_data[bit_depth]["rect"][key].append(metrics[key])
                    if bit_depth == preferred_bit_depth:
                        float_data["rect"]["samples"].append(samples)
                        for key in ("e_f64", "mean_f64", "e_inf"): float_data["rect"][key].append(metrics[key])
                    frame_rows.append({"Case": case_dir.name, "Frame": frame, "BitDepth": bit_depth, "Method": "rect", "Param": param, "Samples": samples, "Reference": f"rect:{ref_param}", **metrics})
                    continue
                image = _load_pair(case_dir, "rect", param, bit_depth, frame)
                if image is None:
                    continue
                image_float, image_digitised = image
                metrics = image_error_metrics(image_float, ref_float, bit_depth, quantise_camera)
                digitised_data[bit_depth]["rect"]["samples"].append(samples)
                for key in ("e_b", "mean_eb", "max_eb", "delta_b", "severe_b", "p95_eb", "p99_eb"): digitised_data[bit_depth]["rect"][key].append(metrics[key])
                if bit_depth == preferred_bit_depth:
                    float_data["rect"]["samples"].append(samples)
                    for key in ("e_f64", "mean_f64", "e_inf"): float_data["rect"][key].append(metrics[key])
                frame_rows.append({"Case": case_dir.name, "Frame": frame, "BitDepth": bit_depth, "Method": "rect", "Param": param, "Samples": samples, "Reference": f"rect:{ref_param}", **metrics})
                del image, image_float, image_digitised
        if frame_rows:
            plot_bespoke_four_panel(
                case_dir.name,
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
    return rows


def _analyse_case_job(case_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Process one deformation case independently for the common harness."""
    rows = analyse_case(case_dir)
    rect_rows = (
        analyse_rectangular_self_convergence(case_dir, Path(f"{RESULTS_DIR}_rectconv"))
        if WRITE_RECTCONV else []
    )
    release_batch()
    return rows, rect_rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = ["Case", "Frame", "BitDepth", "Method", "Param", "Samples", "Reference", "e_f64", "mean_f64", "e_inf", "e_b", "mean_eb", "delta_b", "severe_b", "p95_eb", "p99_eb", "max_eb"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not custom_enabled("eggbox"):
        print("Experiment 1 eggbox analysis disabled by CUSTOM_RENDER_CASES; skipping.")
        return
    if not analysis_should_run(RESULTS_DIR, "Experiment 1 eggbox analysis"):
        return
    _include_completed_integration_levels()
    timer = ScriptTimer(__file__)
    if CLEAR_DIR:
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rectconv_dir = Path(f"{RESULTS_DIR}_rectconv")
    if not WRITE_RECTCONV:
        shutil.rmtree(rectconv_dir, ignore_errors=True)
    elif CLEAR_DIR:
        shutil.rmtree(rectconv_dir, ignore_errors=True)
    if WRITE_RECTCONV:
        rectconv_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    rectconv_rows: list[dict[str, object]] = []
    case_dirs: list[Path] = []
    for case_name in DEFORMATION_CASES:
        output_name = output_case_name(case_name, TARG_PX_X)
        case_dir = OUTPUT_DIR / output_name
        if not case_dir.exists():
            print(f"Skipping unavailable analysis directory: {case_dir}.")
            continue
        case_dirs.append(case_dir)
    for rows, rect_rows in run_analysis_jobs("Experiment 1 eggbox analysis", case_dirs, _analyse_case_job):
        all_rows.extend(rows)
        rectconv_rows.extend(rect_rows)
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    if WRITE_RECTCONV:
        _write_rows(rectconv_dir / "summary.csv", rectconv_rows)
    mark_analysis_complete(RESULTS_DIR)
    print("Experiment 1 grid analysis completed.")


if __name__ == "__main__":
    main()
