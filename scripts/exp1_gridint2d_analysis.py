# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Convergence analysis for Experiment 1 bespoke grid renderers."""

from __future__ import annotations

import csv
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
from exp1common import output_case_name
from analysis_memory import release_batch
from expplots import plot_bespoke_four_panel, samples_for_method
from script_timing import ScriptTimer, timed_call

RESULTS_DIR = exp1_output_dir("exp1_gridint2d_analysis")
RENDER_SUFFIX = ""


def _paths(directory: Path, method: str, param: int, bit_depth: int, frame: int) -> tuple[Path, Path]:
    prefix = f"targ_px{TARG_PX_X}_int_{method}_param_{param}{RENDER_SUFFIX}_b{bit_depth}_frame{frame:02d}"
    return directory / f"{prefix}.npy", directory / f"{prefix}.tiff"


def _load_pair(directory: Path, method: str, param: int, bit_depth: int, frame: int):
    npy_path, tiff_path = _paths(directory, method, param, bit_depth, frame)
    if not npy_path.exists() or not tiff_path.exists():
        return None
    with Image.open(tiff_path) as image:
        digitised = np.asarray(image, dtype=np.float64)
    return np.load(npy_path) / float(2**bit_depth - 1), digitised


def _empty_float(methods: list[str]) -> dict[str, dict[str, list[float]]]:
    return {method: {"samples": [], "e_f64": [], "e_inf": []} for method in methods}


def _empty_digitised(methods: list[str]) -> dict[int, dict[str, dict[str, list[float]]]]:
    return {bit_depth: {method: {"samples": [], "max_eb": [], "delta_b": []} for method in methods} for bit_depth in BIT_DEPTHS}


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
            print(f"Warning: {case_dir.name}, frame {frame:02d}: no analytic or Gauss reference.")
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
                float_diff = image_float - ref_float
                digitised_diff = image_digitised - ref_digitised
                e_f64 = float(np.sqrt(np.mean(float_diff**2)))
                e_inf = float(np.max(np.abs(float_diff)))
                max_eb = float(np.max(np.abs(digitised_diff)))
                delta_b = float(np.mean(image_digitised != ref_digitised))
                digitised_data[bit_depth][method]["samples"].append(samples)
                digitised_data[bit_depth][method]["max_eb"].append(max_eb)
                digitised_data[bit_depth][method]["delta_b"].append(delta_b)
                if bit_depth == preferred_bit_depth:
                    float_data[method]["samples"].append(samples)
                    float_data[method]["e_f64"].append(e_f64)
                    float_data[method]["e_inf"].append(e_inf)
                rows.append({"Case": case_dir.name, "Frame": frame, "BitDepth": bit_depth, "Method": method, "Param": param, "Samples": samples, "Reference": f"{ref_method}:{ref_param}", "e_f64": e_f64, "e_inf": e_inf, "e_b": float(np.sqrt(np.mean(digitised_diff**2))), "delta_b": delta_b, "max_eb": max_eb})
                del image, image_float, image_digitised, float_diff, digitised_diff
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
            print(f"Warning: {case_dir.name}, frame {frame:02d}: no rectangular reference.")
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
            if param == ref_param:
                continue
            samples = samples_for_method("rect", param)
            for bit_depth, (ref_float, ref_digitised) in references.items():
                image = _load_pair(case_dir, "rect", param, bit_depth, frame)
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
                frame_rows.append({"Case": case_dir.name, "Frame": frame, "BitDepth": bit_depth, "Method": "rect", "Param": param, "Samples": samples, "Reference": f"rect:{ref_param}", "e_f64": e_f64, "e_inf": e_inf, "e_b": e_b, "delta_b": delta_b, "max_eb": max_eb})
                del image, image_float, image_digitised, float_diff, digitised_diff
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


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["Case", "Frame", "BitDepth", "Method", "Param", "Samples", "Reference", "e_f64", "e_inf", "e_b", "delta_b", "max_eb"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    timer = ScriptTimer(__file__)
    if CLEAR_DIR:
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rectconv_dir = Path(f"{RESULTS_DIR}_rectconv")
    if CLEAR_DIR:
        shutil.rmtree(rectconv_dir, ignore_errors=True)
    rectconv_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    rectconv_rows: list[dict[str, object]] = []
    for case_name in DEFORMATION_CASES:
        output_name = output_case_name(case_name, TARG_PX_X)
        case_dir = OUTPUT_DIR / output_name
        if not case_dir.exists():
            print(f"Warning: {case_dir} does not exist. Skipping.")
            continue
        all_rows.extend(timed_call(timer, output_name, analyse_case, case_dir))
        rectconv_rows.extend(timed_call(timer, f"{output_name}_rectconv", analyse_rectangular_self_convergence, case_dir, rectconv_dir))
        release_batch()
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    _write_rows(rectconv_dir / "summary.csv", rectconv_rows)
    print("Experiment 1 grid analysis completed.")


if __name__ == "__main__":
    main()
