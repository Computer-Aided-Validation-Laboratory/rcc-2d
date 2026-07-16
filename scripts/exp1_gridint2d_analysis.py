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

from exp1params import ACTIVE_FRAMES, BIT_DEPTHS, CLEAR_DIR, DEFORMATION_CASES, INTEGRATION_METHODS, OUTPUT_DIR
from expplots import plot_bespoke_convergence, samples_for_method

RESULTS_DIR = Path("./out/exp1_gridint2d_analysis")


def _paths(directory: Path, method: str, param: int, bit_depth: int, frame: int) -> tuple[Path, Path]:
    prefix = f"targ_px256_int_{method}_param_{param}_b{bit_depth}_frame{frame:02d}"
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


def analyse_case(case_dir: Path) -> list[dict[str, object]]:
    methods = sorted({method for method, _ in INTEGRATION_METHODS if method != "analytic"})
    rows: list[dict[str, object]] = []
    for frame in ACTIVE_FRAMES:
        references = {bit_depth: _load_pair(case_dir, "analytic", 0, bit_depth, frame) for bit_depth in BIT_DEPTHS}
        references = {bit_depth: value for bit_depth, value in references.items() if value is not None}
        if not references:
            print(f"Warning: {case_dir.name}, frame {frame:02d}: analytic reference missing.")
            continue
        float_data = _empty_float(methods)
        digitised_data = _empty_digitised(methods)
        preferred_bit_depth = 16 if 16 in references else max(references)
        for method, param in INTEGRATION_METHODS:
            if method == "analytic":
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
                rows.append({"Case": case_dir.name, "Frame": frame, "BitDepth": bit_depth, "Method": method, "Param": param, "Samples": samples, "Reference": "analytic:0", "e_f64": e_f64, "e_inf": e_inf, "e_b": float(np.sqrt(np.mean(digitised_diff**2))), "delta_b": delta_b, "max_eb": max_eb})
        paths = plot_bespoke_convergence(case_dir.name, frame, "Analytic Reference", RESULTS_DIR, float_data, digitised_data, sorted(references))
        for path in paths:
            print(f"Saved {path}")
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["Case", "Frame", "BitDepth", "Method", "Param", "Samples", "Reference", "e_f64", "e_inf", "e_b", "delta_b", "max_eb"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if CLEAR_DIR:
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for case_name in DEFORMATION_CASES:
        case_dir = OUTPUT_DIR / case_name
        if not case_dir.exists():
            print(f"Warning: {case_dir} does not exist. Skipping.")
            continue
        all_rows.extend(analyse_case(case_dir))
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    print("Experiment 1 grid analysis completed.")


if __name__ == "__main__":
    main()
