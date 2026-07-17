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
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from exp2params import ACTIVE_FRAMES, BIT_DEPTHS, DEFORMATION_CASES, OUTPUT_DIR
from expplots import plot_bespoke_convergence, samples_for_method


RESULTS_DIR = Path("./out/exp2_speckint2d_analysis")
JOB_RE = re.compile(
    r"^(?P<pattern>.+)_int_(?P<method>analytic|rect|gauss|mc)_param_(?P<param>\d+)$"
)


def _image_pair(directory: Path, method: str, param: int, bit_depth: int, frame: int):
    prefix = f"targ_px256_int_{method}_param_{param}_frame{frame:02d}"
    npy_path = directory / f"{prefix}.npy"
    tiff_path = directory / f"{prefix}_b{bit_depth}.tiff"
    if npy_path.exists() and tiff_path.exists():
        with Image.open(tiff_path) as image:
            digitised = np.asarray(image, dtype=np.float64)
        return np.load(npy_path), digitised

    # Support output produced before the f64 texture convention, whose NumPy
    # files held digitised code values separately for every bit depth.
    legacy_prefix = f"{prefix}_b{bit_depth}"
    legacy_npy_path = directory / f"{legacy_prefix}.npy"
    legacy_tiff_path = directory / f"{legacy_prefix}.tiff"
    if not legacy_npy_path.exists() or not legacy_tiff_path.exists():
        return None
    with Image.open(legacy_tiff_path) as image:
        digitised = np.asarray(image, dtype=np.float64)
    return np.load(legacy_npy_path) / float(2**bit_depth - 1), digitised


def _discover_jobs() -> dict[str, dict[tuple[str, int], Path]]:
    """Group rendered jobs by ``<case>_<pattern-tag>``."""
    groups: dict[str, dict[tuple[str, int], Path]] = defaultdict(dict)
    if not OUTPUT_DIR.exists():
        return groups
    for directory in OUTPUT_DIR.iterdir():
        if not directory.is_dir():
            continue
        for case_name in DEFORMATION_CASES:
            prefix = f"{case_name}_"
            if not directory.name.startswith(prefix):
                continue
            match = JOB_RE.match(directory.name[len(prefix):])
            if match is None:
                continue
            group_name = f"{case_name}_{match.group('pattern')}"
            groups[group_name][(match.group("method"), int(match.group("param")))] = directory
            break
    return groups


def _reference_job(jobs: dict[tuple[str, int], Path], frame: int):
    """Prefer an analytic image; otherwise select the highest available Gauss rule."""
    analytic = jobs.get(("analytic", 0))
    if analytic is not None and any(
        _image_pair(analytic, "analytic", 0, bit_depth, frame) is not None
        for bit_depth in BIT_DEPTHS
    ):
        return ("analytic", 0), analytic, "Analytic Reference"
    gaussian_jobs = sorted(
        ((param, directory) for (method, param), directory in jobs.items() if method == "gauss"),
        reverse=True,
    )
    for param, directory in gaussian_jobs:
        if any(
            _image_pair(directory, "gauss", param, bit_depth, frame) is not None
            for bit_depth in BIT_DEPTHS
        ):
            return ("gauss", param), directory, f"Gauss Quadrature Reference ({param}x{param})"
    return None


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
        selected = _reference_job(jobs, frame)
        if selected is None:
            print(f"Warning: {group_name}, frame {frame:02d}: no analytic or Gaussian reference.")
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
        paths = plot_bespoke_convergence(group_name, frame, ref_name, output_dir, float_data, digitised_data, sorted(references))
        for path in paths:
            print(f"Saved {path}")
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
            print(f"Warning: {group_name}, frame {frame:02d}: no rectangular reference.")
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
            if param == ref_param:
                continue
            directory = jobs[("rect", param)]
            samples = samples_for_method("rect", param)
            for bit_depth, (ref_float, ref_digitised) in references.items():
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
        if frame_rows:
            plot_bespoke_convergence(
                group_name,
                frame,
                f"Rectangular SSAA Reference ({ref_param}x{ref_param})",
                output_dir,
                float_data,
                digitised_data,
                sorted(references),
            )
            rows.extend(frame_rows)
    _write_rows(output_dir / "summary.csv", rows)
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rectconv_dir = Path(f"{RESULTS_DIR}_rectconv")
    rectconv_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    rectconv_rows: list[dict[str, object]] = []
    for group_name, jobs in sorted(_discover_jobs().items()):
        print(f"Analysing {group_name}")
        all_rows.extend(analyse_group(group_name, jobs))
        rectconv_rows.extend(analyse_rectangular_self_convergence(group_name, jobs))
    _write_rows(RESULTS_DIR / "summary.csv", all_rows)
    _write_rows(rectconv_dir / "summary.csv", rectconv_rows)
    print("Experiment 2 grid analysis completed.")


if __name__ == "__main__":
    main()
