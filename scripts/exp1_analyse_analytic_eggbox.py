# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from exp1params import (
    BIT_DEPTHS,
    INTEGRATION_METHODS,
    OUTPUT_DIR,
    DEFORMATION_CASES,
    ACTIVE_FRAMES,
)

# Output directory for results / plots
RESULTS_DIR = Path("./out/exp1_results")


def analyze_case(case_dir: Path) -> tuple[list, list]:
    """Analyze convergence metrics for a specific case."""
    case_name = case_dir.name
    print(80 * "=")
    print(f"Analyzing case: {case_name}")
    print(80 * "=")

    frames_to_analyze = ACTIVE_FRAMES
    float_rows = []
    bit_rows = []

    for ff in frames_to_analyze:
        print(f"\n--- Frame {ff:02d} ---")

        ref_method, ref_param = "gauss", 128
        ref_name = "Gauss-128 Reference"

        # Float data container: {method: {samples: [], e_f64: [], e_inf: []}}
        float_data = {
            "rect": {"samples": [], "e_f64": [], "e_inf": []},
            "mc": {"samples": [], "e_f64": [], "e_inf": []},
            "gauss": {"samples": [], "e_f64": [], "e_inf": []},
        }

        # Digitised data container
        digitised_data = {
            bb: {
                "rect": {
                    "samples": [],
                    "e_b": [],
                    "delta_b": [],
                    "max_eb": [],
                },
                "mc": {
                    "samples": [],
                    "e_b": [],
                    "delta_b": [],
                    "max_eb": [],
                },
                "gauss": {
                    "samples": [],
                    "e_b": [],
                    "delta_b": [],
                    "max_eb": [],
                },
            }
            for bb in BIT_DEPTHS
        }

        # Find reference float & digitised images
        ref_float_by_bb = {}
        ref_dig_by_bb = {}

        for bb in BIT_DEPTHS:
            max_val = float(2**bb - 1)
            ref_prefix = (
                f"targ_px256_int_{ref_method}_param_{ref_param}"
                f"_b{bb}_frame{ff:02d}"
            )
            ref_npy_path = case_dir / f"{ref_prefix}.npy"
            ref_tiff_path = case_dir / f"{ref_prefix}.tiff"

            if not ref_npy_path.exists() or not ref_tiff_path.exists():
                continue

            ref_float_by_bb[bb] = np.load(ref_npy_path) / max_val
            with Image.open(ref_tiff_path) as img:
                ref_dig_by_bb[bb] = np.array(img, dtype=np.float64)

        if not ref_float_by_bb:
            print(f"Warning: Reference files not found for Frame {ff:02d}.")
            continue

        # Evaluate all methods and parameters
        for method, param in INTEGRATION_METHODS:
            if method == "analytic":
                continue

            if method == "rect":
                samples = param * param
            elif method == "gauss":
                samples = param * param
            else:
                samples = param

            # Continuous floating point evaluation
            bb_for_float = 16 if 16 in ref_float_by_bb else BIT_DEPTHS[0]
            max_val_f = float(2**bb_for_float - 1)
            prefix_f = (
                f"targ_px256_int_{method}_param_{param}"
                f"_b{bb_for_float}_frame{ff:02d}"
            )
            npy_path_f = case_dir / f"{prefix_f}.npy"

            if npy_path_f.exists():
                img_float = np.load(npy_path_f) / max_val_f
                ref_float = ref_float_by_bb[bb_for_float]
                diff = img_float - ref_float
                e_f64 = np.sqrt(np.mean(diff**2))
                e_inf = np.max(np.abs(diff))

                float_data[method]["samples"].append(samples)
                float_data[method]["e_f64"].append(e_f64)
                float_data[method]["e_inf"].append(e_inf)

            # Digitised evaluation for each bit depth
            for bb in BIT_DEPTHS:
                if bb not in ref_dig_by_bb:
                    continue
                prefix = (
                    f"targ_px256_int_{method}_param_{param}"
                    f"_b{bb}_frame{ff:02d}"
                )
                tiff_path = case_dir / f"{prefix}.tiff"

                if tiff_path.exists():
                    with Image.open(tiff_path) as img:
                        img_dig = np.array(img, dtype=np.float64)
                    ref_dig = ref_dig_by_bb[bb]

                    diff_dig = img_dig - ref_dig
                    e_b = np.sqrt(np.mean(diff_dig**2))
                    delta_b = np.mean(img_dig != ref_dig)
                    max_eb = np.max(np.abs(diff_dig))

                    digitised_data[bb][method]["samples"].append(samples)
                    digitised_data[bb][method]["e_b"].append(e_b)
                    digitised_data[bb][method]["delta_b"].append(delta_b)
                    digitised_data[bb][method]["max_eb"].append(max_eb)

        # Print tables for each bit depth
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            print(
                f"\n--- Bit Depth: {bb}-bit (Reference: {ref_name}) ---"
            )
            print(
                f"{'Method':<12} {'Param':<6} {'Samples':<8} "
                f"{'e_f64':<11} {'e_inf':<11} {'e_b (bits)':<11} "
                f"{'delta_b':<11} {'max_eb (bits)':<8}"
            )
            print(85 * "-")

            for method, param in INTEGRATION_METHODS:
                if method == "analytic":
                    continue

                if method == "rect":
                    samples = param * param
                elif method == "gauss":
                    samples = param * param
                else:
                    samples = param

                d_info = digitised_data[bb][method]
                if samples not in d_info["samples"]:
                    continue
                idx = d_info["samples"].index(samples)
                e_b = d_info["e_b"][idx]
                delta_b = d_info["delta_b"][idx]
                max_eb = d_info["max_eb"][idx]

                f_info = float_data[method]
                if samples in f_info["samples"]:
                    f_idx = f_info["samples"].index(samples)
                    e_f64 = f_info["e_f64"][f_idx]
                    e_inf = f_info["e_inf"][f_idx]
                else:
                    e_f64, e_inf = 0.0, 0.0

                is_ref = (method == ref_method and param == ref_param)
                method_str = f"{method} (ref)" if is_ref else method

                print(
                    f"{method_str:<12} {param:<6d} {samples:<8d} "
                    f"{e_f64:<11.4e} {e_inf:<11.4e} {e_b:<11.4f} "
                    f"{delta_b:<11.4f} {max_eb:<8.1f}"
                )

        # Write Float Convergence CSV (Plot 1 data)
        float_csv_path = (
            RESULTS_DIR
            / f"convergence_{case_name}_float_frame{ff:02d}.csv"
        )
        with open(float_csv_path, "w") as f:
            f.write("Method,Param,Samples,e_f64,e_inf\n")
            for method, param in INTEGRATION_METHODS:
                if method == "analytic":
                    continue

                if method == "rect":
                    samples = param * param
                elif method == "gauss":
                    samples = param * param
                else:
                    samples = param

                f_info = float_data[method]
                if samples in f_info["samples"]:
                    f_idx = f_info["samples"].index(samples)
                    e_f64 = f_info["e_f64"][f_idx]
                    e_inf = f_info["e_inf"][f_idx]
                    is_ref = (
                        method == ref_method and param == ref_param
                    )
                    method_str = (
                        f"{method} (ref)" if is_ref else method
                    )
                    f.write(
                        f"{method_str},{param},{samples},"
                        f"{e_f64:.4e},{e_inf:.4e}\n"
                    )
                    # Accumulate for summary
                    float_rows.append(
                        (
                            case_name,
                            ff,
                            method_str,
                            param,
                            samples,
                            e_f64,
                            e_inf,
                        )
                    )

        # Write Digitised Convergence CSVs (Plot 2 data)
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            bit_csv_path = (
                RESULTS_DIR
                / f"convergence_{case_name}_b{bb}_frame{ff:02d}.csv"
            )
            with open(bit_csv_path, "w") as f:
                f.write(
                    "Method,Param,Samples,e_f64,e_inf,e_b,delta_b,max_eb\n"
                )
                for method, param in INTEGRATION_METHODS:
                    if method == "analytic":
                        continue

                    if method == "rect":
                        samples = param * param
                    elif method == "gauss":
                        samples = param * param
                    else:
                        samples = param

                    d_info = digitised_data[bb][method]
                    if samples not in d_info["samples"]:
                        continue
                    idx = d_info["samples"].index(samples)
                    e_b = d_info["e_b"][idx]
                    delta_b = d_info["delta_b"][idx]
                    max_eb = d_info["max_eb"][idx]

                    f_info = float_data[method]
                    if samples in f_info["samples"]:
                        f_idx = f_info["samples"].index(samples)
                        e_f64 = f_info["e_f64"][f_idx]
                        e_inf = f_info["e_inf"][f_idx]
                    else:
                        e_f64, e_inf = 0.0, 0.0

                    is_ref = (
                        method == ref_method and param == ref_param
                    )
                    method_str = (
                        f"{method} (ref)" if is_ref else method
                    )

                    f.write(
                        f"{method_str},{param},{samples},"
                        f"{e_f64:.4e},{e_inf:.4e},"
                        f"{e_b:.4f},{delta_b:.4f},{max_eb:.1f}\n"
                    )
                    # Accumulate for summary
                    bit_rows.append(
                        (
                            case_name,
                            ff,
                            bb,
                            method_str,
                            param,
                            samples,
                            e_f64,
                            e_inf,
                            e_b,
                            delta_b,
                            max_eb,
                        )
                    )

        # Plot 1: Continuous convergence (e_f64)
        plt.figure(figsize=(10, 6))
        colors = {
            "rect": "#1f77b4",
            "gauss": "#2ca02c",
            "mc": "#ff7f0e",
        }
        markers = {
            "rect": "o",
            "gauss": "s",
            "mc": "^",
        }
        labels = {
            "rect": "Rectangular / SSAA",
            "gauss": "Gauss Quadrature",
            "mc": "Monte Carlo",
        }

        for m_name, m_info in float_data.items():
            if not m_info["samples"]:
                continue
            idx = np.argsort(m_info["samples"])
            s_sorted = np.array(m_info["samples"])[idx]
            ef_sorted = np.array(m_info["e_f64"])[idx]

            # Filter out zero errors to prevent log(0) warnings
            valid = ef_sorted > 0.0
            if not np.any(valid):
                continue
            s_sorted = s_sorted[valid]
            ef_sorted = ef_sorted[valid]

            plt.loglog(
                s_sorted,
                ef_sorted,
                marker=markers[m_name],
                color=colors[m_name],
                label=labels[m_name],
                linewidth=2,
                markersize=8,
            )

        plt.title(
            f"Continuous Floating-Point Convergence ($e_{{f64}}$):\n"
            f"{case_name} (Frame {ff:02d}) | Reference: {ref_name}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Floating-Point RMSE ($e_{f64}$)", fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        plot_name1 = f"convergence_{case_name}_float_frame{ff:02d}.png"
        plt.savefig(RESULTS_DIR / plot_name1, dpi=150)
        plt.close()

        # Plot 2: Bit Resolution Convergence (delta_b)
        plt.figure(figsize=(11, 7))
        linestyles = {8: "-", 12: "--", 16: ":"}
        floor_val = 5e-6

        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            for m_name in ["rect", "gauss", "mc"]:
                m_info = digitised_data[bb][m_name]
                if not m_info["samples"]:
                    continue
                idx = np.argsort(m_info["samples"])
                s_sorted = np.array(m_info["samples"])[idx]
                delta_sorted = np.array(m_info["delta_b"])[idx]

                delta_plot = np.where(
                    delta_sorted == 0.0, floor_val, delta_sorted
                )

                plt.loglog(
                    s_sorted,
                    delta_plot,
                    linestyle=linestyles[bb],
                    marker=markers[m_name],
                    color=colors[m_name],
                    label=f"{labels[m_name]} ({bb}-bit)",
                    linewidth=1.8,
                    markersize=7,
                )

        # Plot horizontal line at exactly 1 pixel mismatch
        one_px_thresh = 1.0 / 65536.0
        plt.axhline(
            one_px_thresh,
            color="red",
            linestyle="-.",
            alpha=0.6,
            label="1 Pixel Mismatch Threshold ($1.53 \\times 10^{-5}$)",
        )

        plt.title(
            f"Bit Resolution Convergence (Fraction of Differing Pixels "
            f"$\\delta_b$):\n"
            f"{case_name} (Frame {ff:02d}) | Reference: {ref_name}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Fraction of Differing Pixels ($\\delta_b$)", fontsize=10)
        plt.ylim(floor_val * 0.8, 1.2)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="lower left",
            fontsize=9,
            ncol=2,
        )
        plt.tight_layout()

        plot_name2 = f"convergence_{case_name}_bits_frame{ff:02d}.png"
        plt.savefig(RESULTS_DIR / plot_name2, dpi=150)
        plt.close()
        print(f"Saved float convergence plot: {RESULTS_DIR / plot_name1}")
        print(f"Saved bit-res convergence plot: {RESULTS_DIR / plot_name2}")

    return float_rows, bit_rows


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Convergence Analysis of Integration Methods")
    print(80 * "=")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [OUTPUT_DIR / name for name in DEFORMATION_CASES]
    all_float_rows = []
    all_bit_rows = []

    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: Directory {case_path} does not exist. Skipping.")
            continue
        float_rows, bit_rows = analyze_case(case_path)
        all_float_rows.extend(float_rows)
        all_bit_rows.extend(bit_rows)

    # Write summary_float.csv
    float_summary_path = RESULTS_DIR / "summary_float.csv"
    with open(float_summary_path, "w") as f:
        f.write("Case,Frame,Method,Param,Samples,e_f64,e_inf\n")
        for row in all_float_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},"
                f"{row[5]:.4e},{row[6]:.4e}\n"
            )

    # Write summary_bits.csv
    bit_summary_path = RESULTS_DIR / "summary_bits.csv"
    with open(bit_summary_path, "w") as f:
        f.write(
            "Case,Frame,BitDepth,Method,Param,Samples,"
            "e_f64,e_inf,e_b,delta_b,max_eb\n"
        )
        for row in all_bit_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},"
                f"{row[6]:.4e},{row[7]:.4e},"
                f"{row[8]:.4f},{row[9]:.4f},{row[10]:.1f}\n"
            )

    # Write unified summary.csv
    unified_summary_path = RESULTS_DIR / "summary.csv"
    with open(unified_summary_path, "w") as f:
        f.write(
            "Case,Frame,BitDepth,Method,Param,Samples,"
            "e_f64,e_inf,e_b,delta_b,max_eb\n"
        )
        for row in all_bit_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},"
                f"{row[6]:.4e},{row[7]:.4e},"
                f"{row[8]:.4f},{row[9]:.4f},{row[10]:.1f}\n"
            )

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
