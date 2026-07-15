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

def analyze_case(case_dir: Path) -> tuple[list, list]:
    """Analyze convergence metrics for a specific case."""
    case_name = case_dir.name
    print(80 * "=")
    print(f"Analyzing case: {case_name}")
    print(80 * "=")

    frames_to_analyze = ACTIVE_FRAMES
    float_rows = []
    bit_rows = []
    ssaa_ticks = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]

    for ff in frames_to_analyze:
        print(f"\n--- Frame {ff:02d} ---")

        ref_method, ref_param = "analytic", 0
        ref_name = "Analytic Reference"

        # Float data container: {bb: {method: {samples: ...}}}
        float_data = {
            bb: {
                "rect": {"samples": [], "e_f64": [], "e_inf": []},
                "mc": {"samples": [], "e_f64": [], "e_inf": []},
                "gauss": {"samples": [], "e_f64": [], "e_inf": []},
            }
            for bb in BIT_DEPTHS
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
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                max_val_f = float(2**bb - 1)
                prefix_f = (
                    f"targ_px256_int_{method}_param_{param}"
                    f"_b{bb}_frame{ff:02d}"
                )
                npy_path_f = case_dir / f"{prefix_f}.npy"

                if npy_path_f.exists():
                    img_float = np.load(npy_path_f) / max_val_f
                    ref_float = ref_float_by_bb[bb]
                    diff = img_float - ref_float
                    e_f64 = np.sqrt(np.mean(diff**2))
                    e_inf = np.max(np.abs(diff))

                    float_data[bb][method]["samples"].append(samples)
                    float_data[bb][method]["e_f64"].append(e_f64)
                    float_data[bb][method]["e_inf"].append(e_inf)

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

                f_info = float_data[bb][method]
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

        # Accumulate float convergence data
        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            for method, param in INTEGRATION_METHODS:
                if method == "analytic":
                    continue

                if method == "rect":
                    samples = param * param
                elif method == "gauss":
                    samples = param * param
                else:
                    samples = param

                f_info = float_data[bb][method]
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
                    float_rows.append(
                        (
                            case_name,
                            ff,
                            bb,
                            method_str,
                            param,
                            samples,
                            e_f64,
                            e_inf,
                        )
                    )

        # Accumulate digitized convergence data
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
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

                f_info = float_data[bb][method]
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
        plt.figure(figsize=(11, 7))
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

        # We only plot a single bit depth (16-bit) to avoid redundant curves
        bb_for_float = 16 if 16 in ref_float_by_bb else BIT_DEPTHS[-1]

        for m_name in ["rect", "gauss", "mc"]:
            m_info = float_data[bb_for_float][m_name]
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
                linewidth=2.0,
                markersize=8,
            )

        # Plot horizontal threshold lines for each bit depth
        # 1) LSB line: 1.0 / (2**bb - 1)
        # 2) No pixels different limit (0.5 LSB): 0.5 / (2**bb - 1)
        linestyles_ref = {8: "-", 12: "--", 16: ":"}
        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            max_val_bb = float(2**bb - 1)

            # LSB Line (black)
            plt.axhline(
                1.0 / max_val_bb,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Line",
            )

            # No pixels different line (red)
            plt.axhline(
                0.5 / max_val_bb,
                color="red",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
            )

        plt.title(
            f"Continuous Floating-Point RMSE ($e_{{f64}}$) Convergence:\n"
            f"{case_name} (Frame {ff:02d}) | Reference: {ref_name}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Floating-Point RMSE ($e_{f64}$)", fontsize=10)
        plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
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

        plot_name1a = (
            f"convergence_{case_name}_float_rmse_frame{ff:02d}.png"
        )
        plt.savefig(RESULTS_DIR / plot_name1a, dpi=150)
        plt.close()

        # Plot 1b: Continuous Max Error convergence (e_inf)
        plt.figure(figsize=(11, 7))

        for m_name in ["rect", "gauss", "mc"]:
            m_info = float_data[bb_for_float][m_name]
            if not m_info["samples"]:
                continue
            idx = np.argsort(m_info["samples"])
            s_sorted = np.array(m_info["samples"])[idx]
            einf_sorted = np.array(m_info["e_inf"])[idx]

            # Filter out zero errors to prevent log(0) warnings
            valid = einf_sorted > 0.0
            if not np.any(valid):
                continue
            s_sorted = s_sorted[valid]
            einf_sorted = einf_sorted[valid]

            plt.loglog(
                s_sorted,
                einf_sorted,
                marker=markers[m_name],
                color=colors[m_name],
                label=labels[m_name],
                linewidth=2.0,
                markersize=8,
            )

        # Plot horizontal threshold lines for each bit depth
        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            max_val_bb = float(2**bb - 1)

            # LSB Line (black)
            plt.axhline(
                1.0 / max_val_bb,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Line",
            )

            # No pixels different line (red)
            plt.axhline(
                0.5 / max_val_bb,
                color="red",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
            )

        plt.title(
            f"Continuous Floating-Point Max Error ($e_{{\\infty}}$) "
            f"Convergence:\n"
            f"{case_name} (Frame {ff:02d}) | Reference: {ref_name}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Floating-Point Max Error ($e_{\\infty}$)", fontsize=10)
        plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
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

        plot_name1b = (
            f"convergence_{case_name}_float_max_frame{ff:02d}.png"
        )
        plt.savefig(RESULTS_DIR / plot_name1b, dpi=150)
        plt.close()

        # Plot 2: Digitised Maximum Error Convergence (Raw LSB, Log scale)
        plt.figure(figsize=(11, 7))
        linestyles = {8: "-", 12: "--", 16: ":"}
        floor_val = 0.2

        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            for m_name in ["rect", "gauss", "mc"]:
                m_info = digitised_data[bb][m_name]
                if not m_info["samples"]:
                    continue
                idx = np.argsort(m_info["samples"])
                s_sorted = np.array(m_info["samples"])[idx]
                max_eb_sorted = np.array(m_info["max_eb"])[idx]

                max_eb_plot = np.where(
                    max_eb_sorted == 0.0, floor_val, max_eb_sorted
                )

                plt.loglog(
                    s_sorted,
                    max_eb_plot,
                    linestyle=linestyles[bb],
                    marker=markers[m_name],
                    color=colors[m_name],
                    label=f"{labels[m_name]} ({bb}-bit)",
                    linewidth=1.8,
                    markersize=7,
                )

        # Plot horizontal line at exactly 1 LSB mismatch
        plt.axhline(
            1.0,
            color="black",
            linestyle="-",
            alpha=0.6,
            linewidth=1.2,
            label="1 LSB Mismatch Line",
        )

        # Plot horizontal line at floor (no pixels different / 0 LSB)
        plt.axhline(
            floor_val,
            color="red",
            linestyle=":",
            alpha=0.6,
            linewidth=1.2,
            label="No Mismatch (0 LSB)",
        )

        plt.title(
            f"Digitised Maximum Error Convergence (LSB Mismatch):\n"
            f"{case_name} (Frame {ff:02d}) | Reference: {ref_name}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Maximum Digitised Mismatch (LSB levels)", fontsize=10)
        plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])

        # Set explicit y-ticks on log scale for readability
        y_ticks = [0.2, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        y_labels = [
            "0", "1", "2", "5", "10", "20", "50", "100",
            "200", "500", "1000", "2000", "5000"
        ]
        plt.yticks(y_ticks, y_labels)
        plt.ylim(floor_val * 0.8, 10000.0)
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

        plot_name2 = f"convergence_{case_name}_max_eb_frame{ff:02d}.png"
        plt.savefig(RESULTS_DIR / plot_name2, dpi=150)
        plt.close()

        # Plot 3: Fraction of Differing Pixels (delta_b, Linear y-axis)
        plt.figure(figsize=(11, 7))

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

                plt.plot(
                    s_sorted,
                    delta_sorted,
                    linestyle=linestyles[bb],
                    marker=markers[m_name],
                    color=colors[m_name],
                    label=f"{labels[m_name]} ({bb}-bit)",
                    linewidth=1.8,
                    markersize=7,
                )

        # Plot horizontal line at exactly 0.0 (no pixels different)
        plt.axhline(
            0.0,
            color="red",
            linestyle=":",
            alpha=0.6,
            linewidth=1.2,
            label="No Pixels Different (0 pixels)",
        )

        plt.xscale("log")
        plt.title(
            f"Fraction of Differing Pixels ($\\delta_b$):\n"
            f"{case_name} (Frame {ff:02d}) | Reference: {ref_name}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Fraction of Differing Pixels ($\\delta_b$)", fontsize=10)
        plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
        plt.ylim(-0.05, 1.05)
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

        plot_name3 = f"convergence_{case_name}_bits_frame{ff:02d}.png"
        plt.savefig(RESULTS_DIR / plot_name3, dpi=150)
        plt.close()

        print(
            f"Saved float RMSE convergence plot: "
            f"{RESULTS_DIR / plot_name1a}"
        )
        print(
            f"Saved float max convergence plot: "
            f"{RESULTS_DIR / plot_name1b}"
        )
        print(
            f"Saved bit-res max mismatch plot: "
            f"{RESULTS_DIR / plot_name2}"
        )
        print(
            f"Saved bit-res fraction plot: "
            f"{RESULTS_DIR / plot_name3}"
        )

    # Write case-specific float CSV
    float_csv_path = RESULTS_DIR / f"convergence_{case_name}_float.csv"
    with open(float_csv_path, "w") as f:
        f.write("Frame,BitDepth,Method,Param,Samples,e_f64,e_inf\n")
        for row in float_rows:
            f.write(
                f"{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},"
                f"{row[6]:.4e},{row[7]:.4e}\n"
            )

    # Write case-specific bit depth CSVs
    for bb in BIT_DEPTHS:
        bit_csv_path = RESULTS_DIR / f"convergence_{case_name}_b{bb}.csv"
        with open(bit_csv_path, "w") as f:
            f.write(
                "Frame,Method,Param,Samples,e_f64,e_inf,e_b,delta_b,max_eb\n"
            )
            for row in bit_rows:
                if row[2] == bb:
                    f.write(
                        f"{row[1]},{row[3]},{row[4]},{row[5]},"
                        f"{row[6]:.4e},{row[7]:.4e},"
                        f"{row[8]:.4f},{row[9]:.4f},{row[10]:.1f}\n"
                    )

    return float_rows, bit_rows


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Convergence Analysis of Integration Methods")
    print(80 * "=")

    import shutil
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
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
        f.write("Case,Frame,BitDepth,Method,Param,Samples,e_f64,e_inf\n")
        for row in all_float_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},"
                f"{row[6]:.4e},{row[7]:.4e}\n"
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
