# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from exp1common import parse_case_params
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    BIT_DEPTHS,
    DEFORMATION_CASES,
    ACTIVE_FRAMES,
    OUTPUT_DIR,
    TEX_OVERSAMPLES,
)

SSAA_LEVELS = [1, 2, 4, 8, 16]
RESULTS_DIR_FUNC = Path("./out/exp1_riley_func")
RILEY_FUNC_WORLD_DIR = Path("./out/exp1_riley_func_world")
RESULTS_DIR_TEX = Path("./out/exp1_riley_tex")


def analyze_riley_case(case_name: str) -> None:
    print(80 * "=")
    print(f"Analyzing Riley vs Custom: {case_name}")
    print(80 * "=")

    case_dir = OUTPUT_DIR / case_name
    ssaa_ticks = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]

    for ff in ACTIVE_FRAMES:
        print(f"\n--- Frame {ff:02d} ---")

        # Load analytic reference images
        ref_float_by_bb = {}
        ref_dig_by_bb = {}

        for bb in BIT_DEPTHS:
            max_val = float(2**bb - 1)
            ref_prefix = (
                f"targ_px256_int_analytic_param_0_b{bb}_frame{ff:02d}"
            )
            ref_npy = case_dir / f"{ref_prefix}.npy"
            ref_tiff = case_dir / f"{ref_prefix}.tiff"

            if ref_npy.exists() and ref_tiff.exists():
                ref_float_by_bb[bb] = np.load(ref_npy) / max_val
                with Image.open(ref_tiff) as img:
                    ref_dig_by_bb[bb] = np.array(img, dtype=np.float64)

        if not ref_float_by_bb:
            print(
                f"Warning: Reference not found for Frame {ff:02d}. Skipping."
            )
            continue

        # Data structures for Plotting
        # 1. Custom Renderer (rect & gauss)
        custom_data = {
            bb: {
                "rect": {
                    "samples": [],
                    "e_f64": [],
                    "e_inf": [],
                    "delta_b": [],
                    "max_eb": [],
                },
                "gauss": {
                    "samples": [],
                    "e_f64": [],
                    "e_inf": [],
                    "delta_b": [],
                    "max_eb": [],
                },
            }
            for bb in BIT_DEPTHS
        }

        # Load Custom Renderer data
        # Check params from 1 to 512 for rect, 2 to 128 for gauss
        rect_params = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        gauss_params = [2, 4, 8, 16, 32, 64, 128]

        for method, params in [("rect", rect_params), ("gauss", gauss_params)]:
            for param in params:
                samples = param * param
                for bb in BIT_DEPTHS:
                    if bb not in ref_float_by_bb:
                        continue
                    max_val = float(2**bb - 1)
                    prefix = (
                        f"targ_px256_int_{method}_param_{param}"
                        f"_b{bb}_frame{ff:02d}"
                    )
                    npy_path = case_dir / f"{prefix}.npy"
                    tiff_path = case_dir / f"{prefix}.tiff"

                    if npy_path.exists() and tiff_path.exists():
                        # Float metrics
                        img_float = np.load(npy_path) / max_val
                        diff = img_float - ref_float_by_bb[bb]
                        e_f64 = np.sqrt(np.mean(diff**2))
                        e_inf = np.max(np.abs(diff))

                        # Digitised metrics
                        with Image.open(tiff_path) as img:
                            img_dig = np.array(img, dtype=np.float64)
                        diff_dig = img_dig - ref_dig_by_bb[bb]
                        delta_b = np.mean(img_dig != ref_dig_by_bb[bb])
                        max_eb = np.max(np.abs(diff_dig))

                        custom_data[bb][method]["samples"].append(samples)
                        custom_data[bb][method]["e_f64"].append(e_f64)
                        custom_data[bb][method]["e_inf"].append(e_inf)
                        custom_data[bb][method]["delta_b"].append(delta_b)
                        custom_data[bb][method]["max_eb"].append(max_eb)

        # 2. Riley Function Shader Data
        riley_func = {
            bb: {
                "samples": [],
                "e_f64": [],
                "e_inf": [],
                "delta_b": [],
                "max_eb": [],
            }
            for bb in BIT_DEPTHS
        }

        # Load Riley Function Shader data
        func_dir_base = RILEY_FUNC_WORLD_DIR / case_name
        for ss in SSAA_LEVELS:
            samples = ss * ss
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                max_val = float(2**bb - 1)
                case_out = func_dir_base / f"ss{ss}_b{bb}"
                npy_path = case_out / f"image_c00_f{ff:02d}.npy"
                tiff_path = case_out / f"cam0_frame{ff}_field0.tiff"

                if npy_path.exists() and tiff_path.exists():
                    img_float = np.load(npy_path) / max_val
                    diff = img_float - ref_float_by_bb[bb]
                    e_f64 = np.sqrt(np.mean(diff**2))
                    e_inf = np.max(np.abs(diff))

                    with Image.open(tiff_path) as img:
                        img_dig = np.array(img, dtype=np.float64)
                    diff_dig = img_dig - ref_dig_by_bb[bb]
                    delta_b = np.mean(img_dig != ref_dig_by_bb[bb])
                    max_eb = np.max(np.abs(diff_dig))

                    riley_func[bb]["samples"].append(samples)
                    riley_func[bb]["e_f64"].append(e_f64)
                    riley_func[bb]["e_inf"].append(e_inf)
                    riley_func[bb]["delta_b"].append(delta_b)
                    riley_func[bb]["max_eb"].append(max_eb)

        # 3. Riley Texture Shader Data
        riley_tex = {
            bb: {
                oversamp: {
                    "samples": [],
                    "e_f64": [],
                    "e_inf": [],
                    "delta_b": [],
                    "max_eb": [],
                }
                for oversamp in TEX_OVERSAMPLES
            }
            for bb in BIT_DEPTHS
        }

        # Load Riley Texture Shader data
        tex_dir_base = Path("./out") / f"riley_{case_name}_tex"
        for ss in SSAA_LEVELS:
            samples = ss * ss
            for bb in BIT_DEPTHS:
                if bb not in ref_float_by_bb:
                    continue
                max_val = float(2**bb - 1)
                for oversamp in TEX_OVERSAMPLES:
                    case_out = (
                        tex_dir_base / f"ss{ss}_b{bb}_oversamp{oversamp}"
                    )
                    npy_path = case_out / f"image_c00_f{ff:02d}.npy"
                    tiff_path = case_out / f"cam0_frame{ff}_field0.tiff"

                    if npy_path.exists() and tiff_path.exists():
                        img_float = np.load(npy_path) / max_val
                        diff = img_float - ref_float_by_bb[bb]
                        e_f64 = np.sqrt(np.mean(diff**2))
                        e_inf = np.max(np.abs(diff))

                        with Image.open(tiff_path) as img:
                            img_dig = np.array(img, dtype=np.float64)
                        diff_dig = img_dig - ref_dig_by_bb[bb]
                        delta_b = np.mean(img_dig != ref_dig_by_bb[bb])
                        max_eb = np.max(np.abs(diff_dig))

                        r_tex = riley_tex[bb][oversamp]
                        r_tex["samples"].append(samples)
                        r_tex["e_f64"].append(e_f64)
                        r_tex["e_inf"].append(e_inf)
                        r_tex["delta_b"].append(delta_b)
                        r_tex["max_eb"].append(max_eb)

        # ------------------------------------------------------------------
        # GENERATE PLOTS - GROUP A: Riley Function Shader vs Custom Renderer
        # ------------------------------------------------------------------
        bb_float = 16 if 16 in ref_float_by_bb else BIT_DEPTHS[-1]
        max_val_float = float(2**bb_float - 1)

        # Plot A1: Float RMSE Convergence
        plt.figure(figsize=(11, 7))
        # Custom Rect (blue)
        r_info = custom_data[bb_float]["rect"]
        if r_info["samples"]:
            idx = np.argsort(r_info["samples"])
            plt.loglog(
                np.array(r_info["samples"])[idx],
                np.array(r_info["e_f64"])[idx],
                marker="o",
                color="#1f77b4",
                label="Custom Rect / SSAA",
                linewidth=2.0,
                markersize=8,
            )
        # Custom Gauss (green)
        g_info = custom_data[bb_float]["gauss"]
        if g_info["samples"]:
            idx = np.argsort(g_info["samples"])
            plt.loglog(
                np.array(g_info["samples"])[idx],
                np.array(g_info["e_f64"])[idx],
                marker="s",
                color="#2ca02c",
                label="Custom Gauss Quadrature",
                linewidth=2.0,
                markersize=8,
            )
        # Riley Func (black, plotted last)
        f_info = riley_func[bb_float]
        if f_info["samples"]:
            idx = np.argsort(f_info["samples"])
            plt.loglog(
                np.array(f_info["samples"])[idx],
                np.array(f_info["e_f64"])[idx],
                marker="x",
                color="black",
                label="Riley Function Shader",
                linewidth=2.0,
                linestyle="--",
                markersize=9,
            )

        # LSB and 0.5 LSB lines
        linestyles_ref = {8: "-", 12: "--", 16: ":"}
        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            mv = float(2**bb - 1)
            plt.axhline(
                1.0 / mv,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Line",
            )
            plt.axhline(
                0.5 / mv,
                color="red",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
            )

        plt.title(
            f"Riley Func Shader vs Custom Renderer: Floating-Point RMSE\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
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
        plt.savefig(
            RESULTS_DIR_FUNC
            / f"convergence_{case_name}_func_float_rmse_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # Plot A2: Float Max Error Convergence
        plt.figure(figsize=(11, 7))
        if r_info["samples"]:
            idx = np.argsort(r_info["samples"])
            plt.loglog(
                np.array(r_info["samples"])[idx],
                np.array(r_info["e_inf"])[idx],
                marker="o",
                color="#1f77b4",
                label="Custom Rect / SSAA",
                linewidth=2.0,
                markersize=8,
            )
        if g_info["samples"]:
            idx = np.argsort(g_info["samples"])
            plt.loglog(
                np.array(g_info["samples"])[idx],
                np.array(g_info["e_inf"])[idx],
                marker="s",
                color="#2ca02c",
                label="Custom Gauss Quadrature",
                linewidth=2.0,
                markersize=8,
            )
        if f_info["samples"]:
            idx = np.argsort(f_info["samples"])
            plt.loglog(
                np.array(f_info["samples"])[idx],
                np.array(f_info["e_inf"])[idx],
                marker="x",
                color="black",
                label="Riley Function Shader",
                linewidth=2.0,
                linestyle="--",
                markersize=9,
            )

        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            mv = float(2**bb - 1)
            plt.axhline(
                1.0 / mv,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Line",
            )
            plt.axhline(
                0.5 / mv,
                color="red",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
            )

        plt.title(
            f"Riley Func Shader vs Custom Renderer: Floating-Point Max Error\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
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
        plt.savefig(
            RESULTS_DIR_FUNC
            / f"convergence_{case_name}_func_float_max_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # Plot A3: Digitised Mismatch Fraction (delta_b)
        plt.figure(figsize=(11, 7))
        bit_colors = {8: "#1f77b4", 12: "#2ca02c", 16: "#ff7f0e"}
        bit_markers = {8: "o", 12: "s", 16: "^"}

        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            # Custom Rect
            cr = custom_data[bb]["rect"]
            if cr["samples"]:
                idx = np.argsort(cr["samples"])
                plt.plot(
                    np.array(cr["samples"])[idx],
                    np.array(cr["delta_b"])[idx],
                    marker=bit_markers[bb],
                    color=bit_colors[bb],
                    label=f"Custom Rect {bb}-bit",
                    linewidth=1.5,
                    markersize=6,
                )
            # Riley Func
            rf = riley_func[bb]
            if rf["samples"]:
                idx = np.argsort(rf["samples"])
                plt.plot(
                    np.array(rf["samples"])[idx],
                    np.array(rf["delta_b"])[idx],
                    marker="x",
                    color="black",
                    label=f"Riley Func {bb}-bit",
                    linewidth=1.5,
                    linestyle="--",
                    markersize=8,
                )

        plt.xscale("log")
        plt.title(
            f"Riley Func Shader vs Custom: Digitised Mismatch Fraction\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
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
        plt.savefig(
            RESULTS_DIR_FUNC
            / f"convergence_{case_name}_func_bits_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # Plot A4: Max Digitised Mismatch (max_eb)
        plt.figure(figsize=(11, 7))
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            cr = custom_data[bb]["rect"]
            if cr["samples"]:
                idx = np.argsort(cr["samples"])
                s_s = np.array(cr["samples"])[idx]
                m_s = np.array(cr["max_eb"])[idx]
                valid = m_s > 0.0
                plt.loglog(
                    s_s[valid],
                    m_s[valid],
                    marker=bit_markers[bb],
                    color=bit_colors[bb],
                    label=f"Custom Rect {bb}-bit",
                    linewidth=1.5,
                    markersize=6,
                )
            rf = riley_func[bb]
            if rf["samples"]:
                idx = np.argsort(rf["samples"])
                s_s = np.array(rf["samples"])[idx]
                m_s = np.array(rf["max_eb"])[idx]
                valid = m_s > 0.0
                plt.loglog(
                    s_s[valid],
                    m_s[valid],
                    marker="x",
                    color="black",
                    label=f"Riley Func {bb}-bit",
                    linewidth=1.5,
                    linestyle="--",
                    markersize=8,
                )

        # Reference horizontal threshold lines
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            plt.axhline(
                1.0,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Limit",
            )

        plt.title(
            f"Riley Func Shader vs Custom: Max Digitised Mismatch\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Maximum Digitised Mismatch (LSB levels)", fontsize=10)
        plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
        # Explicit integer y-ticks
        y_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        plt.yticks(y_ticks, [str(yt) for yt in y_ticks])
        plt.ylim(0.8, 4096.0)
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
        plt.savefig(
            RESULTS_DIR_FUNC
            / f"convergence_{case_name}_func_max_eb_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # ------------------------------------------------------------------
        # GENERATE PLOTS - GROUP B: Riley Texture Shader vs Custom Baseline
        # ------------------------------------------------------------------
        # Plot B1: Float RMSE Convergence (Texture Shader)
        plt.figure(figsize=(11, 7))
        # Custom Rect Baseline
        if r_info["samples"]:
            idx = np.argsort(r_info["samples"])
            plt.loglog(
                np.array(r_info["samples"])[idx],
                np.array(r_info["e_f64"])[idx],
                marker="o",
                color="#1f77b4",
                label="Custom Rect (No Texture)",
                linewidth=2.0,
                markersize=8,
            )

        tex_colors = {
            1: "#ff7f0e",
            2: "#bcbd22",
            4: "#9467bd",
            8: "#17becf",
            16: "#e377c2",
            32: "#8c564b",
        }
        tex_markers = {
            1: "^",
            2: "v",
            4: "<",
            8: ">",
            16: "p",
            32: "h",
        }

        for oversamp in TEX_OVERSAMPLES:
            rt_info = riley_tex[bb_float][oversamp]
            if rt_info["samples"]:
                idx = np.argsort(rt_info["samples"])
                plt.loglog(
                    np.array(rt_info["samples"])[idx],
                    np.array(rt_info["e_f64"])[idx],
                    marker=tex_markers[oversamp],
                    color=tex_colors[oversamp],
                    label=f"Riley Tex (Oversamp={oversamp})",
                    linewidth=1.5,
                    markersize=7,
                )

        # LSB and 0.5 LSB lines
        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            mv = float(2**bb - 1)
            plt.axhline(
                1.0 / mv,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Line",
            )
            plt.axhline(
                0.5 / mv,
                color="red",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
            )

        plt.title(
            f"Riley Tex Shader vs Custom: Floating-Point RMSE\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
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
        plt.savefig(
            RESULTS_DIR_TEX
            / f"convergence_{case_name}_tex_float_rmse_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # Plot B2: Float Max Error Convergence (Texture Shader)
        plt.figure(figsize=(11, 7))
        if r_info["samples"]:
            idx = np.argsort(r_info["samples"])
            plt.loglog(
                np.array(r_info["samples"])[idx],
                np.array(r_info["e_inf"])[idx],
                marker="o",
                color="#1f77b4",
                label="Custom Rect (No Texture)",
                linewidth=2.0,
                markersize=8,
            )

        for oversamp in TEX_OVERSAMPLES:
            rt_info = riley_tex[bb_float][oversamp]
            if rt_info["samples"]:
                idx = np.argsort(rt_info["samples"])
                plt.loglog(
                    np.array(rt_info["samples"])[idx],
                    np.array(rt_info["e_inf"])[idx],
                    marker=tex_markers[oversamp],
                    color=tex_colors[oversamp],
                    label=f"Riley Tex (Oversamp={oversamp})",
                    linewidth=1.5,
                    markersize=7,
                )

        for bb in BIT_DEPTHS:
            if bb not in ref_float_by_bb:
                continue
            mv = float(2**bb - 1)
            plt.axhline(
                1.0 / mv,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Line",
            )
            plt.axhline(
                0.5 / mv,
                color="red",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit No Pixels Diff (0.5 LSB)",
            )

        plt.title(
            f"Riley Tex Shader vs Custom: Floating-Point Max Error\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
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
        plt.savefig(
            RESULTS_DIR_TEX
            / f"convergence_{case_name}_tex_float_max_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # Plot B3: Digitised Mismatch Fraction (Texture Shader)
        plt.figure(figsize=(11, 7))
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            cr = custom_data[bb]["rect"]
            if cr["samples"]:
                idx = np.argsort(cr["samples"])
                plt.plot(
                    np.array(cr["samples"])[idx],
                    np.array(cr["delta_b"])[idx],
                    marker=bit_markers[bb],
                    color=bit_colors[bb],
                    label=f"Custom Rect {bb}-bit",
                    linewidth=1.5,
                    markersize=6,
                )

            # We plot oversamp=8 for each bit depth to show best texture case
            rt = riley_tex[bb][8]
            if rt["samples"]:
                idx = np.argsort(rt["samples"])
                plt.plot(
                    np.array(rt["samples"])[idx],
                    np.array(rt["delta_b"])[idx],
                    marker="x",
                    color=bit_colors[bb],
                    label=f"Riley Tex (Oversamp=8) {bb}-bit",
                    linewidth=1.5,
                    linestyle="--",
                    markersize=8,
                )

        plt.xscale("log")
        plt.title(
            f"Riley Tex Shader vs Custom: Digitised Mismatch Fraction\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
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
        plt.savefig(
            RESULTS_DIR_TEX
            / f"convergence_{case_name}_tex_bits_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()

        # Plot B4: Max Digitised Mismatch (Texture Shader)
        plt.figure(figsize=(11, 7))
        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            cr = custom_data[bb]["rect"]
            if cr["samples"]:
                idx = np.argsort(cr["samples"])
                s_s = np.array(cr["samples"])[idx]
                m_s = np.array(cr["max_eb"])[idx]
                valid = m_s > 0.0
                plt.loglog(
                    s_s[valid],
                    m_s[valid],
                    marker=bit_markers[bb],
                    color=bit_colors[bb],
                    label=f"Custom Rect {bb}-bit",
                    linewidth=1.5,
                    markersize=6,
                )

            rt = riley_tex[bb][8]
            if rt["samples"]:
                idx = np.argsort(rt["samples"])
                s_s = np.array(rt["samples"])[idx]
                m_s = np.array(rt["max_eb"])[idx]
                valid = m_s > 0.0
                plt.loglog(
                    s_s[valid],
                    m_s[valid],
                    marker="x",
                    color=bit_colors[bb],
                    label=f"Riley Tex (Oversamp=8) {bb}-bit",
                    linewidth=1.5,
                    linestyle="--",
                    markersize=8,
                )

        for bb in BIT_DEPTHS:
            if bb not in ref_dig_by_bb:
                continue
            plt.axhline(
                1.0,
                color="black",
                linestyle=linestyles_ref[bb],
                alpha=0.6,
                linewidth=1.2,
                label=f"{bb}-bit LSB Limit",
            )

        plt.title(
            f"Riley Tex Shader vs Custom: Max Digitised Mismatch\n"
            f"{case_name} (Frame {ff:02d}) | Reference: Analytic",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Total Samples per Pixel", fontsize=10)
        plt.ylabel("Maximum Digitised Mismatch (LSB levels)", fontsize=10)
        plt.xticks(ssaa_ticks, [str(t) for t in ssaa_ticks])
        plt.yticks(y_ticks, [str(yt) for yt in y_ticks])
        plt.ylim(0.8, 4096.0)
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
        plt.savefig(
            RESULTS_DIR_TEX
            / f"convergence_{case_name}_tex_max_eb_frame{ff:02d}.png",
            dpi=150,
        )
        plt.close()


def main() -> None:
    shutil.rmtree(RESULTS_DIR_FUNC, ignore_errors=True)
    RESULTS_DIR_FUNC.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(RESULTS_DIR_TEX, ignore_errors=True)
    RESULTS_DIR_TEX.mkdir(parents=True, exist_ok=True)

    for case_name in DEFORMATION_CASES:
        analyze_riley_case(case_name)

    print("\nRiley analysis completed successfully.")


if __name__ == "__main__":
    main()
