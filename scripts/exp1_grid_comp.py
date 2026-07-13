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
from PIL import Image

from exp1params import (
    BIT_DEPTHS,
    SSAA_LEVELS,
    TEX_OVERSAMPLES,
)

# Output directory paths
ANALYTIC_DIR = Path(
    "./out/exp1_2d_analytic_render_world/plate260_cam256_quad9_rigid"
)
RILEY_FUNC_DIR = Path(
    "./out/exp1_riley_render_world/plate260_cam256_quad9_rigid"
)
RILEY_TEX_DIR = Path(
    "./out/riley_plate260_cam256_quad9_rigid_tex"
)


def compare_images(
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> tuple[float, float]:
    """Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE)."""
    diff = np.abs(arr1 - arr2)
    mae = float(np.mean(diff))
    mse = float(np.mean(diff**2))
    return mae, mse


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Analytic vs. Riley Convergence Comparison")
    print(80 * "=")

    if not ANALYTIC_DIR.exists():
        print(f"Error: Analytic outputs not found at {ANALYTIC_DIR}")
        print("Please run scripts/exp1_sin_grid_gen.py first.")
        sys.exit(1)

    if not RILEY_FUNC_DIR.exists():
        print(f"Error: Riley func outputs not found at {RILEY_FUNC_DIR}")
        print("Please run scripts/exp1_riley_func_sin.py first.")
        sys.exit(1)

    num_frames = 11  # Standard for the rigid case

    print("\n--- Function Shader Comparison (Analytic vs. Riley Func) ---")
    print(
        f"{'SSAA':<6} {'Bits':<6} {'Frame':<6} "
        f"{'Float MAE':<12} {'Float MSE':<12} "
        f"{'TIFF MAE':<10} {'TIFF MSE':<10}"
    )
    print(70 * "-")

    for ss in SSAA_LEVELS:
        for bb in BIT_DEPTHS:
            func_case_dir = RILEY_FUNC_DIR / f"ss{ss}_b{bb}"
            if not func_case_dir.exists():
                continue

            for ff in range(num_frames):
                # 1. Load analytic NPY (float) and TIFF (quantized)
                prefix_a = (
                    f"targ_px256_int_rect_param_{ss}_b{bb}"
                    f"_frame{ff:02d}"
                )
                npy_path = ANALYTIC_DIR / f"{prefix_a}.npy"
                tiff_path_a = ANALYTIC_DIR / f"{prefix_a}.tiff"

                if not npy_path.exists() or not tiff_path_a.exists():
                    continue

                analytic_float = np.load(npy_path)
                with Image.open(tiff_path_a) as img:
                    analytic_tiff = np.array(img, dtype=np.float64)

                # 2. Load Riley Func NPY (float) and TIFF (quantized)
                npy_path_r = (
                    func_case_dir / f"image_c00_f{ff:02d}.npy"
                )
                tiff_path_r = (
                    func_case_dir / f"cam0_frame{ff}_field0.tiff"
                )

                if not npy_path_r.exists() or not tiff_path_r.exists():
                    continue

                riley_float = np.load(npy_path_r)
                with Image.open(tiff_path_r) as img:
                    riley_tiff = np.array(img, dtype=np.float64)

                # Compare
                f_mae, f_mse = compare_images(analytic_float, riley_float)
                t_mae, t_mse = compare_images(analytic_tiff, riley_tiff)

                print(
                    f"{ss:<6d} {bb:<6d} {ff:<6d} "
                    f"{f_mae:<12.5e} {f_mse:<12.5e} "
                    f"{t_mae:<10.3f} {t_mse:<10.3f}"
                )

    print("\n--- Texture Shader Comparison (Analytic vs. Riley Tex) ---")
    print(
        f"{'SSAA':<5} {'Bits':<5} {'Oversamp':<9} {'Frame':<6} "
        f"{'Float MAE':<12} {'Float MSE':<12} "
        f"{'TIFF MAE':<10} {'TIFF MSE':<10}"
    )
    print(80 * "-")

    for ss in SSAA_LEVELS:
        for bb in BIT_DEPTHS:
            for oversamp in TEX_OVERSAMPLES:
                tex_case_dir = (
                    RILEY_TEX_DIR / f"ss{ss}_b{bb}_oversamp{oversamp}"
                )
                if not tex_case_dir.exists():
                    continue

                for ff in range(num_frames):
                    prefix_a = (
                        f"targ_px256_int_rect_param_{ss}_b{bb}"
                        f"_frame{ff:02d}"
                    )
                    npy_path = ANALYTIC_DIR / f"{prefix_a}.npy"
                    tiff_path_a = ANALYTIC_DIR / f"{prefix_a}.tiff"

                    if not npy_path.exists() or not tiff_path_a.exists():
                        continue

                    analytic_float = np.load(npy_path)
                    with Image.open(tiff_path_a) as img:
                        analytic_tiff = np.array(img, dtype=np.float64)

                    npy_path_r = (
                        tex_case_dir / f"image_c00_f{ff:02d}.npy"
                    )
                    tiff_path_r = (
                        tex_case_dir / f"cam0_frame{ff}_field0.tiff"
                    )

                    if not npy_path_r.exists() or not tiff_path_r.exists():
                        continue

                    riley_float = np.load(npy_path_r)
                    with Image.open(tiff_path_r) as img:
                        riley_tiff = np.array(img, dtype=np.float64)

                    f_mae, f_mse = compare_images(
                        analytic_float, riley_float
                    )
                    t_mae, t_mse = compare_images(
                        analytic_tiff, riley_tiff
                    )

                    print(
                        f"{ss:<5d} {bb:<5d} {oversamp:<9d} {ff:<6d} "
                        f"{f_mae:<12.5e} {f_mse:<12.5e} "
                        f"{t_mae:<10.3f} {t_mse:<10.3f}"
                    )


if __name__ == "__main__":
    main()
