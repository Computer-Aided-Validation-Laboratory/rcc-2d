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
from multiprocessing import Pool

import numpy as np
import pyvista as pv
from PIL import Image

from exp1common import (
    build_pv_mesh,
    parse_case_params,
    compute_padded_uvs,
    get_integration_points_2d,
    evaluate_eggbox_analytic_average,
)
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    TEX_OVERSAMPLES,
    BIT_DEPTHS,
    INTEGRATION_METHODS,
    P_PIXELS,
    I0,
    GAMMA,
    OUTPUT_DIR,
)


def process_chunk(args):
    """Helper function to process a chunk of pixel rows in a separate process."""
    (
        y_start,
        y_end,
        method,
        param,
        pixel_size,
        roi_size,
        p_phys,
        coords,
        connect,
        disp_x_ff,
        disp_y_ff,
    ) = args

    cur_num_y = y_end - y_start
    pts_x, pts_y, weights = get_integration_points_2d(
        method,
        param,
        start_x=-roi_size / 2.0,
        start_y=-roi_size / 2.0 + y_start * pixel_size,
        pixel_size=pixel_size,
        num_px_x=TARG_PX_X,
        num_px_y=cur_num_y,
    )

    num_pts = len(weights)
    query_pts = np.zeros(
        (cur_num_y * TARG_PX_X * num_pts, 3),
        dtype=np.float64,
    )
    query_pts[:, 0] = pts_x.ravel()
    query_pts[:, 1] = pts_y.ravel()

    # Rebuild mesh locally to avoid sharing VTK pointers
    mesh = build_pv_mesh(coords, connect)
    mesh.point_data["disp_x"] = disp_x_ff
    mesh.point_data["disp_y"] = disp_y_ff

    # Sub-chunk PyVista sampling to keep memory extremely low (<150MB per worker)
    N_total = len(query_pts)
    sub_chunk_size = 2000000
    u_x = np.zeros(N_total, dtype=np.float64)
    u_y = np.zeros(N_total, dtype=np.float64)

    for i in range(0, N_total, sub_chunk_size):
        end_i = min(i + sub_chunk_size, N_total)
        sub_pts = query_pts[i:end_i]

        grid_query = pv.PolyData(sub_pts)
        sampled = grid_query.sample(mesh)

        sub_ux = sampled.point_data["disp_x"]
        sub_uy = sampled.point_data["disp_y"]
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        sub_ux[~valid] = 0.0
        sub_uy[~valid] = 0.0

        u_x[i:end_i] = sub_ux
        u_y[i:end_i] = sub_uy

    x_def = query_pts[:, 0] - u_x
    y_def = query_pts[:, 1] - u_y

    cos_x = np.cos(2.0 * np.pi * x_def / p_phys)
    cos_y = np.cos(2.0 * np.pi * y_def / p_phys)
    sub_intensity = (
        I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
    )

    sub_intensity = sub_intensity.reshape(
        cur_num_y,
        TARG_PX_X,
        num_pts,
    )
    chunk_raw = np.sum(
        sub_intensity * weights[None, None, :],
        axis=2,
    )
    return y_start, y_end, chunk_raw


def generate_texture(
    method: str,
    param: int,
    bb: int,
    oversamp: int,
) -> None:
    """Generate and save reference texture image and float files."""
    tex_w: int = oversamp * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h: int = oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD)

    roi_size: float = float(max(TARG_PX_X, TARG_PX_Y))
    pixel_size: float = roi_size / float(
        max(TARG_PX_X, TARG_PX_Y)
    )

    pixel_size_tex: float = pixel_size / float(oversamp)

    half_roi: float = roi_size / 2.0
    pad_phys_x: float = float(TEX_PX_PAD) * pixel_size
    pad_phys_y: float = float(TEX_PX_PAD) * pixel_size

    start_x = -half_roi - pad_phys_x
    start_y = -half_roi - pad_phys_y
    p_phys: float = P_PIXELS * pixel_size

    if method == "analytic":
        pixel_raw = evaluate_eggbox_analytic_average(
            start_x=start_x,
            start_y=start_y,
            pixel_size=pixel_size_tex,
            num_px_x=tex_w,
            num_px_y=tex_h,
            p_phys=p_phys,
            i0=I0,
            gamma=GAMMA,
        )
    else:
        pixel_raw = np.zeros((tex_h, tex_w), dtype=np.float64)
        
        chunk_size_y = 64
        for y_start in range(0, tex_h, chunk_size_y):
            y_end = min(y_start + chunk_size_y, tex_h)
            cur_num_y = y_end - y_start
            
            pts_x, pts_y, weights = get_integration_points_2d(
                method,
                param,
                start_x=start_x,
                start_y=start_y + y_start * pixel_size_tex,
                pixel_size=pixel_size_tex,
                num_px_x=tex_w,
                num_px_y=cur_num_y,
            )
            
            cos_x = np.cos(2.0 * np.pi * pts_x / p_phys)
            cos_y = np.cos(2.0 * np.pi * pts_y / p_phys)
            sub_intensity = (
                I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
            )
            
            pixel_raw[y_start:y_end, :] = np.sum(
                sub_intensity * weights[None, None, :], axis=2
            )

    pixel_raw_flipped: np.ndarray = np.flipud(pixel_raw)

    max_val_bb: float = float(2**bb - 1)
    pixel_bb: np.ndarray = np.round(pixel_raw_flipped * max_val_bb)
    pixel_bb = np.clip(pixel_bb, 0.0, max_val_bb)

    tex_out_dir: Path = OUTPUT_DIR / "textures"
    tex_out_dir.mkdir(parents=True, exist_ok=True)

    p_val: int = max(TARG_PX_X, TARG_PX_Y)
    prefix: str = (
        f"tex_px{p_val}_int_{method}_param_{param}_b{bb}"
        f"_pad{TEX_PX_PAD}_oversamp{oversamp}"
    )

    if bb == 8:
        pixel_8: np.ndarray = pixel_bb.astype(np.uint8)
        img: Image.Image = Image.fromarray(pixel_8)
        img.save(tex_out_dir / f"{prefix}.tiff")
    else:
        pixel_16: np.ndarray = pixel_bb.astype(np.uint16)
        img = Image.fromarray(pixel_16)
        img.save(tex_out_dir / f"{prefix}.tiff")


def generate_grid_images(
    case_dir: Path,
    method: str,
    param: int,
) -> None:
    """Load mesh, interpolate displacements, and generate target images."""
    case_name: str = case_dir.name
    print(
        f"\nProcessing: {case_name} ({method}={param})"
    )

    coords: np.ndarray = np.loadtxt(
        case_dir / "coords.csv", delimiter=","
    )
    connect: np.ndarray = np.loadtxt(
        case_dir / "connectivity.csv", delimiter=",", dtype=int
    )
    disp_x: np.ndarray = np.loadtxt(
        case_dir / "field_disp_x.csv", delimiter=","
    )
    disp_y: np.ndarray = np.loadtxt(
        case_dir / "field_disp_y.csv", delimiter=","
    )

    if connect.ndim == 1:
        connect = connect.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x = disp_x.reshape(-1, 1)
    if disp_y.ndim == 1:
        disp_y = disp_y.reshape(-1, 1)

    camera_pixels: int
    roi_size: float
    camera_pixels, roi_size = parse_case_params(case_dir)
    pixel_size: float = roi_size / camera_pixels
    num_frames: int = disp_x.shape[1]

    uvs_path = case_dir / "uvs_exp1_sin_grid.csv"
    if not uvs_path.exists():
        uvs: np.ndarray = compute_padded_uvs(
            coords, roi_size, camera_pixels, TEX_PX_PAD
        )
        np.savetxt(uvs_path, uvs, delimiter=",")

    case_out_dir: Path = OUTPUT_DIR / case_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    p_phys: float = P_PIXELS * pixel_size

    for ff in range(num_frames):
        is_rect_base = (method == "rect" and param in [1, 2, 4])
        is_needed_frame = (ff in [0, 5])
        if not (is_rect_base or is_needed_frame):
            continue

        if method == "analytic":
            if ff > 0:
                continue
            pixel_raw = evaluate_eggbox_analytic_average(
                start_x=-roi_size / 2.0,
                start_y=-roi_size / 2.0,
                pixel_size=pixel_size,
                num_px_x=TARG_PX_X,
                num_px_y=TARG_PX_Y,
                p_phys=p_phys,
                i0=I0,
                gamma=GAMMA,
            )
        elif ff == 0:
            pixel_raw = np.zeros(
                (TARG_PX_Y, TARG_PX_X),
                dtype=np.float64,
            )
            if method == "rect" or method == "gauss":
                S = param * param
            else:
                S = param
            chunk_size_y = min(TARG_PX_Y, max(1, 65536 // S))
            for y_start in range(0, TARG_PX_Y, chunk_size_y):
                y_end = min(y_start + chunk_size_y, TARG_PX_Y)
                cur_num_y = y_end - y_start

                pts_x, pts_y, weights = get_integration_points_2d(
                    method,
                    param,
                    start_x=-roi_size / 2.0,
                    start_y=-roi_size / 2.0 + y_start * pixel_size,
                    pixel_size=pixel_size,
                    num_px_x=TARG_PX_X,
                    num_px_y=cur_num_y,
                )

                cos_x = np.cos(2.0 * np.pi * pts_x / p_phys)
                cos_y = np.cos(2.0 * np.pi * pts_y / p_phys)
                sub_intensity = (
                    I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
                )

                pixel_raw[y_start:y_end, :] = np.sum(
                    sub_intensity * weights[None, None, :],
                    axis=2,
                )
        else:
            if method == "rect" or method == "gauss":
                S = param * param
            else:
                S = param
            chunk_size_y = min(TARG_PX_Y, max(1, 65536 // S))

            tasks = []
            for y_start in range(0, TARG_PX_Y, chunk_size_y):
                y_end = min(y_start + chunk_size_y, TARG_PX_Y)
                tasks.append(
                    (
                        y_start,
                        y_end,
                        method,
                        param,
                        pixel_size,
                        roi_size,
                        p_phys,
                        coords,
                        connect,
                        disp_x[:, ff],
                        disp_y[:, ff],
                    )
                )

            with Pool(processes=4) as pool:
                results = pool.map(process_chunk, tasks)

            pixel_raw = np.zeros(
                (TARG_PX_Y, TARG_PX_X),
                dtype=np.float64,
            )
            for y_start, y_end, chunk_raw in results:
                pixel_raw[y_start:y_end, :] = chunk_raw

        pixel_raw_flipped: np.ndarray = np.flipud(pixel_raw)

        for bb in BIT_DEPTHS:
            max_val_bb: float = float(2**bb - 1)
            pixel_bb: np.ndarray = np.round(pixel_raw_flipped * max_val_bb)
            pixel_bb = np.clip(pixel_bb, 0.0, max_val_bb)

            p_val: int = max(TARG_PX_X, TARG_PX_Y)
            prefix: str = (
                f"targ_px{p_val}_int_{method}_param_{param}"
                f"_b{bb}_frame{ff:02d}"
            )

            if bb == 8:
                pixel_8: np.ndarray = pixel_bb.astype(np.uint8)
                img: Image.Image = Image.fromarray(pixel_8)
                img.save(case_out_dir / f"{prefix}.tiff")
            else:
                pixel_16: np.ndarray = pixel_bb.astype(np.uint16)
                img = Image.fromarray(pixel_16)
                img.save(case_out_dir / f"{prefix}.tiff")

            np.save(
                case_out_dir / f"{prefix}.npy",
                pixel_raw_flipped * max_val_bb,
            )


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Sinusoidal Grid Generator (Integration Methods)")
    print(80 * "=")

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [
            Path("data/plate260_cam256_quad9_rigid"),
            # Path("data/plate260_cam256_quad9_affine"),
        ]

    print("\nGenerating reference textures...")
    for method, param in INTEGRATION_METHODS:
        if method != "rect" or param not in [1, 2, 4]:
            continue
        for bb in BIT_DEPTHS:
            for oversamp in TEX_OVERSAMPLES:
                print(
                    f"  Texture: {method}={param}, bb={bb}, "
                    f"oversamp={oversamp}"
                )
                generate_texture(method, param, bb, oversamp)

    print("\nGenerating deformed target images...")
    for case_path in cases:
        if not case_path.exists():
            print(
                f"Warning: Case directory {case_path} "
                "does not exist. Skipping."
            )
            continue
        for method, param in INTEGRATION_METHODS:
            generate_grid_images(case_path, method, param)

    print("\nAll generations completed successfully!")


if __name__ == "__main__":
    # Safe start method for parallel PyVista/VTK multiprocessing
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
