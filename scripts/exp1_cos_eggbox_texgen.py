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
    NUM_PROCESSES,
    DEFORMATION_CASES,
    ACTIVE_FRAMES,
)


_worker_mesh = None


def init_worker(coords, connect, disp_x_ff, disp_y_ff) -> None:
    """Initialize worker process with mesh and displacement field."""
    global _worker_mesh
    if coords is not None and connect is not None:
        _worker_mesh = build_pv_mesh(coords, connect)
        _worker_mesh.point_data["disp_x"] = disp_x_ff
        _worker_mesh.point_data["disp_y"] = disp_y_ff
    else:
        _worker_mesh = None


def process_pixel_chunk(args):
    """Process a chunk of pixels in a worker process."""
    (
        start_idx,
        end_idx,
        method,
        param,
        pixel_size,
        roi_size,
        p_phys,
    ) = args

    # Generate flat pixel indices for this chunk
    pixel_indices = np.arange(start_idx, end_idx)
    num_pixels = len(pixel_indices)

    # Get integration offsets and weights for 1 pixel
    if method == "rect":
        offsets = (np.arange(param) + 0.5) * (pixel_size / param)
        dx_grid, dy_grid = np.meshgrid(offsets, offsets)
        dx = dx_grid.ravel()
        dy = dy_grid.ravel()
        weights = (
            np.ones(param * param, dtype=np.float64) / (param * param)
        )
    # 1. Integration offsets relative to pixel bottom-left
    if method == "rect":
        offsets = (np.arange(param) + 0.5) * (pixel_size / param)
        dx_grid, dy_grid = np.meshgrid(offsets, offsets)
        dx = dx_grid.ravel()
        dy = dy_grid.ravel()
        weights = (
            np.ones(param * param, dtype=np.float64) / (param * param)
        )
    elif method == "gauss":
        pts_1d, wts_1d = np.polynomial.legendre.leggauss(param)
        offsets = 0.5 * (pts_1d + 1.0) * pixel_size
        dx_grid, dy_grid = np.meshgrid(offsets, offsets)
        dx = dx_grid.ravel()
        dy = dy_grid.ravel()
        wts_norm = wts_1d * 0.5
        w_grid = wts_norm[:, None] * wts_norm[None, :]
        weights = w_grid.ravel()
    elif method == "mc":
        dx = np.random.uniform(0.0, pixel_size, param)
        dy = np.random.uniform(0.0, pixel_size, param)
        weights = np.ones(param, dtype=np.float64) / param
    elif method == "analytic":
        dx = np.array([0.0, pixel_size, 0.0, pixel_size])
        dy = np.array([0.0, 0.0, pixel_size, pixel_size])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
    else:
        raise ValueError(f"Unknown integration method: {method}")

    # Compute pixel grid coordinates
    px_indices = pixel_indices % TARG_PX_X
    py_indices = pixel_indices // TARG_PX_X

    start_x = -roi_size / 2.0
    start_y = -roi_size / 2.0
    px_x = start_x + px_indices * pixel_size
    px_y = start_y + py_indices * pixel_size

    # Get local affine mapping from 4 corner displacements
    h = 0.5 * pixel_size
    global _worker_mesh
    if _worker_mesh is not None:
        # Query corners only (4 points per pixel)
        dx_c = np.array([0.0, pixel_size, 0.0, pixel_size])
        dy_c = np.array([0.0, 0.0, pixel_size, pixel_size])
        pts_xc = px_x[:, None] + dx_c[None, :]
        pts_yc = px_y[:, None] + dy_c[None, :]

        query_c = np.zeros((num_pixels * 4, 3), dtype=np.float64)
        query_c[:, 0] = pts_xc.ravel()
        query_c[:, 1] = pts_yc.ravel()

        grid_query = pv.PolyData(query_c)
        sampled = grid_query.sample(_worker_mesh)

        sub_ux = sampled.point_data["disp_x"]
        sub_uy = sampled.point_data["disp_y"]
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        sub_ux[~valid] = 0.0
        sub_uy[~valid] = 0.0

        ux00 = sub_ux[0::4]
        ux10 = sub_ux[1::4]
        ux01 = sub_ux[2::4]
        ux11 = sub_ux[3::4]

        uy00 = sub_uy[0::4]
        uy10 = sub_uy[1::4]
        uy01 = sub_uy[2::4]
        uy11 = sub_uy[3::4]

        M11 = (-ux00 + ux10 - ux01 + ux11) / (4.0 * h)
        M12 = (-ux00 - ux10 + ux01 + ux11) / (4.0 * h)
        M21 = (-uy00 + uy10 - uy01 + uy11) / (4.0 * h)
        M22 = (-uy00 - uy10 + uy01 + uy11) / (4.0 * h)

        u0x = (ux00 + ux10 + ux01 + ux11) / 4.0
        u0y = (uy00 + uy10 + uy01 + uy11) / 4.0

        J11 = 1.0 - M11
        J12 = -M12
        J21 = -M21
        J22 = 1.0 - M22
    else:
        u0x = np.zeros(num_pixels)
        u0y = np.zeros(num_pixels)
        J11 = np.ones(num_pixels)
        J12 = np.zeros(num_pixels)
        J21 = np.zeros(num_pixels)
        J22 = np.ones(num_pixels)

    if method == "analytic":
        xc = px_x + h
        yc = px_y + h
        cx = xc - u0x
        cy = yc - u0y

        k = 2.0 * np.pi / p_phys

        def integrate_local_wave(
            wx: np.ndarray, wy: np.ndarray, phi: np.ndarray
        ) -> np.ndarray:
            eps = 1e-12
            cond_both = (np.abs(wx) > eps) & (np.abs(wy) > eps)
            cond_xonly = (np.abs(wx) > eps) & (np.abs(wy) <= eps)
            cond_yonly = (np.abs(wx) <= eps) & (np.abs(wy) > eps)
            cond_none = (np.abs(wx) <= eps) & (np.abs(wy) <= eps)

            val = np.zeros_like(wx)

            if np.any(cond_both):
                wxb = wx[cond_both]
                wyb = wy[cond_both]
                phib = phi[cond_both]
                factor = -1.0 / (wxb * wyb)
                c4 = factor * np.cos(wxb * h + wyb * h + phib)
                c3 = factor * np.cos(-wxb * h + wyb * h + phib)
                c2 = factor * np.cos(wxb * h - wyb * h + phib)
                c1 = factor * np.cos(-wxb * h - wyb * h + phib)
                val[cond_both] = c4 - c3 - c2 + c1

            if np.any(cond_xonly):
                wxx = wx[cond_xonly]
                phix = phi[cond_xonly]
                factor = h / wxx
                c4 = factor * np.sin(wxx * h + phix)
                c3 = factor * np.sin(-wxx * h + phix)
                c2 = -factor * np.sin(wxx * h + phix)
                c1 = -factor * np.sin(-wxx * h + phix)
                val[cond_xonly] = c4 - c3 - c2 + c1

            if np.any(cond_yonly):
                wyy = wy[cond_yonly]
                phiy = phi[cond_yonly]
                factor = h / wyy
                c4 = factor * np.sin(wyy * h + phiy)
                c3 = -factor * np.sin(wyy * h + phiy)
                c2 = factor * np.sin(-wyy * h + phiy)
                c1 = -factor * np.sin(-wyy * h + phiy)
                val[cond_yonly] = c4 - c3 - c2 + c1

            if np.any(cond_none):
                phin = phi[cond_none]
                val[cond_none] = 4.0 * h * h * np.cos(phin)

            return val

        # Term 1: Constant
        val_const = (I0 - 0.5 * GAMMA) * (4.0 * h * h)

        # Term 2: cos(k * x_ref)
        val_t2 = 0.5 * GAMMA * integrate_local_wave(k * J11, k * J12, k * cx)

        # Term 3: cos(k * y_ref)
        val_t3 = 0.5 * GAMMA * integrate_local_wave(k * J21, k * J22, k * cy)

        # Term 4: 0.25 * cos(k * (x_ref + y_ref))
        val_t4 = 0.25 * GAMMA * integrate_local_wave(
            k * (J11 + J21), k * (J12 + J22), k * (cx + cy)
        )

        # Term 5: 0.25 * cos(k * (x_ref - y_ref))
        val_t5 = 0.25 * GAMMA * integrate_local_wave(
            k * (J11 - J21), k * (J12 - J22), k * (cx - cy)
        )

        total_int = val_const + val_t2 + val_t3 + val_t4 + val_t5
        chunk_raw = total_int / (pixel_size * pixel_size)
    else:
        # Local coordinates relative to pixel center
        x_local = dx - h
        y_local = dy - h

        # Map to reference space using local affine parameters
        x_ref = (
            px_x[:, None]
            + h
            - u0x[:, None]
            + J11[:, None] * x_local[None, :]
            + J12[:, None] * y_local[None, :]
        )
        y_ref = (
            px_y[:, None]
            + h
            - u0y[:, None]
            + J21[:, None] * x_local[None, :]
            + J22[:, None] * y_local[None, :]
        )

        cos_x = np.cos(2.0 * np.pi * x_ref / p_phys)
        cos_y = np.cos(2.0 * np.pi * y_ref / p_phys)
        sub_intensity = (
            I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
        )

        chunk_raw = np.sum(sub_intensity * weights[None, :], axis=1)

    return start_idx, end_idx, chunk_raw


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
    tex_w_uv = camera_pixels + 2 * TEX_PX_PAD
    tex_h_uv = camera_pixels + 2 * TEX_PX_PAD
    px_bbox = (
        float(TEX_PX_PAD),
        float(TEX_PX_PAD),
        float(camera_pixels + TEX_PX_PAD),
        float(camera_pixels + TEX_PX_PAD),
    )
    import riley
    uvs = riley.project_uvs_planar_bbox(
        coords,
        (tex_w_uv, tex_h_uv),
        px_bbox,
        riley.ProjectionPlane.xy,
    )
    np.savetxt(uvs_path, uvs, delimiter=",")

    case_out_dir: Path = OUTPUT_DIR / case_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    p_phys: float = P_PIXELS * pixel_size

    for ff in range(num_frames):
        is_rect_base = (method == "rect" and param in [1, 2, 4])
        is_needed_frame = (ff in ACTIVE_FRAMES)
        if not (is_rect_base or is_needed_frame):
            continue

        if method == "rect" or method == "gauss":
            S = param * param
        elif method == "analytic":
            S = 1
        else:
            S = param

        # Limit memory usage by restricting integration points per chunk
        MAX_PTS_PER_CHUNK = 1000000
        pixels_per_chunk = max(1, MAX_PTS_PER_CHUNK // S)
        total_pixels = TARG_PX_X * TARG_PX_Y

        tasks = []
        for start_idx in range(0, total_pixels, pixels_per_chunk):
            end_idx = min(start_idx + pixels_per_chunk, total_pixels)
            tasks.append(
                (
                    start_idx,
                    end_idx,
                    method,
                    param,
                    pixel_size,
                    roi_size,
                    p_phys,
                )
            )

        if ff == 0:
            init_args = (None, None, None, None)
        else:
            init_args = (coords, connect, disp_x[:, ff], disp_y[:, ff])

        with Pool(
            processes=NUM_PROCESSES,
            initializer=init_worker,
            initargs=init_args,
        ) as pool:
            results = pool.map(process_pixel_chunk, tasks)

        pixel_raw_flat = np.zeros(total_pixels, dtype=np.float64)
        for start_idx, end_idx, chunk_raw in results:
            pixel_raw_flat[start_idx:end_idx] = chunk_raw
        pixel_raw = pixel_raw_flat.reshape(TARG_PX_Y, TARG_PX_X)

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
        cases = [Path("data") / name for name in DEFORMATION_CASES]

    print("\nGenerating reference textures...")
    tex_dir_ref = OUTPUT_DIR / "textures"
    if tex_dir_ref.exists():
        for p in tex_dir_ref.glob("*_int_rect_*"):
            p.unlink()

    for bb in BIT_DEPTHS:
        for oversamp in TEX_OVERSAMPLES:
            print(
                f"  Texture: analytic=0, bb={bb}, "
                f"oversamp={oversamp}"
            )
            generate_texture("analytic", 0, bb, oversamp)

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
