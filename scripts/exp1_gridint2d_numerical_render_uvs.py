# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import sys
import os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pyvista as pv
from PIL import Image
from script_timing import ScriptTimer, timed_call

from exp1common import (
    build_pv_mesh,
    compute_riley_bbox_uvs,
    get_riley_bbox_uv_transform,
    parse_case_params,
    output_case_name,
)
from exp1params import (
    BACKGROUND,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    BIT_DEPTHS,
    INTEGRATION_METHODS,
    P_PIXELS,
    I0,
    GAMMA,
    NUM_PROCESSES,
    AFFINE_MAX_POINTS_PER_CHUNK,
    VTK_MAX_POINTS_PER_CHUNK,
    NEWTON_MAX_POINTS_PER_CHUNK,
    mapping_mode_for_case,
    DEFORMATION_CASES,
    ACTIVE_FRAMES,
    FORCE_RENDER_OVER,
    exp1_output_dir,
)

OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs")
# INTEGRATION_METHODS = [
#     ("rect", 1),
#     ("rect", 2),
#     ("rect", 4),
#     ("rect", 8),
#     ("rect", 16),
#     # ("rect", 32),
#     # ("rect", 64),
#     # ("rect", 128),
#     # ("rect", 256),
#     # ("rect", 512),
#     # ("rect", 1024), TODO: check we have the RAM for this
#     ("gauss", 2),
#     ("gauss", 4),
#     ("gauss", 8),
#     ("gauss", 16),
#     # ("gauss", 32),
#     # ("gauss", 64),
#     # ("gauss", 128),
#     ("analytic", 0),
# ]

_worker_mesh = None
_worker_reference_coords = None
_worker_deformed_coords = None
_worker_connect = None
NUM_PROCESSES_RUN = int(
    os.environ.get("EXP1_NUM_PROCESSES", str(NUM_PROCESSES))
)


def max_points_per_chunk(mapping_mode: str) -> int:
    """Return the RAM-safe worker point cap for the selected map evaluator."""
    default = (
        VTK_MAX_POINTS_PER_CHUNK if mapping_mode == "vtk" else
        NEWTON_MAX_POINTS_PER_CHUNK if mapping_mode == "newton" else
        AFFINE_MAX_POINTS_PER_CHUNK
    )
    # The mode-specific override prevents a nonlinear run accidentally inheriting
    # the large affine cap.  The legacy override remains useful for either
    # mode when deliberately set for a controlled benchmark.
    value = os.environ.get(
        f"EXP1_{mapping_mode.upper()}_MAX_PTS_PER_CHUNK",
        os.environ.get("EXP1_MAX_PTS_PER_CHUNK", str(default)),
    )
    return max(1, int(value))


def get_active_frames() -> set[int]:
    """Allow narrow verification runs without editing exp1params.py."""
    frames_str = os.environ.get("EXP1_ACTIVE_FRAMES")
    if not frames_str:
        return set(ACTIVE_FRAMES)
    return {int(val.strip()) for val in frames_str.split(",") if val.strip()}


def get_integration_methods() -> list[tuple[str, int]]:
    """Allow narrow verification runs without editing exp1params.py."""
    methods_str = os.environ.get("EXP1_METHODS")
    if not methods_str:
        return INTEGRATION_METHODS
    methods: list[tuple[str, int]] = []
    for item in methods_str.split(","):
        method, param = item.split(":")
        methods.append((method.strip(), int(param.strip())))
    return methods


def init_worker(coords, connect, disp_x_ff, disp_y_ff, mapping_mode: str | None = None) -> None:
    """Initialize worker process with mesh and displacement field."""
    global _worker_mesh, _worker_reference_coords, _worker_deformed_coords, _worker_connect
    if coords is not None and connect is not None:
        coords_def = np.array(coords, copy=True)
        coords_def[:, 0] += disp_x_ff
        coords_def[:, 1] += disp_y_ff
        _worker_reference_coords = coords
        _worker_deformed_coords = coords_def
        _worker_connect = connect
        if mapping_mode == "newton":
            _worker_mesh = None
        else:
            _worker_mesh = build_pv_mesh(coords_def, connect)
            _worker_mesh.point_data["x_ref"] = coords[:, 0]
            _worker_mesh.point_data["y_ref"] = coords[:, 1]
    else:
        _worker_mesh = None
        _worker_reference_coords = None
        _worker_deformed_coords = None
        _worker_connect = None


def process_pixel_chunk(args) -> tuple[int, int, np.ndarray]:
    """Process a chunk of pixels in a worker process."""
    (
        start_idx,
        end_idx,
        method,
        param,
        pixel_size,
        roi_size,
        p_phys,
        uv_scale,
        u_offset,
        v_offset,
        mapping_mode,
    ) = args

    pixel_indices = np.arange(start_idx, end_idx)
    num_pixels = len(pixel_indices)

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

    # Map deformed quadrature points to reference coordinates.  The affine
    # mode is an exact fast path only for global-affine deformation; ``newton``
    # evaluates every requested point through the exact Quad9 inverse map.
    h = 0.5 * pixel_size
    pixel_valid = np.ones(num_pixels, dtype=bool)
    global _worker_mesh, _worker_reference_coords, _worker_deformed_coords, _worker_connect
    if mapping_mode == "newton" and _worker_deformed_coords is not None and method != "analytic":
        from quad9_newton import inverse_map_quad9

        x_ref_flat, y_ref_flat, valid = inverse_map_quad9(
            (px_x[:, None] + dx[None, :]).ravel(),
            (px_y[:, None] + dy[None, :]).ravel(),
            _worker_deformed_coords,
            _worker_reference_coords,
            _worker_connect,
        )
        x_ref_samples = x_ref_flat.reshape(num_pixels, len(dx))
        y_ref_samples = y_ref_flat.reshape(num_pixels, len(dx))
        valid_samples = valid.reshape(num_pixels, len(dx))
        pixel_valid = valid_samples.all(axis=1)
    elif _worker_mesh is not None and mapping_mode == "vtk" and method != "analytic":
        query = np.zeros((num_pixels * len(dx), 3), dtype=np.float64)
        query[:, 0] = (px_x[:, None] + dx[None, :]).ravel()
        query[:, 1] = (px_y[:, None] + dy[None, :]).ravel()
        sampled = pv.PolyData(query).sample(_worker_mesh)
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        x_ref_samples = sampled.point_data["x_ref"].reshape(num_pixels, len(dx))
        y_ref_samples = sampled.point_data["y_ref"].reshape(num_pixels, len(dx))
        valid_samples = valid.reshape(num_pixels, len(dx))
        pixel_valid = valid_samples.all(axis=1)
    elif _worker_mesh is not None:
        dx_c = np.array([0.0, pixel_size, 0.0, pixel_size])
        dy_c = np.array([0.0, 0.0, pixel_size, pixel_size])
        pts_xc = px_x[:, None] + dx_c[None, :]
        pts_yc = px_y[:, None] + dy_c[None, :]

        query_c = np.zeros((num_pixels * 4, 3), dtype=np.float64)
        query_c[:, 0] = pts_xc.ravel()
        query_c[:, 1] = pts_yc.ravel()

        grid_query = pv.PolyData(query_c)
        sampled = grid_query.sample(_worker_mesh)

        sub_x_ref = sampled.point_data["x_ref"]
        sub_y_ref = sampled.point_data["y_ref"]
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        pixel_valid = valid.reshape(num_pixels, 4).all(axis=1)
        sub_x_ref[~valid] = 0.0
        sub_y_ref[~valid] = 0.0

        x00 = sub_x_ref[0::4]
        x10 = sub_x_ref[1::4]
        x01 = sub_x_ref[2::4]
        x11 = sub_x_ref[3::4]

        y00 = sub_y_ref[0::4]
        y10 = sub_y_ref[1::4]
        y01 = sub_y_ref[2::4]
        y11 = sub_y_ref[3::4]

        A11 = (-x00 + x10 - x01 + x11) / (4.0 * h)
        A12 = (-x00 - x10 + x01 + x11) / (4.0 * h)
        A21 = (-y00 + y10 - y01 + y11) / (4.0 * h)
        A22 = (-y00 - y10 + y01 + y11) / (4.0 * h)

        x0_ref = (x00 + x10 + x01 + x11) / 4.0
        y0_ref = (y00 + y10 + y01 + y11) / 4.0
    else:
        x0_ref = px_x + h
        y0_ref = px_y + h
        A11 = np.ones(num_pixels)
        A12 = np.zeros(num_pixels)
        A21 = np.zeros(num_pixels)
        A22 = np.ones(num_pixels)

    if method == "analytic":
        if mapping_mode in {"vtk", "newton"} and _worker_mesh is not None:
            raise ValueError(
                "The closed-form analytic rule assumes an affine inverse map; "
                "use rect/gauss numerical integration for nonlinear mapping."
            )
        pitch_uv = uv_scale * p_phys
        k_uv = 2.0 * np.pi / pitch_uv

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

        # Term 2: cos(2pi * u_ref / pitch_uv)
        val_t2 = 0.5 * GAMMA * integrate_local_wave(
            k_uv * uv_scale * A11,
            k_uv * uv_scale * A12,
            k_uv * uv_scale * x0_ref,
        )

        # Term 3: cos(2pi * v_ref / pitch_uv)
        val_t3 = 0.5 * GAMMA * integrate_local_wave(
            -k_uv * uv_scale * A21,
            -k_uv * uv_scale * A22,
            -k_uv * uv_scale * y0_ref,
        )

        # Term 4: 0.25 * cos(2pi * (u_ref + v_ref) / pitch_uv)
        val_t4 = 0.25 * GAMMA * integrate_local_wave(
            k_uv * uv_scale * (A11 - A21),
            k_uv * uv_scale * (A12 - A22),
            k_uv * uv_scale * (x0_ref - y0_ref),
        )

        # Term 5: 0.25 * cos(2pi * (u_ref - v_ref) / pitch_uv)
        val_t5 = 0.25 * GAMMA * integrate_local_wave(
            k_uv * uv_scale * (A11 + A21),
            k_uv * uv_scale * (A12 + A22),
            k_uv * uv_scale * (x0_ref + y0_ref),
        )

        total_int = val_const + val_t2 + val_t3 + val_t4 + val_t5
        chunk_raw = total_int / (pixel_size * pixel_size)
    elif (mapping_mode == "newton" and _worker_deformed_coords is not None) or (
        mapping_mode == "vtk" and _worker_mesh is not None
    ):
        u_ref = uv_scale * x_ref_samples
        v_ref = -uv_scale * y_ref_samples
        pitch_uv = uv_scale * p_phys
        sub_intensity = (
            I0
            + 0.5 * GAMMA * (1.0 + np.cos(2.0 * np.pi * u_ref / pitch_uv))
            * (1.0 + np.cos(2.0 * np.pi * v_ref / pitch_uv))
            - GAMMA
        )
        sub_intensity[~valid_samples] = BACKGROUND
        chunk_raw = np.sum(sub_intensity * weights[None, :], axis=1)
    else:
        x_local = dx - h
        y_local = dy - h

        x_ref = (
            x0_ref[:, None]
            + A11[:, None] * x_local[None, :]
            + A12[:, None] * y_local[None, :]
        )
        y_ref = (
            y0_ref[:, None]
            + A21[:, None] * x_local[None, :]
            + A22[:, None] * y_local[None, :]
        )

        u_ref = uv_scale * x_ref
        v_ref = -uv_scale * y_ref
        pitch_uv = uv_scale * p_phys
        cos_x = np.cos(2.0 * np.pi * u_ref / pitch_uv)
        cos_y = np.cos(2.0 * np.pi * v_ref / pitch_uv)
        sub_intensity = (
            I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
        )

        chunk_raw = np.sum(sub_intensity * weights[None, :], axis=1)

    chunk_raw[~pixel_valid] = BACKGROUND
    return start_idx, end_idx, chunk_raw


def generate_grid_images(case_dir: Path, method: str, param: int) -> None:
    """Load mesh, interpolate displacements, and generate target images."""
    timer = ScriptTimer(__file__)
    case_name: str = output_case_name(case_dir.name, TARG_PX_X)
    mapping_mode = os.environ.get(
        "EXP1_MAPPING_MODE", mapping_mode_for_case(case_dir.name)
    )
    if mapping_mode not in {"affine", "vtk", "newton"}:
        raise ValueError(f"Unsupported EXP1 mapping mode: {mapping_mode!r}")
    print(f"\nProcessing: {case_name} ({method}={param}, mapping={mapping_mode})")

    coords: np.ndarray = np.loadtxt(case_dir / "coords.csv", delimiter=",")
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

    _case_camera_pixels, roi_size = parse_case_params(case_dir)
    if TARG_PX_X != TARG_PX_Y:
        raise ValueError("Experiment 1 currently requires square target dimensions.")
    pixel_size: float = roi_size / TARG_PX_X
    num_frames: int = disp_x.shape[1]

    uvs = compute_riley_bbox_uvs(coords, TARG_PX_X, TEX_PX_PAD)
    np.savetxt(case_dir / "uvs_exp1_sin_grid_uvs.csv", uvs, delimiter=",")

    case_out_dir: Path = OUTPUT_DIR / case_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    p_phys: float = P_PIXELS * pixel_size
    uv_scale, u_offset, v_offset = get_riley_bbox_uv_transform(
        coords, TARG_PX_X, TEX_PX_PAD
    )
    active_frames = get_active_frames()

    for ff in range(num_frames):
        is_rect_base = method == "rect" and param in [1, 2, 4]
        is_needed_frame = ff in active_frames
        if not (is_rect_base or is_needed_frame):
            continue

        p_val: int = max(TARG_PX_X, TARG_PX_Y)
        expected_outputs = [
            (
                case_out_dir
                / f"targ_px{p_val}_int_{method}_param_{param}_b{bb}_frame{ff:02d}.tiff",
                case_out_dir
                / f"targ_px{p_val}_int_{method}_param_{param}_b{bb}_frame{ff:02d}.npy",
            )
            for bb in BIT_DEPTHS
        ]
        if not FORCE_RENDER_OVER and all(
            tiff_path.exists() and npy_path.exists()
            for tiff_path, npy_path in expected_outputs
        ):
            print(f"    frame {ff:02d}: outputs exist; skipping.")
            continue

        if method == "rect" or method == "gauss":
            S = param * param
        elif method == "analytic":
            S = 1
        else:
            S = param

        point_cap = max_points_per_chunk(mapping_mode)
        pixels_per_chunk = max(1, point_cap // S)
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
                    uv_scale,
                    u_offset,
                    v_offset,
                    mapping_mode,
                )
            )

        print(
            f"    frame {ff:02d}/{num_frames - 1:02d}: rendering "
            f"{method} param={param} ({len(tasks):,} chunk"
            f"{'s' if len(tasks) != 1 else ''}; "
            f"worker point cap={point_cap:,}).",
            flush=True,
        )

        if ff == 0:
            init_args = (None, None, None, None, mapping_mode)
        else:
            init_args = (coords, connect, disp_x[:, ff], disp_y[:, ff], mapping_mode)

        with Pool(
            processes=NUM_PROCESSES_RUN,
            initializer=init_worker,
            initargs=init_args,
        ) as pool:
            results = timed_call(
                timer, f"{case_dir.name}_int_{method}_param_{param}_frame{ff:02d}",
                pool.map, process_pixel_chunk, tasks,
            )

        pixel_raw_flat = np.zeros(total_pixels, dtype=np.float64)
        for start_idx, end_idx, chunk_raw in results:
            pixel_raw_flat[start_idx:end_idx] = chunk_raw
        pixel_raw = pixel_raw_flat.reshape(TARG_PX_Y, TARG_PX_X)

        pixel_raw_flipped: np.ndarray = np.flipud(pixel_raw)

        for bb in BIT_DEPTHS:
            max_val_bb: float = float(2**bb - 1)
            pixel_bb: np.ndarray = np.round(pixel_raw_flipped * max_val_bb)
            pixel_bb = np.clip(pixel_bb, 0.0, max_val_bb)

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
    print("Experiment 1: Custom Renderer Numerical Integration (UVs)")
    print(80 * "=")

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / name for name in DEFORMATION_CASES]

    for case_path in cases:
        if not case_path.exists():
            print(
                f"Warning: Case directory {case_path} "
                "does not exist. Skipping."
            )
            continue
        mapping_mode = os.environ.get(
            "EXP1_MAPPING_MODE", mapping_mode_for_case(case_path.name)
        )
        for method, param in get_integration_methods():
            # The analytic eggbox integral is closed form only after an affine
            # inverse map.  A nonlinear-mapped element (e.g. quadratic saddle) must
            # therefore use one of the numerical quadrature rules instead.
            if method == "analytic" and mapping_mode in {"vtk", "newton"}:
                print(
                    f"Skipping {case_path.name} analytic integration: "
                    f"mapping={mapping_mode} requires numerical rect/gauss quadrature."
                )
                continue
            generate_grid_images(case_path, method, param)

    print("\nAll numerical integrations completed successfully!")


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
