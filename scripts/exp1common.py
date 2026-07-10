# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple

import numpy as np
import pyvista as pv


def get_pv_cell_type(nodes_per_elem: int) -> pv.CellType:
    """Determine PyVista cell type from connectivity width."""
    if nodes_per_elem == 3:
        return pv.CellType.TRIANGLE
    if nodes_per_elem == 6:
        return pv.CellType.QUADRATIC_TRIANGLE
    if nodes_per_elem == 4:
        return pv.CellType.QUAD
    if nodes_per_elem == 8:
        return pv.CellType.QUADRATIC_QUAD
    if nodes_per_elem == 9:
        return pv.CellType.BIQUADRATIC_QUAD
    raise ValueError(
        f"Unsupported element type with {nodes_per_elem} nodes."
    )


def build_pv_mesh(
    coords: np.ndarray,
    connect: np.ndarray,
) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid mesh from coords and connect."""
    num_elems, nodes_per_elem = connect.shape
    cell_type = get_pv_cell_type(nodes_per_elem)

    cells = np.hstack(
        [np.full((num_elems, 1), nodes_per_elem), connect]
    ).ravel()
    cell_types = np.full(num_elems, cell_type, dtype=np.uint8)

    mesh = pv.UnstructuredGrid(cells, cell_types, coords)
    return mesh


def parse_case_params(case_dir: Path) -> Tuple[int, float]:
    """Parse resolution and ROI scale from directory name."""
    folder_name = case_dir.name
    parts = folder_name.split("_")
    camera_pixels = 256
    for pp in parts:
        if pp.startswith("cam"):
            try:
                camera_pixels = int(pp.replace("cam", ""))
            except ValueError:
                pass
    # For square plates, the camera FOV (ROI) size equals camera_pixels
    roi_size = float(camera_pixels)
    return camera_pixels, roi_size


def compute_padded_uvs(
    coords: np.ndarray,
    roi_size: float,
    camera_pixels: int,
    pad: int,
) -> np.ndarray:
    """Compute normalized UV coordinates incorporating padding."""
    uvs: np.ndarray = np.zeros((len(coords), 2), dtype=np.float64)
    half_roi: float = roi_size / 2.0
    u_unpadded: np.ndarray = (coords[:, 0] + half_roi) / roi_size
    v_unpadded: np.ndarray = (coords[:, 1] + half_roi) / roi_size
    uvs[:, 0] = (
        u_unpadded * camera_pixels + pad
    ) / (camera_pixels + 2 * pad)
    uvs[:, 1] = (
        v_unpadded * camera_pixels + pad
    ) / (camera_pixels + 2 * pad)
    return uvs


def compute_riley_bbox_uvs(
    coords: np.ndarray,
    camera_pixels: int,
    pad: int,
) -> np.ndarray:
    """Match Riley's planar bbox UV projection for the xy plane."""
    x_proj = coords[:, 0]
    y_proj = coords[:, 1]

    x_min = np.min(x_proj)
    x_max = np.max(x_proj)
    y_min = np.min(y_proj)
    y_max = np.max(y_proj)

    mesh_w = x_max - x_min
    mesh_h = y_max - y_min
    px_x_l = float(pad)
    px_y_l = float(pad)
    px_x_u = float(camera_pixels + pad)
    px_y_u = float(camera_pixels + pad)
    px_w = px_x_u - px_x_l
    px_h = px_y_u - px_y_l

    scale_x = px_w / mesh_w if mesh_w > 0.0 else 1.0
    scale_y = px_h / mesh_h if mesh_h > 0.0 else 1.0
    scale = 0.5 * (scale_x + scale_y)

    mesh_cx = 0.5 * (x_min + x_max)
    mesh_cy = 0.5 * (y_min + y_max)
    px_cx = 0.5 * (px_x_l + px_x_u)
    px_cy = 0.5 * (px_y_l + px_y_u)

    px_x = px_cx + (x_proj - mesh_cx) * scale
    px_y = px_cy + (y_proj - mesh_cy) * scale

    tex_w = float(camera_pixels + 2 * pad)
    tex_h = float(camera_pixels + 2 * pad)
    uvs = np.zeros((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = px_x / (tex_w - 1.0)
    uvs[:, 1] = 1.0 - (px_y / (tex_h - 1.0))
    return np.ascontiguousarray(uvs, dtype=np.float64)


def get_riley_bbox_uv_transform(
    coords: np.ndarray,
    camera_pixels: int,
    pad: int,
) -> tuple[float, float, float]:
    """Return affine terms for Riley bbox UVs on the xy plane.

    For reference world coordinates x_ref, y_ref:
    u = uv_scale * x_ref + u_offset
    v = -uv_scale * y_ref + v_offset
    """
    x_proj = coords[:, 0]
    y_proj = coords[:, 1]

    x_min = np.min(x_proj)
    x_max = np.max(x_proj)
    y_min = np.min(y_proj)
    y_max = np.max(y_proj)

    mesh_w = x_max - x_min
    mesh_h = y_max - y_min
    px_x_l = float(pad)
    px_y_l = float(pad)
    px_x_u = float(camera_pixels + pad)
    px_y_u = float(camera_pixels + pad)
    px_w = px_x_u - px_x_l
    px_h = px_y_u - px_y_l

    scale_x = px_w / mesh_w if mesh_w > 0.0 else 1.0
    scale_y = px_h / mesh_h if mesh_h > 0.0 else 1.0
    scale = 0.5 * (scale_x + scale_y)

    mesh_cx = 0.5 * (x_min + x_max)
    mesh_cy = 0.5 * (y_min + y_max)
    px_cx = 0.5 * (px_x_l + px_x_u)
    px_cy = 0.5 * (px_y_l + px_y_u)

    tex_w = float(camera_pixels + 2 * pad)
    tex_h = float(camera_pixels + 2 * pad)
    uv_scale = scale / (tex_w - 1.0)
    u_offset = (px_cx - mesh_cx * scale) / (tex_w - 1.0)
    v_offset = 1.0 - (px_cy - mesh_cy * scale) / (tex_h - 1.0)
    return uv_scale, u_offset, v_offset


def get_integration_points_2d(
    method: str,
    param: int,
    start_x: float,
    start_y: float,
    pixel_size: float,
    num_px_x: int,
    num_px_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D integration points and weights for a grid of pixels."""
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
    else:
        raise ValueError(f"Unknown integration method: {method}")

    # Generate grid of pixel corners
    px_x = start_x + np.arange(num_px_x) * pixel_size
    px_y = start_y + np.arange(num_px_y) * pixel_size
    grid_px_x, grid_px_y = np.meshgrid(px_x, px_y)

    pts_x = grid_px_x[:, :, None] + dx[None, None, :]
    pts_y = grid_px_y[:, :, None] + dy[None, None, :]

    return pts_x, pts_y, weights


def evaluate_eggbox_analytic_average(
    start_x: float,
    start_y: float,
    pixel_size: float,
    num_px_x: int,
    num_px_y: int,
    p_phys: float,
    i0: float,
    gamma: float,
) -> np.ndarray:
    """Compute the true analytic average over the pixel box for eggbox."""
    k = 2.0 * np.pi / p_phys

    px_x = start_x + np.arange(num_px_x + 1) * pixel_size
    px_y = start_y + np.arange(num_px_y + 1) * pixel_size
    grid_x, grid_y = np.meshgrid(px_x, px_y)

    term1 = (i0 - 0.5 * gamma) * grid_x * grid_y
    term2 = (gamma / (2.0 * k)) * grid_y * np.sin(k * grid_x)
    term3 = (gamma / (2.0 * k)) * grid_x * np.sin(k * grid_y)
    term4 = (
        (gamma / (2.0 * k**2))
        * np.sin(k * grid_x)
        * np.sin(k * grid_y)
    )
    a_val = term1 + term2 + term3 + term4

    int_val = (
        a_val[1:, 1:] - a_val[1:, :-1] - a_val[:-1, 1:] + a_val[:-1, :-1]
    )
    return int_val / (pixel_size * pixel_size)
