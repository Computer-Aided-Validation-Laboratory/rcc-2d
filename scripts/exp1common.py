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
