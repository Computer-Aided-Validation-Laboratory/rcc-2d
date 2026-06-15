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
