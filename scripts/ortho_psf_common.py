"""High-resolution bespoke orthographic rendering for image-plane PSFs."""

from __future__ import annotations

import multiprocessing as mp
import os
import numpy as np
import pyvista as pv
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve

from psf_render_common import gaussian_kernel_1d, psf_radius_subpixels

_BAND_CONTEXT: dict[str, object] | None = None


def _horizontal_filter(raw: np.ndarray, kernel: np.ndarray, background: float) -> np.ndarray:
    """Filter independent rows; use FFT once the sampled support is large."""
    if kernel.size <= 257:
        return convolve1d(raw, kernel, axis=1, mode="constant", cval=background)
    # The outer support columns are already background, so zero-pad only after
    # subtracting it; this is the same constant-boundary convolution.
    return fftconvolve(raw - background, kernel[None, :], mode="same", axes=1) + background


def _render_horizontal_band(task: tuple[int, int, int]) -> tuple[int, int, np.ndarray]:
    """Fork-worker: shade one row band, filter horizontally, and x-reduce."""
    if _BAND_CONTEXT is None:
        raise RuntimeError("PSF worker context was not initialised.")
    c = _BAND_CONTEXT
    frame, row_start, row_stop = task
    raw_h, raw_w = c["raw_shape"]  # type: ignore[index]
    radius, ssaa = c["radius"], c["ssaa"]  # type: ignore[index]
    x = c["x"]  # type: ignore[index]
    y = c["y"]  # type: ignore[index]
    background = c["background"]  # type: ignore[index]
    xx, yy = np.meshgrid(x, y[row_start:row_stop])
    # The scratch raster includes the full PSF support outside the final image.
    # Map and shade those samples too; only crop after PSF convolution.
    xr, yr, valid = _map_reference(xx.ravel(), yy.ravel(), c["coords"], c["connect"], c["disp_x"], c["disp_y"], frame, c["mapping_mode"])  # type: ignore[index]
    values = np.full(xr.size, c["invalid_value"], dtype=np.float64)  # type: ignore[index]
    if np.any(valid): values[valid] = c["evaluate_reference"](xr[valid], yr[valid])  # type: ignore[index,operator]
    raw = values.reshape(row_stop-row_start, raw_w)
    filtered = _horizontal_filter(raw, c["kernel"], background)  # type: ignore[index]
    width = c["final_shape"][1]  # type: ignore[index]
    core = filtered[:, radius:radius + width * ssaa]
    return row_start, row_stop, core.reshape(row_stop-row_start, width, ssaa).mean(axis=2)


def _map_reference(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    connect: np.ndarray,
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    frame: int,
    mapping_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target-camera samples to the reference element configuration."""
    if frame == 0:
        valid = (
            (x >= np.min(coords[:, 0])) & (x <= np.max(coords[:, 0])) &
            (y >= np.min(coords[:, 1])) & (y <= np.max(coords[:, 1]))
        )
        return x, y, valid
    coords_def = np.array(coords, copy=True)
    coords_def[:, 0] += disp_x[:, frame]
    coords_def[:, 1] += disp_y[:, frame]
    if mapping_mode == "affine":
        # Exact for the manufactured rigid/global-affine fields.
        coeff, *_ = np.linalg.lstsq(
            np.column_stack((coords_def[:, :2], np.ones(coords_def.shape[0]))),
            coords[:, :2], rcond=None,
        )
        mapped = np.column_stack((x, y, np.ones(x.size))) @ coeff
        valid = (
            (mapped[:, 0] >= np.min(coords[:, 0])) &
            (mapped[:, 0] <= np.max(coords[:, 0])) &
            (mapped[:, 1] >= np.min(coords[:, 1])) &
            (mapped[:, 1] <= np.max(coords[:, 1]))
        )
        return mapped[:, 0], mapped[:, 1], valid
    if mapping_mode == "newton":
        from quad9_newton import inverse_map_quad9
        xr, yr, valid = inverse_map_quad9(x, y, coords_def, coords, connect)
        return xr, yr, valid
    if mapping_mode == "vtk":
        from exp1common import build_pv_mesh
        mesh = build_pv_mesh(coords_def, connect)
        mesh.point_data["x_ref"] = coords[:, 0]
        mesh.point_data["y_ref"] = coords[:, 1]
        query = np.zeros((x.size, 3), dtype=np.float64)
        query[:, 0] = x
        query[:, 1] = y
        sampled = pv.PolyData(query).sample(mesh)
        return (
            np.asarray(sampled.point_data["x_ref"]),
            np.asarray(sampled.point_data["y_ref"]),
            np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool),
        )
    raise ValueError(f"Unsupported mapping mode {mapping_mode!r}.")


def render_psf_frame(
    *,
    evaluate_reference,
    invalid_value: float,
    roi_size: float,
    image_shape: tuple[int, int],
    ssaa: int,
    sigma_px: float,
    support_radius_px: float,
    background: float,
    coords: np.ndarray,
    connect: np.ndarray,
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    frame: int,
    mapping_mode: str,
    max_points_per_batch: int = 250_000,
    processes: int | None = None,
) -> np.ndarray:
    """Shade, filter, and pixel-integrate one camera frame.

    The evaluator receives flattened reference coordinates and returns the raw
    field to be filtered (intensity for Exp1, unclamped coverage for Exp2).
    """
    height, width = image_shape
    if height != width:
        raise ValueError("PSF renderer currently requires square final images.")
    radius = psf_radius_subpixels(ssaa, support_radius_px)
    raw_h = height * ssaa + 2 * radius
    raw_w = width * ssaa + 2 * radius
    pixel_size = roi_size / width
    x = -0.5 * roi_size + ((np.arange(raw_w) - radius + 0.5) / ssaa) * pixel_size
    y = -0.5 * roi_size + ((np.arange(raw_h) - radius + 0.5) / ssaa) * pixel_size
    kernel = gaussian_kernel_1d(ssaa, sigma_px, support_radius_px)
    # Keep only high-resolution rows by final-image columns after the first
    # separable pass.  This is exact and avoids a full H*SSAA by W*SSAA array.
    horizontal = np.empty((raw_h, width), dtype=np.float64)
    rows_per_band = max(1, max_points_per_batch // raw_w)
    tasks = [(frame, start, min(start + rows_per_band, raw_h)) for start in range(0, raw_h, rows_per_band)]
    global _BAND_CONTEXT
    _BAND_CONTEXT = {
        "raw_shape": (raw_h, raw_w), "radius": radius, "ssaa": ssaa,
        "x": x, "y": y, "background": background, "invalid_value": invalid_value,
        "kernel": kernel, "final_shape": image_shape, "evaluate_reference": evaluate_reference,
        "coords": coords, "connect": connect, "disp_x": disp_x, "disp_y": disp_y,
        "mapping_mode": mapping_mode,
    }
    workers = max(1, int(os.environ.get("ORTHO_PSF_PROCESSES", str(processes or 1))))
    # Linux fork keeps immutable geometry/pattern arrays resident without
    # pickling them into every task.  Fall back to serial where fork is absent.
    if workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(processes=workers) as pool:
            # Each task returns only a reduced (rows x final-width) array.  A
            # moderate map chunk amortises IPC without increasing peak RAM.
            chunksize = max(1, len(tasks) // (workers * 8))
            results = pool.map(_render_horizontal_band, tasks, chunksize=chunksize)
    else:
        results = [_render_horizontal_band(task) for task in tasks]
    for row_start, row_stop, reduced in results:
        horizontal[row_start:row_stop] = reduced
    _BAND_CONTEXT = None
    # Vertical pass operates on just ``width`` columns.  Direct convolution is
    # now cheap even at high SSAA; it remains the exact sampled Riley kernel.
    vertical = convolve1d(horizontal, kernel, axis=0, mode="constant", cval=background)
    centre = vertical[radius:radius + height * ssaa]
    return centre.reshape(height, ssaa, width).mean(axis=1)
