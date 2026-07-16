# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Shared deterministic speckle generation and gridint2d rendering tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from exp2params import (
    BIT_DEPTHS,
    GAMMA,
    I0,
    NUM_PROCESSES,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
)

MAX_PTS_PER_CHUNK = int(
    os.environ.get("EXP2_MAX_PTS_PER_CHUNK", "2000000")
)
NUM_PROCESSES_RUN = int(
    os.environ.get("EXP2_NUM_PROCESSES", str(NUM_PROCESSES))
)
_worker_mesh: Any = None
_worker_pattern: SpecklePattern | None = None


def _circle_primitive(x: np.ndarray, radius: float) -> np.ndarray:
    """Antiderivative of ``sqrt(radius**2 - x**2)`` on the circle."""
    clipped = np.clip(x, -radius, radius)
    return 0.5 * (
        clipped * np.sqrt(np.maximum(radius * radius - clipped * clipped, 0.0))
        + radius * radius * np.arcsin(clipped / radius)
    )


def _disk_box_area(
    x0: np.ndarray,
    y0: np.ndarray,
    width: float,
    height: float,
    radius: float,
) -> np.ndarray:
    """Return exact circle--rectangle areas for translated boxes."""
    left = x0
    right = x0 + width
    bottom = y0
    top = y0 + height
    lo = np.maximum(left, -radius)
    hi = np.minimum(right, radius)
    valid = hi > lo
    roots = []
    for edge in (bottom, top):
        root = np.sqrt(np.maximum(radius * radius - edge * edge, 0.0))
        roots.extend((-root, root))
    points = np.stack(
        [lo, hi, *[np.clip(root, lo, hi) for root in roots]],
        axis=1,
    )
    points.sort(axis=1)
    area = np.zeros_like(x0)

    for index in range(points.shape[1] - 1):
        start = points[:, index]
        end = points[:, index + 1]
        mid = 0.5 * (start + end)
        half_height = np.sqrt(
            np.maximum(radius * radius - mid * mid, 0.0)
        )
        upper_is_arc = half_height < top
        lower_is_arc = -half_height > bottom
        primitive = _circle_primitive(end, radius) - _circle_primitive(
            start,
            radius,
        )
        integral = np.where(
            upper_is_arc & lower_is_arc,
            2.0 * primitive,
            np.where(
                upper_is_arc,
                primitive - bottom * (end - start),
                np.where(
                    lower_is_arc,
                    top * (end - start) + primitive,
                    (top - bottom) * (end - start),
                ),
            ),
        )
        overlap = np.minimum(top, half_height) - np.maximum(
            bottom,
            -half_height,
        )
        area += np.where((overlap > 0.0) & valid, integral, 0.0)

    return area


@dataclass(frozen=True)
class SpecklePattern:
    """A finite, jittered lattice of disk and Gaussian speckle centres."""

    pattern_type: str
    pitch: float
    equivalent_diameter: float
    centers: np.ndarray
    grid_shape: tuple[int, int]
    lattice_origin: tuple[float, float]
    max_jitter: float
    cutoff_sigmas: float
    i0: float
    gamma: float

    @property
    def radius(self) -> float:
        return 0.5 * self.equivalent_diameter

    @property
    def sigma(self) -> float:
        return self.radius / self.cutoff_sigmas

    @property
    def support_radius(self) -> float:
        if self.pattern_type == "gausscont":
            # Tail is below 1.3e-14 at eight standard deviations.
            return 8.0 * self.sigma
        return self.radius

    def evaluate_coverage(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate additive blob coverage using nearby lattice cells."""
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        ny, nx = self.grid_shape
        centres_grid = self.centers.reshape(ny, nx, 2)
        origin_x, origin_y = self.lattice_origin

        # Search enough neighbouring cells for the finite support and jitter.
        reach = int(
            np.ceil((self.support_radius + self.max_jitter) / self.pitch)
        )
        base_ix = np.rint((x_flat - origin_x) / self.pitch).astype(np.int64)
        base_iy = np.rint((y_flat - origin_y) / self.pitch).astype(np.int64)
        coverage = np.zeros(x_flat.size, dtype=np.float64)

        for oy in range(-reach, reach + 1):
            iy = base_iy + oy
            valid_y = (iy >= 0) & (iy < ny)
            for ox in range(-reach, reach + 1):
                ix = base_ix + ox
                valid = valid_y & (ix >= 0) & (ix < nx)
                if not np.any(valid):
                    continue
                centers = centres_grid[iy[valid], ix[valid]]
                dx = x_flat[valid] - centers[:, 0]
                dy = y_flat[valid] - centers[:, 1]
                r2 = dx * dx + dy * dy
                if self.pattern_type in {"disk", "diskaddsat"}:
                    coverage[valid] += (r2 <= self.radius * self.radius)
                else:
                    within_support = r2 <= self.support_radius**2
                    contribution = np.zeros_like(r2)
                    contribution[within_support] = np.exp(
                        -0.5 * r2[within_support] / (self.sigma * self.sigma)
                    )
                    coverage[valid] += contribution

        return coverage.reshape(np.shape(x))

    def intensity_from_coverage(
        self,
        coverage: np.ndarray,
    ) -> np.ndarray:
        """Clamp coverage and map black-to-white coverage to intensity."""
        raw = 1.0 - np.clip(coverage, 0.0, 1.0)
        # Match Exp 1: GAMMA is the half-range around I0, not full contrast.
        intensity = np.clip(
            self.i0 + self.gamma * (2.0 * raw - 1.0), 0.0, 1.0
        )
        return intensity

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the pointwise, clamped intensity field."""
        return self.intensity_from_coverage(self.evaluate_coverage(x, y))

    def evaluate_gausscont_box_average(
        self,
        start_x: np.ndarray,
        start_y: np.ndarray,
        width: float,
        height: float,
    ) -> np.ndarray:
        """Integrate untruncated Gaussian coverage over axis-aligned boxes."""
        if self.pattern_type != "gausscont":
            raise ValueError("Analytic box integration requires gausscont.")

        from scipy.special import erf

        x_flat = np.asarray(start_x, dtype=np.float64).ravel()
        y_flat = np.asarray(start_y, dtype=np.float64).ravel()
        ny, nx = self.grid_shape
        centres_grid = self.centers.reshape(ny, nx, 2)
        origin_x, origin_y = self.lattice_origin
        reach = int(
            np.ceil((self.support_radius + self.max_jitter) / self.pitch)
        )
        base_ix = np.rint((x_flat - origin_x) / self.pitch).astype(np.int64)
        base_iy = np.rint((y_flat - origin_y) / self.pitch).astype(np.int64)
        coverage = np.zeros(x_flat.size, dtype=np.float64)
        scale = np.sqrt(2.0) * self.sigma
        factor = self.sigma * np.sqrt(np.pi / 2.0)

        for oy in range(-reach, reach + 1):
            iy = base_iy + oy
            valid_y = (iy >= 0) & (iy < ny)
            for ox in range(-reach, reach + 1):
                ix = base_ix + ox
                valid = valid_y & (ix >= 0) & (ix < nx)
                if not np.any(valid):
                    continue
                centers = centres_grid[iy[valid], ix[valid]]
                x0 = x_flat[valid]
                y0 = y_flat[valid]
                int_x = factor * (
                    erf((x0 + width - centers[:, 0]) / scale)
                    - erf((x0 - centers[:, 0]) / scale)
                )
                int_y = factor * (
                    erf((y0 + height - centers[:, 1]) / scale)
                    - erf((y0 - centers[:, 1]) / scale)
                )
                coverage[valid] += int_x * int_y / (width * height)

        return coverage.reshape(np.shape(start_x))

    def evaluate_diskaddsat_box_average(
        self,
        start_x: np.ndarray,
        start_y: np.ndarray,
        width: float,
        height: float,
    ) -> np.ndarray:
        """Integrate additive unit disks over axis-aligned boxes exactly."""
        if self.pattern_type != "diskaddsat":
            raise ValueError("Analytic box integration requires diskaddsat.")

        x_flat = np.asarray(start_x, dtype=np.float64).ravel()
        y_flat = np.asarray(start_y, dtype=np.float64).ravel()
        ny, nx = self.grid_shape
        centers_grid = self.centers.reshape(ny, nx, 2)
        origin_x, origin_y = self.lattice_origin
        reach = int(
            np.ceil((self.radius + self.max_jitter) / self.pitch)
        )
        base_ix = np.rint((x_flat - origin_x) / self.pitch).astype(np.int64)
        base_iy = np.rint((y_flat - origin_y) / self.pitch).astype(np.int64)
        coverage = np.zeros(x_flat.size, dtype=np.float64)

        for oy in range(-reach, reach + 1):
            iy = base_iy + oy
            valid_y = (iy >= 0) & (iy < ny)
            for ox in range(-reach, reach + 1):
                ix = base_ix + ox
                valid = valid_y & (ix >= 0) & (ix < nx)
                if not np.any(valid):
                    continue
                coverage[valid] += _disk_box_area(
                    x_flat[valid] - centers_grid[iy[valid], ix[valid], 0],
                    y_flat[valid] - centers_grid[iy[valid], ix[valid], 1],
                    width,
                    height,
                    self.radius,
                ) / (width * height)

        return coverage.reshape(np.shape(start_x))


def make_speckle_pattern(
    pattern_type: str,
    px_per_speck: float,
    black_area_fraction: float,
    perturbation_distribution: str,
    perturbation_fraction: float,
    random_seed: int,
    cutoff_sigmas: float,
    bounds: tuple[float, float, float, float],
    i0: float = I0,
    gamma: float = GAMMA,
) -> SpecklePattern:
    """Create a repeatable jittered lattice covering bounds plus support."""
    if pattern_type not in {
        "disk",
        "diskaddsat",
        "gausstrunc",
        "gausscont",
    }:
        raise ValueError(
            "pattern_type must be disk, diskaddsat, gausstrunc, or gausscont"
        )
    if perturbation_distribution not in {"uniform", "gaussian"}:
        raise ValueError(
            "perturbation distribution must be 'uniform' or 'gaussian'"
        )
    if not 0.0 < black_area_fraction < 1.0:
        raise ValueError("black_area_fraction must be between zero and one")
    if perturbation_fraction < 0.0:
        raise ValueError("perturbation_fraction must be non-negative")

    diameter = float(px_per_speck)
    pitch = diameter * np.sqrt(np.pi / (4.0 * black_area_fraction))
    radius = 0.5 * diameter
    xmin, xmax, ymin, ymax = bounds
    margin = radius + 4.0 * perturbation_fraction * pitch
    if pattern_type == "gausscont":
        margin = 8.0 * (radius / cutoff_sigmas) + (
            4.0 * perturbation_fraction * pitch
        )
    ix = np.arange(
        np.floor((xmin - margin) / pitch),
        np.ceil((xmax + margin) / pitch) + 1,
    )
    iy = np.arange(
        np.floor((ymin - margin) / pitch),
        np.ceil((ymax + margin) / pitch) + 1,
    )
    grid_x, grid_y = np.meshgrid(ix * pitch, iy * pitch)
    centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    lattice_origin = tuple(centers[0])

    rng = np.random.default_rng(random_seed)
    if perturbation_distribution == "uniform":
        offsets = rng.uniform(
            -perturbation_fraction,
            perturbation_fraction,
            centers.shape,
        )
    else:
        offsets = rng.normal(0.0, perturbation_fraction, centers.shape)
    centers += offsets * pitch
    max_jitter = float(np.max(np.linalg.norm(offsets * pitch, axis=1)))
    return SpecklePattern(
        pattern_type,
        pitch,
        diameter,
        centers,
        grid_x.shape,
        lattice_origin,
        max_jitter,
        cutoff_sigmas,
        i0,
        gamma,
    )


def integration_rule(
    method: str,
    param: int,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "rect":
        offsets = (np.arange(param) + 0.5) * pixel_size / param
        dx, dy = np.meshgrid(offsets, offsets)
        weights = np.full(param * param, 1.0 / (param * param))
        return dx.ravel(), dy.ravel(), weights

    if method == "gauss":
        points, weights = np.polynomial.legendre.leggauss(param)
        offsets = 0.5 * (points + 1.0) * pixel_size
        dx, dy = np.meshgrid(offsets, offsets)
        tensor_weights = (
            weights[:, None] * weights[None, :] * 0.25
        ).ravel()
        return dx.ravel(), dy.ravel(), tensor_weights

    raise ValueError(f"Unsupported integration method: {method}")


def init_worker(
    coords,
    connect,
    disp_x,
    disp_y,
    pattern: SpecklePattern,
) -> None:
    global _worker_mesh, _worker_pattern
    _worker_pattern = pattern
    if coords is None:
        _worker_mesh = None
        return

    from exp1common import build_pv_mesh

    coords_def = np.array(coords, copy=True)
    coords_def[:, 0] += disp_x
    coords_def[:, 1] += disp_y
    _worker_mesh = build_pv_mesh(coords_def, connect)
    _worker_mesh.point_data["x_ref"] = coords[:, 0]
    _worker_mesh.point_data["y_ref"] = coords[:, 1]


def process_pixel_chunk(
    args: tuple[int, int, str, int, float, float],
) -> tuple[int, int, np.ndarray]:
    start_idx, end_idx, method, param, pixel_size, roi_size = args
    dx, dy, weights = integration_rule(method, param, pixel_size)
    indices = np.arange(start_idx, end_idx)
    px = -0.5 * roi_size + (indices % TARG_PX_X) * pixel_size
    py = -0.5 * roi_size + (indices // TARG_PX_X) * pixel_size
    xx = px[:, None] + dx[None, :]
    yy = py[:, None] + dy[None, :]

    global _worker_mesh, _worker_pattern
    if _worker_pattern is None:
        raise RuntimeError("Speckle worker has not been initialized.")

    if _worker_mesh is not None:
        import pyvista as pv

        query = np.zeros((xx.size, 3), dtype=np.float64)
        query[:, 0] = xx.ravel()
        query[:, 1] = yy.ravel()
        sampled = pv.PolyData(query).sample(_worker_mesh)
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        x_ref = sampled.point_data["x_ref"].reshape(xx.shape)
        y_ref = sampled.point_data["y_ref"].reshape(yy.shape)
        if _worker_pattern.pattern_type == "gausscont":
            values = np.zeros(xx.size, dtype=np.float64)
            values[valid] = _worker_pattern.evaluate_coverage(
                x_ref.ravel()[valid],
                y_ref.ravel()[valid],
            )
        else:
            values = np.full(
                xx.size,
                _worker_pattern.i0 + _worker_pattern.gamma,
            )
            values[valid] = _worker_pattern.evaluate(
                x_ref.ravel()[valid],
                y_ref.ravel()[valid],
            )
        values = values.reshape(xx.shape)
    else:
        if _worker_pattern.pattern_type == "gausscont":
            values = _worker_pattern.evaluate_coverage(xx, yy)
        else:
            values = _worker_pattern.evaluate(xx, yy)

    pixel_average = np.sum(values * weights[None, :], axis=1)
    if _worker_pattern.pattern_type == "gausscont":
        pixel_average = _worker_pattern.intensity_from_coverage(pixel_average)

    return start_idx, end_idx, pixel_average


def save_image(image: np.ndarray, output_dir: Path, prefix: str) -> None:
    """Save float-scaled NumPy data and each requested TIFF quantization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flipped = np.flipud(image)
    for bits in BIT_DEPTHS:
        max_value = float(2**bits - 1)
        quantized = np.clip(np.round(flipped * max_value), 0.0, max_value)
        image_data = quantized.astype(
            np.uint8 if bits == 8 else np.uint16
        )
        Image.fromarray(image_data).save(output_dir / f"{prefix}_b{bits}.tiff")
        np.save(output_dir / f"{prefix}_b{bits}.npy", flipped * max_value)


def render_case(
    case_dir: Path,
    output_dir: Path,
    pattern: SpecklePattern,
    method: str,
    param: int,
    active_frames: set[int],
) -> None:
    from exp1common import parse_case_params

    coords = np.loadtxt(case_dir / "coords.csv", delimiter=",")
    connect = np.loadtxt(
        case_dir / "connectivity.csv",
        delimiter=",",
        dtype=int,
    )
    disp_x = np.loadtxt(case_dir / "field_disp_x.csv", delimiter=",")
    disp_y = np.loadtxt(case_dir / "field_disp_y.csv", delimiter=",")
    if connect.ndim == 1:
        connect = connect.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x = disp_x.reshape(-1, 1)
        disp_y = disp_y.reshape(-1, 1)
    camera_pixels, roi_size = parse_case_params(case_dir)
    if (camera_pixels, camera_pixels) != (TARG_PX_X, TARG_PX_Y):
        raise ValueError(
            "Experiment 2 requires square target dimensions matching "
            "the case camera."
        )

    pixel_size = roi_size / camera_pixels
    samples = param * param
    chunk_pixels = max(1, MAX_PTS_PER_CHUNK // samples)
    tasks = [
        (
            start,
            min(start + chunk_pixels, TARG_PX_X * TARG_PX_Y),
            method,
            param,
            pixel_size,
            roi_size,
        )
        for start in range(0, TARG_PX_X * TARG_PX_Y, chunk_pixels)
    ]
    for frame in range(disp_x.shape[1]):
        if frame not in active_frames:
            continue
        if frame == 0:
            initargs = (None, None, None, None, pattern)
        else:
            initargs = (
                coords,
                connect,
                disp_x[:, frame],
                disp_y[:, frame],
                pattern,
            )

        with Pool(
            NUM_PROCESSES_RUN,
            initializer=init_worker,
            initargs=initargs,
        ) as pool:
            results = pool.map(process_pixel_chunk, tasks)
        flat = np.empty(TARG_PX_X * TARG_PX_Y, dtype=np.float64)
        for start, end, values in results:
            flat[start:end] = values
        image = flat.reshape(TARG_PX_Y, TARG_PX_X)
        prefix = (
            f"targ_px{TARG_PX_X}_int_{method}_param_{param}_frame{frame:02d}"
        )
        save_image(image, output_dir, prefix)
