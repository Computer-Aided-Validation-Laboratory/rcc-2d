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
from script_timing import ScriptTimer, timed_call

from exp2params import (
    BACKGROUND,
    BIT_DEPTHS,
    GAMMA,
    FORCE_RENDER_OVER,
    GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
    GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION,
    I0,
    NUM_PROCESSES,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
)

MAX_PTS_PER_CHUNK = int(
    os.environ.get("EXP2_MAX_PTS_PER_CHUNK", "2000000")
)
MAX_PIXELS_PER_CHUNK = int(
    os.environ.get("EXP2_MAX_PIXELS_PER_CHUNK", "1000000")
)
NUM_PROCESSES_RUN = max(1, min(
    NUM_PROCESSES,
    int(os.environ.get("EXP2_NUM_PROCESSES", str(NUM_PROCESSES))),
))
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
    gaussian_equivalent_disk_edge_fraction: float
    continuous_tail_sigmas: float
    i0: float
    gamma: float

    @property
    def radius(self) -> float:
        return 0.5 * self.equivalent_diameter

    @property
    def sigma(self) -> float:
        if self.pattern_type == "gausscont":
            return self.radius / np.sqrt(
                -2.0 * np.log(self.gaussian_equivalent_disk_edge_fraction)
            )
        return self.radius / self.cutoff_sigmas

    @property
    def support_radius(self) -> float:
        if self.pattern_type == "gausscont":
            return self.continuous_tail_sigmas * self.sigma
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
    gaussian_equivalent_disk_edge_fraction: float = (
        GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION
    ),
    continuous_tail_sigmas: float = GAUSSIAN_CONTINUOUS_TAIL_SIGMAS,
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
    if continuous_tail_sigmas <= 0.0:
        raise ValueError("continuous_tail_sigmas must be positive")
    if not 0.0 < gaussian_equivalent_disk_edge_fraction < 1.0:
        raise ValueError(
            "gaussian_equivalent_disk_edge_fraction must be between zero and one"
        )

    diameter = float(px_per_speck)
    pitch = diameter * np.sqrt(np.pi / (4.0 * black_area_fraction))
    radius = 0.5 * diameter
    xmin, xmax, ymin, ymax = bounds
    margin = radius + 4.0 * perturbation_fraction * pitch
    if pattern_type == "gausscont":
        sigma = radius / np.sqrt(-2.0 * np.log(gaussian_equivalent_disk_edge_fraction))
        margin = continuous_tail_sigmas * sigma + (
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
        gaussian_equivalent_disk_edge_fraction,
        continuous_tail_sigmas,
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

    raise ValueError(f"Unsupported numerical integration method: {method}")


def _candidate_reach(pattern: SpecklePattern, a: np.ndarray, h: float) -> int:
    """Return a conservative lattice-cell search radius for local maps."""
    mapped_half_diagonal = h * np.max(np.linalg.norm(a, axis=(1, 2)))
    return int(
        np.ceil(
            (pattern.support_radius + pattern.max_jitter + mapped_half_diagonal)
            / pattern.pitch
        )
        + 1
    )


def _analytic_gaussian_coverage(
    r0: np.ndarray,
    a: np.ndarray,
    pattern: SpecklePattern,
    pixel_size: float,
    rigid: bool,
) -> np.ndarray:
    """Return additive Gaussian pixel averages in locally affine pixels."""
    from scipy.special import erf
    from scipy.stats import multivariate_normal

    count = len(r0)
    coverage = np.zeros(count, dtype=np.float64)
    ny, nx = pattern.grid_shape
    centres_grid = pattern.centers.reshape(ny, nx, 2)
    origin_x, origin_y = pattern.lattice_origin
    base_ix = np.rint((r0[:, 0] - origin_x) / pattern.pitch).astype(np.int64)
    base_iy = np.rint((r0[:, 1] - origin_y) / pattern.pitch).astype(np.int64)
    reach = _candidate_reach(pattern, a, 0.5 * pixel_size)
    factor = pattern.sigma * np.sqrt(np.pi / 2.0)
    h = 0.5 * pixel_size

    # The supplied affine plate has one constant map. Grouping also supports
    # piecewise-affine meshes without requiring one CDF setup per pixel.
    rounded_a = np.round(a.reshape(count, 4), decimals=12)
    unique_a, group_ids = np.unique(rounded_a, axis=0, return_inverse=True)
    for group_id, a_values in enumerate(unique_a):
        group = np.flatnonzero(group_ids == group_id)
        a_group = a_values.reshape(2, 2)
        if not rigid:
            inv_a = np.linalg.inv(a_group)
            covariance = pattern.sigma**2 * np.linalg.inv(a_group.T @ a_group)

        for oy in range(-reach, reach + 1):
            iy = base_iy[group] + oy
            valid_y = (iy >= 0) & (iy < ny)
            for ox in range(-reach, reach + 1):
                ix = base_ix[group] + ox
                valid = valid_y & (ix >= 0) & (ix < nx)
                if not np.any(valid):
                    continue
                indices = group[valid]
                centres = centres_grid[iy[valid], ix[valid]]
                # q-coordinate of the blob centre, with r(q) = r0 + A q.
                mu = (centres - r0[indices]) @ inv_a.T if not rigid else (
                    (centres - r0[indices]) @ a_group
                )
                if rigid:
                    int_x = factor * (
                        erf((h - mu[:, 0]) / (np.sqrt(2.0) * pattern.sigma))
                        - erf((-h - mu[:, 0]) / (np.sqrt(2.0) * pattern.sigma))
                    )
                    int_y = factor * (
                        erf((h - mu[:, 1]) / (np.sqrt(2.0) * pattern.sigma))
                        - erf((-h - mu[:, 1]) / (np.sqrt(2.0) * pattern.sigma))
                    )
                    coverage[indices] += int_x * int_y / pixel_size**2
                else:
                    upper = np.full_like(mu, h)
                    lower = np.full_like(mu, -h)
                    probability = (
                        multivariate_normal.cdf(upper - mu, cov=covariance)
                        - multivariate_normal.cdf(
                            np.column_stack((lower[:, 0], upper[:, 1])) - mu,
                            cov=covariance,
                        )
                        - multivariate_normal.cdf(
                            np.column_stack((upper[:, 0], lower[:, 1])) - mu,
                            cov=covariance,
                        )
                        + multivariate_normal.cdf(lower - mu, cov=covariance)
                    )
                    coverage[indices] += (
                        2.0
                        * np.pi
                        * pattern.sigma**2
                        * probability
                        / (abs(np.linalg.det(a_group)) * pixel_size**2)
                    )
    return coverage


def _analytic_disk_coverage(
    r0: np.ndarray,
    a: np.ndarray,
    pattern: SpecklePattern,
    pixel_size: float,
) -> np.ndarray:
    """Return exact additive disk averages for locally rigid target pixels."""
    count = len(r0)
    coverage = np.zeros(count, dtype=np.float64)
    ny, nx = pattern.grid_shape
    centres_grid = pattern.centers.reshape(ny, nx, 2)
    origin_x, origin_y = pattern.lattice_origin
    base_ix = np.rint((r0[:, 0] - origin_x) / pattern.pitch).astype(np.int64)
    base_iy = np.rint((r0[:, 1] - origin_y) / pattern.pitch).astype(np.int64)
    reach = _candidate_reach(pattern, a, 0.5 * pixel_size)
    h = 0.5 * pixel_size

    for oy in range(-reach, reach + 1):
        iy = base_iy + oy
        valid_y = (iy >= 0) & (iy < ny)
        for ox in range(-reach, reach + 1):
            ix = base_ix + ox
            valid = valid_y & (ix >= 0) & (ix < nx)
            if not np.any(valid):
                continue
            indices = np.flatnonzero(valid)
            centres = centres_grid[iy[valid], ix[valid]]
            # For an orthogonal A, rotate the disk centre into the square q box.
            mu = np.einsum("nij,nj->ni", a[indices].transpose(0, 2, 1), centres - r0[indices])
            coverage[indices] += _disk_box_area(
                -h - mu[:, 0],
                -h - mu[:, 1],
                pixel_size,
                pixel_size,
                pattern.radius,
            ) / pixel_size**2
    return coverage


def _local_affine_map(
    px: np.ndarray,
    py: np.ndarray,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit r(q)=r0+Aq from the mapped target-pixel corners."""
    count = len(px)
    if _worker_mesh is None:
        r0 = np.column_stack((px + 0.5 * pixel_size, py + 0.5 * pixel_size))
        a = np.broadcast_to(np.eye(2), (count, 2, 2)).copy()
        valid = np.ones(count, dtype=bool)
        return r0, a, valid, np.zeros(count, dtype=bool)

    import pyvista as pv

    corner_x = np.array([0.0, pixel_size, 0.0, pixel_size])
    corner_y = np.array([0.0, 0.0, pixel_size, pixel_size])
    query = np.zeros((count * 4, 3), dtype=np.float64)
    query[:, 0] = (px[:, None] + corner_x).ravel()
    query[:, 1] = (py[:, None] + corner_y).ravel()
    sampled = pv.PolyData(query).sample(_worker_mesh)
    corner_valid = sampled.point_data["vtkValidPointMask"].astype(bool).reshape(count, 4)
    valid = np.all(corner_valid, axis=1)
    x = sampled.point_data["x_ref"].reshape(count, 4)
    y = sampled.point_data["y_ref"].reshape(count, 4)
    r0 = np.column_stack((np.mean(x, axis=1), np.mean(y, axis=1)))
    a = np.empty((count, 2, 2), dtype=np.float64)
    a[:, 0, 0] = (-x[:, 0] + x[:, 1] - x[:, 2] + x[:, 3]) / (2.0 * pixel_size)
    a[:, 0, 1] = (-x[:, 0] - x[:, 1] + x[:, 2] + x[:, 3]) / (2.0 * pixel_size)
    a[:, 1, 0] = (-y[:, 0] + y[:, 1] - y[:, 2] + y[:, 3]) / (2.0 * pixel_size)
    a[:, 1, 1] = (-y[:, 0] - y[:, 1] + y[:, 2] + y[:, 3]) / (2.0 * pixel_size)
    determinant = np.linalg.det(a)
    condition = np.linalg.cond(a)
    ill_conditioned = (~np.isfinite(determinant)) | (np.abs(determinant) < 1e-12) | (~np.isfinite(condition)) | (condition > 1e8)
    return r0, a, valid & ~ill_conditioned, ill_conditioned & valid


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
) -> tuple[int, int, np.ndarray, int, int, int]:
    start_idx, end_idx, method, param, pixel_size, roi_size = args
    indices = np.arange(start_idx, end_idx)
    px = -0.5 * roi_size + (indices % TARG_PX_X) * pixel_size
    py = -0.5 * roi_size + (indices // TARG_PX_X) * pixel_size

    global _worker_mesh, _worker_pattern
    if _worker_pattern is None:
        raise RuntimeError("Speckle worker has not been initialized.")

    if method == "analytic":
        r0, a, valid, ill_conditioned = _local_affine_map(px, py, pixel_size)
        pixel_average = np.full(len(indices), BACKGROUND, dtype=np.float64)
        if _worker_pattern.pattern_type == "diskaddsat":
            identity = np.eye(2)
            rigid = np.all(
                np.isclose(
                    np.einsum("nji,njk->nik", a[valid], a[valid]),
                    identity,
                    rtol=1e-9,
                    atol=1e-9,
                ),
                axis=(1, 2),
            )
            supported = np.flatnonzero(valid)[rigid]
            if len(supported):
                coverage = _analytic_disk_coverage(
                    r0[supported], a[supported], _worker_pattern, pixel_size
                )
                pixel_average[supported] = _worker_pattern.intensity_from_coverage(
                    coverage
                )
            skipped = int(np.count_nonzero(valid) - len(supported))
        elif _worker_pattern.pattern_type == "gausscont":
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices):
                gram = np.einsum("nji,njk->nik", a[valid], a[valid])
                rigid = np.all(
                    np.isclose(gram, np.eye(2), rtol=1e-9, atol=1e-9),
                    axis=(1, 2),
                )
                coverage = np.empty(len(valid_indices), dtype=np.float64)
                if np.any(rigid):
                    coverage[rigid] = _analytic_gaussian_coverage(
                        r0[valid_indices[rigid]],
                        a[valid_indices[rigid]],
                        _worker_pattern,
                        pixel_size,
                        rigid=True,
                    )
                if np.any(~rigid):
                    coverage[~rigid] = _analytic_gaussian_coverage(
                        r0[valid_indices[~rigid]],
                        a[valid_indices[~rigid]],
                        _worker_pattern,
                        pixel_size,
                        rigid=False,
                    )
                pixel_average[valid_indices] = _worker_pattern.intensity_from_coverage(
                    coverage
                )
            skipped = 0
        else:
            raise ValueError(
                "Analytic rendering only supports diskaddsat and gausscont."
            )
        invalid = int(len(indices) - np.count_nonzero(valid))
        return start_idx, end_idx, pixel_average, invalid, int(np.count_nonzero(ill_conditioned)), skipped

    dx, dy, weights = integration_rule(method, param, pixel_size)
    xx = px[:, None] + dx[None, :]
    yy = py[:, None] + dy[None, :]

    coverage_first = _worker_pattern.pattern_type in {
        "diskaddsat",
        "gausscont",
    }
    if _worker_mesh is not None:
        import pyvista as pv

        query = np.zeros((xx.size, 3), dtype=np.float64)
        query[:, 0] = xx.ravel()
        query[:, 1] = yy.ravel()
        sampled = pv.PolyData(query).sample(_worker_mesh)
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        x_ref = sampled.point_data["x_ref"].reshape(xx.shape)
        y_ref = sampled.point_data["y_ref"].reshape(yy.shape)
        if coverage_first:
            values = np.zeros(xx.size, dtype=np.float64)
            values[valid] = _worker_pattern.evaluate_coverage(
                x_ref.ravel()[valid],
                y_ref.ravel()[valid],
            )
            background_coverage = (
                _worker_pattern.i0 + _worker_pattern.gamma - BACKGROUND
            ) / (2.0 * _worker_pattern.gamma)
            values[~valid] = background_coverage
        else:
            values = np.full(xx.size, BACKGROUND)
            values[valid] = _worker_pattern.evaluate(
                x_ref.ravel()[valid],
                y_ref.ravel()[valid],
            )
        values = values.reshape(xx.shape)
    else:
        if coverage_first:
            values = _worker_pattern.evaluate_coverage(xx, yy)
        else:
            values = _worker_pattern.evaluate(xx, yy)

    pixel_average = np.sum(values * weights[None, :], axis=1)
    if coverage_first:
        pixel_average = _worker_pattern.intensity_from_coverage(pixel_average)

    return start_idx, end_idx, pixel_average, 0, 0, 0


def save_image(
    image: np.ndarray,
    output_dir: Path,
    prefix: str,
    float_texture: np.ndarray | None = None,
) -> None:
    """Save raw f64 texture data and digitised TIFF intensity outputs.

    ``image`` is the display/render intensity to quantise for TIFF.  When
    supplied, ``float_texture`` is saved as the primary f64 ``.npy`` file
    without clipping or scaling; additive speckle generators pass their
    pixel-integrated coverage here so overlapping blobs remain visible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if float_texture is None:
        float_texture = image
    if float_texture.shape != image.shape:
        raise ValueError("float_texture and image must have identical shapes.")

    flipped = np.ascontiguousarray(np.flipud(image), dtype=np.float64)
    raw_flipped = np.ascontiguousarray(np.flipud(float_texture), dtype=np.float64)
    np.save(output_dir / f"{prefix}.npy", raw_flipped)
    for bits in BIT_DEPTHS:
        max_value = float(2**bits - 1)
        quantized = np.clip(np.round(flipped * max_value), 0.0, max_value)
        image_data = quantized.astype(
            np.uint8 if bits == 8 else np.uint16
        )
        Image.fromarray(image_data).save(output_dir / f"{prefix}_b{bits}.tiff")


def image_outputs_complete(output_dir: Path, prefix: str) -> bool:
    """Return whether the f64 image and every digitised image are present."""
    return (output_dir / f"{prefix}.npy").exists() and all(
        (output_dir / f"{prefix}_b{bits}.tiff").exists()
        for bits in BIT_DEPTHS
    )


def save_raw_coverage(
    coverage: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save unclamped additive coverage as float data and direct 8-bit counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flipped = np.flipud(coverage)
    np.save(output_dir / f"{prefix}_raw.npy", flipped)
    raw_8bit = np.rint(flipped)
    if np.any(raw_8bit < 0.0) or np.any(raw_8bit > 255.0):
        raise ValueError(
            "Raw coverage cannot be represented as an 8-bit overlap count."
        )
    Image.fromarray(raw_8bit.astype(np.uint8)).save(
        output_dir / f"{prefix}_raw_b8.tiff"
    )


def render_case(
    case_dir: Path,
    output_dir: Path,
    pattern: SpecklePattern,
    method: str,
    param: int,
    active_frames: set[int],
    timer: ScriptTimer | None = None,
) -> None:
    from exp1common import parse_case_params
    timer = timer or ScriptTimer(__file__)

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
    _case_camera_pixels, roi_size = parse_case_params(case_dir)
    if TARG_PX_X != TARG_PX_Y:
        raise ValueError(
            "Experiment 2 currently requires square target dimensions."
        )

    pixel_size = roi_size / TARG_PX_X
    samples = 1 if method == "analytic" else param * param
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
        prefix = (
            f"targ_px{TARG_PX_X}_int_{method}_param_{param}_frame{frame:02d}"
        )
        if not FORCE_RENDER_OVER and image_outputs_complete(output_dir, prefix):
            print(f"    frame {frame:02d}: outputs exist; skipping.")
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
            results = timed_call(
                timer,
                f"{case_dir.name}_int_{method}_param_{param}_frame{frame:02d}",
                pool.map, process_pixel_chunk, tasks,
            )
        flat = np.empty(TARG_PX_X * TARG_PX_Y, dtype=np.float64)
        invalid_pixels = 0
        ill_conditioned_pixels = 0
        skipped_pixels = 0
        for start, end, values, invalid, ill_conditioned, skipped in results:
            flat[start:end] = values
            invalid_pixels += invalid
            ill_conditioned_pixels += ill_conditioned
            skipped_pixels += skipped
        if invalid_pixels:
            print(
                f"    frame {frame:02d}: {invalid_pixels} invalid affine-map "
                f"pixels set to BACKGROUND={BACKGROUND:g}."
            )
        if ill_conditioned_pixels:
            print(
                f"    frame {frame:02d}: {ill_conditioned_pixels} singular or "
                "ill-conditioned affine maps set to background."
            )
        if skipped_pixels:
            print(
                f"    frame {frame:02d}: skipped {skipped_pixels} non-rigid "
                "diskaddsat pixels; affine disk integration is unsupported."
            )
        if method == "analytic" and pattern.pattern_type == "diskaddsat" and (
            skipped_pixels + invalid_pixels == TARG_PX_X * TARG_PX_Y
        ):
            print(f"    frame {frame:02d}: no analytic disk output written.")
            continue
        image = flat.reshape(TARG_PX_Y, TARG_PX_X)
        save_image(image, output_dir, prefix)
