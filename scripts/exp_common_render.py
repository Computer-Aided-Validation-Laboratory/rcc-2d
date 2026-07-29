"""Renderer-independent numerical and analytic integration primitives.

This module deliberately contains no experiment parameters, paths, camera
sizes, or output naming.  Experiments provide those details through their
``expN_render_utils`` layer.  The speckle helpers operate on the small common
interface implemented by :class:`exp2speckint2d.SpecklePattern`.
"""

from __future__ import annotations

import numpy as np


def disk_box_area(
    x0: np.ndarray, y0: np.ndarray, width: float, height: float, radius: float
) -> np.ndarray:
    """Exact area of a radius-``radius`` disk overlapping translated boxes."""
    def primitive(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -radius, radius)
        return 0.5 * (
            clipped * np.sqrt(np.maximum(radius * radius - clipped * clipped, 0.0))
            + radius * radius * np.arcsin(clipped / radius)
        )

    left, right = x0, x0 + width
    bottom, top = y0, y0 + height
    lo, hi = np.maximum(left, -radius), np.minimum(right, radius)
    valid = hi > lo
    roots = []
    for edge in (bottom, top):
        root = np.sqrt(np.maximum(radius * radius - edge * edge, 0.0))
        roots.extend((-root, root))
    points = np.stack([lo, hi, *[np.clip(root, lo, hi) for root in roots]], axis=1)
    points.sort(axis=1)
    area = np.zeros_like(x0)
    for index in range(points.shape[1] - 1):
        start, end = points[:, index], points[:, index + 1]
        half_height = np.sqrt(np.maximum(radius * radius - ((start + end) * 0.5) ** 2, 0.0))
        upper_is_arc, lower_is_arc = half_height < top, -half_height > bottom
        integral = np.where(
            upper_is_arc & lower_is_arc,
            2.0 * (primitive(end) - primitive(start)),
            np.where(
                upper_is_arc,
                primitive(end) - primitive(start) - bottom * (end - start),
                np.where(
                    lower_is_arc,
                    top * (end - start) + primitive(end) - primitive(start),
                    (top - bottom) * (end - start),
                ),
            ),
        )
        overlap = np.minimum(top, half_height) - np.maximum(bottom, -half_height)
        area += np.where((overlap > 0.0) & valid, integral, 0.0)
    return area


def _candidate_reach(pattern, affine: np.ndarray, half_pixel: float) -> int:
    mapped_half_diagonal = half_pixel * np.max(np.linalg.norm(affine, axis=(1, 2)))
    return int(np.ceil((pattern.support_radius + pattern.max_jitter + mapped_half_diagonal) / pattern.pitch) + 1)


def is_rigid_inverse(affine: np.ndarray, *, atol: float = 1e-9) -> np.ndarray:
    """Return one boolean per map: is the inverse map orthogonal?"""
    gram = np.einsum("nji,njk->nik", affine, affine)
    return np.all(np.isclose(gram, np.eye(2), rtol=1e-9, atol=atol), axis=(1, 2))


def analytic_disk_coverage(
    reference_centres: np.ndarray, affine: np.ndarray, pattern, pixel_size: float
) -> np.ndarray:
    """Exact additive-disk average for rigid square camera pixels.

    A non-rigid affine map turns a square camera pixel into a parallelogram;
    that is intentionally rejected rather than silently centre-sampled.
    """
    if not np.all(is_rigid_inverse(affine)):
        raise ValueError("Exact disk integration requires a rigid inverse pixel map.")
    count = len(reference_centres)
    coverage = np.zeros(count, dtype=np.float64)
    ny, nx = pattern.grid_shape
    centres_grid = pattern.centers.reshape(ny, nx, 2)
    origin_x, origin_y = pattern.lattice_origin
    base_ix = np.rint((reference_centres[:, 0] - origin_x) / pattern.pitch).astype(np.int64)
    base_iy = np.rint((reference_centres[:, 1] - origin_y) / pattern.pitch).astype(np.int64)
    reach = _candidate_reach(pattern, affine, 0.5 * pixel_size)
    half = 0.5 * pixel_size
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
            # Orthogonality preserves a disk when the target square is rotated
            # into reference coordinates.
            mu = np.einsum("nij,nj->ni", affine[indices].transpose(0, 2, 1), centres - reference_centres[indices])
            coverage[indices] += disk_box_area(
                -half - mu[:, 0], -half - mu[:, 1], pixel_size, pixel_size, pattern.radius
            ) / pixel_size**2
    return coverage


def analytic_gaussian_coverage(
    reference_centres: np.ndarray, affine: np.ndarray, pattern, pixel_size: float
) -> np.ndarray:
    """Analytic additive Gaussian average for square affine camera pixels."""
    from scipy.special import erf
    from scipy.stats import multivariate_normal

    count = len(reference_centres)
    coverage = np.zeros(count, dtype=np.float64)
    ny, nx = pattern.grid_shape
    centres_grid = pattern.centers.reshape(ny, nx, 2)
    origin_x, origin_y = pattern.lattice_origin
    base_ix = np.rint((reference_centres[:, 0] - origin_x) / pattern.pitch).astype(np.int64)
    base_iy = np.rint((reference_centres[:, 1] - origin_y) / pattern.pitch).astype(np.int64)
    rigid_all = is_rigid_inverse(affine)
    reach = _candidate_reach(pattern, affine, 0.5 * pixel_size)
    factor, half = pattern.sigma * np.sqrt(np.pi / 2.0), 0.5 * pixel_size
    rounded = np.round(affine.reshape(count, 4), decimals=12)
    unique, ids = np.unique(rounded, axis=0, return_inverse=True)
    for group_id, values in enumerate(unique):
        group = np.flatnonzero(ids == group_id)
        matrix = values.reshape(2, 2)
        rigid = bool(rigid_all[group[0]])
        if not rigid:
            inv = np.linalg.inv(matrix)
            covariance = pattern.sigma**2 * np.linalg.inv(matrix.T @ matrix)
        for oy in range(-reach, reach + 1):
            iy = base_iy[group] + oy
            valid_y = (iy >= 0) & (iy < ny)
            for ox in range(-reach, reach + 1):
                ix = base_ix[group] + ox
                valid = valid_y & (ix >= 0) & (ix < nx)
                if not np.any(valid):
                    continue
                indices, centres = group[valid], centres_grid[iy[valid], ix[valid]]
                # Row-vector convention: for a rigid inverse A, q-centres are
                # ``(r_c-r_0) A``; the non-rigid derivation uses inv(A)^T.
                mu = ((centres - reference_centres[indices]) @ matrix
                      if rigid else (centres - reference_centres[indices]) @ inv.T)
                if rigid:
                    ix_avg = factor * (erf((half - mu[:, 0]) / (np.sqrt(2.0) * pattern.sigma)) - erf((-half - mu[:, 0]) / (np.sqrt(2.0) * pattern.sigma)))
                    iy_avg = factor * (erf((half - mu[:, 1]) / (np.sqrt(2.0) * pattern.sigma)) - erf((-half - mu[:, 1]) / (np.sqrt(2.0) * pattern.sigma)))
                    coverage[indices] += ix_avg * iy_avg / pixel_size**2
                else:
                    upper, lower = np.full_like(mu, half), np.full_like(mu, -half)
                    probability = (multivariate_normal.cdf(upper - mu, cov=covariance)
                        - multivariate_normal.cdf(np.column_stack((lower[:, 0], upper[:, 1])) - mu, cov=covariance)
                        - multivariate_normal.cdf(np.column_stack((upper[:, 0], lower[:, 1])) - mu, cov=covariance)
                        + multivariate_normal.cdf(lower - mu, cov=covariance))
                    coverage[indices] += 2.0 * np.pi * pattern.sigma**2 * probability / (abs(np.linalg.det(matrix)) * pixel_size**2)
    return coverage
