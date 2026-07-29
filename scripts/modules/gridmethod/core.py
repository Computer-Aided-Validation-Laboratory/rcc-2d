"""Windowed-Fourier grid-method displacement measurement.

Ported from the compact GridMethodToolbox core: ``build_window.m``, ``LSA.m``,
``unwrap2D.cpp``, ``func_temporalUnwrap.m`` and ``calculate_U_EPS.m``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from skimage.restoration import unwrap_phase as skimage_unwrap_phase

TAU = 2.0 * np.pi


@dataclass(frozen=True)
class GridMethodConfig:
    period_px: float
    window_width_periods: float = 2.0
    window: str = "gaussian"  # gaussian, triangular, rect
    displacement_method: str = "iterative"  # direct, iterative
    unwrap: str = "reliability"  # reliability, skimage
    max_iterations: int = 50


@dataclass
class GridMethodResult:
    phase_x: np.ndarray
    phase_y: np.ndarray
    modulation_x: np.ndarray
    modulation_y: np.ndarray
    displacement_x: np.ndarray
    displacement_y: np.ndarray
    temporal_turns_x: np.ndarray
    temporal_turns_y: np.ndarray


def strain_and_rotation(displacement_x: np.ndarray, displacement_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``eps_xx, eps_yy, eps_xy, w_xy`` as in ``calculate_U_EPS.m``.

    Arrays are indexed ``[row, column]``: columns are image/grid X and rows
    are image/grid Y.  For physical coordinates with Y positive upwards,
    pass a sign-converted ``displacement_y`` and reverse the row derivative
    as appropriate to the caller's coordinate convention.
    """
    dux_drow, dux_dx = np.gradient(displacement_x)
    duy_drow, duy_dx = np.gradient(displacement_y)
    eps_xx = dux_dx
    eps_yy = duy_drow
    eps_xy = 0.5 * (dux_drow + duy_dx)
    w_xy = 0.5 * (dux_drow - duy_dx)
    return eps_xx, eps_yy, eps_xy, w_xy


def analysis_window(config: GridMethodConfig) -> np.ndarray:
    """Match MATLAB's normalised Gaussian/triangular analysis windows."""
    half_width = config.window_width_periods * config.period_px
    if config.window == "gaussian":
        radius = int(np.ceil(4.0 * half_width))
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        window = np.exp(-(xx * xx + yy * yy) / (2.0 * half_width**2))
    elif config.window == "triangular":
        radius = int(np.floor(half_width))
        line = 1.0 - np.abs(np.linspace(-1.0, 1.0, 2 * radius - 1))
        window = np.outer(line, line)
    elif config.window == "rect":
        radius = int(np.floor(half_width))
        window = np.ones((2 * radius, 2 * radius), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported window {config.window!r}.")
    return window / window.sum()


def local_spectrum(image: np.ndarray, window: np.ndarray, period_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB LSA equivalent using FFT convolution at the two grid carriers."""
    image = np.asarray(image, dtype=np.float64)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    carrier = TAU / period_px
    x_demod = fftconvolve(image * np.exp(-1j * carrier * xx), window, mode="same")
    y_demod = fftconvolve(image * np.exp(-1j * carrier * yy), window, mode="same")
    return np.angle(x_demod), np.angle(y_demod), np.abs(x_demod), np.abs(y_demod)


@njit(cache=True)
def _find(parent: np.ndarray, potential: np.ndarray, node: int) -> tuple[int, int]:
    """Return root and integer turns from ``node`` to that root.

    Union-by-size keeps these trees shallow.  Avoiding path compression here
    makes the signed-potential bookkeeping unambiguous and is still fast for
    image-sized phase maps (tree depth is logarithmic).
    """
    turns = 0
    while parent[node] != node:
        turns += potential[node]
        node = parent[node]
    return node, turns


@njit(cache=True)
def _unwrap_reliability(wrapped: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray, edge_turn: np.ndarray) -> np.ndarray:
    """Reliability-sorted union-find implementation of the supplied MEX."""
    count = wrapped.size
    parent = np.arange(count)
    potential = np.zeros(count, dtype=np.int64)  # turns to parent
    size = np.ones(count, dtype=np.int64)
    for edge in range(edge_a.size):
        a, b = edge_a[edge], edge_b[edge]
        root_a, turns_a = _find(parent, potential, a)
        root_b, turns_b = _find(parent, potential, b)
        if root_a == root_b:
            continue
        # Required k_b-k_a for the wrapped neighbour difference.
        relation = edge_turn[edge] - turns_b + turns_a
        if size[root_a] < size[root_b]:
            parent[root_a] = root_b
            potential[root_a] = -relation
            size[root_b] += size[root_a]
        else:
            parent[root_b] = root_a
            potential[root_b] = relation
            size[root_a] += size[root_b]
    result = np.empty(count, dtype=np.float64)
    for node in range(count):
        _root, turns = _find(parent, potential, node)
        result[node] = wrapped[node] + TAU * turns
    return result


def unwrap_reliability(phase: np.ndarray) -> np.ndarray:
    """Faithful 2D reliability unwrap (Herráez et al., 2002/MATLAB MEX)."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase unwrap requires a 2D array.")
    wrapped = np.angle(np.exp(1j * phase))
    # Reliability is the sum of squared wrapped second differences in the
    # horizontal, vertical, and both diagonal directions, as in unwrap2D.cpp.
    reliability = np.full(wrapped.shape, np.inf)
    centre = wrapped[1:-1, 1:-1]
    def second(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.angle(np.exp(1j * (a - centre))) - np.angle(np.exp(1j * (centre - b)))
    h = second(wrapped[1:-1, :-2], wrapped[1:-1, 2:])
    v = second(wrapped[:-2, 1:-1], wrapped[2:, 1:-1])
    d1 = second(wrapped[:-2, :-2], wrapped[2:, 2:])
    d2 = second(wrapped[:-2, 2:], wrapped[2:, :-2])
    reliability[1:-1, 1:-1] = h*h + v*v + d1*d1 + d2*d2
    height, width = phase.shape
    grid = np.arange(height * width).reshape(height, width)
    a = np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel()))
    b = np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel()))
    quality = np.concatenate(((reliability[:, :-1] + reliability[:, 1:]).ravel(), (reliability[:-1, :] + reliability[1:, :]).ravel()))
    wa, wb = wrapped.ravel()[a], wrapped.ravel()[b]
    turns = np.rint((wa - wb) / TAU).astype(np.int64)
    order = np.argsort(quality, kind="stable")
    return _unwrap_reliability(wrapped.ravel(), a[order], b[order], turns[order]).reshape(phase.shape)


def unwrap_spatial(phase: np.ndarray, method: str) -> np.ndarray:
    if method == "reliability":
        return unwrap_reliability(phase)
    if method == "skimage":
        return np.asarray(skimage_unwrap_phase(phase), dtype=np.float64)
    raise ValueError(f"Unknown unwrap method {method!r}.")


def temporal_unwrap(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correct whole-frame 2π jumps from the ROI-average phase trajectory."""
    mean = phases.mean(axis=(1, 2))
    turns = np.zeros(len(mean), dtype=np.int64)
    for frame in range(1, len(mean)):
        turns[frame] = turns[frame - 1] - int(np.rint((mean[frame] - mean[frame - 1]) / TAU))
    return phases + TAU * turns[:, None, None], turns


def displacement(reference_x: np.ndarray, reference_y: np.ndarray, current_x: np.ndarray, current_y: np.ndarray, config: GridMethodConfig) -> tuple[np.ndarray, np.ndarray]:
    ux = -(current_x - reference_x) * config.period_px / TAU
    uy = -(current_y - reference_y) * config.period_px / TAU
    if config.displacement_method == "direct":
        return ux, uy
    yy, xx = np.indices(ux.shape, dtype=np.float64)
    for _ in range(config.max_iterations):
        sample_x = map_coordinates(current_x, [yy + uy, xx + ux], order=3, mode="constant", cval=np.nan)
        sample_y = map_coordinates(current_y, [yy + uy, xx + ux], order=3, mode="constant", cval=np.nan)
        new_ux = -(sample_x - reference_x) * config.period_px / TAU
        new_uy = -(sample_y - reference_y) * config.period_px / TAU
        interior = (slice(int(config.period_px), -int(config.period_px)),) * 2
        change = np.nanmax(np.abs(new_ux[interior] - ux[interior]))
        ux, uy = new_ux, new_uy
        if change < 5e-4:
            break
    return ux, uy


def analyse_sequence(images: np.ndarray, config: GridMethodConfig) -> GridMethodResult:
    """Measure a time sequence with image shape ``(frame, row, column)``."""
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError("images must have shape (frame, row, column).")
    window = analysis_window(config)
    phases_x=[]; phases_y=[]; mod_x=[]; mod_y=[]
    for image in images:
        px, py, mx, my = local_spectrum(image, window, config.period_px)
        phases_x.append(unwrap_spatial(px, config.unwrap)); phases_y.append(unwrap_spatial(py, config.unwrap))
        mod_x.append(mx); mod_y.append(my)
    phase_x, tx = temporal_unwrap(np.asarray(phases_x)); phase_y, ty = temporal_unwrap(np.asarray(phases_y))
    ux=np.zeros_like(phase_x); uy=np.zeros_like(phase_y)
    for frame in range(1, len(images)):
        ux[frame], uy[frame] = displacement(phase_x[0], phase_y[0], phase_x[frame], phase_y[frame], config)
    return GridMethodResult(phase_x, phase_y, np.asarray(mod_x), np.asarray(mod_y), ux, uy, tx, ty)
