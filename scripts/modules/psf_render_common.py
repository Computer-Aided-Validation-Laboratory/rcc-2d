"""Shared discrete camera-PSF operations for bespoke orthographic renders.

The kernel construction deliberately mirrors Riley: evaluate the Gaussian on
the regular SSAA lattice over a finite support, normalise the sampled kernel,
filter the image-plane SSAA raster, then average each final-pixel block.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve1d


def psf_radius_subpixels(ssaa: int, support_radius_px: float) -> int:
    """Return Riley's inclusive Gaussian support radius on the SSAA lattice."""
    if ssaa < 1:
        raise ValueError("SSAA must be positive.")
    return int(math.ceil(support_radius_px * ssaa))


def gaussian_kernel_1d(
    ssaa: int,
    sigma_px: float,
    support_radius_px: float,
) -> np.ndarray:
    """Build Riley-equivalent sampled, normalised isotropic Gaussian weights."""
    if sigma_px <= 0.0:
        raise ValueError("PSF sigma must be positive.")
    if support_radius_px <= 0.0:
        raise ValueError("PSF support radius must be positive.")
    radius = psf_radius_subpixels(ssaa, support_radius_px)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    distances_px = offsets / float(ssaa)
    weights = np.exp(-0.5 * (distances_px / sigma_px) ** 2)
    return weights / np.sum(weights)


def filter_and_average(
    shaded_with_halo: np.ndarray,
    ssaa: int,
    sigma_px: float,
    support_radius_px: float,
    final_shape: tuple[int, int],
    background: float,
) -> np.ndarray:
    """Apply the image-plane PSF and resolve the central final-image raster.

    ``shaded_with_halo`` contains exactly one PSF support radius of sampled
    image-plane shading on every side.  Any samples beyond that finite raster
    are the camera background, matching Riley's scratch-raster resolver.
    """
    height, width = final_shape
    radius = psf_radius_subpixels(ssaa, support_radius_px)
    expected = (height * ssaa + 2 * radius, width * ssaa + 2 * radius)
    if shaded_with_halo.shape != expected:
        raise ValueError(
            f"Expected SSAA raster with halo {expected}, got "
            f"{shaded_with_halo.shape}."
        )
    kernel = gaussian_kernel_1d(ssaa, sigma_px, support_radius_px)
    horizontal = convolve1d(
        shaded_with_halo, kernel, axis=1, mode="constant", cval=background
    )
    filtered = convolve1d(
        horizontal, kernel, axis=0, mode="constant", cval=background
    )
    centre = filtered[radius:radius + height * ssaa, radius:radius + width * ssaa]
    return centre.reshape(height, ssaa, width, ssaa).mean(axis=(1, 3))
