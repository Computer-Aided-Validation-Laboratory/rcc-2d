"""Single source of spatial conventions for Experiments 1 and 2.

All coordinates are expressed directly in final-camera pixel units.  The
camera sees the central 32x32 pixel ROI; the 42x42 material plate provides a
five-pixel guard band for the four-pixel PSF halo plus one pixel of motion.
"""
from __future__ import annotations

import numpy as np


ROI_PIXELS = 32
MAX_MOTION_PIXELS = 1.0
PSF_HALO_PIXELS = 4
GUARD_PIXELS = int(PSF_HALO_PIXELS + MAX_MOTION_PIXELS)
PLATE_PIXELS = ROI_PIXELS + 2 * GUARD_PIXELS
TEXTURE_PAD_PIXELS = GUARD_PIXELS
FRAME_STEP_PIXELS = 0.1


def roi_corners() -> np.ndarray:
    """Return the final-camera ROI corners in reference pixel coordinates."""
    half = ROI_PIXELS / 2.0
    return np.array(((-half, -half, 0.0), (half, -half, 0.0),
                     (half, half, 0.0), (-half, half, 0.0)), dtype=np.float64)


def texture_world_uvs(coords: np.ndarray, oversamp: int) -> np.ndarray:
    """Map reference material coordinates to rows-flipped texel-centre UVs."""
    width = oversamp * PLATE_PIXELS
    texel = 1.0 / oversamp
    x_min = -PLATE_PIXELS / 2.0
    y_max = PLATE_PIXELS / 2.0
    result = np.empty((coords.shape[0], 2), dtype=np.float64)
    result[:, 0] = ((coords[:, 0] - x_min) / texel - 0.5) / (width - 1.0)
    result[:, 1] = ((y_max - coords[:, 1]) / texel - 0.5) / (width - 1.0)
    return np.ascontiguousarray(result)
