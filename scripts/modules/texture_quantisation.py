"""Input-precision simulation for unbounded floating additive textures."""
from __future__ import annotations

import numpy as np


def quantise_texture_f64(texture: np.ndarray, bits: int) -> np.ndarray:
    """Round raw texture texels to a normalised ``b``-bit-equivalent increment.

    This deliberately does *not* clip to ``[0, 1]``.  Additive coverage may
    exceed one where blobs overlap; saturation belongs after Riley's pixel or
    PSF integration.  The returned f64 array therefore models finite input
    precision while retaining Riley's floating texture pipeline.
    """
    if bits < 1:
        raise ValueError(f"Texture precision must be positive, got {bits}.")
    values = np.asarray(texture, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Cannot quantise a texture containing non-finite values.")
    maximum = float((1 << int(bits)) - 1)
    return np.ascontiguousarray(np.rint(values * maximum) / maximum)
