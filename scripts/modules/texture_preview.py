"""Small, on-demand 8-bit previews of floating-point source textures."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image


def write_preview_b8(
    float_path: Path,
    preview_path: Path,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> bool:
    """Write a clamped 8-bit TIFF preview only when it is absent.

    ``transform`` supports generators such as Exp2 whose durable f64 texture
    is unbounded coverage rather than display intensity.
    """
    if preview_path.is_file():
        return False
    values = np.load(float_path, mmap_mode="r")
    if transform is not None:
        values = transform(values)
    codes = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(codes).save(preview_path, format="TIFF")
    return True
