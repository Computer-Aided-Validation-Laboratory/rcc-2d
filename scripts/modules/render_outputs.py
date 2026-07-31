"""Canonical float-image persistence and cheap camera-depth derivation."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


# Float NPY is the source of truth.  TIFF is deliberately only an 8-bit
# visualisation/debug preview; higher camera depths are derived in memory by
# analysis when required and must never be recreated by a render no-op check.
PREVIEW_BIT_DEPTH = 8


def camera_tiff_path(float_path: Path, bits: int) -> Path:
    """Return the bit-depth companion for a canonical float ``.npy`` image."""
    return float_path.with_name(f"{float_path.stem}_b{bits}.tiff")


def quantise_camera(image: np.ndarray, bits: int) -> np.ndarray:
    """Clamp a normalised float image and return its camera code values."""
    maximum = (1 << bits) - 1
    codes = np.rint(np.clip(image, 0.0, 1.0) * maximum)
    return codes.astype(np.uint8 if bits <= 8 else np.uint16)


def write_camera_depths(float_path: Path, bit_depths: Iterable[int]) -> None:
    """Create only the canonical 8-bit TIFF preview from a float image."""
    image = np.asarray(np.load(float_path, mmap_mode="r"), dtype=np.float64)
    output = camera_tiff_path(float_path, PREVIEW_BIT_DEPTH)
    if not output.exists():
        Image.fromarray(quantise_camera(image, PREVIEW_BIT_DEPTH)).save(output)


def float_and_depths_complete(float_path: Path, bit_depths: Iterable[int]) -> bool:
    return float_path.is_file() and camera_tiff_path(float_path, PREVIEW_BIT_DEPTH).is_file()


def save_float_and_depths(float_path: Path, image: np.ndarray, bit_depths: Iterable[int]) -> None:
    """Persist one canonical normalised float image then derive camera TIFFs."""
    float_path.parent.mkdir(parents=True, exist_ok=True)
    if not float_path.exists():
        np.save(float_path, np.ascontiguousarray(image, dtype=np.float64))
    write_camera_depths(float_path, bit_depths)
