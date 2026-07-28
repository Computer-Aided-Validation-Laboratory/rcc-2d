"""Bounded consistency checks for Experiment 3 smoke renders.

Exact parity is expected only for the undeformed eggbox function shader.  A
texture includes reconstruction error; a VTK inverse-map of a curved Quad9
mesh is likewise not bit-identical to Riley's native isoparametric inversion.
Those cases are reported as magnitudes, not treated as failed parity tests.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def image(path: Path) -> np.ndarray:
    value = np.load(path)
    # Riley's saved in-memory image is in its output integer scale.  Exp3
    # bespoke files are saved in the same top-row-first convention.
    return value / 255.0 if value.max() > 1.0 else value


def compare(label: str, left: Path, right: Path) -> tuple[float, float]:
    a, b = image(left), image(right)
    difference = np.abs(a - b)
    maximum, mean = float(difference.max()), float(difference.mean())
    print(f"{label}: max={maximum:.3e}, mean={mean:.3e}")
    return maximum, mean


def main() -> None:
    root = Path("out")
    checks = [
        ("rigid frame-0 eggbox function parity",
         root / "exp3_gridint2d_render_ssaa_im512x512/plate516_cam512_quad9_rigid/eggbox_ss4_b8/frame00.npy",
         root / "exp3_riley_render_func_im512x512/plate516_cam512_quad9_rigid/eggbox_func_ss4_b8/image_c00_f00.npy"),
        ("finite-star frame-0 eggbox function parity",
         root / "exp3_gridint2d_render_ssaa_im1020x252/plate260x65_cam256_quad9_chirp/eggbox_ss4_b8/frame00.npy",
         root / "exp3_riley_render_func_im1020x252/plate260x65_cam256_quad9_chirp/eggbox_func_ss4_b8/image_c00_f00.npy"),
    ]
    for label, custom, riley in checks:
        if custom.exists() and riley.exists():
            maximum, _ = compare(label, custom, riley)
            if maximum > 1e-10:
                raise AssertionError(f"{label} exceeds exact-parity tolerance")
    # The structured Newton path should reproduce Riley's native Quad9 inverse
    # apart from 8-bit output quantisation and a handful of camera-edge taps.
    curved = checks[1][1].with_name("frame01.npy")
    riley_curved = checks[1][2].with_name("image_c00_f01.npy")
    if curved.exists() and riley_curved.exists():
        maximum, mean = compare("finite-star frame-1 structured-Newton/Riley parity", curved, riley_curved)
        if mean > 1.0 / 255.0:
            raise AssertionError("finite-star structured Newton is not within one output LSB of Riley")
    # At a fixed OS/SSAA the only intended texfloat/texuint difference is the
    # source texture quantisation.  It must therefore remain on the existing
    # Exp1/2 scale: no more than a small multiple of one output LSB.
    for pattern in ("eggbox", "diskaddsat", "gausscont"):
        tag = pattern if pattern == "eggbox" else (
            f"{pattern}_blackfrac0.6_"
            f"{'uniform_j0.25' if pattern == 'diskaddsat' else 'gaussian_j0.12'}_seed3"
        )
        floating = root / "exp3_riley_render_texfloat_im512x512/plate516_cam512_quad9_rigid" / f"{tag}_float_nearest_os4_ss4_b8/image_c00_f00.npy"
        integer = root / "exp3_riley_render_texuint_im512x512/plate516_cam512_quad9_rigid" / f"{tag}_uint_nearest_os4_ss4_b8/image_c00_f00.npy"
        if floating.exists() and integer.exists():
            maximum, _ = compare(f"{pattern} texfloat/texuint quantisation diagnostic", floating, integer)
            if maximum > 3.0 / 255.0:
                raise AssertionError(f"{pattern}: texture-storage difference exceeds 3 output LSBs")


if __name__ == "__main__":
    main()
