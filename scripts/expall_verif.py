"""Independent pre-sweep consistency checks for the Exp1/2 pixel convention.

This script is intentionally not invoked by any all-render or all-analysis
launcher.  Run it explicitly after data generation and before an expensive
render campaign.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from modules.exp12_geometry import (
    FRAME_STEP_PIXELS, GUARD_PIXELS, PLATE_PIXELS, ROI_PIXELS,
    TEXTURE_PAD_PIXELS, texture_world_uvs,
)


DATA = Path("data")
OUT = Path("out")
CASES = ("rigid", "affine", "quadsaddle")
RIGID_NAME = f"pt{PLATE_PIXELS}_cam{ROI_PIXELS}_q9_rig"
EPS = 1.0e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_source_data() -> None:
    for case in CASES:
        root = DATA / f"plate{PLATE_PIXELS}_cam{ROI_PIXELS}_quad9_{case}"
        require(root.is_dir(), f"Missing generated case: {root}")
        coords = np.loadtxt(root / "coords.csv", delimiter=",")
        require(np.isclose(coords[:, 0].min(), -PLATE_PIXELS / 2.0), f"{case}: wrong plate xmin")
        require(np.isclose(coords[:, 0].max(), PLATE_PIXELS / 2.0), f"{case}: wrong plate xmax")
    rigid = DATA / f"plate{PLATE_PIXELS}_cam{ROI_PIXELS}_quad9_rigid"
    dx = np.loadtxt(rigid / "field_disp_x.csv", delimiter=",")
    dy = np.loadtxt(rigid / "field_disp_y.csv", delimiter=",")
    require(np.allclose(dx[:, -1], 1.0), "Rigid frame 10 x displacement is not exactly 1 px")
    require(np.allclose(dy[:, -1], 1.0), "Rigid frame 10 y displacement is not exactly 1 px")
    require(np.allclose(dx[:, 1], FRAME_STEP_PIXELS), "Rigid frame 1 x displacement is not 0.1 px")
    require(np.allclose(dy[:, 1], FRAME_STEP_PIXELS), "Rigid frame 1 y displacement is not 0.1 px")


def check_uv_mapping() -> None:
    # Camera pixel centres map to the inner 32x32 texels of the OS=1 42x42
    # texture.  This catches a half-texel error or a vertical orientation flip.
    xx, yy = np.meshgrid(np.arange(-15.5, 16.0), np.arange(15.5, -16.0, -1.0))
    coords = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    uv = texture_world_uvs(coords, 1)
    col = uv[:, 0] * (PLATE_PIXELS - 1)
    row = uv[:, 1] * (PLATE_PIXELS - 1)
    expected_col = np.tile(np.arange(GUARD_PIXELS, GUARD_PIXELS + ROI_PIXELS), ROI_PIXELS)
    expected_row = np.repeat(np.arange(GUARD_PIXELS, GUARD_PIXELS + ROI_PIXELS), ROI_PIXELS)
    require(np.allclose(col, expected_col), "UV x mapping is not texel-centred")
    require(np.allclose(row, expected_row), "UV y mapping is flipped or not texel-centred")
    require(TEXTURE_PAD_PIXELS == GUARD_PIXELS, "Texture and geometry guard bands diverged")


def load(path: Path) -> np.ndarray:
    require(path.is_file(), f"Missing verification render: {path}")
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)


def check_close(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    error = float(np.max(np.abs(actual - expected)))
    require(error <= EPS, f"{label}: max error {error:.3e} exceeds {EPS:.1e}")


def check_integer_translation(label: str, frame0: np.ndarray, frame10: np.ndarray) -> None:
    # Positive physical x/y motion appears as (+column, -row) in stored image
    # coordinates.  Exclude the newly exposed image edge; it is not periodic.
    check_close(label, frame10[:-1, 1:], frame0[1:, :-1])


def check_render_parity() -> None:
    """Check the deliberately small OS=1/SSAA=1 rigid smoke render set."""
    custom = OUT / "exp1_grid2d_render_uvs" / RIGID_NAME
    func = OUT / "exp1_riley_render_func_uvs" / RIGID_NAME / "ss1_f"
    texture = OUT / "exp1_riley_render_texf" / f"{RIGID_NAME}_line" / "ss1_os1_f"
    for frame in (0, 10):
        bespoke = load(custom / f"targ_px32_int_rect_param_1_frame{frame:02d}.npy")
        analytic = load(custom / f"targ_px32_int_analytic_param_0_frame{frame:02d}.npy")
        riley_func = load(func / f"image_c00_f{frame:02d}.npy")
        riley_tex = load(texture / f"image_c00_f{frame:02d}.npy")
        check_close(f"Exp1 function parity frame {frame:02d}", riley_func, bespoke)
        check_close(f"Exp1 float texture/analytic frame {frame:02d}", riley_tex, analytic)
    check_integer_translation(
        "Exp1 Riley float texture rigid translation",
        load(texture / "image_c00_f00.npy"), load(texture / "image_c00_f10.npy"),
    )

    for pattern in ("diskadd_seed3", "gaussadd_seed3"):
        analytic_dir = OUT / "exp2_speck2d_render_uvs" / f"{RIGID_NAME}_{pattern}_analytic_0"
        riley_dir = OUT / "exp2_riley_render_texf" / f"{RIGID_NAME}_{pattern}_line" / "ss1_os1"
        for frame in (0, 10):
            analytic = load(analytic_dir / f"targ_px32_int_analytic_param_0_frame{frame:02d}.npy")
            riley = load(riley_dir / f"image_c00_f{frame:02d}_clamped.npy")
            check_close(f"Exp2 {pattern} float texture/analytic frame {frame:02d}", riley, analytic)
        check_integer_translation(
            f"Exp2 {pattern} Riley float texture rigid translation",
            load(riley_dir / "image_c00_f00_clamped.npy"),
            load(riley_dir / "image_c00_f10_clamped.npy"),
        )


def main() -> None:
    check_source_data()
    check_uv_mapping()
    check_render_parity()
    print("PASS: Exp1/2 source, UV, function, texture, and rigid-shift checks")


if __name__ == "__main__":
    main()
