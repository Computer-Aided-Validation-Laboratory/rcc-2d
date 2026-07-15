# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from exp1common import evaluate_eggbox_analytic_average
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    TEX_OVERSAMPLES,
    BIT_DEPTHS,
    P_PIXELS,
    I0,
    GAMMA,
    TEXTURE_OUTPUT_DIR,
)


def generate_texture(
    method: str,
    param: int,
    bb: int,
    oversamp: int,
) -> None:
    """Generate and save reference texture image and float files."""
    tex_w: int = oversamp * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h: int = oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD)

    roi_size: float = float(max(TARG_PX_X, TARG_PX_Y))
    pixel_size: float = roi_size / float(max(TARG_PX_X, TARG_PX_Y))

    pixel_size_tex: float = pixel_size / float(oversamp)

    half_roi: float = roi_size / 2.0
    pad_phys_x: float = float(TEX_PX_PAD) * pixel_size
    pad_phys_y: float = float(TEX_PX_PAD) * pixel_size

    start_x = -half_roi - pad_phys_x
    start_y = -half_roi - pad_phys_y
    p_phys: float = P_PIXELS * pixel_size

    if method == "analytic":
        pixel_raw = evaluate_eggbox_analytic_average(
            start_x=start_x,
            start_y=start_y,
            pixel_size=pixel_size_tex,
            num_px_x=tex_w,
            num_px_y=tex_h,
            p_phys=p_phys,
            i0=I0,
            gamma=GAMMA,
        )
    else:
        raise ValueError(
            f"Unsupported texture generation method: {method}. "
            "Only 'analytic' is supported."
        )

    pixel_raw_flipped: np.ndarray = np.flipud(pixel_raw)

    max_val_bb: float = float(2**bb - 1)
    pixel_bb: np.ndarray = np.round(pixel_raw_flipped * max_val_bb)
    pixel_bb = np.clip(pixel_bb, 0.0, max_val_bb)

    tex_out_dir = TEXTURE_OUTPUT_DIR
    tex_out_dir.mkdir(parents=True, exist_ok=True)

    p_val: int = max(TARG_PX_X, TARG_PX_Y)
    prefix: str = (
        f"tex_px{p_val}_int_{method}_param_{param}_b{bb}"
        f"_pad{TEX_PX_PAD}_oversamp{oversamp}"
    )

    if bb == 8:
        pixel_8: np.ndarray = pixel_bb.astype(np.uint8)
        img: Image.Image = Image.fromarray(pixel_8)
        img.save(tex_out_dir / f"{prefix}.tiff")
    else:
        pixel_16: np.ndarray = pixel_bb.astype(np.uint16)
        img = Image.fromarray(pixel_16)
        img.save(tex_out_dir / f"{prefix}.tiff")


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Sinusoidal Grid Texture Generator")
    print(80 * "=")

    print("\nGenerating reference textures...")
    tex_dir_ref = TEXTURE_OUTPUT_DIR
    if tex_dir_ref.exists():
        for p in tex_dir_ref.glob("*_int_rect_*"):
            p.unlink()

    for bb in BIT_DEPTHS:
        for oversamp in TEX_OVERSAMPLES:
            print(
                f"  Texture: analytic=0, bb={bb}, "
                f"oversamp={oversamp}"
            )
            generate_texture("analytic", 0, bb, oversamp)

    print("\nAll reference textures generated successfully!")


if __name__ == "__main__":
    main()
