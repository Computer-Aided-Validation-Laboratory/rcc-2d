# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import os
import shutil
from multiprocessing import Pool
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
    CLEAR_DIR,
    NUM_PROCESSES,
    P_PIXELS,
    I0,
    GAMMA,
    TEXTURE_OUTPUT_DIR,
)

# Process bounded horizontal bands, so a high-oversampling texture never needs
# a full floating point image in RAM.  These can be overridden for a particular
# workstation run without changing the shared experiment parameters.
NUM_PROCESSES_RUN: int = int(
    os.environ.get(
        "EXP1_TEXGEN_NUM_PROCESSES",
        os.environ.get("EXP1_NUM_PROCESSES", str(NUM_PROCESSES)),
    )
)
MAX_TEXELS_PER_BATCH: int = int(
    os.environ.get("EXP1_TEXGEN_MAX_TEXELS_PER_BATCH", "1000000")
)

# Pillow's decompression-bomb guard also applies to some very large image
# operations.  These textures are generated locally and their dimensions are
# intentional.
Image.MAX_IMAGE_PIXELS = None


def _generate_quantized_rows(
    task: tuple[
        int, int, int, float, float, float, float, float, float, float, int
    ],
) -> tuple[int, int, np.ndarray]:
    """Evaluate and quantize one horizontal texture band in a worker."""
    (
        row_start,
        row_stop,
        tex_w,
        start_x,
        start_y,
        pixel_size_tex,
        p_phys,
        max_val_bb,
        i0,
        gamma,
        bb,
    ) = task
    pixel_raw = evaluate_eggbox_analytic_average(
        start_x=start_x,
        start_y=start_y + row_start * pixel_size_tex,
        pixel_size=pixel_size_tex,
        num_px_x=tex_w,
        num_px_y=row_stop - row_start,
        p_phys=p_phys,
        i0=i0,
        gamma=gamma,
    )
    pixel_bb = np.clip(np.rint(pixel_raw * max_val_bb), 0.0, max_val_bb)
    dtype = np.uint8 if bb == 8 else np.uint16
    return row_start, row_stop, pixel_bb.astype(dtype)


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

    if method != "analytic":
        raise ValueError(
            f"Unsupported texture generation method: {method}. "
            "Only 'analytic' is supported."
        )

    tex_out_dir = TEXTURE_OUTPUT_DIR
    tex_out_dir.mkdir(parents=True, exist_ok=True)

    p_val: int = max(TARG_PX_X, TARG_PX_Y)
    prefix: str = (
        f"tex_px{p_val}_int_{method}_param_{param}_b{bb}"
        f"_pad{TEX_PX_PAD}_oversamp{oversamp}"
    )
    output_path = tex_out_dir / f"{prefix}.tiff"

    dtype = np.dtype(np.uint8 if bb == 8 else np.uint16)
    texture_bytes = tex_w * tex_h * dtype.itemsize
    free_bytes = shutil.disk_usage(tex_out_dir).free
    # The raw staging file and the TIFF coexist while saving.  Do not abort on
    # this estimate, since filesystem compression and an existing replacement
    # file can make it conservative, but make the requirement explicit.
    if free_bytes < 2 * texture_bytes:
        print(
            "  WARNING: free disk space may be insufficient for staging and "
            f"saving {output_path.name}: need about {2 * texture_bytes / 2**30:.2f} GiB, "
            f"have {free_bytes / 2**30:.2f} GiB."
        )

    rows_per_batch = max(1, MAX_TEXELS_PER_BATCH // tex_w)
    tasks = [
        (
            row_start,
            min(row_start + rows_per_batch, tex_h),
            tex_w,
            start_x,
            start_y,
            pixel_size_tex,
            p_phys,
            float(2**bb - 1),
            I0,
            GAMMA,
            bb,
        )
        for row_start in range(0, tex_h, rows_per_batch)
    ]
    staging_path = tex_out_dir / f".{prefix}.raw"
    pixel_bb = np.memmap(
        staging_path, mode="w+", dtype=dtype, shape=(tex_h, tex_w)
    )
    print(
        f"    {len(tasks)} row batches, {NUM_PROCESSES_RUN} workers, "
        f"up to {rows_per_batch} rows/batch"
    )
    try:
        with Pool(processes=NUM_PROCESSES_RUN) as pool:
            for row_start, row_stop, pixel_rows in pool.imap_unordered(
                _generate_quantized_rows, tasks
            ):
                # The generated coordinate system grows upward, while image
                # rows grow downward.  Flip each completed band into place.
                pixel_bb[tex_h - row_stop : tex_h - row_start] = pixel_rows[::-1]
        pixel_bb.flush()

        img: Image.Image = Image.fromarray(pixel_bb)
        img.save(output_path, format="TIFF", big_tiff=texture_bytes > 2**32)
    finally:
        # Release the mapping before removing its backing file, including when
        # a worker or TIFF save raises an error.
        del pixel_bb
        if staging_path.exists():
            staging_path.unlink()



def main() -> None:
    print(80 * "=")
    print("Experiment 1: Sinusoidal Grid Texture Generator")
    print(80 * "=")

    print("\nGenerating reference textures...")
    tex_dir_ref = TEXTURE_OUTPUT_DIR
    if CLEAR_DIR and tex_dir_ref.exists():
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
