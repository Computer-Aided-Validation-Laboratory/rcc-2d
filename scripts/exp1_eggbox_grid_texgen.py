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
    FORCE_RENDER_OVER,
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
    task: tuple[int, int, int, float, float, float, float, float],
) -> tuple[int, int, np.ndarray]:
    """Evaluate one horizontal f64 texture band in a worker."""
    (
        row_start,
        row_stop,
        tex_w,
        start_x,
        start_y,
        pixel_size_tex,
        p_phys,
        i0,
    ) = task
    pixel_raw = evaluate_eggbox_analytic_average(
        start_x=start_x,
        start_y=start_y + row_start * pixel_size_tex,
        pixel_size=pixel_size_tex,
        num_px_x=tex_w,
        num_px_y=row_stop - row_start,
        p_phys=p_phys,
        i0=i0,
        gamma=GAMMA,
    )
    return row_start, row_stop, pixel_raw


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
    float_prefix = (
        f"tex_px{p_val}_int_{method}_param_{param}"
        f"_pad{TEX_PX_PAD}_oversamp{oversamp}"
    )
    float_path = tex_out_dir / f"{float_prefix}.npy"

    if (
        not FORCE_RENDER_OVER
        and output_path.exists()
        and float_path.exists()
    ):
        print(f"    outputs exist; skipping: {output_path.name}")
        return

    dtype = np.dtype(np.uint8 if bb == 8 else np.uint16)
    texture_bytes = tex_w * tex_h * dtype.itemsize
    float_bytes = tex_w * tex_h * np.dtype(np.float64).itemsize
    free_bytes = shutil.disk_usage(tex_out_dir).free
    # The f64 NumPy texture, raw TIFF staging buffer, and TIFF coexist while
    # saving.  Do not abort on this conservative estimate, but make it clear.
    required_bytes = float_bytes + 2 * texture_bytes
    if free_bytes < required_bytes:
        print(
            "  WARNING: free disk space may be insufficient for staging and "
            f"saving {output_path.name}: need about {required_bytes / 2**30:.2f} GiB, "
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
            I0,
        )
        for row_start in range(0, tex_h, rows_per_batch)
    ]
    staging_path = tex_out_dir / f".{prefix}.raw"
    if float_path.exists():
        pixel_float = np.load(float_path, mmap_mode="r")
        if pixel_float.shape != (tex_h, tex_w) or pixel_float.dtype != np.float64:
            raise ValueError(
                f"Existing texture {float_path} is not a {tex_h}x{tex_w} f64 array."
            )
        print(f"    Reusing f64 texture: {float_path.name}")
    else:
        pixel_float = np.lib.format.open_memmap(
            float_path, mode="w+", dtype=np.float64, shape=(tex_h, tex_w)
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
                    pixel_float[tex_h - row_stop : tex_h - row_start] = pixel_rows[::-1]
            pixel_float.flush()
        except Exception:
            del pixel_float
            float_path.unlink(missing_ok=True)
            raise
        finally:
            if "pixel_float" in locals():
                del pixel_float
        pixel_float = np.load(float_path, mmap_mode="r")

    pixel_bb = np.memmap(staging_path, mode="w+", dtype=dtype, shape=(tex_h, tex_w))
    try:
        max_val_bb = float(2**bb - 1)
        for row_start in range(0, tex_h, rows_per_batch):
            row_stop = min(row_start + rows_per_batch, tex_h)
            pixel_bb[row_start:row_stop] = np.clip(
                np.rint(pixel_float[row_start:row_stop] * max_val_bb),
                0.0,
                max_val_bb,
            ).astype(dtype)
        pixel_bb.flush()
        Image.fromarray(pixel_bb).save(
            output_path, format="TIFF", big_tiff=texture_bytes > 2**32
        )
    finally:
        del pixel_bb
        del pixel_float
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
