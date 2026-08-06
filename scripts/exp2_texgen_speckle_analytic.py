# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Generate exact texel averages for additive-saturation speckle fields."""

import multiprocessing
import os

import numpy as np

from exp2params import (
    ANALYTIC_SPECKLE_TYPES,
    BLACK_AREA_FRACTIONS,
    FORCE_RENDER_OVER,
    GAUSSIAN_CUTOFF_SIGMAS,
    GAMMA,
    I0,
    NUM_PROCESSES,
    additive_jitter_for,
    PX_PER_SPECK,
    RANDOM_SEED,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_OVERSAMPLES,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
)
from modules.exp2speckint2d import (
    MAX_PIXELS_PER_CHUNK,
    image_outputs_complete,
    make_speckle_pattern,
    save_image,
)
from modules.script_timing import ScriptTimer, timed_call
from modules.render_selection import uint_textures_enabled
from modules.render_logging import render_log
from modules.texture_preview import write_preview_b8
from modules.output_naming import config_name
from exp2params import BIT_DEPTHS, ENABLE_TRUE_UINT_TEXTURES

NUM_PROCESSES_RUN = max(1, min(
    NUM_PROCESSES,
    int(os.environ.get("EXP2_NUM_PROCESSES", str(NUM_PROCESSES))),
))

_worker_pattern = None
_worker_pattern_type: str | None = None
_worker_x: np.ndarray | None = None
_worker_start_y: float | None = None
_worker_texel_size: float | None = None


def _init_texture_worker(
    pattern_type: str,
    speckle_size: float,
    black_fraction: float,
    distribution: str,
    fraction: float,
    bounds: tuple[float, float, float, float],
    tex_w: int,
    texel_size: float,
) -> None:
    """Create one immutable pattern per pixel-batch worker."""
    global _worker_pattern, _worker_pattern_type, _worker_x
    global _worker_start_y, _worker_texel_size
    _worker_pattern = make_speckle_pattern(
        pattern_type,
        speckle_size,
        black_fraction,
        distribution,
        fraction,
        RANDOM_SEED,
        GAUSSIAN_CUTOFF_SIGMAS,
        bounds,
        I0,
        GAMMA,
    )
    _worker_pattern_type = pattern_type
    _worker_x = bounds[0] + np.arange(tex_w) * texel_size
    _worker_start_y = bounds[2]
    _worker_texel_size = texel_size


def _process_texture_rows(
    task: tuple[int, int],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Evaluate exact coverage and intensity for one texture row batch."""
    if (
        _worker_pattern is None
        or _worker_x is None
        or _worker_start_y is None
        or _worker_texel_size is None
        or _worker_pattern_type is None
    ):
        raise RuntimeError("Analytic texture worker was not initialised.")
    start_row, end_row = task
    y = _worker_start_y + np.arange(start_row, end_row) * _worker_texel_size
    xx, yy = np.meshgrid(_worker_x, y)
    if _worker_pattern_type == "diskaddsat":
        coverage = _worker_pattern.evaluate_diskaddsat_box_average(
            xx, yy, _worker_texel_size, _worker_texel_size
        )
    else:
        coverage = _worker_pattern.evaluate_gausscont_box_average(
            xx, yy, _worker_texel_size, _worker_texel_size
        )
    return start_row, end_row, coverage, _worker_pattern.intensity_from_coverage(coverage)


def tag(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
) -> str:
    return config_name(
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_seed{RANDOM_SEED}"
    )


def generate_texture(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
    oversample: int,
    bit_depths: tuple[int, ...] | None = None,
) -> None:
    """Generate exact axis-aligned texel averages for one analytic model."""
    prefix = config_name(
        f"tex_px{TARG_PX_X}_"
        f"{tag(pattern_type, black_fraction, distribution, fraction)}"
        f"_pad{TEX_PX_PAD}_oversamp{oversample}_analytic"
    )
    texture_bits = (
        (BIT_DEPTHS if ENABLE_TRUE_UINT_TEXTURES and uint_textures_enabled() else ())
        if bit_depths is None else bit_depths
    )
    float_path = TEXTURE_OUTPUT_DIR / f"{prefix}.npy"
    preview_path = TEXTURE_OUTPUT_DIR / f"{prefix}_preview_b8.tiff"
    # The durable Exp2 f64 asset is coverage, not display intensity.
    coverage_to_intensity = lambda coverage: np.clip(
        I0 + GAMMA * (2.0 * (1.0 - np.clip(coverage, 0.0, 1.0)) - 1.0),
        0.0, 1.0,
    )
    required_paths = [float_path] + [
        TEXTURE_OUTPUT_DIR / f"{prefix}_b{bits}.tiff" for bits in texture_bits
    ]
    if not FORCE_RENDER_OVER and all(path.exists() for path in required_paths):
        if oversample == 1 and write_preview_b8(float_path, preview_path, coverage_to_intensity):
            print(f"    wrote texture preview: {preview_path.name}")
        print("    outputs exist; skipping.")
        return
    if float_path.exists() and not FORCE_RENDER_OVER:
        # A float texture is the expensive, canonical asset.  A newly requested
        # uint depth must be derived from it, never trigger another analytic
        # texture evaluation.
        raw_coverage = np.ascontiguousarray(np.flipud(np.load(float_path)), dtype=np.float64)
        image = coverage_to_intensity(raw_coverage)
        save_image(
            image, TEXTURE_OUTPUT_DIR, prefix,
            float_texture=raw_coverage, bit_depths=texture_bits,
        )
        if oversample == 1 and write_preview_b8(float_path, preview_path, coverage_to_intensity):
            print(f"    wrote texture preview: {preview_path.name}")
        print("    generated missing digitised texture assets from the existing f64 texture.")
        return
    roi_size = float(max(TARG_PX_X, TARG_PX_Y))
    pixel_size = roi_size / max(TARG_PX_X, TARG_PX_Y)
    texel_size = pixel_size / oversample
    tex_w = oversample * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h = oversample * (TARG_PX_Y + 2 * TEX_PX_PAD)
    bounds = (
        -0.5 * roi_size - TEX_PX_PAD * pixel_size,
        0.5 * roi_size + TEX_PX_PAD * pixel_size,
        -0.5 * roi_size - TEX_PX_PAD * pixel_size,
        0.5 * roi_size + TEX_PX_PAD * pixel_size,
    )
    image = np.empty((tex_h, tex_w), dtype=np.float64)
    raw_coverage = np.empty((tex_h, tex_w), dtype=np.float64)
    rows_per_batch = max(1, MAX_PIXELS_PER_CHUNK // tex_w)
    tasks = [
        (start_row, min(start_row + rows_per_batch, tex_h))
        for start_row in range(0, tex_h, rows_per_batch)
    ]
    print(
        f"    {len(tasks)} pixel batches, {NUM_PROCESSES_RUN} workers, "
        f"up to {rows_per_batch} rows/batch"
    )
    initargs = (
        pattern_type,
        PX_PER_SPECK * pixel_size,
        black_fraction,
        distribution,
        fraction,
        bounds,
        tex_w,
        texel_size,
    )
    with multiprocessing.Pool(
        NUM_PROCESSES_RUN,
        initializer=_init_texture_worker,
        initargs=initargs,
    ) as pool:
        for start_row, end_row, coverage, intensity in pool.imap_unordered(
            _process_texture_rows, tasks
        ):
            raw_coverage[start_row:end_row] = coverage
            image[start_row:end_row] = intensity

    # Save pixel-integrated coverage as the primary f64 texture.  It is not
    # clamped: overlapping disks/Gaussians can and should exceed one.
    save_image(image, TEXTURE_OUTPUT_DIR, prefix, float_texture=raw_coverage, bit_depths=texture_bits)
    if oversample == 1 and write_preview_b8(float_path, preview_path, coverage_to_intensity):
        print(f"    wrote texture preview: {preview_path.name}")


def get_texture_oversamples() -> list[int]:
    """Return configured texture oversamples, optionally restricted by env."""
    value = os.environ.get("EXP2_TEX_OVERSAMPLES")
    if not value:
        return TEX_OVERSAMPLES
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    print("Experiment 2: analytic additive-saturation texture generator")
    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timer = ScriptTimer(__file__)
    for pattern_type in ANALYTIC_SPECKLE_TYPES:
        if pattern_type not in {"diskaddsat", "gausscont"}:
            raise ValueError(f"Unsupported analytic type: {pattern_type}")
        for black_fraction in BLACK_AREA_FRACTIONS:
            for distribution, fraction in (additive_jitter_for(pattern_type),):
                    for oversample in get_texture_oversamples():
                        pattern_name = tag(
                            pattern_type,
                            black_fraction,
                            distribution,
                            fraction,
                        )
                        render_log("EXP2", "texgen", pattern_type,
                                   f"starting OS={oversample}")
                        timed_call(
                            timer, f"{pattern_name}_oversamp{oversample}",
                            generate_texture, pattern_type, black_fraction,
                            distribution, fraction, oversample,
                        )


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
