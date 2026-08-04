# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Render Exp2 digitised coverage textures with Riley.

The float renderer samples the analytic f64 coverage textures directly.  This
companion renderer instead samples their 8-, 12-, and 16-bit TIFF versions.
It writes normalised coverage and the corresponding clamped intensity so its
outputs have the same representation as ``exp2_riley_render_texfloat.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import Image
import riley

from modules.exp1common import output_case_name, parse_case_params
from exp2params import (
    ACTIVE_FRAMES,
    ANALYTIC_SPECKLE_TYPES,
    BLACK_AREA_FRACTIONS,
    BIT_DEPTHS,
    DEFORMATION_CASES,
    FORCE_RENDER_OVER,
    RILEY_RASTER_THREADS,
    PSF_SIGMA_FINAL_PX,
    PSF_SUPPORT_SIGMAS,
    TEX_INTERPOLATORS,
    RILEY_TEXTURE_SAMPLERS,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
    TARG_PX_X,
    TARG_PX_Y,
    exp2_output_dir,
)
from modules.psf_riley_common import camera_kwargs, configure_raster_config, enabled as psf_enabled, output_name as psf_output_name
from exp2_riley_render_texfloat import (
    compute_texture_world_uvs,
    get_riley_mesh_type,
    get_ssaa_levels,
    get_texture_interpolators,
    get_texture_oversamples,
    intensity_from_coverage,
    pattern_tag,
    should_render_pair,
)
from exp2params import additive_jitter_for
from exp2_texgen_speckle_analytic import generate_texture
from modules.script_timing import ScriptTimer, timed_call
from modules.render_selection import riley_enabled
from modules.render_logging import case_label, render_log
from modules.output_naming import config_name
from modules.exp12_geometry import roi_corners


# Exp2 deliberately creates large oversampled textures.  They are local,
# trusted data, so Pillow's decompression-bomb safeguard is not applicable.
Image.MAX_IMAGE_PIXELS = None
OUTPUT_ROOT = exp2_output_dir(
    psf_output_name("exp2_riley_render_texuint_psf")
    if psf_enabled() else "exp2_riley_render_texuint"
)


def get_bit_depths() -> list[int]:
    value = os.environ.get("EXP2_BIT_DEPTHS")
    if not value:
        return list(BIT_DEPTHS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ensure_uint_texture(
    path: Path,
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
    oversamp: int,
    bit_depth: int,
) -> None:
    """Generate one missing digitised coverage texture dependency on demand."""
    if not path.exists():
        print(f"  Generating missing texture dependency: {path.name}")
        generate_texture(
            pattern_type, black_fraction, distribution, fraction, oversamp,
            bit_depths=(bit_depth,),
        )
    if not path.exists():
        raise FileNotFoundError(f"Required uint texture was not generated: {path}")


def load_uint_texture(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, riley.TextureStorage]:
    """Load a digitised coverage texture with its native Riley storage type."""
    with Image.open(path) as image:
        if image.size != (expected_shape[1], expected_shape[0]):
            raise ValueError(
                f"Texture {path} has shape {(image.height, image.width)}; "
                f"expected {expected_shape}."
            )
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            return np.ascontiguousarray(np.asarray(image), dtype=np.uint16), riley.TextureStorage.u16
        return np.ascontiguousarray(np.asarray(image.convert("L")), dtype=np.uint8), riley.TextureStorage.u8


def render_exists(case_out: Path, frames: range) -> bool:
    return all(
        (case_out / f"image_c00_f{frame:02d}_raw.npy").exists()
        and (case_out / f"image_c00_f{frame:02d}_clamped.npy").exists()
        and (case_out / f"image_c00_f{frame:02d}_clamped_b8.tiff").exists()
        for frame in frames
    )


def main() -> None:
    if not riley_enabled("texuint_psf" if psf_enabled() else "texuint"):
        print("Experiment 2 Riley uint textures disabled by RILEY_RENDER_CASES; skipping.")
        return
    print("Experiment 2: Riley digitised coverage texture render")
    timer = ScriptTimer(__file__)
    cases = [Path(sys.argv[1])] if len(sys.argv) > 1 else [
        Path("data") / name for name in DEFORMATION_CASES
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    texture_pixels = max(TARG_PX_X, TARG_PX_Y)

    for case_path in cases:
        if not case_path.exists():
            raise FileNotFoundError(f"Required deformation case does not exist: {case_path}")
        coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
        connect = np.loadtxt(case_path / "connectivity.csv", delimiter=",", dtype=np.uintp)
        disp_x = np.loadtxt(case_path / "field_disp_x.csv", delimiter=",")
        disp_y = np.loadtxt(case_path / "field_disp_y.csv", delimiter=",")
        if connect.ndim == 1:
            connect = connect.reshape(1, -1)
        if disp_x.ndim == 1:
            disp_x = disp_x.reshape(-1, 1)
        if disp_y.ndim == 1:
            disp_y = disp_y.reshape(-1, 1)
        connect = np.ascontiguousarray(connect, dtype=np.uintp)
        num_nodes, num_frames = disp_x.shape
        disp = np.zeros((num_frames, num_nodes, 3), dtype=np.float64)
        disp[:, :, 0] = disp_x.T
        disp[:, :, 1] = disp_y.T
        frames = range(num_frames)
        _camera_pixels, roi_size = parse_case_params(case_path)
        roi_coords = roi_corners()
        camera_pos = riley.pos_fill_frame_from_rot(
            roi_coords, (TARG_PX_X, TARG_PX_Y), (1.0, 1.0), 1000.0,
            (0.0, 0.0, 0.0), 1.0,
        )
        roi_pos = tuple(riley.roi_cent_from_coords(roi_coords))
        mesh_type = get_riley_mesh_type(connect.shape[1])
        case_name = output_case_name(case_path.name, TARG_PX_X)

        pattern_types = ["diskaddsat"] if psf_enabled() else ANALYTIC_SPECKLE_TYPES
        for pattern_type in pattern_types:
            for black_fraction in BLACK_AREA_FRACTIONS:
                distribution, fraction = additive_jitter_for(pattern_type)
                tag = pattern_tag(pattern_type, black_fraction, distribution, fraction)
                for interp_name in get_texture_interpolators():
                    for ssaa in get_ssaa_levels():
                        for bit_depth in get_bit_depths():
                            maximum = float(2**bit_depth - 1)
                            for oversamp in get_texture_oversamples():
                                if not should_render_pair(ssaa, oversamp):
                                    continue
                                case_out = (
                                    OUTPUT_ROOT / f"{case_name}_{tag}_{config_name(interp_name)}"
                                    / config_name(f"ss{ssaa}_b{bit_depth}_oversamp{oversamp}")
                                )
                                if not FORCE_RENDER_OVER and render_exists(case_out, frames):
                                    print(f"  {case_out.name}: outputs exist; skipping.")
                                    continue
                                texture_path = TEXTURE_OUTPUT_DIR / config_name(
                                    f"tex_px{texture_pixels}_{tag}_pad{TEX_PX_PAD}"
                                    f"_oversamp{oversamp}_analytic_b{bit_depth}.tiff"
                                )
                                if not texture_path.exists():
                                    ensure_uint_texture(
                                        texture_path, pattern_type, black_fraction,
                                        distribution, fraction, oversamp, bit_depth,
                                    )
                                tex_size = oversamp * (texture_pixels + 2 * TEX_PX_PAD)
                                texture, texture_storage = load_uint_texture(
                                    texture_path, (tex_size, tex_size)
                                )
                                render_log("EXP2", "riley-texuint", case_label(case_name),
                                           f"starting {pattern_type}; interp={interp_name}; SSAA={ssaa}; OS={oversamp}; texture-bits={bit_depth}")
                                mesh = riley.Mesh(
                                    mesh_type=mesh_type, coords=coords, connect=connect, disp=disp,
                                    shader_type=riley.ShaderType.tex,
                                    uvs=compute_texture_world_uvs(
                                        coords, roi_size, texture_pixels, TEX_PX_PAD, oversamp
                                    ),
                                    texture=texture, texture_storage=texture_storage,
                                    sample=RILEY_TEXTURE_SAMPLERS[interp_name],
                                    sample_mode=riley.TextureSampleMode.direct,
                                    bits=bit_depth, scaling_type=riley.ScaleStrategy.fixed,
                                    scaling_min=0.0, scaling_max=maximum,
                                )
                                camera = riley.Camera(
                                    pixels_num=(TARG_PX_X, TARG_PX_Y), pixels_size=(1.0, 1.0),
                                    pos_world=camera_pos, rot_world=(0.0, 0.0, 0.0),
                                    roi_cent_world=roi_pos, focal_length=1000.0,
                                    sub_sample=ssaa, coord_sys=riley.CameraCoordSys.opengl,
                                    **camera_kwargs(PSF_SIGMA_FINAL_PX, PSF_SUPPORT_SIGMAS),
                                )
                                config = riley.create_raster_config(
                                    num_frames=num_frames,
                                    total_threads=RILEY_RASTER_THREADS,
                                    save_strategy=riley.SaveStrategy.memory,
                                )
                                config.frame_batch_size_per_group = 1
                                config.max_geom_jobs_in_flight_per_group = 1
                                config.max_geom_workers_per_job = 1
                                config.max_raster_workers_per_job = RILEY_RASTER_THREADS
                                config.tile_size_min = 1
                                configure_raster_config(
                                    config, workers=RILEY_RASTER_THREADS,
                                    sub_sample=ssaa,
                                )
                                images = timed_call(timer, str(case_out), riley.raster, [mesh], [camera], config)
                                if images is None:
                                    raise RuntimeError("Riley returned no in-memory image data.")
                                case_out.mkdir(parents=True, exist_ok=True)
                                for frame in frames:
                                    coverage = np.asarray(images[0, frame, 0], dtype=np.float64) / maximum
                                    np.save(case_out / f"image_c00_f{frame:02d}_raw.npy", coverage)
                                    clamped = intensity_from_coverage(coverage)
                                    np.save(case_out / f"image_c00_f{frame:02d}_clamped.npy", clamped)
                                    counts = np.rint(np.flipud(clamped) * 255.0).astype(np.uint8)
                                    Image.fromarray(counts).save(
                                        case_out / f"image_c00_f{frame:02d}_clamped_b8.tiff"
                                    )
    print("All digitised Riley coverage renders completed.")


if __name__ == "__main__":
    main()
