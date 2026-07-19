# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Render Experiment 1 textures from normalised f64 NumPy texture files."""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import riley
from script_timing import ScriptTimer, timed_call

from exp1common import output_case_name, parse_case_params
from exp1params import (
    BIT_DEPTHS,
    CLEAR_DIR,
    DEFORMATION_CASES,
    FORCE_RENDER_OVER,
    RILEY_RASTER_THREADS,
    SSAA_LEVELS,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_OVERSAMPLES,
    TEX_INTERPOLATORS,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
    exp1_output_dir,
)


OUTPUT_ROOT = exp1_output_dir("exp1_riley_render_texfloat")


def get_ssaa_levels() -> list[int]:
    """Return configured SSAA levels, optionally restricted by the env."""
    levels_str = os.environ.get("EXP1_SSAA_LEVELS")
    if not levels_str:
        return SSAA_LEVELS
    return [int(value.strip()) for value in levels_str.split(",") if value.strip()]


def get_bit_depths() -> list[int]:
    """Return configured output bit depths, optionally restricted by env."""
    bits_str = os.environ.get("EXP1_BIT_DEPTHS")
    if not bits_str:
        return BIT_DEPTHS
    return [int(value.strip()) for value in bits_str.split(",") if value.strip()]


def get_texture_oversamples() -> list[int]:
    """Return configured texture oversamples, optionally restricted by env."""
    oversamp_str = os.environ.get("EXP1_TEX_OVERSAMPLES")
    if not oversamp_str:
        return TEX_OVERSAMPLES
    return [int(value.strip()) for value in oversamp_str.split(",") if value.strip()]


def get_texture_interpolators() -> list[str]:
    """Return named Riley texture filters, optionally restricted by env."""
    interps_str = os.environ.get("EXP1_TEX_INTERPOLATORS")
    interps = (
        list(TEX_INTERPOLATORS)
        if not interps_str
        else [value.strip() for value in interps_str.split(",") if value.strip()]
    )
    invalid = [interp for interp in interps if interp not in TEX_INTERPOLATORS]
    if invalid:
        raise ValueError(
            f"Unsupported texture interpolator(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(TEX_INTERPOLATORS)}"
        )
    return interps


def get_riley_mesh_type(nodes_per_elem: int) -> riley.MeshType:
    """Determine Riley MeshType from connectivity width."""
    mesh_types = {
        3: riley.MeshType.tri3,
        6: riley.MeshType.tri6,
        4: riley.MeshType.quad4ibi,
        8: riley.MeshType.quad8,
        9: riley.MeshType.quad9,
    }
    try:
        return mesh_types[nodes_per_elem]
    except KeyError as error:
        raise ValueError(f"Unsupported element type with {nodes_per_elem} nodes.") from error


def compute_texture_world_uvs(
    coords: np.ndarray,
    roi_size: float,
    texture_pixels: int,
    pad: int,
    oversamp: int,
) -> np.ndarray:
    """Map world coordinates to centres of the generated texture texels."""
    tex_w = oversamp * (texture_pixels + 2 * pad)
    tex_h = oversamp * (texture_pixels + 2 * pad)
    pixel_size = roi_size / texture_pixels
    texel_size = pixel_size / oversamp
    x_start = -0.5 * roi_size - pad * pixel_size
    y_end = 0.5 * roi_size + pad * pixel_size

    uvs = np.empty((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = ((coords[:, 0] - x_start) / texel_size - 0.5) / (tex_w - 1.0)
    uvs[:, 1] = ((y_end - coords[:, 1]) / texel_size - 0.5) / (tex_h - 1.0)
    return np.ascontiguousarray(uvs)


def render_exists(case_out: Path, num_frames: int) -> bool:
    """Return whether Riley wrote both saved representations for every frame."""
    return all(
        (case_out / f"cam0_frame{frame}_field0.tiff").exists()
        and (case_out / f"image_c00_f{frame:02d}.npy").exists()
        for frame in range(num_frames)
    )


def load_float_texture(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a normalised f64 analytic texture for Riley floating storage."""
    texture = np.load(path, mmap_mode="r")
    if texture.shape != expected_shape:
        raise ValueError(
            f"Texture {path} has shape {texture.shape}; expected {expected_shape}."
        )
    if texture.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Texture {path} must be float32 or float64, not {texture.dtype}.")
    if not np.isfinite(texture).all():
        raise ValueError(f"Texture {path} contains non-finite values.")
    return np.ascontiguousarray(texture, dtype=np.float64)


def main() -> None:
    print(80 * "=")
    print("Riley Floating Texture Shader Render (Experiment 1)")
    print(80 * "=")
    timer = ScriptTimer(__file__)

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / case_name for case_name in DEFORMATION_CASES]

    p_val = max(TARG_PX_X, TARG_PX_Y)
    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist. Skipping.")
            continue

        case_name = output_case_name(case_path.name, TARG_PX_X)
        print(f"\nProcessing case: {case_name}")
        is_subset_render = any(
            os.environ.get(name)
            for name in (
                "EXP1_SSAA_LEVELS",
                "EXP1_BIT_DEPTHS",
                "EXP1_TEX_OVERSAMPLES",
                "EXP1_TEX_INTERPOLATORS",
            )
        )
        if CLEAR_DIR and not is_subset_render:
            shutil.rmtree(OUTPUT_ROOT / case_name, ignore_errors=True)
            for old_out in OUTPUT_ROOT.glob(f"{case_name}_*"):
                shutil.rmtree(old_out, ignore_errors=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
        connect_raw = np.loadtxt(
            case_path / "connectivity.csv", delimiter=",", dtype=np.uintp
        )
        disp_x = np.loadtxt(case_path / "field_disp_x.csv", delimiter=",")
        disp_y = np.loadtxt(case_path / "field_disp_y.csv", delimiter=",")
        if connect_raw.ndim == 1:
            connect_raw = connect_raw.reshape(1, -1)
        if disp_x.ndim == 1:
            disp_x = disp_x.reshape(-1, 1)
        if disp_y.ndim == 1:
            disp_y = disp_y.reshape(-1, 1)
        connect = np.ascontiguousarray(connect_raw, dtype=np.uintp)
        num_nodes, num_frames = disp_x.shape
        disp = np.zeros((num_frames, num_nodes, 3), dtype=np.float64)
        disp[:, :, 0] = disp_x.T
        disp[:, :, 1] = disp_y.T

        _camera_pixels, roi_size = parse_case_params(case_path)
        texture_pixels = max(TARG_PX_X, TARG_PX_Y)
        roi_coords = np.array(
            [
                [-128.0, -128.0, 0.0],
                [128.0, -128.0, 0.0],
                [128.0, 128.0, 0.0],
                [-128.0, 128.0, 0.0],
            ],
            dtype=np.float64,
        )
        camera_pos = riley.pos_fill_frame_from_rot(
            roi_coords,
            (TARG_PX_X, TARG_PX_Y),
            (1.0, 1.0),
            1000.0,
            (0.0, 0.0, 0.0),
            1.0,
        )
        roi_pos = tuple(riley.roi_cent_from_coords(roi_coords))
        mtype = get_riley_mesh_type(connect.shape[1])

        for tex_interp in get_texture_interpolators():
            tex_sample = TEX_INTERPOLATORS[tex_interp]
            for ssaa in get_ssaa_levels():
                for bits in get_bit_depths():
                    for oversamp in get_texture_oversamples():
                        case_out = OUTPUT_ROOT / f"{case_name}_{tex_interp}" / (
                            f"ss{ssaa}_b{bits}_oversamp{oversamp}"
                        )
                        print(
                            "  Running Riley floating texture render: "
                            f"interp={tex_interp}, SSAA={ssaa}, bits={bits}, "
                            f"oversamp={oversamp}"
                        )
                        if not FORCE_RENDER_OVER and render_exists(case_out, num_frames):
                            print("    outputs exist; skipping.")
                            continue

                        tex_path = TEXTURE_OUTPUT_DIR / (
                            f"tex_px{p_val}_int_analytic_param_0"
                            f"_pad{TEX_PX_PAD}_oversamp{oversamp}.npy"
                        )
                        if not tex_path.exists():
                            print(f"Warning: {tex_path.name} does not exist. Skipping.")
                            continue
                        tex_size = oversamp * (texture_pixels + 2 * TEX_PX_PAD)
                        texture = load_float_texture(tex_path, (tex_size, tex_size))
                        uvs = compute_texture_world_uvs(
                            coords, roi_size, texture_pixels, TEX_PX_PAD, oversamp
                        )
                        case_out.mkdir(parents=True, exist_ok=True)
                        mesh = riley.Mesh(
                            mesh_type=mtype,
                            coords=coords,
                            connect=connect,
                            disp=disp,
                            shader_type=riley.ShaderType.tex,
                            uvs=uvs,
                            texture=texture,
                            texture_storage=riley.TextureStorage.floating,
                            sample=tex_sample,
                            sample_mode=riley.TextureSampleMode.direct,
                            bits=bits,
                            scaling_type=riley.ScaleStrategy.fixed,
                            scaling_min=0.0,
                            scaling_max=1.0,
                        )
                        camera = riley.Camera(
                            pixels_num=(TARG_PX_X, TARG_PX_Y),
                            pixels_size=(1.0, 1.0),
                            pos_world=camera_pos,
                            rot_world=(0.0, 0.0, 0.0),
                            roi_cent_world=roi_pos,
                            focal_length=1000.0,
                            sub_sample=ssaa,
                            coord_sys=riley.CameraCoordSys.opengl,
                        )
                        config = riley.create_raster_config(
                            num_frames=num_frames,
                            total_threads=RILEY_RASTER_THREADS,
                            save_strategy=riley.SaveStrategy.both,
                        )
                        config.frame_batch_size_per_group = 1
                        config.max_geom_jobs_in_flight_per_group = 1
                        config.max_geom_workers_per_job = 1
                        config.max_raster_workers_per_job = RILEY_RASTER_THREADS
                        config.tile_size_min = 1
                        config.save_format = riley.ImageFormat.tiff
                        config.save_bits = 16 if bits in (12, 16) else 8
                        config.save_scaling = riley.ScaleStrategy.none
                        images = timed_call(timer, str(case_out), riley.raster,
                            [mesh], [camera], config, out_dir=str(case_out)
                        )
                        if images is not None:
                            for frame in range(num_frames):
                                np.save(
                                    case_out / f"image_c00_f{frame:02d}.npy",
                                    images[0, frame, 0],
                                )

    print("All renders completed.")


if __name__ == "__main__":
    main()
