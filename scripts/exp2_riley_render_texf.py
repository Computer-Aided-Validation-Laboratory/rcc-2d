# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Render raw Exp2 coverage textures with Riley and save f64 NumPy outputs.

Riley samples and pixel-integrates the unbounded additive coverage texture in
memory only.  Each saved ``*_raw.npy`` is that direct Riley result; each
``*_clamped.npy`` applies Exp2's coverage-to-intensity transform afterwards.
"""

import os
import sys
from pathlib import Path

import numpy as np
import riley

from exp1common import parse_case_params
from exp2params import (
    ACTIVE_FRAMES,
    ANALYTIC_SPECKLE_TYPES,
    BLACK_AREA_FRACTIONS,
    DEFORMATION_CASES,
    FORCE_RENDER_OVER,
    GAMMA,
    I0,
    PERTURBATION_DISTRIBUTIONS,
    PERTURBATION_FRACTIONS,
    RANDOM_SEED,
    RILEY_RASTER_THREADS,
    RILEY_SSAA_LEVLES,
    TEX_OVERSAMPLES,
    TEX_INTERPOLATORS,
    TEX_PX_PAD,
    TEXTURE_OUTPUT_DIR,
    TARG_PX_X,
    TARG_PX_Y,
)


OUTPUT_ROOT = Path("./out/exp2_riley_render_texf")


def get_ssaa_levels() -> list[int]:
    value = os.environ.get("EXP2_RILEY_SSAA_LEVELS")
    if not value:
        return list(RILEY_SSAA_LEVLES)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def get_texture_oversamples() -> list[int]:
    value = os.environ.get("EXP2_TEX_OVERSAMPLES")
    if not value:
        return TEX_OVERSAMPLES
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def get_texture_interpolators() -> list[str]:
    value = os.environ.get("EXP2_TEX_INTERPOLATORS")
    interps = list(TEX_INTERPOLATORS) if not value else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    invalid = [interp for interp in interps if interp not in TEX_INTERPOLATORS]
    if invalid:
        raise ValueError(
            f"Unsupported texture interpolator(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(TEX_INTERPOLATORS)}"
        )
    return interps


def pattern_tag(
    pattern_type: str,
    black_fraction: float,
    distribution: str,
    fraction: float,
) -> str:
    return (
        f"{pattern_type}_blackfrac{black_fraction:g}_"
        f"{distribution}_j{fraction:g}_seed{RANDOM_SEED}"
    )


def get_riley_mesh_type(nodes_per_elem: int) -> riley.MeshType:
    mesh_types = {
        3: riley.MeshType.tri3,
        4: riley.MeshType.quad4ibi,
        6: riley.MeshType.tri6,
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
    camera_pixels: int,
    pad: int,
    oversamp: int,
) -> np.ndarray:
    """Map world coordinates to centres of rows-flipped texture texels."""
    tex_w = oversamp * (camera_pixels + 2 * pad)
    tex_h = oversamp * (camera_pixels + 2 * pad)
    pixel_size = roi_size / camera_pixels
    texel_size = pixel_size / oversamp
    x_start = -0.5 * roi_size - pad * pixel_size
    y_end = 0.5 * roi_size + pad * pixel_size

    uvs = np.empty((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = ((coords[:, 0] - x_start) / texel_size - 0.5) / (tex_w - 1.0)
    uvs[:, 1] = ((y_end - coords[:, 1]) / texel_size - 0.5) / (tex_h - 1.0)
    return np.ascontiguousarray(uvs)


def load_raw_texture(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load one raw, unbounded, f64 coverage texture without rescaling it."""
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


def intensity_from_coverage(coverage: np.ndarray) -> np.ndarray:
    """Apply the exact Exp2 post-integration clamp and invert-colour map."""
    inverse_coverage = 1.0 - np.clip(coverage, 0.0, 1.0)
    return np.clip(I0 + GAMMA * (2.0 * inverse_coverage - 1.0), 0.0, 1.0)


def render_exists(case_out: Path, frames: range) -> bool:
    """Return whether both f64 arrays exist for every Riley frame."""
    return all(
        (case_out / f"image_c00_f{frame:02d}_raw.npy").exists()
        and (case_out / f"image_c00_f{frame:02d}_clamped.npy").exists()
        for frame in frames
    )


def main() -> None:
    print("Experiment 2: Riley raw floating coverage texture render")
    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / name for name in DEFORMATION_CASES]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    p_val = max(TARG_PX_X, TARG_PX_Y)
    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist. Skipping.")
            continue
        coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
        connect = np.loadtxt(
            case_path / "connectivity.csv", delimiter=",", dtype=np.uintp
        )
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
        frame_range = range(num_frames)

        camera_pixels, roi_size = parse_case_params(case_path)
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
        mesh_type = get_riley_mesh_type(connect.shape[1])

        for pattern_type in ANALYTIC_SPECKLE_TYPES:
            for black_fraction in BLACK_AREA_FRACTIONS:
                for distribution in PERTURBATION_DISTRIBUTIONS:
                    for fraction in PERTURBATION_FRACTIONS:
                        tag = pattern_tag(
                            pattern_type, black_fraction, distribution, fraction
                        )
                        for interp_name in get_texture_interpolators():
                            for ssaa in get_ssaa_levels():
                                for oversamp in get_texture_oversamples():
                                    case_out = (
                                        OUTPUT_ROOT
                                        / f"{case_path.name}_{tag}_{interp_name}"
                                        / f"ss{ssaa}_oversamp{oversamp}"
                                    )
                                    if not FORCE_RENDER_OVER and render_exists(
                                        case_out, frame_range
                                    ):
                                        print(f"  {case_out.name}: outputs exist; skipping.")
                                        continue
                                    texture_path = TEXTURE_OUTPUT_DIR / (
                                        f"tex_px{p_val}_{tag}_pad{TEX_PX_PAD}"
                                        f"_oversamp{oversamp}_analytic.npy"
                                    )
                                    if not texture_path.exists():
                                        print(f"Warning: {texture_path.name} does not exist. Skipping.")
                                        continue
                                    tex_size = oversamp * (camera_pixels + 2 * TEX_PX_PAD)
                                    texture = load_raw_texture(
                                        texture_path, (tex_size, tex_size)
                                    )
                                    print(
                                        f"  {tag}, interp={interp_name}, "
                                        f"SSAA={ssaa}, oversamp={oversamp}"
                                    )
                                    mesh = riley.Mesh(
                                        mesh_type=mesh_type,
                                        coords=coords,
                                        connect=connect,
                                        disp=disp,
                                        shader_type=riley.ShaderType.tex,
                                        uvs=compute_texture_world_uvs(
                                            coords,
                                            roi_size,
                                            camera_pixels,
                                            TEX_PX_PAD,
                                            oversamp,
                                        ),
                                        texture=texture,
                                        texture_storage=riley.TextureStorage.floating,
                                        sample=TEX_INTERPOLATORS[interp_name],
                                        sample_mode=riley.TextureSampleMode.direct,
                                        bits=16,
                                        scaling_type=riley.ScaleStrategy.none,
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
                                        save_strategy=riley.SaveStrategy.memory,
                                    )
                                    config.frame_batch_size_per_group = 1
                                    config.max_geom_jobs_in_flight_per_group = 1
                                    config.max_geom_workers_per_job = 1
                                    config.max_raster_workers_per_job = RILEY_RASTER_THREADS
                                    config.tile_size_min = 1
                                    images = riley.raster([mesh], [camera], config)
                                    if images is None:
                                        raise RuntimeError("Riley returned no in-memory image data.")
                                    case_out.mkdir(parents=True, exist_ok=True)
                                    for frame in frame_range:
                                        raw = np.asarray(images[0, frame, 0], dtype=np.float64)
                                        np.save(case_out / f"image_c00_f{frame:02d}_raw.npy", raw)
                                        np.save(
                                            case_out / f"image_c00_f{frame:02d}_clamped.npy",
                                            intensity_from_coverage(raw),
                                        )

    print("All raw and clamped Riley coverage renders completed.")


if __name__ == "__main__":
    main()
