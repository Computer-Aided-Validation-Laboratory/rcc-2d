# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
# --------------------------------------------------------------------------

"""Common input, UV, and memory-planning helpers for Exp3 Riley renders."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from PIL import Image
import riley

from exp1common import parse_case_params
from exp3params import (
    RILEY_MAX_FILTER_HALO_PX,
    RILEY_MAX_THREADS,
    RILEY_MEMORY_BUDGET_BYTES,
    RILEY_RESERVED_MEMORY_BYTES,
    RILEY_SCRATCH_BYTES_PER_SUBPIXEL,
    RILEY_TEXTURE_BINDING_OVERHEAD,
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
)


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class RasterPlan:
    """Memory-safe Riley worker allocation for one texture/SSAA combination."""

    threads: int
    texture_bytes: int
    scratch_bytes_per_worker: int


def selected_ints(env_name: str, configured: list[int]) -> list[int]:
    value = os.environ.get(env_name)
    return list(configured) if not value else [
        int(item.strip()) for item in value.split(",") if item.strip()
    ]


def selected_names(
    env_name: str, configured: dict[str, riley.TextureSample]
) -> list[str]:
    value = os.environ.get(env_name)
    names = list(configured) if not value else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    invalid = set(names).difference(configured)
    if invalid:
        raise ValueError(
            f"Unsupported texture interpolator(s): {', '.join(sorted(invalid))}. "
            f"Choose from: {', '.join(configured)}"
        )
    return names


def texture_shape(oversamp: int) -> tuple[int, int]:
    return (
        oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD),
        oversamp * (TARG_PX_X + 2 * TEX_PX_PAD),
    )


def raster_plan(oversamp: int, ssaa: int, texture_item_bytes: int) -> RasterPlan:
    """Allocate as many workers as fit the conservative Exp3 RAM budget."""
    tex_h, tex_w = texture_shape(oversamp)
    texture_bytes = tex_h * tex_w * texture_item_bytes
    scratch_width = 1 + 2 * RILEY_MAX_FILTER_HALO_PX
    scratch_per_worker = (
        RILEY_SCRATCH_BYTES_PER_SUBPIXEL * (scratch_width * ssaa) ** 2
    )
    free_for_scratch = (
        RILEY_MEMORY_BUDGET_BYTES
        - RILEY_RESERVED_MEMORY_BYTES
        - int(texture_bytes * RILEY_TEXTURE_BINDING_OVERHEAD)
    )
    threads = min(RILEY_MAX_THREADS, free_for_scratch // scratch_per_worker)
    if threads < 1:
        raise MemoryError(
            "Exp3 memory budget cannot support "
            f"oversamp={oversamp}, SSAA={ssaa}: texture uses "
            f"{texture_bytes / 2**30:.2f} GiB and one Riley worker needs "
            f"{scratch_per_worker / 2**30:.2f} GiB."
        )
    return RasterPlan(
        threads=int(threads),
        texture_bytes=texture_bytes,
        scratch_bytes_per_worker=scratch_per_worker,
    )


def compute_texture_world_uvs(
    coords: np.ndarray, roi_size: float, camera_pixels: int, oversamp: int
) -> np.ndarray:
    """Map world coordinates to centres of the rows-flipped padded texture."""
    tex_h, tex_w = texture_shape(oversamp)
    pixel_size = roi_size / camera_pixels
    texel_size = pixel_size / oversamp
    x_start = -0.5 * roi_size - TEX_PX_PAD * pixel_size
    y_end = 0.5 * roi_size + TEX_PX_PAD * pixel_size
    uvs = np.empty((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = ((coords[:, 0] - x_start) / texel_size - 0.5) / (tex_w - 1.0)
    uvs[:, 1] = ((y_end - coords[:, 1]) / texel_size - 0.5) / (tex_h - 1.0)
    return np.ascontiguousarray(uvs)


def mesh_type(nodes_per_element: int) -> riley.MeshType:
    types = {3: riley.MeshType.tri3, 4: riley.MeshType.quad4ibi,
             6: riley.MeshType.tri6, 8: riley.MeshType.quad8,
             9: riley.MeshType.quad9}
    try:
        return types[nodes_per_element]
    except KeyError as error:
        raise ValueError(f"Unsupported mesh connectivity width: {nodes_per_element}") from error


def load_case(case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Load an Exp1/2 mesh and displacement sequence for a 32-pixel camera."""
    coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
    connect = np.loadtxt(case_path / "connectivity.csv", delimiter=",", dtype=np.uintp)
    disp_x = np.loadtxt(case_path / "field_disp_x.csv", delimiter=",")
    disp_y = np.loadtxt(case_path / "field_disp_y.csv", delimiter=",")
    if connect.ndim == 1:
        connect = connect.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x, disp_y = disp_x.reshape(-1, 1), disp_y.reshape(-1, 1)
    disp = np.zeros((disp_x.shape[1], disp_x.shape[0], 3), dtype=np.float64)
    disp[:, :, 0], disp[:, :, 1] = disp_x.T, disp_y.T
    _, roi_size = parse_case_params(case_path)
    return coords, np.ascontiguousarray(connect), disp, roi_size, disp.shape[0]


def camera_for_roi(roi_size: float, ssaa: int) -> riley.Camera:
    """Create a 32-pixel camera covering the case's world-space ROI."""
    half = 0.5 * roi_size
    roi_coords = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0],
         [half, half, 0.0], [-half, half, 0.0]], dtype=np.float64,
    )
    pos = riley.pos_fill_frame_from_rot(
        roi_coords, (TARG_PX_X, TARG_PX_Y), (1.0, 1.0), 1000.0,
        (0.0, 0.0, 0.0), 1.0,
    )
    roi_pos = tuple(riley.roi_cent_from_coords(roi_coords))
    return riley.Camera(
        pixels_num=(TARG_PX_X, TARG_PX_Y), pixels_size=(1.0, 1.0),
        pos_world=pos, rot_world=(0.0, 0.0, 0.0), roi_cent_world=roi_pos,
        focal_length=1000.0, sub_sample=ssaa, coord_sys=riley.CameraCoordSys.opengl,
    )


def load_float_texture(path: Path, shape: tuple[int, int]) -> np.ndarray:
    texture = np.load(path, mmap_mode="r")
    if texture.shape != shape or texture.dtype not in (np.float32, np.float64):
        raise ValueError(f"Texture {path} must be a {shape} float32/float64 array.")
    if not np.isfinite(texture).all():
        raise ValueError(f"Texture {path} contains non-finite values.")
    return np.ascontiguousarray(texture, dtype=np.float64)


def load_digitised_texture(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, riley.TextureStorage]:
    with Image.open(path) as image:
        texture = np.ascontiguousarray(np.asarray(image))
    if texture.shape != shape:
        raise ValueError(f"Texture {path} has shape {texture.shape}; expected {shape}.")
    if texture.dtype == np.uint8:
        return texture, riley.TextureStorage.u8
    if texture.dtype == np.uint16:
        return texture, riley.TextureStorage.u16
    raise ValueError(f"Texture {path} must be uint8 or uint16, not {texture.dtype}.")


def complete(case_out: Path, frames: int, suffixes: tuple[str, ...]) -> bool:
    return all(
        (case_out / f"image_c00_f{frame:02d}{suffix}.npy").exists()
        for frame in range(frames) for suffix in suffixes
    )


def make_config(frames: int, plan: RasterPlan) -> riley.RasterConfig:
    config = riley.create_raster_config(
        num_frames=frames, total_threads=plan.threads,
        save_strategy=riley.SaveStrategy.memory,
    )
    config.frame_batch_size_per_group = 1
    config.max_geom_jobs_in_flight_per_group = 1
    config.max_geom_workers_per_job = 1
    config.max_raster_workers_per_job = plan.threads
    config.tile_size_min = 1
    return config
