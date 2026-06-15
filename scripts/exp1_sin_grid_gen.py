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
import pyvista as pv
from PIL import Image

from exp1common import (
    build_pv_mesh,
    parse_case_params,
    compute_padded_uvs,
)
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    TEX_OVERSAMPLES,
    BIT_DEPTHS,
    SSAA_LEVELS,
    P_PIXELS,
    I0,
    GAMMA,
    OUTPUT_DIR,
)


def generate_texture(
    ss: int,
    bb: int,
    oversamp: int,
) -> None:
    """Generate and save reference texture image and float files."""
    tex_w: int = oversamp * (TARG_PX_X + 2 * TEX_PX_PAD)
    tex_h: int = oversamp * (TARG_PX_Y + 2 * TEX_PX_PAD)

    sub_w: int = tex_w * ss
    sub_h: int = tex_h * ss

    roi_size: float = float(max(TARG_PX_X, TARG_PX_Y))
    pixel_size: float = roi_size / float(
        max(TARG_PX_X, TARG_PX_Y)
    )

    pixel_size_sub: float = pixel_size / float(oversamp * ss)

    half_roi: float = roi_size / 2.0
    pad_phys_x: float = float(TEX_PX_PAD) * pixel_size
    pad_phys_y: float = float(TEX_PX_PAD) * pixel_size

    xs: np.ndarray = (
        -half_roi - pad_phys_x + (np.arange(sub_w) + 0.5) * pixel_size_sub
    )
    ys: np.ndarray = (
        -half_roi - pad_phys_y + (np.arange(sub_h) + 0.5) * pixel_size_sub
    )
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_x, grid_y = np.meshgrid(xs, ys)

    p_phys: float = P_PIXELS * pixel_size

    cos_x: np.ndarray = np.cos(2.0 * np.pi * grid_x / p_phys)
    cos_y: np.ndarray = np.cos(2.0 * np.pi * grid_y / p_phys)
    sub_intensity: np.ndarray = (
        I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
    )

    reshaped: np.ndarray = sub_intensity.reshape(
        tex_h, ss,
        tex_w, ss
    )
    pixel_raw: np.ndarray = reshaped.mean(axis=(1, 3))
    pixel_raw_flipped: np.ndarray = np.flipud(pixel_raw)

    max_val_bb: float = float(2**bb - 1)
    pixel_bb: np.ndarray = np.round(pixel_raw_flipped * max_val_bb)
    pixel_bb = np.clip(pixel_bb, 0.0, max_val_bb)

    tex_out_dir: Path = OUTPUT_DIR / "textures"
    tex_out_dir.mkdir(parents=True, exist_ok=True)

    p_val: int = max(TARG_PX_X, TARG_PX_Y)
    prefix: str = (
        f"tex_px{p_val}_b{bb}_ss{ss}"
        f"_pad{TEX_PX_PAD}_oversamp{oversamp}"
    )

    if bb == 8:
        pixel_8: np.ndarray = pixel_bb.astype(np.uint8)
        img: Image.Image = Image.fromarray(pixel_8)
        img.save(tex_out_dir / f"{prefix}.tiff")
    else:
        pixel_16: np.ndarray = np.round(
            pixel_bb * (65535.0 / max_val_bb)
        ).astype(np.uint16)
        img: Image.Image = Image.fromarray(pixel_16)
        img.save(tex_out_dir / f"{prefix}.tiff")



def generate_grid_images(
    case_dir: Path,
    ss: int,
    bb: int,
) -> None:
    """Load mesh, interpolate displacements, and generate grid images."""
    case_name: str = case_dir.name
    print(f"\nProcessing case: {case_name} (SSAA={ss}, bit depth={bb})")

    coords: np.ndarray = np.loadtxt(
        case_dir / "coords.csv", delimiter=","
    )
    connect: np.ndarray = np.loadtxt(
        case_dir / "connectivity.csv", delimiter=",", dtype=int
    )
    disp_x: np.ndarray = np.loadtxt(
        case_dir / "field_disp_x.csv", delimiter=","
    )
    disp_y: np.ndarray = np.loadtxt(
        case_dir / "field_disp_y.csv", delimiter=","
    )

    if connect.ndim == 1:
        connect = connect.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x = disp_x.reshape(-1, 1)
    if disp_y.ndim == 1:
        disp_y = disp_y.reshape(-1, 1)

    camera_pixels: int
    roi_size: float
    camera_pixels, roi_size = parse_case_params(case_dir)
    pixel_size: float = roi_size / camera_pixels
    num_frames: int = disp_x.shape[1]

    uvs: np.ndarray = compute_padded_uvs(
        coords, roi_size, camera_pixels, TEX_PX_PAD
    )
    np.savetxt(
        case_dir / "uvs_exp1_sin_grid.csv",
        uvs,
        delimiter=",",
    )

    case_out_dir: Path = OUTPUT_DIR / case_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    sub_h: int = TARG_PX_Y * ss
    sub_w: int = TARG_PX_X * ss
    pixel_size_sub: float = pixel_size / ss

    xs: np.ndarray = (
        -roi_size / 2.0 + (np.arange(sub_w) + 0.5) * pixel_size_sub
    )
    ys: np.ndarray = (
        -roi_size / 2.0 + (np.arange(sub_h) + 0.5) * pixel_size_sub
    )
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_x, grid_y = np.meshgrid(xs, ys)

    query_pts: np.ndarray = np.zeros(
        (sub_h * sub_w, 3), dtype=np.float64
    )
    query_pts[:, 0] = grid_x.ravel()
    query_pts[:, 1] = grid_y.ravel()

    grid_query: pv.PolyData = pv.PolyData(query_pts)
    p_phys: float = P_PIXELS * pixel_size

    mesh: pv.UnstructuredGrid = build_pv_mesh(coords, connect)

    for ff in range(num_frames):
        mesh.point_data["disp_x"] = disp_x[:, ff]
        mesh.point_data["disp_y"] = disp_y[:, ff]

        sampled: pv.PolyData = grid_query.sample(mesh)
        u_x: np.ndarray = sampled.point_data["disp_x"]
        u_y: np.ndarray = sampled.point_data["disp_y"]

        valid: np.ndarray = sampled.point_data[
            "vtkValidPointMask"
        ].astype(bool)
        u_x[~valid] = 0.0
        u_y[~valid] = 0.0

        x_def: np.ndarray = grid_x.ravel() + u_x
        y_def: np.ndarray = grid_y.ravel() + u_y

        cos_x: np.ndarray = np.cos(2.0 * np.pi * x_def / p_phys)
        cos_y: np.ndarray = np.cos(2.0 * np.pi * y_def / p_phys)
        sub_intensity: np.ndarray = (
            I0 + 0.5 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y) - GAMMA
        )
        sub_intensity = sub_intensity.reshape(sub_h, sub_w)

        reshaped: np.ndarray = sub_intensity.reshape(
            TARG_PX_Y, ss,
            TARG_PX_X, ss
        )
        pixel_raw: np.ndarray = reshaped.mean(axis=(1, 3))

        pixel_raw_flipped: np.ndarray = np.flipud(pixel_raw)

        max_val_bb: float = float(2**bb - 1)
        pixel_bb: np.ndarray = np.round(pixel_raw_flipped * max_val_bb)
        pixel_bb = np.clip(pixel_bb, 0.0, max_val_bb)

        p_val: int = max(TARG_PX_X, TARG_PX_Y)
        prefix: str = f"targ_px{p_val}_ss{ss}_b{bb}_frame{ff:02d}"

        if bb == 8:
            pixel_8: np.ndarray = pixel_bb.astype(np.uint8)
            img: Image.Image = Image.fromarray(pixel_8)
            img.save(case_out_dir / f"{prefix}.tiff")
        else:
            pixel_16: np.ndarray = np.round(
                pixel_bb * (65535.0 / max_val_bb)
            ).astype(np.uint16)
            img = Image.fromarray(pixel_16)
            img.save(case_out_dir / f"{prefix}.tiff")

        np.save(
            case_out_dir / f"{prefix}.npy",
            pixel_raw_flipped * float(2**bb),
        )


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Sinusoidal Grid Generator")
    print(80 * "=")

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [
            Path("data/plate260_cam256_quad9_rigid"),
            Path("data/plate260_cam256_quad9_affine"),
        ]

    print("\nGenerating reference textures...")
    for ss in SSAA_LEVELS:
        for bb in BIT_DEPTHS:
            for oversamp in TEX_OVERSAMPLES:
                print(
                    f"  Texture: ss={ss}, bb={bb}, oversamp={oversamp}"
                )
                generate_texture(ss, bb, oversamp)

    print("\nGenerating deformed target images...")
    for case_path in cases:
        if not case_path.exists():
            print(
                f"Warning: Case directory {case_path} "
                "does not exist. Skipping."
            )
            continue
        for ss in SSAA_LEVELS:
            for bb in BIT_DEPTHS:
                generate_grid_images(case_path, ss, bb)

    print("\nAll generations completed successfully!")


if __name__ == "__main__":
    main()
