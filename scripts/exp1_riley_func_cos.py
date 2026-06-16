# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import sys
import shutil
from pathlib import Path
import numpy as np
import riley

from exp1common import parse_case_params
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    P_PIXELS,
    I0,
    GAMMA,
    BIT_DEPTHS,
    SSAA_LEVELS,
)

# Configurable constants
CASE_DIR: Path = Path("data/plate260_cam256_quad9_rigid")


def get_riley_mesh_type(nodes_per_elem: int) -> riley.MeshType:
    """Determine Riley MeshType from connectivity width."""
    if nodes_per_elem == 3:
        return riley.MeshType.tri3
    if nodes_per_elem == 6:
        return riley.MeshType.tri6
    if nodes_per_elem == 4:
        return riley.MeshType.quad4ibi
    if nodes_per_elem == 8:
        return riley.MeshType.quad8
    if nodes_per_elem == 9:
        return riley.MeshType.quad9
    raise ValueError(
        f"Unsupported element type with {nodes_per_elem} nodes."
    )


def main() -> None:
    print(80 * "=")
    print("Riley Function Shader Render (Experiment 1)")
    print(80 * "=")

    if not CASE_DIR.exists():
        print(f"Error: Case directory {CASE_DIR} does not exist.")
        sys.exit(1)

    case_name: str = CASE_DIR.name
    out_base: Path = Path(f"./out/riley_{case_name}_func")
    shutil.rmtree(out_base, ignore_errors=True)
    out_base.mkdir(parents=True, exist_ok=True)

    # Load coordinates, connectivity, displacements, and UVs
    coords: np.ndarray = np.loadtxt(
        CASE_DIR / "coords.csv", delimiter=","
    )
    connect_raw: np.ndarray = np.loadtxt(
        CASE_DIR / "connectivity.csv", delimiter=",", dtype=np.uintp
    )
    disp_x: np.ndarray = np.loadtxt(
        CASE_DIR / "field_disp_x.csv", delimiter=","
    )
    disp_y: np.ndarray = np.loadtxt(
        CASE_DIR / "field_disp_y.csv", delimiter=","
    )
    uvs: np.ndarray = np.loadtxt(
        CASE_DIR / "uvs_exp1_sin_grid.csv", delimiter=","
    )

    if connect_raw.ndim == 1:
        connect_raw = connect_raw.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x = disp_x.reshape(-1, 1)
    if disp_y.ndim == 1:
        disp_y = disp_y.reshape(-1, 1)

    connect: np.ndarray = np.ascontiguousarray(
        connect_raw, dtype=np.uintp
    )
    num_nodes: int
    num_frames: int
    num_nodes, num_frames = disp_x.shape

    # Pad displacements to 3D: (num_frames, num_nodes, 3)
    disp: np.ndarray = np.zeros(
        (num_frames, num_nodes, 3), dtype=np.float64
    )
    disp[:, :, 0] = disp_x.T
    disp[:, :, 1] = disp_y.T

    # Setup the mesh input
    mtype: riley.MeshType = get_riley_mesh_type(connect.shape[1])
    
    # Grid pitch in normalized UV space
    # Total width of the padded coordinate system is TARG_PX_X + 2 * pad
    total_pad_px: float = float(TARG_PX_X + 2 * TEX_PX_PAD)
    pitch_u: float = P_PIXELS / total_pad_px
    pitch_v: float = P_PIXELS / total_pad_px

    func_params: riley.FuncShaderParams = riley.FuncShaderParams(
        eggbox_mean=I0,
        eggbox_contrast=GAMMA,
        eggbox_pitch=(pitch_u, pitch_v),
        eggbox_phase=(-0.5, -0.5),
    )

    # Auto-placement of camera
    # We define dummy ROI coordinates centered at origin (256x256 units)
    roi_coords: np.ndarray = np.array([
        [-128.0, -128.0, 0.0],
        [ 128.0, -128.0, 0.0],
        [ 128.0,  128.0, 0.0],
        [-128.0,  128.0, 0.0],
    ], dtype=np.float64)

    camera_pixels, roi_size = parse_case_params(CASE_DIR)
    pixels_num: tuple[int, int] = (TARG_PX_X, TARG_PX_Y)
    pixels_size: tuple[float, float] = (1.0, 1.0)
    focal_length: float = 1000.0

    camera_pos: tuple[float, float, float] = (
        riley.pos_fill_frame_from_rot(
            roi_coords,
            pixels_num,
            pixels_size,
            focal_length,
            (0.0, 0.0, 0.0),
            1.0,
        )
    )

    roi_pos: tuple[float, float, float] = tuple(
        riley.roi_cent_from_coords(roi_coords)
    )

    for ss in SSAA_LEVELS:
        for bb in BIT_DEPTHS:
            print(
                f"Running Riley function render: SSAA={ss}, bits={bb}"
            )
            case_out = out_base / f"ss{ss}_b{bb}"
            case_out.mkdir(parents=True, exist_ok=True)

            mesh = riley.Mesh(
                mesh_type=mtype,
                coords=coords,
                connect=connect,
                disp=disp,
                shader_type=riley.ShaderType.func,
                uvs=uvs,
                func_shader_builtin=riley.FuncShaderBuiltin.eggbox,
                func_shader_coord_mode=riley.FuncCoordMode.uv,
                func_shader_params=func_params,
                bits=bb,
                scaling_type=riley.ScaleStrategy.fixed,
                scaling_min=0.0,
                scaling_max=1.0,
            )

            camera = riley.Camera(
                pixels_num=pixels_num,
                pixels_size=pixels_size,
                pos_world=camera_pos,
                rot_world=(0.0, 0.0, 0.0),
                roi_cent_world=roi_pos,
                focal_length=focal_length,
                sub_sample=ss,
                coord_sys=riley.CameraCoordSys.opengl,
            )

            config = riley.build_config(
                num_frames=num_frames,
                total_threads=4,
                save_strategy=riley.SaveStrategy.both,
            )
            config.save_format = riley.ImageFormat.tiff
            config.save_bits = 16 if bb in (12, 16) else 8
            config.save_scaling = riley.ScaleStrategy.none

            images = riley.raster(
                [mesh],
                [camera],
                config,
                out_dir=str(case_out),
            )

            if images is not None:
                for ff in range(num_frames):
                    frame_img = images[0, ff, 0]
                    np.save(
                        case_out / f"image_c00_f{ff:02d}.npy",
                        frame_img,
                    )

    print("All renders completed.")


if __name__ == "__main__":
    main()
