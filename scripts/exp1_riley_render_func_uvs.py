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
import os
from pathlib import Path
import numpy as np
import riley

from exp1common import (
    compute_riley_bbox_uvs,
    get_riley_bbox_uv_transform,
    parse_case_params,
)
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    P_PIXELS,
    I0,
    GAMMA,
    BIT_DEPTHS,
    CLEAR_DIR,
    DEFORMATION_CASES,
    RILEY_RASTER_THREADS,
    SSAA_LEVELS,
)

OUTPUT_ROOT = Path("./out/exp1_riley_render_func_uvs")


def get_ssaa_levels() -> list[int]:
    levels_str = os.environ.get("EXP1_SSAA_LEVELS")
    if not levels_str:
        return SSAA_LEVELS
    return [int(val.strip()) for val in levels_str.split(",") if val.strip()]


def get_bit_depths() -> list[int]:
    bits_str = os.environ.get("EXP1_BIT_DEPTHS")
    if not bits_str:
        return BIT_DEPTHS
    return [int(val.strip()) for val in bits_str.split(",") if val.strip()]


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
    print("Riley Function Shader Render (Experiment 1, UVs)")
    print(80 * "=")

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / c for c in DEFORMATION_CASES]

    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist. Skipping.")
            continue

        case_name = case_path.name
        print(f"\nProcessing case: {case_name}")

        out_base = OUTPUT_ROOT / case_name
        if CLEAR_DIR:
            shutil.rmtree(out_base, ignore_errors=True)
        out_base.mkdir(parents=True, exist_ok=True)

        # Load coordinates, connectivity, displacements, and UVs
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

        # Pad displacements to 3D: (num_frames, num_nodes, 3)
        disp = np.zeros((num_frames, num_nodes, 3), dtype=np.float64)
        disp[:, :, 0] = disp_x.T
        disp[:, :, 1] = disp_y.T

        camera_pixels, roi_size = parse_case_params(case_path)
        uvs_path = case_path / "uvs_exp1_sin_grid_uvs.csv"
        if uvs_path.exists():
            uvs = np.loadtxt(uvs_path, delimiter=",")
        else:
            uvs = compute_riley_bbox_uvs(coords, camera_pixels, TEX_PX_PAD)

        # Setup the mesh input
        mtype = get_riley_mesh_type(connect.shape[1])
        p_phys = P_PIXELS * (roi_size / camera_pixels)
        uv_scale, u_offset, v_offset = get_riley_bbox_uv_transform(
            coords, camera_pixels, TEX_PX_PAD
        )
        pitch_uv = uv_scale * p_phys

        func_params = riley.FuncShaderParams(
            eggbox_mean=I0,
            eggbox_contrast=GAMMA,
            eggbox_pitch=(pitch_uv, pitch_uv),
            eggbox_phase=(-u_offset, -v_offset),
        )

        # Auto-placement of camera
        roi_coords = np.array(
            [
                [-128.0, -128.0, 0.0],
                [128.0, -128.0, 0.0],
                [128.0, 128.0, 0.0],
                [-128.0, 128.0, 0.0],
            ],
            dtype=np.float64,
        )

        pixels_num = (TARG_PX_X, TARG_PX_Y)
        pixels_size = (1.0, 1.0)
        focal_length = 1000.0

        camera_pos = riley.pos_fill_frame_from_rot(
            roi_coords,
            pixels_num,
            pixels_size,
            focal_length,
            (0.0, 0.0, 0.0),
            1.0,
        )

        roi_pos = tuple(riley.roi_cent_from_coords(roi_coords))

        for ss in get_ssaa_levels():
            for bb in get_bit_depths():
                print(
                    f"  Running Riley function render: "
                    f"SSAA={ss}, bits={bb}"
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

                config = riley.create_raster_config(
                    num_frames=num_frames,
                    total_threads=RILEY_RASTER_THREADS,
                    save_strategy=riley.SaveStrategy.both,
                )
                # The Python wrapper creates one render group. Keep geometry
                # serial and give that group's workers to the raster loop.
                config.frame_batch_size_per_group = 1
                config.max_geom_jobs_in_flight_per_group = 1
                config.max_geom_workers_per_job = 1
                config.max_raster_workers_per_job = RILEY_RASTER_THREADS
                config.tile_size_min = 1
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
