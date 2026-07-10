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
from PIL import Image
import riley

from exp1common import parse_case_params
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    BIT_DEPTHS,
    TEX_OVERSAMPLES,
    DEFORMATION_CASES,
)

SSAA_LEVELS = [1, 2, 4, 8, 16]


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
    print("Riley Texture Shader Render (Experiment 1)")
    print(80 * "=")

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / c for c in DEFORMATION_CASES]

    tex_dir = Path("./out/exp1_analytic_grid/textures")
    p_val = max(TARG_PX_X, TARG_PX_Y)

    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist. Skipping.")
            continue

        case_name = case_path.name
        print(f"\nProcessing case: {case_name}")

        out_base = Path(f"./out/riley_{case_name}_tex")
        shutil.rmtree(out_base, ignore_errors=True)
        out_base.mkdir(parents=True, exist_ok=True)

        # Load coordinates, connectivity, displacements, and UVs
        coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
        connect_raw = np.loadtxt(
            case_path / "connectivity.csv", delimiter=",", dtype=np.uintp
        )
        disp_x = np.loadtxt(case_path / "field_disp_x.csv", delimiter=",")
        disp_y = np.loadtxt(case_path / "field_disp_y.csv", delimiter=",")
        uvs = np.loadtxt(
            case_path / "uvs_exp1_sin_grid.csv", delimiter=","
        )

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

        # Setup the mesh input type
        mtype = get_riley_mesh_type(connect.shape[1])

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

        camera_pixels, roi_size = parse_case_params(case_path)
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

        for ss in SSAA_LEVELS:
            for bb in BIT_DEPTHS:
                for oversamp in TEX_OVERSAMPLES:
                    print(
                        f"  Running Riley texture render: "
                        f"SSAA={ss}, bits={bb}, oversamp={oversamp}"
                    )

                    # Load texture
                    tex_filename = (
                        f"tex_px{p_val}_int_analytic_param_0_b{bb}"
                        f"_pad{TEX_PX_PAD}_oversamp{oversamp}.tiff"
                    )
                    tex_path = tex_dir / tex_filename

                    if not tex_path.exists():
                        print(
                            f"Warning: {tex_path.name} "
                            f"does not exist. Skipping."
                        )
                        continue

                    with Image.open(tex_path) as img_in:
                        if img_in.mode in ("I;16", "I;16B", "I;16L", "I"):
                            img_np = np.asarray(img_in)
                        else:
                            img_grey = img_in.convert("L")
                            img_np = np.asarray(img_grey)

                    # Normalize texture to [0.0, 1.0]
                    max_val_bb = float(2**bb - 1)
                    texture_raw = np.ascontiguousarray(
                        img_np, dtype=np.float64
                    )
                    texture = texture_raw / max_val_bb

                    case_out = (
                        out_base / f"ss{ss}_b{bb}_oversamp{oversamp}"
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
                        sample=riley.TextureSample.cubic_catmull_rom,
                        sample_mode=riley.TextureSampleMode.lut_lerp,
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
