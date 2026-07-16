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
from PIL import Image
import riley

from exp1common import parse_case_params
from exp1params import (
    TARG_PX_X,
    TARG_PX_Y,
    TEX_PX_PAD,
    BIT_DEPTHS,
    TEX_OVERSAMPLES,
    TEXTURE_OUTPUT_DIR,
    DEFORMATION_CASES,
    SSAA_LEVELS,
)


TEX_INTERPOLATORS: dict[str, riley.TextureSample] = {
    "nearest": riley.TextureSample.nearest,
    "linear": riley.TextureSample.linear,
    "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
    "cubic_mitchell_netravali": riley.TextureSample.cubic_mitchell_netravali,
    "lanczos3": riley.TextureSample.lanczos3,
}


def get_ssaa_levels() -> list[int]:
    """Return configured SSAA levels, optionally restricted by the env."""
    levels_str = os.environ.get("EXP1_SSAA_LEVELS")
    if not levels_str:
        return SSAA_LEVELS
    return [int(val.strip()) for val in levels_str.split(",") if val.strip()]


def get_bit_depths() -> list[int]:
    """Return configured output bit depths, optionally restricted by env."""
    bits_str = os.environ.get("EXP1_BIT_DEPTHS")
    if not bits_str:
        return BIT_DEPTHS
    return [int(val.strip()) for val in bits_str.split(",") if val.strip()]


def get_texture_oversamples() -> list[int]:
    """Return configured texture oversamples, optionally restricted by env."""
    oversamp_str = os.environ.get("EXP1_TEX_OVERSAMPLES")
    if not oversamp_str:
        return TEX_OVERSAMPLES
    return [int(val.strip()) for val in oversamp_str.split(",") if val.strip()]


def get_texture_interpolators() -> list[str]:
    """Return named Riley texture filters, optionally restricted by env."""
    interps_str = os.environ.get("EXP1_TEX_INTERPOLATORS")
    interps = (
        list(TEX_INTERPOLATORS)
        if not interps_str
        else [val.strip() for val in interps_str.split(",") if val.strip()]
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


def compute_texture_world_uvs(
    coords: np.ndarray,
    roi_size: float,
    camera_pixels: int,
    pad: int,
    oversamp: int,
) -> np.ndarray:
    """Map world coordinates to centres of the generated texture texels.

    The eggbox texture covers the fixed padded camera ROI, rather than the
    mesh bounding box.  Its rows are flipped before loading into Riley, so
    increasing world y maps to decreasing texture-row coordinate.
    """
    tex_w = oversamp * (camera_pixels + 2 * pad)
    tex_h = oversamp * (camera_pixels + 2 * pad)
    pixel_size = roi_size / camera_pixels
    texel_size = pixel_size / oversamp
    x_start = -0.5 * roi_size - pad * pixel_size
    y_end = 0.5 * roi_size + pad * pixel_size

    uvs = np.empty((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = (
        (coords[:, 0] - x_start) / texel_size - 0.5
    ) / (tex_w - 1.0)
    uvs[:, 1] = (
        (y_end - coords[:, 1]) / texel_size - 0.5
    ) / (tex_h - 1.0)
    return np.ascontiguousarray(uvs)


def main() -> None:
    print(80 * "=")
    print("Riley Texture Shader Render (Experiment 1)")
    print(80 * "=")

    if len(sys.argv) > 1:
        cases = [Path(sys.argv[1])]
    else:
        cases = [Path("data") / c for c in DEFORMATION_CASES]

    tex_dir = TEXTURE_OUTPUT_DIR
    p_val = max(TARG_PX_X, TARG_PX_Y)

    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist. Skipping.")
            continue

        case_name = case_path.name
        print(f"\nProcessing case: {case_name}")

        output_root = Path("./out/exp1_riley_render_tex")
        is_subset_render = any(
            os.environ.get(name)
            for name in (
                "EXP1_SSAA_LEVELS",
                "EXP1_BIT_DEPTHS",
                "EXP1_TEX_OVERSAMPLES",
                "EXP1_TEX_INTERPOLATORS",
            )
        )
        if not is_subset_render:
            shutil.rmtree(output_root / case_name, ignore_errors=True)
            for old_out in output_root.glob(f"{case_name}_*"):
                shutil.rmtree(old_out, ignore_errors=True)
        output_root.mkdir(parents=True, exist_ok=True)

        # Load coordinates, connectivity, and displacements.
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

        for tex_interp in get_texture_interpolators():
            tex_sample = TEX_INTERPOLATORS[tex_interp]
            for ss in get_ssaa_levels():
                for bb in get_bit_depths():
                    for oversamp in get_texture_oversamples():
                        print(
                            "  Running Riley texture render: "
                            f"interp={tex_interp}, SSAA={ss}, bits={bb}, "
                            f"oversamp={oversamp}"
                        )
                        tex_filename = (
                            f"tex_px{p_val}_int_analytic_param_0_b{bb}"
                            f"_pad{TEX_PX_PAD}_oversamp{oversamp}.tiff"
                        )
                        tex_path = tex_dir / tex_filename
                        if not tex_path.exists():
                            print(f"Warning: {tex_path.name} does not exist. Skipping.")
                            continue
                        with Image.open(tex_path) as img_in:
                            if img_in.mode in ("I;16", "I;16B", "I;16L", "I"):
                                img_np = np.asarray(img_in)
                            else:
                                img_np = np.asarray(img_in.convert("L"))

                        max_val_bb = float(2**bb - 1)
                        texture_raw = np.ascontiguousarray(img_np, dtype=np.float64)
                        if bb == 16:
                            texture = texture_raw
                            texture_scale_max = max_val_bb
                        else:
                            texture = texture_raw * (255.0 / max_val_bb)
                            texture_scale_max = 255.0
                        uvs = compute_texture_world_uvs(
                            coords, roi_size, camera_pixels, TEX_PX_PAD, oversamp
                        )
                        case_out = output_root / (
                            f"{case_name}_{tex_interp}_ss{ss}_b{bb}"
                            f"_oversamp{oversamp}"
                        )
                        case_out.mkdir(parents=True, exist_ok=True)
                        mesh = riley.Mesh(
                            mesh_type=mtype, coords=coords, connect=connect, disp=disp,
                            shader_type=riley.ShaderType.tex, uvs=uvs, texture=texture,
                            sample=tex_sample,
                            sample_mode=riley.TextureSampleMode.direct,
                            bits=bb, scaling_type=riley.ScaleStrategy.fixed,
                            scaling_min=0.0, scaling_max=texture_scale_max,
                        )
                        camera = riley.Camera(
                            pixels_num=pixels_num, pixels_size=pixels_size,
                            pos_world=camera_pos, rot_world=(0.0, 0.0, 0.0),
                            roi_cent_world=roi_pos, focal_length=focal_length,
                            sub_sample=ss, coord_sys=riley.CameraCoordSys.opengl,
                        )
                        config = riley.create_raster_config(
                            num_frames=num_frames, total_threads=4,
                            save_strategy=riley.SaveStrategy.both,
                        )
                        config.tile_size_min = 1
                        config.save_format = riley.ImageFormat.tiff
                        config.save_bits = 16 if bb in (12, 16) else 8
                        config.save_scaling = riley.ScaleStrategy.none
                        images = riley.raster(
                            [mesh], [camera], config, out_dir=str(case_out)
                        )
                        if images is not None:
                            for ff in range(num_frames):
                                np.save(
                                    case_out / f"image_c00_f{ff:02d}.npy",
                                    images[0, ff, 0],
                                )

    print("All renders completed.")


if __name__ == "__main__":
    main()
