"""Render Exp1 eggboxes with a Riley-equivalent image-plane Gaussian PSF."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from modules.exp1common import compute_riley_bbox_uvs, get_riley_bbox_uv_transform, output_case_name, parse_case_params
from exp1params import (
    ACTIVE_FRAMES, BACKGROUND, BIT_DEPTHS, DEFORMATION_CASES, FORCE_RENDER_OVER,
    GAMMA, I0, INTEGRATION_METHODS, P_PIXELS, PSF_SIGMA_FINAL_PX,
    PSF_SUPPORT_SIGMAS, TARG_PX_X, TARG_PX_Y, TEX_PX_PAD, exp1_output_dir,
    mapping_mode_for_case, NUM_PROCESSES,
)
from modules.ortho_psf_common import render_psf_frame
from modules.render_outputs import save_float_and_depths, write_camera_depths, float_and_depths_complete
from modules.render_selection import custom_enabled
from modules.render_logging import case_label, render_log

OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")


def _active_frames() -> set[int]:
    value = os.environ.get("EXP1_ACTIVE_FRAMES")
    return set(ACTIVE_FRAMES) if not value else {int(item) for item in value.split(",") if item}


def _methods() -> list[tuple[str, int]]:
    value = os.environ.get("EXP1_METHODS")
    methods = INTEGRATION_METHODS if not value else [
        (part.split(":")[0], int(part.split(":")[1])) for part in value.split(",")
    ]
    return [(name, param) for name, param in methods if name == "rect"]


def main() -> None:
    if not custom_enabled("eggbox_psf"):
        print("Experiment 1 eggbox-PSF renderer disabled by CUSTOM_RENDER_CASES; skipping.")
        return
    print("Experiment 1: bespoke orthographic eggbox render with Gaussian PSF")
    cases = [Path(sys.argv[1])] if len(sys.argv) > 1 else [Path("data") / name for name in DEFORMATION_CASES]
    support_px = PSF_SIGMA_FINAL_PX * PSF_SUPPORT_SIGMAS
    force_render = FORCE_RENDER_OVER or os.environ.get("EXP1_FORCE_RENDER_OVER") == "1"
    for case_path in cases:
        if not case_path.exists():
            print(f"Warning: {case_path} does not exist; skipping.")
            continue
        coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
        connect = np.loadtxt(case_path / "connectivity.csv", delimiter=",", dtype=int)
        disp_x = np.loadtxt(case_path / "field_disp_x.csv", delimiter=",")
        disp_y = np.loadtxt(case_path / "field_disp_y.csv", delimiter=",")
        if connect.ndim == 1: connect = connect.reshape(1, -1)
        if disp_x.ndim == 1: disp_x = disp_x.reshape(-1, 1)
        if disp_y.ndim == 1: disp_y = disp_y.reshape(-1, 1)
        _, roi_size = parse_case_params(case_path)
        uv_scale, _, _ = get_riley_bbox_uv_transform(coords, TARG_PX_X, TEX_PX_PAD)
        pitch_uv = uv_scale * P_PIXELS * (roi_size / TARG_PX_X)
        case_out = OUTPUT_DIR / output_case_name(case_path.name, TARG_PX_X)
        case_out.mkdir(parents=True, exist_ok=True)
        mapping = os.environ.get("EXP1_MAPPING_MODE", mapping_mode_for_case(case_path.name))
        for method, ssaa in _methods():
            for frame in range(disp_x.shape[1]):
                if frame not in _active_frames(): continue
                canonical = case_out / f"targ_px{TARG_PX_X}_int_{method}_param_{ssaa}_psf_frame{frame:02d}.npy"
                if canonical.exists() and not force_render:
                    write_camera_depths(canonical, BIT_DEPTHS)
                if not force_render and float_and_depths_complete(canonical, BIT_DEPTHS):
                    continue
                render_log("EXP1", "gridint2d-psf", case_label(case_out.name),
                           f"frame={frame:02d}; SSAA={ssaa}; mapping={mapping}")
                def eggbox(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                    u, v = uv_scale * x, -uv_scale * y
                    return I0 + 0.5 * GAMMA * (1.0 + np.cos(2.0 * np.pi * u / pitch_uv)) * (1.0 + np.cos(2.0 * np.pi * v / pitch_uv)) - GAMMA
                image = render_psf_frame(
                    evaluate_reference=eggbox, invalid_value=0.0, roi_size=roi_size,
                    image_shape=(TARG_PX_Y, TARG_PX_X), ssaa=ssaa,
                    sigma_px=PSF_SIGMA_FINAL_PX, support_radius_px=support_px,
                    background=0.0, coords=coords, connect=connect, disp_x=disp_x,
                    disp_y=disp_y, frame=frame, mapping_mode=mapping,
                    processes=NUM_PROCESSES,
                )
                image = np.flipud(image)
                save_float_and_depths(canonical, image, BIT_DEPTHS)


if __name__ == "__main__":
    main()
