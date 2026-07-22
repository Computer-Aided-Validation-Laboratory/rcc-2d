"""Render Exp2 additive disks with a pre-clamp image-plane Gaussian PSF."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from exp1common import output_case_name, parse_case_params
from exp2_speckint2d_render_uvs import pattern_tag
from exp2params import (
    ACTIVE_FRAMES, ADDITIVE_DISK_JITTER_DISTRIBUTION, ADDITIVE_DISK_JITTER_FRACTION,
    BACKGROUND, BLACK_AREA_FRACTIONS, DEFORMATION_CASES, FORCE_RENDER_OVER,
    GAUSSIAN_CUTOFF_SIGMAS, GAMMA, I0, INTEGRATION_METHODS, PSF_SIGMA_FINAL_PX,
    PSF_SUPPORT_SIGMAS, PX_PER_SPECK, RANDOM_SEED, TARG_PX_X, TARG_PX_Y,
    TEX_PX_PAD, exp2_output_dir, mapping_mode_for_case, NUM_PROCESSES,
)
from exp2speckint2d import make_speckle_pattern, save_image
from ortho_psf_common import render_psf_frame

OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs_psf")


def _active_frames() -> set[int]:
    value = os.environ.get("EXP2_ACTIVE_FRAMES")
    return set(ACTIVE_FRAMES) if not value else {int(item) for item in value.split(",") if item}


def _methods() -> list[tuple[str, int]]:
    value = os.environ.get("EXP2_METHODS")
    methods = INTEGRATION_METHODS if not value else [
        (part.split(":")[0], int(part.split(":")[1])) for part in value.split(",")
    ]
    return [(name, param) for name, param in methods if name == "rect"]


def main() -> None:
    print("Experiment 2: bespoke orthographic additive-disk render with Gaussian PSF")
    cases = [Path(sys.argv[1])] if len(sys.argv) > 1 else [Path("data") / name for name in DEFORMATION_CASES]
    support_px = PSF_SIGMA_FINAL_PX * PSF_SUPPORT_SIGMAS
    force_render = FORCE_RENDER_OVER or os.environ.get("EXP2_FORCE_RENDER_OVER") == "1"
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
        pixel_size = roi_size / TARG_PX_X
        bounds = (-0.5 * roi_size - TEX_PX_PAD * pixel_size, 0.5 * roi_size + TEX_PX_PAD * pixel_size, -0.5 * roi_size - TEX_PX_PAD * pixel_size, 0.5 * roi_size + TEX_PX_PAD * pixel_size)
        mapping = os.environ.get("EXP2_MAPPING_MODE", mapping_mode_for_case(case_path.name))
        for black_fraction in BLACK_AREA_FRACTIONS:
            tag = pattern_tag("diskaddsat", black_fraction, ADDITIVE_DISK_JITTER_DISTRIBUTION, ADDITIVE_DISK_JITTER_FRACTION)
            pattern = make_speckle_pattern("diskaddsat", PX_PER_SPECK * pixel_size, black_fraction, ADDITIVE_DISK_JITTER_DISTRIBUTION, ADDITIVE_DISK_JITTER_FRACTION, RANDOM_SEED, GAUSSIAN_CUTOFF_SIGMAS, bounds, I0, GAMMA)
            background_coverage = (I0 + GAMMA - BACKGROUND) / (2.0 * GAMMA)
            for method, ssaa in _methods():
                out_dir = OUTPUT_DIR / f"{output_case_name(case_path.name, TARG_PX_X)}_{tag}_int_{method}_param_{ssaa}_psf"
                out_dir.mkdir(parents=True, exist_ok=True)
                for frame in range(disp_x.shape[1]):
                    if frame not in _active_frames(): continue
                    prefix = f"targ_px{TARG_PX_X}_int_{method}_param_{ssaa}_psf_frame{frame:02d}"
                    if not force_render and (out_dir / f"{prefix}.npy").exists(): continue
                    print(f"  {out_dir.name}: frame {frame:02d}, SSAA={ssaa}", flush=True)
                    coverage = render_psf_frame(
                        evaluate_reference=pattern.evaluate_coverage, invalid_value=0.0,
                        roi_size=roi_size, image_shape=(TARG_PX_Y, TARG_PX_X), ssaa=ssaa,
                        sigma_px=PSF_SIGMA_FINAL_PX, support_radius_px=support_px,
                        # Match Riley's zero-valued unshaded PSF scratch samples.
                        background=0.0, coords=coords, connect=connect,
                        disp_x=disp_x, disp_y=disp_y, frame=frame, mapping_mode=mapping,
                        processes=NUM_PROCESSES,
                    )
                    intensity = pattern.intensity_from_coverage(np.flipud(coverage))
                    save_image(intensity, out_dir, prefix)


if __name__ == "__main__":
    main()
