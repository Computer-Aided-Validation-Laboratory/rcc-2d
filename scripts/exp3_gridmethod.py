#!/usr/bin/env python3
"""Measure the Exp3 rigid eggbox sequence with the Python grid method.

The input is the bespoke SSAA render, whose continuous eggbox has a known
five-final-pixel carrier.  The script writes one two-component displacement
field image per frame, numerical fields, a CSV summary, and an independent
scikit-image phase-unwrapping cross-check.

Environment overrides:
``EXP3_GRIDMETHOD_CASE`` and ``EXP3_GRIDMETHOD_SSAA``.
"""
from __future__ import annotations

import csv
import gc
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp3params import EGGBOX_PERIOD_FINAL_PX, output_dir
from gridmethod import GridMethodConfig, analyse_sequence


CASE = os.environ.get("EXP3_GRIDMETHOD_CASE", "plate516_cam512_quad9_rigid")
SSAA = int(os.environ.get("EXP3_GRIDMETHOD_SSAA", "16"))
RENDER_ROOT = output_dir("exp3_gridint2d_render_ssaa", CASE)
INPUT_DIR = RENDER_ROOT / CASE / f"eggbox_ss{SSAA}_b8"
OUTPUT_DIR = Path("out") / "exp3_gridmethod" / CASE


def expected_motion(case: str, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the prescribed uniform displacement from the FE input fields."""
    data_dir = Path("data") / case
    x = np.loadtxt(data_dir / "field_disp_x.csv", delimiter=",")
    y = np.loadtxt(data_dir / "field_disp_y.csv", delimiter=",")
    if x.ndim == 1:
        x = x[None, :]
        y = y[None, :]
    return x[:, :frames].mean(axis=0), y[:, :frames].mean(axis=0)


def crop_for_measurement(array: np.ndarray, period: float, window_periods: float) -> np.ndarray:
    # Exclude finite-window and iterative-back-deformation boundaries.  This
    # crop is for scalar validation only; images retain the complete field.
    halo = int(np.ceil(4.0 * period * window_periods + 2.0 * period))
    return array[halo:-halo, halo:-halo]


def save_field_image(
    path: Path,
    ux: np.ndarray,
    uy: np.ndarray,
    expected_x: float,
    expected_y: float,
    frame: int,
    grid_pitch_px: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, field, name, expected in zip(axes, (ux, uy), ("$u_x$", "$u_y$"), (expected_x, expected_y)):
        # Deliberately leave colour limits automatic.  These diagnostic plots
        # should reveal the small spatial variation about the nominal rigid
        # displacement, rather than using the global displacement magnitude.
        image = axis.imshow(field, cmap="coolwarm", origin="upper")
        axis.set_title(f"frame {frame:02d}: {name} (expected {expected:.3f} px)")
        axis.set_xlabel("column [px]")
        axis.set_ylabel("row [px]")
        # This is the one-pitch interior used by the MATLAB toolbox when it
        # checks iterative back-deformation convergence.
        rows, cols = field.shape
        axis.add_patch(Rectangle(
            (grid_pitch_px, grid_pitch_px),
            cols - 1 - 2.0 * grid_pitch_px,
            rows - 1 - 2.0 * grid_pitch_px,
            fill=False, edgecolor="black", linewidth=1.2,
        ))
        fig.colorbar(image, ax=axis, label="displacement [px]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    if "rigid" not in CASE:
        raise ValueError("exp3_gridmethod.py currently validates the rigid-body sequence only.")
    files = sorted(INPUT_DIR.glob("frame*.npy"))
    if not files:
        raise FileNotFoundError(f"No rigid eggbox frames found in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = np.stack([np.load(path, mmap_mode="r") for path in files]).astype(np.float64, copy=False)
    expected_x, expected_y = expected_motion(CASE, len(files))
    config = GridMethodConfig(
        period_px=EGGBOX_PERIOD_FINAL_PX,
        window_width_periods=2.0,
        window="gaussian",
        displacement_method="iterative",
        unwrap="reliability",
    )
    print(f"Grid method: {len(files)} frames from {INPUT_DIR}; p={config.period_px:g} px, unwrap=reliability")
    primary = analyse_sequence(images, config)
    # A separate implementation provides a practical validation of the most
    # delicate step (spatial phase unwrapping), not a replacement for it.
    crosscheck = analyse_sequence(images, replace(config, unwrap="skimage"))

    rows: list[dict[str, float | int]] = []
    for frame, _file in enumerate(files):
        ux = primary.displacement_x[frame]
        # Image rows increase downwards while Exp3's physical Y increases
        # upwards, hence the sign conversion for reporting/comparison.
        uy_physical = -primary.displacement_y[frame]
        ux_roi = crop_for_measurement(ux, config.period_px, config.window_width_periods)
        uy_roi = crop_for_measurement(uy_physical, config.period_px, config.window_width_periods)
        sx_roi = crop_for_measurement(crosscheck.displacement_x[frame], config.period_px, config.window_width_periods)
        sy_roi = crop_for_measurement(-crosscheck.displacement_y[frame], config.period_px, config.window_width_periods)
        row = {
            "frame": frame,
            "expected_ux_px": float(expected_x[frame]),
            "expected_uy_px": float(expected_y[frame]),
            "measured_ux_px": float(np.nanmean(ux_roi)),
            "measured_uy_px": float(np.nanmean(uy_roi)),
            "error_ux_px": float(np.nanmean(ux_roi) - expected_x[frame]),
            "error_uy_px": float(np.nanmean(uy_roi) - expected_y[frame]),
            "skimage_ux_px": float(np.nanmean(sx_roi)),
            "skimage_uy_px": float(np.nanmean(sy_roi)),
            "unwrap_max_delta_px": float(max(np.nanmax(np.abs(ux_roi - sx_roi)), np.nanmax(np.abs(uy_roi - sy_roi)))),
        }
        rows.append(row)
        np.savez_compressed(OUTPUT_DIR / f"displacement_frame{frame:02d}.npz", ux=ux, uy=uy_physical)
        save_field_image(
            OUTPUT_DIR / f"displacement_frame{frame:02d}.png",
            ux,
            uy_physical,
            expected_x[frame],
            expected_y[frame],
            frame,
            config.period_px,
        )

    with (OUTPUT_DIR / "rigid_motion_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frame = np.asarray([row["frame"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, constrained_layout=True)
    for axis, component in zip(axes, ("ux", "uy")):
        expected = np.asarray([row[f"expected_{component}_px"] for row in rows])
        measured = np.asarray([row[f"measured_{component}_px"] for row in rows])
        skimage = np.asarray([row[f"skimage_{component}_px"] for row in rows])
        axis.plot(frame, expected, "k--", label="prescribed")
        axis.plot(frame, measured, "o-", label="toolbox-compatible")
        axis.plot(frame, skimage, "x:", label="scikit-image cross-check")
        axis.set_title(f"{component} rigid displacement")
        axis.set_xlabel("frame")
        axis.set_ylabel("displacement [px]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / "rigid_motion_summary.png", dpi=170)
    plt.close(fig)

    maximum_error = max(max(abs(float(row["error_ux_px"])), abs(float(row["error_uy_px"]))) for row in rows)
    unwrap_delta = max(float(row["unwrap_max_delta_px"]) for row in rows)
    print(f"Wrote {OUTPUT_DIR}")
    print(f"Maximum ROI mean rigid-motion error: {maximum_error:.6g} px")
    print(f"Maximum reliability vs scikit-image field delta: {unwrap_delta:.6g} px")
    del images, primary, crosscheck
    gc.collect()


if __name__ == "__main__":
    main()
