"""Generate the 512-square rigid and affine Quad9 cases for Experiment 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PLATE_SIZE = 516.0
ROI_SIZE = 512.0
CAMERA_PIXELS = 512
FRAMES = 11


def _save(path: Path, values: np.ndarray, *, integer: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%d" if integer else "%.10f")


def _quad9() -> tuple[np.ndarray, np.ndarray]:
    half = PLATE_SIZE / 2.0
    coords = np.array((
        (-half, -half, 0.0), (half, -half, 0.0),
        (half, half, 0.0), (-half, half, 0.0),
        (0.0, -half, 0.0), (half, 0.0, 0.0),
        (0.0, half, 0.0), (-half, 0.0, 0.0), (0.0, 0.0, 0.0),
    ), dtype=np.float64)
    return coords, np.array(((0, 1, 2, 3, 4, 5, 6, 7, 8),), dtype=int)


def _write_case(name: str, disp_x: np.ndarray, disp_y: np.ndarray) -> None:
    coords, connect = _quad9()
    half_roi = ROI_SIZE / 2.0
    uvs = np.empty((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = (coords[:, 0] + half_roi) / ROI_SIZE
    uvs[:, 1] = (coords[:, 1] + half_roi) / ROI_SIZE
    root = Path("data") / name
    _save(root / "coords.csv", coords)
    _save(root / "connectivity.csv", connect, integer=True)
    _save(root / "connect.csv", connect, integer=True)
    _save(root / "field_disp_x.csv", disp_x)
    _save(root / "field_disp_y.csv", disp_y)
    _save(root / "field_disp_z.csv", np.zeros_like(disp_x))
    _save(root / "uvs.csv", uvs)
    print(f"Generated {root}")


def main() -> None:
    coords, _ = _quad9()
    ramp_px = np.arange(FRAMES, dtype=np.float64) * 0.1
    rigid_x = np.broadcast_to(ramp_px, (coords.shape[0], FRAMES)).copy()
    rigid_y = np.broadcast_to(ramp_px, (coords.shape[0], FRAMES)).copy()
    _write_case("plate516_cam512_quad9_rigid", rigid_x, rigid_y)

    affine_x = np.outer((coords[:, 0] + coords[:, 1]) / ROI_SIZE, ramp_px)
    affine_y = np.outer(coords[:, 1] / ROI_SIZE, ramp_px)
    _write_case("plate516_cam512_quad9_affine", affine_x, affine_y)


if __name__ == "__main__":
    main()
