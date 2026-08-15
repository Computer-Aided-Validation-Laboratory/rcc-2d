"""Generate PixInt2D's compact, distributed multi-element finite-star case."""

from __future__ import annotations

from pathlib import Path

import numpy as np


PLATE_SIZE = (260.0, 65.0)
ROI_SIZE = (256.0, 64.0)
FINAL_CAMERA_PIXELS = (1020, 252)
ELEMENTS = (12, 16)
X_GRADING_RATIO = 1.05
WAVELENGTH_FINAL_PIXELS = (64.0, 128.0)
PEAK_DISPLACEMENT_FINAL_PIXELS = 0.5
OUTPUT = Path("src/pixint2d/data/finite_star_coarse_q9")


def widths(count: int, length: float, ratio: float) -> np.ndarray:
    if np.isclose(ratio, 1.0): return np.full(count, length/count)
    first = length*(ratio-1)/(ratio**count-1)
    return first*ratio**np.arange(count)


def q9_connectivity(ix: int, iy: int, nx: int) -> list[int]:
    columns = 2*nx+1; base = 2*iy*columns+2*ix
    return [base, base+2, base+2+2*columns, base+2*columns,
            base+1, base+2+columns, base+2*columns+1, base+columns,
            base+columns+1]


def main() -> None:
    nx, ny = ELEMENTS; lx, ly = PLATE_SIZE
    xb = np.r_[-lx/2, -lx/2+np.cumsum(widths(nx,lx,X_GRADING_RATIO))]
    yb = np.linspace(-ly/2, ly/2, ny+1)
    xq = np.empty(2*nx+1); yq = np.empty(2*ny+1)
    xq[0::2], yq[0::2] = xb, yb
    xq[1::2], yq[1::2] = .5*(xb[:-1]+xb[1:]), .5*(yb[:-1]+yb[1:])
    xx, yy = np.meshgrid(xq,yq); coords = np.column_stack((xx.ravel(),yy.ravel()))
    connectivity = np.asarray([q9_connectivity(i,j,nx) for j in range(ny) for i in range(nx)], dtype=int)
    px_y = ROI_SIZE[1]/FINAL_CAMERA_PIXELS[1]
    amplitude = PEAK_DISPLACEMENT_FINAL_PIXELS*px_y
    lambda_min = WAVELENGTH_FINAL_PIXELS[0]*px_y
    lambda_max = WAVELENGTH_FINAL_PIXELS[1]*px_y
    wavelength = lambda_min+(lambda_max-lambda_min)*(coords[:,0]+lx/2)/lx
    ux = np.zeros((len(coords),2)); uy = np.zeros((len(coords),2))
    uy[:,1] = amplitude*np.cos(2*np.pi*coords[:,1]/wavelength)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savetxt(OUTPUT/"coords.csv",coords,delimiter=",",fmt="%.12g")
    np.savetxt(OUTPUT/"connectivity.csv",connectivity,delimiter=",",fmt="%d")
    np.savetxt(OUTPUT/"field_disp_x.csv",ux,delimiter=",",fmt="%.12g")
    np.savetxt(OUTPUT/"field_disp_y.csv",uy,delimiter=",",fmt="%.12g")
    (OUTPUT/"metadata.json").write_text(
        "{\n"
        "  \"camera_pixels\": [1020, 252],\n"
        "  \"roi_size\": [256.0, 64.0],\n"
        "  \"wavelength_final_pixels\": [64.0, 128.0],\n"
        "  \"elements\": [12, 16],\n"
        "  \"peak_displacement_final_pixels\": 0.5\n"
        "}\n"
    )
    print(f"Wrote {OUTPUT}: {len(coords)} nodes, {len(connectivity)} Quad9 elements.")


if __name__ == "__main__": main()
