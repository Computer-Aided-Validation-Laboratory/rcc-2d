# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

import os
import sys
import numpy as np
import pyvista as pv
from PIL import Image

# Configurable constants at the top of the file
SSAA_LEVEL = 4          # Sub-pixel anti-aliasing level (N x N sub-pixels)
CAMERA_PIXELS_X = 256   # Camera resolution in X
CAMERA_PIXELS_Y = 256   # Camera resolution in Y
P_PIXELS = 5.0          # Grid period (pitch) in pixels
I0 = 0.5                # Mean intensity fraction of dynamic range
GAMMA = 0.2             # Amplitude fraction of dynamic range
BIT_DEPTH = 10          # Bit depth for image quantization
OUTPUT_DIR = "./out/exp1_analytic_grid"


def get_pv_cell_type(nodes_per_elem):
    """Determine PyVista cell type from connectivity width."""
    if nodes_per_elem == 3:
        return pv.CellType.TRIANGLE
    elif nodes_per_elem == 6:
        return pv.CellType.QUADRATIC_TRIANGLE
    elif nodes_per_elem == 4:
        return pv.CellType.QUAD
    elif nodes_per_elem == 8:
        return pv.CellType.QUADRATIC_QUAD
    elif nodes_per_elem == 9:
        return pv.CellType.BIQUADRATIC_QUAD
    else:
        raise ValueError(
            f"Unsupported element type with {nodes_per_elem} nodes."
        )


def build_pv_mesh(coords, connect):
    """Build a PyVista UnstructuredGrid mesh from coords and connect."""
    num_elems, nodes_per_elem = connect.shape
    cell_type = get_pv_cell_type(nodes_per_elem)

    cells = np.hstack(
        [np.full((num_elems, 1), nodes_per_elem), connect]
    ).ravel()
    cell_types = np.full(num_elems, cell_type, dtype=np.uint8)

    mesh = pv.UnstructuredGrid(cells, cell_types, coords)
    return mesh


def parse_case_params(case_dir):
    """Parse resolution and ROI scale from directory name."""
    folder_name = os.path.basename(os.path.normpath(case_dir))
    parts = folder_name.split("_")
    camera_pixels = 256
    for p in parts:
        if p.startswith("cam"):
            try:
                camera_pixels = int(p.replace("cam", ""))
            except ValueError:
                pass
    # For square plates, the camera FOV (ROI) size equals camera_pixels
    roi_size = float(camera_pixels)
    return camera_pixels, roi_size


def generate_grid_images(case_dir):
    """Load mesh, interpolate displacements, and generate grid images."""
    case_name = os.path.basename(os.path.normpath(case_dir))
    print(f"\nProcessing case: {case_name}")

    # Load coordinates, connectivity, and displacements
    coords = np.loadtxt(os.path.join(case_dir, "coords.csv"), delimiter=",")
    connect = np.loadtxt(
        os.path.join(case_dir, "connectivity.csv"), delimiter=",", dtype=int
    )
    disp_x = np.loadtxt(
        os.path.join(case_dir, "field_disp_x.csv"), delimiter=","
    )
    disp_y = np.loadtxt(
        os.path.join(case_dir, "field_disp_y.csv"), delimiter=","
    )

    if connect.ndim == 1:
        connect = connect.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x = disp_x.reshape(-1, 1)
    if disp_y.ndim == 1:
        disp_y = disp_y.reshape(-1, 1)

    camera_pixels, roi_size = parse_case_params(case_dir)
    pixel_size = roi_size / camera_pixels
    num_frames = disp_x.shape[1]

    # Setup the output directory for this case
    case_out_dir = os.path.join(OUTPUT_DIR, case_name)
    os.makedirs(case_out_dir, exist_ok=True)

    # 1. Generate reference sub-pixel coordinates
    sub_h = CAMERA_PIXELS_Y * SSAA_LEVEL
    sub_w = CAMERA_PIXELS_X * SSAA_LEVEL
    pixel_size_sub = pixel_size / SSAA_LEVEL

    # Center sub-pixels in the physical camera coordinate system
    xs = -roi_size / 2.0 + (np.arange(sub_w) + 0.5) * pixel_size_sub
    ys = -roi_size / 2.0 + (np.arange(sub_h) + 0.5) * pixel_size_sub
    X, Y = np.meshgrid(xs, ys)

    # Prepare query points for PyVista sampling (shape: M x 3)
    query_pts = np.zeros((sub_h * sub_w, 3), dtype=np.float64)
    query_pts[:, 0] = X.ravel()
    query_pts[:, 1] = Y.ravel()

    grid_query = pv.PolyData(query_pts)
    p_phys = P_PIXELS * pixel_size

    # Build base PyVista mesh object
    mesh = build_pv_mesh(coords, connect)

    for f in range(num_frames):
        print(f"  Frame {f}/{num_frames - 1}...")

        # Update point data displacements for the current frame
        mesh.point_data["disp_x"] = disp_x[:, f]
        mesh.point_data["disp_y"] = disp_y[:, f]

        # Use PyVista .sample to interpolate displacements
        # via FE shape functions
        sampled = grid_query.sample(mesh)
        u_x = sampled.point_data["disp_x"]
        u_y = sampled.point_data["disp_y"]

        # Ensure any points outside the mesh get zero displacement
        valid = sampled.point_data["vtkValidPointMask"].astype(bool)
        u_x[~valid] = 0.0
        u_y[~valid] = 0.0

        # 2. Apply deformation to sub-pixel coordinates
        x_def = X.ravel() + u_x
        y_def = Y.ravel() + u_y

        # 3. Calculate unquantized grid intensity at sub-pixels
        cos_x = np.cos(2.0 * np.pi * x_def / p_phys)
        cos_y = np.cos(2.0 * np.pi * y_def / p_phys)
        sub_intensity = I0 + 0.25 * GAMMA * (1.0 + cos_x) * (1.0 + cos_y)
        sub_intensity = sub_intensity.reshape(sub_h, sub_w)

        # 4. Integrate sub-pixels to get pixel values (SSAA block average)
        reshaped = sub_intensity.reshape(
            CAMERA_PIXELS_Y, SSAA_LEVEL,
            CAMERA_PIXELS_X, SSAA_LEVEL
        )
        pixel_raw = reshaped.mean(axis=(1, 3))

        # Flip vertically to align Cartesian layout with standard image layout
        # (where row index 0 is at the top/maximum Y coordinate)
        pixel_raw_flipped = np.flipud(pixel_raw)

        # 5. Quantize to b-bit representation
        max_val_b = float(2**BIT_DEPTH - 1)
        pixel_b = np.round(pixel_raw_flipped * max_val_b)
        pixel_b = np.clip(pixel_b, 0.0, max_val_b)

        # Scale b-bit values to 16-bit TIFF range [0, 65535]
        pixel_16 = np.round(pixel_b * (65535.0 / max_val_b)).astype(np.uint16)

        # Save the 16-bit TIFF image
        img_path = os.path.join(case_out_dir, f"image_{f:02d}.tiff")
        img = Image.fromarray(pixel_16)
        img.save(img_path)

        # Save double-precision unquantized values scaled by 2**b to .npy
        # (no discretization/rounding is performed on these floats)
        npy_path = os.path.join(case_out_dir, f"float_{f:02d}.npy")
        np.save(npy_path, pixel_raw_flipped * float(2**BIT_DEPTH))

    print(f"Saved outputs to: {case_out_dir}")


def main() -> None:
    print(80 * "=")
    print("Experiment 1: Sinusoidal Grid Generator")
    print(80 * "=")

    # Process default cases if none specified
    if len(sys.argv) > 1:
        cases = [sys.argv[1]]
    else:
        cases = [
            "data/plate260_cam256_quad9_rigid",
            "data/plate260_cam256_quad9_affine"
        ]

    for case_path in cases:
        if not os.path.exists(case_path):
            print(
                f"Warning: Case directory {case_path} "
                "does not exist. Skipping."
            )
            continue
        generate_grid_images(case_path)


if __name__ == "__main__":
    main()
