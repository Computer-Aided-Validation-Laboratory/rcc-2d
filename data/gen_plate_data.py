import os
import sys
import numpy as np

# Configurable constants
PLATE_SIZE = 260.0
CAMERA_PIXELS = 256
ROI_SIZE = 256.0

# Calculate physical pixel size
PIXEL_SIZE = ROI_SIZE / CAMERA_PIXELS


def save_csv(path, data, is_int=False):
    """Helper to save data to CSV with formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = "%d" if is_int else "%.10f"
    np.savetxt(path, data, delimiter=",", fmt=fmt)


def get_mesh(etype, plate_size):
    """Generate coordinates and connectivity for the element type."""
    half = plate_size / 2.0
    if etype == "quad4":
        coords = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )
        connect = np.array([[0, 1, 2, 3]], dtype=int)
    elif etype == "quad8":
        coords = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
                [0.0, -half, 0.0],
                [half, 0.0, 0.0],
                [0.0, half, 0.0],
                [-half, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    elif etype == "quad9":
        coords = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
                [0.0, -half, 0.0],
                [half, 0.0, 0.0],
                [0.0, half, 0.0],
                [-half, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=int)
    elif etype == "tri3":
        coords = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )
        connect = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    elif etype == "tri6":
        coords = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
                [0.0, -half, 0.0],
                [half, 0.0, 0.0],
                [0.0, half, 0.0],
                [-half, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        connect = np.array(
            [[0, 1, 2, 4, 5, 8], [0, 2, 3, 8, 6, 7]], dtype=int
        )
    else:
        raise ValueError(f"Unknown element type: {etype}")
    return coords, connect


def compute_uvs(coords, roi_size):
    """Map coords to UV space where ROI is [0, 1]."""
    uvs = np.zeros((len(coords), 2), dtype=np.float64)
    half_roi = roi_size / 2.0
    uvs[:, 0] = (coords[:, 0] + half_roi) / roi_size
    uvs[:, 1] = (coords[:, 1] + half_roi) / roi_size
    return uvs


def save_case(out_dir, coords, connect, disp_x, disp_y, disp_z, uvs):
    """Save all case files to output directory."""
    save_csv(os.path.join(out_dir, "coords.csv"), coords)
    save_csv(os.path.join(out_dir, "connectivity.csv"), connect, is_int=True)
    save_csv(os.path.join(out_dir, "connect.csv"), connect, is_int=True)
    save_csv(os.path.join(out_dir, "field_disp_x.csv"), disp_x)
    save_csv(os.path.join(out_dir, "field_disp_y.csv"), disp_y)
    save_csv(os.path.join(out_dir, "field_disp_z.csv"), disp_z)
    save_csv(os.path.join(out_dir, "uvs.csv"), uvs)


def generate_cases():
    # Import visualizer function dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from data_vis import visualize_case

    etypes = ["tri3", "tri6", "quad4", "quad8", "quad9"]
    num_frames = 11
    vis_dir = os.path.join("data", "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Precompute frames and displacement scales
    disp_pixels = np.arange(num_frames) * 0.1
    disp_vals = disp_pixels * PIXEL_SIZE

    for etype in etypes:
        coords, connect = get_mesh(etype, PLATE_SIZE)
        uvs = compute_uvs(coords, ROI_SIZE)
        num_nodes = len(coords)

        # --- Case 1: Rigid Translation ---
        disp_x_rigid = np.zeros((num_nodes, num_frames), dtype=np.float64)
        disp_y_rigid = np.zeros((num_nodes, num_frames), dtype=np.float64)
        disp_z_rigid = np.zeros((num_nodes, num_frames), dtype=np.float64)

        # Vectorized assignment using broadcasting
        disp_x_rigid[:] = disp_vals
        disp_y_rigid[:] = disp_vals

        case_tag = f"plate{int(PLATE_SIZE)}_cam{CAMERA_PIXELS}_{etype}_rigid"
        out_dir = os.path.join("data", case_tag)
        save_case(
            out_dir,
            coords,
            connect,
            disp_x_rigid,
            disp_y_rigid,
            disp_z_rigid,
            uvs,
        )
        print(f"Generated case: {case_tag}")
        visualize_case(out_dir, vis_dir)

        # --- Case 2: Affine Deformation ---
        disp_z_affine = np.zeros((num_nodes, num_frames), dtype=np.float64)

        # Vectorized calculations using outer products
        x = coords[:, 0]
        y = coords[:, 1]
        disp_x_affine = np.outer((x + y) / ROI_SIZE, disp_vals)
        disp_y_affine = np.outer(y / ROI_SIZE, disp_vals)

        case_tag = f"plate{int(PLATE_SIZE)}_cam{CAMERA_PIXELS}_{etype}_affine"
        out_dir = os.path.join("data", case_tag)
        save_case(
            out_dir,
            coords,
            connect,
            disp_x_affine,
            disp_y_affine,
            disp_z_affine,
            uvs,
        )
        print(f"Generated case: {case_tag}")
        visualize_case(out_dir, vis_dir)


if __name__ == "__main__":
    generate_cases()
