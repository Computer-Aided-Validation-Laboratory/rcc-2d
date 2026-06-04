import os
import sys
import numpy as np
import gmsh

# Configurable constants
PLATE_SIZE_X = 260
PLATE_SIZE_Y = 260/4
CAMERA_PIXELS = 256
ROI_SIZE_X = 256.0
ROI_SIZE_Y = 256.0/4.0

# Chirp grid configuration
CHIRP_NX = 20
CHIRP_NY = 9  # Must be odd
CHIRP_RX = 1.25
CHIRP_RY = 1.25
CHIRP_A0 = 0.5  # Peak displacement in pixels

# Calculate physical pixel size (based on X direction)
PIXEL_SIZE = ROI_SIZE_X / CAMERA_PIXELS


def save_csv(path, data, is_int=False):
    """Helper to save data to CSV with formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = "%d" if is_int else "%.10f"
    np.savetxt(path, data, delimiter=",", fmt=fmt)


def get_geometric_widths(nx, lx, rx):
    """Generate element widths increasing geometrically left-to-right."""
    if abs(rx - 1.0) < 1e-9:
        return np.full(nx, lx / nx, dtype=np.float64)
    cx = lx * (rx - 1.0) / (rx**nx - 1.0)
    widths = cx * (rx ** np.arange(nx))
    return widths


def get_symmetric_heights(ny, ly, ry):
    """Generate element heights symmetric around the center row."""
    jc = (ny - 1) // 2
    if abs(ry - 1.0) < 1e-9:
        return np.full(ny, ly / ny, dtype=np.float64)
    sum_val = 1.0 + 2.0 * ry * (ry**jc - 1.0) / (ry - 1.0)
    cy = ly / sum_val
    heights = cy * (ry ** np.abs(np.arange(ny) - jc))
    return heights


def q2_stencil(i, j, nx):
    """Return the CCW 9-node connectivity for element (i, j)."""
    cols = 2 * nx + 1
    # Corners
    c0 = (2 * j) * cols + 2 * i
    c1 = (2 * j) * cols + 2 * i + 2
    c2 = (2 * j + 2) * cols + 2 * i + 2
    c3 = (2 * j + 2) * cols + 2 * i

    # Mid-edges
    m01 = (2 * j) * cols + 2 * i + 1
    m12 = (2 * j + 1) * cols + 2 * i + 2
    m23 = (2 * j + 2) * cols + 2 * i + 1
    m30 = (2 * j + 1) * cols + 2 * i

    # Center
    center = (2 * j + 1) * cols + 2 * i + 1

    return [c0, c1, c2, c3, m01, m12, m23, m30, center]


def compute_uvs(coords, roi_size_x, roi_size_y):
    """Map coords to UV space where ROI is [0, 1] in X and Y."""
    uvs = np.zeros((len(coords), 2), dtype=np.float64)
    half_roi_x = roi_size_x / 2.0
    half_roi_y = roi_size_y / 2.0
    uvs[:, 0] = (coords[:, 0] + half_roi_x) / roi_size_x
    uvs[:, 1] = (coords[:, 1] + half_roi_y) / roi_size_y
    return uvs


def apply_chirp_disp(coords, num_nodes, a0_phys):
    """Compute continuous chirp displacement field at all nodes."""
    disp_x = np.zeros((num_nodes, 2), dtype=np.float64)
    disp_y = np.zeros((num_nodes, 2), dtype=np.float64)
    disp_z = np.zeros((num_nodes, 2), dtype=np.float64)

    lambda_min = 10.0 * PIXEL_SIZE
    lambda_max = 100.0 * PIXEL_SIZE
    x_min = -PLATE_SIZE_X / 2.0

    # Vectorized calculations over all nodes
    x = coords[:, 0]
    y = coords[:, 1]
    lambda_val = (
        lambda_min + (lambda_max - lambda_min) * (x - x_min) / PLATE_SIZE_X
    )
    disp_y[:, 1] = a0_phys * np.cos(2.0 * np.pi * y / lambda_val)

    return disp_x, disp_y, disp_z


def generate_manual_chirp(a0_phys, vis_fn, vis_dir):
    """Generate the manually graded chirp mesh."""
    nx, ny = CHIRP_NX, CHIRP_NY
    rx, ry = CHIRP_RX, CHIRP_RY

    w = get_geometric_widths(nx, PLATE_SIZE_X, rx)
    h = get_symmetric_heights(ny, PLATE_SIZE_Y, ry)

    x_boundary = np.zeros(nx + 1, dtype=np.float64)
    x_boundary[0] = -PLATE_SIZE_X / 2.0
    for i in range(nx):
        x_boundary[i + 1] = x_boundary[i] + w[i]

    y_boundary = np.zeros(ny + 1, dtype=np.float64)
    y_boundary[0] = -PLATE_SIZE_Y / 2.0
    for j in range(ny):
        y_boundary[j + 1] = y_boundary[j] + h[j]

    x_q2 = np.zeros(2 * nx + 1, dtype=np.float64)
    for i in range(nx):
        x_q2[2 * i] = x_boundary[i]
        x_q2[2 * i + 1] = 0.5 * (x_boundary[i] + x_boundary[i + 1])
    x_q2[2 * nx] = x_boundary[nx]

    y_q2 = np.zeros(2 * ny + 1, dtype=np.float64)
    for j in range(ny):
        y_q2[2 * j] = y_boundary[j]
        y_q2[2 * j + 1] = 0.5 * (y_boundary[j] + y_boundary[j + 1])
    y_q2[2 * ny] = y_boundary[ny]

    coords = []
    for q in range(2 * ny + 1):
        for p in range(2 * nx + 1):
            coords.append([x_q2[p], y_q2[q], 0.0])
    coords = np.array(coords, dtype=np.float64)
    num_nodes = len(coords)

    connect = []
    for j in range(ny):
        for i in range(nx):
            connect.append(q2_stencil(i, j, nx))
    connect = np.array(connect, dtype=int)

    disp_x, disp_y, disp_z = apply_chirp_disp(coords, num_nodes, a0_phys)
    uvs = compute_uvs(coords, ROI_SIZE_X, ROI_SIZE_Y)

    case_tag = (
        f"plate{int(PLATE_SIZE_X)}x{int(PLATE_SIZE_Y)}_"
        f"cam{CAMERA_PIXELS}_quad9_chirp"
    )
    out_dir = os.path.join("data", case_tag)

    save_csv(os.path.join(out_dir, "coords.csv"), coords)
    save_csv(os.path.join(out_dir, "connectivity.csv"), connect, is_int=True)
    save_csv(os.path.join(out_dir, "connect.csv"), connect, is_int=True)
    save_csv(os.path.join(out_dir, "field_disp_x.csv"), disp_x)
    save_csv(os.path.join(out_dir, "field_disp_y.csv"), disp_y)
    save_csv(os.path.join(out_dir, "field_disp_z.csv"), disp_z)
    save_csv(os.path.join(out_dir, "uvs.csv"), uvs)

    print(f"Generated case: {case_tag}")
    vis_fn(out_dir, vis_dir)


def generate_gmsh_chirp(a0_phys, vis_fn, vis_dir):
    """Generate the chirp mesh using the Gmsh Python API."""
    gmsh.initialize()
    gmsh.model.add("chirp_gmsh")

    half_x = PLATE_SIZE_X / 2.0
    half_y = PLATE_SIZE_Y / 2.0

    p1 = gmsh.model.geo.addPoint(-half_x, -half_y, 0.0)
    p2 = gmsh.model.geo.addPoint(half_x, -half_y, 0.0)
    p3 = gmsh.model.geo.addPoint(half_x, half_y, 0.0)
    p4 = gmsh.model.geo.addPoint(-half_x, half_y, 0.0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p1, p4)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, -l3, -l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.mesh.setTransfiniteCurve(
        l1, CHIRP_NX + 1, "Progression", CHIRP_RX
    )
    gmsh.model.geo.mesh.setTransfiniteCurve(
        l3, CHIRP_NX + 1, "Progression", CHIRP_RX
    )
    gmsh.model.geo.mesh.setTransfiniteCurve(
        l2, CHIRP_NY + 1, "Bump", CHIRP_RY
    )
    gmsh.model.geo.mesh.setTransfiniteCurve(
        l4, CHIRP_NY + 1, "Bump", CHIRP_RY
    )

    gmsh.model.geo.mesh.setTransfiniteSurface(s)
    gmsh.model.geo.mesh.setRecombine(2, s)

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords_seq = node_coords.reshape(-1, 3)
    num_nodes = len(node_tags)

    tag_to_index = {int(tag): idx for idx, tag in enumerate(node_tags)}

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
    quad9_nodes = None
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 10:  # 9-node quad element
            quad9_nodes = np.array(enodes).reshape(-1, 9)
            break

    if quad9_nodes is None:
        gmsh.finalize()
        raise ValueError("No 9-node quad elements found in Gmsh mesh.")

    connect_seq = np.zeros_like(quad9_nodes, dtype=int)
    for i in range(quad9_nodes.shape[0]):
        for j in range(quad9_nodes.shape[1]):
            connect_seq[i, j] = tag_to_index[quad9_nodes[i, j]]

    gmsh.finalize()

    disp_x, disp_y, disp_z = apply_chirp_disp(coords_seq, num_nodes, a0_phys)
    uvs = compute_uvs(coords_seq, ROI_SIZE_X, ROI_SIZE_Y)

    case_tag = (
        f"plate{int(PLATE_SIZE_X)}x{int(PLATE_SIZE_Y)}_"
        f"cam{CAMERA_PIXELS}_quad9_chirp_gmsh"
    )
    out_dir = os.path.join("data", case_tag)

    save_csv(os.path.join(out_dir, "coords.csv"), coords_seq)
    save_csv(
        os.path.join(out_dir, "connectivity.csv"), connect_seq, is_int=True
    )
    save_csv(os.path.join(out_dir, "connect.csv"), connect_seq, is_int=True)
    save_csv(os.path.join(out_dir, "field_disp_x.csv"), disp_x)
    save_csv(os.path.join(out_dir, "field_disp_y.csv"), disp_y)
    save_csv(os.path.join(out_dir, "field_disp_z.csv"), disp_z)
    save_csv(os.path.join(out_dir, "uvs.csv"), uvs)

    print(f"Generated Gmsh case: {case_tag}")
    vis_fn(out_dir, vis_dir)


def main():
    a0_phys = CHIRP_A0 * PIXEL_SIZE

    # Import visualizer function dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from data_vis import visualize_case

    vis_dir = os.path.join("data", "vis")
    os.makedirs(vis_dir, exist_ok=True)

    generate_manual_chirp(a0_phys, visualize_case, vis_dir)
    generate_gmsh_chirp(a0_phys, visualize_case, vis_dir)


if __name__ == "__main__":
    main()
