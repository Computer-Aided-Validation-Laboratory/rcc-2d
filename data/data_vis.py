import os
import numpy as np
import pyvista as pv

# Applied after fitting the mesh bounds.  Increase if you want a tighter view.
VIEW_ZOOM = 1.15


def load_case_data(case_dir):
    """Load mesh and displacement data from a case directory."""
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

    # Reshape to 2D if loaded as 1D
    if connect.ndim == 1:
        connect = connect.reshape(1, -1)
    if disp_x.ndim == 1:
        disp_x = disp_x.reshape(-1, 1)
    if disp_y.ndim == 1:
        disp_y = disp_y.reshape(-1, 1)

    return coords, connect, disp_x, disp_y


def get_pv_cell_type(nodes_per_elem):
    """Determine PyVista cell type and node count from connectivity."""
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
    """Build PyVista UnstructuredGrid mesh."""
    num_elems, nodes_per_elem = connect.shape
    cell_type = get_pv_cell_type(nodes_per_elem)

    cells = np.hstack(
        [np.full((num_elems, 1), nodes_per_elem), connect]
    ).ravel()
    cell_types = np.full(num_elems, cell_type, dtype=np.uint8)

    # PyVista expects 3D coordinates (coords is already N x 3)
    mesh = pv.UnstructuredGrid(cells, cell_types, coords)
    return mesh


def save_plot(mesh, title, scalar_name, output_path, *, show_edges):
    """Create and save off-screen PyVista plot."""
    plotter = pv.Plotter(off_screen=True)
    plotter.add_text(title, font_size=12, position="upper_edge")

    # Add mesh colored by scalars if scalar_name is provided
    if scalar_name:
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            show_edges=show_edges,
            cmap="viridis",
            edge_color="black",
            line_width=1.5,
        )
    else:
        plotter.add_mesh(
            mesh,
            color="lightgray",
            show_edges=show_edges,
            edge_color="black",
            line_width=1.5,
        )

    plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.zoom(VIEW_ZOOM)
    plotter.screenshot(output_path)
    plotter.close()


def save_plot_variants(mesh, title, scalar_name, output_stem):
    """Save matching scalar views with and without element-edge overlays."""
    save_plot(
        mesh, title, scalar_name, f"{output_stem}_noedges.png",
        show_edges=False,
    )
    save_plot(
        mesh, f"{title} (element edges)", scalar_name,
        f"{output_stem}_edges.png", show_edges=True,
    )


def visualize_case(case_dir, vis_dir):
    """Visualize a case and save screenshots."""
    case_name = os.path.basename(case_dir)
    print(f"Visualizing: {case_name}")

    try:
        coords, connect, disp_x, disp_y = load_case_data(case_dir)
    except FileNotFoundError as e:
        print(f"Skipping {case_name}: missing file ({e})")
        return

    # Extract final frame displacement
    u_final = disp_x[:, -1]
    v_final = disp_y[:, -1]
    mag_final = np.sqrt(u_final**2 + v_final**2)

    # Build PyVista mesh
    mesh = build_pv_mesh(coords, connect)
    mesh.point_data["U Displacement"] = u_final
    mesh.point_data["V Displacement"] = v_final
    mesh.point_data["Displacement Magnitude"] = mag_final

    # 1. Save Mesh Layout (uncolored)
    save_plot_variants(
        mesh,
        f"Mesh Layout: {case_name}",
        None,
        os.path.join(vis_dir, f"{case_name}_mesh"),
    )

    # 2. Save U Displacement (only if non-trivial or for completeness)
    if np.max(np.abs(u_final)) > 1e-9:
        save_plot_variants(
            mesh,
            f"U Displacement: {case_name}",
            "U Displacement",
            os.path.join(vis_dir, f"{case_name}_disp_x"),
        )

    # 3. Save V Displacement (only if non-trivial or for completeness)
    if np.max(np.abs(v_final)) > 1e-9:
        save_plot_variants(
            mesh,
            f"V Displacement: {case_name}",
            "V Displacement",
            os.path.join(vis_dir, f"{case_name}_disp_y"),
        )

    # 4. Save Displacement Magnitude
    save_plot_variants(
        mesh,
        f"Disp Magnitude: {case_name}",
        "Displacement Magnitude",
        os.path.join(vis_dir, f"{case_name}_disp_mag"),
    )


def main():
    base_dir = "data"
    vis_dir = os.path.join(base_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Gather case directories
    subdirs = []
    for item in os.listdir(base_dir):
        path = os.path.join(base_dir, item)
        if os.path.isdir(path) and item != "vis":
            subdirs.append(path)

    subdirs.sort()

    for case_dir in subdirs:
        visualize_case(case_dir, vis_dir)


if __name__ == "__main__":
    main()
