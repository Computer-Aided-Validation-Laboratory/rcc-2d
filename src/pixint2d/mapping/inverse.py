"""2D inverse FE map evaluators with explicit, validated conventions."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pixint2d.core import DisplacementSeries, MappingMode, Mesh2D
from .structured_quad9 import (
    build_structured_quad9_topology,
    inverse_map_structured_quad9,
)


def deformed_coords(mesh: Mesh2D, displacement: DisplacementSeries, frame: int) -> np.ndarray:
    if displacement.ux.shape[0] != len(mesh.coords):
        raise ValueError("Displacement node count does not match mesh.")
    if not 0 <= frame < displacement.frames:
        raise IndexError(f"frame {frame} is outside displacement series.")
    result = np.array(mesh.coords, copy=True)
    result[:, 0] += displacement.ux[:, frame]
    result[:, 1] += displacement.uy[:, frame]
    return result


def _cell_type(nodes: int) -> pv.CellType:
    types = {3: pv.CellType.TRIANGLE, 4: pv.CellType.QUAD, 6: pv.CellType.QUADRATIC_TRIANGLE,
             8: pv.CellType.QUADRATIC_QUAD, 9: pv.CellType.BIQUADRATIC_QUAD}
    try: return types[nodes]
    except KeyError as exc: raise ValueError(f"Unsupported element with {nodes} nodes.") from exc


def _vtk(mesh: Mesh2D, deformed: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = mesh.connectivity.shape[1]
    cells = np.hstack([np.full((len(mesh.connectivity), 1), nodes), mesh.connectivity]).ravel()
    points = np.column_stack((deformed, np.zeros(len(deformed))))
    grid = pv.UnstructuredGrid(cells, np.full(len(mesh.connectivity), _cell_type(nodes), dtype=np.uint8), points)
    grid.point_data["reference_x"] = mesh.coords[:, 0]
    grid.point_data["reference_y"] = mesh.coords[:, 1]
    query = np.column_stack((x, y, np.zeros(len(x))))
    sampled = pv.PolyData(query).sample(grid)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    return (np.asarray(sampled.point_data["reference_x"], dtype=np.float64),
            np.asarray(sampled.point_data["reference_y"], dtype=np.float64), valid)


def _affine(mesh: Mesh2D, deformed: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack((deformed, np.ones(len(deformed))))
    coeff, *_ = np.linalg.lstsq(design, mesh.coords, rcond=None)
    residual = np.max(np.abs(design @ coeff - mesh.coords))
    if residual > 1e-8:
        raise ValueError("Affine mapping requested for a non-affine displacement field.")
    reference = np.column_stack((x, y, np.ones(len(x)))) @ coeff
    return reference[:, 0], reference[:, 1], np.ones(len(x), dtype=bool)


def _shape_quad9(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.array((.5*xi*(xi-1), 1-xi*xi, .5*xi*(xi+1)))
    psi = np.array((.5*eta*(eta-1), 1-eta*eta, .5*eta*(eta+1)))
    dphi = np.array((xi-.5, -2*xi, xi+.5)); dpsi = np.array((eta-.5, -2*eta, eta+.5))
    return (np.array((phi[0]*psi[0],phi[2]*psi[0],phi[2]*psi[2],phi[0]*psi[2],phi[1]*psi[0],phi[2]*psi[1],phi[1]*psi[2],phi[0]*psi[1],phi[1]*psi[1])),
            np.array((dphi[0]*psi[0],dphi[2]*psi[0],dphi[2]*psi[2],dphi[0]*psi[2],dphi[1]*psi[0],dphi[2]*psi[1],dphi[1]*psi[2],dphi[0]*psi[1],dphi[1]*psi[1])),
            np.array((phi[0]*dpsi[0],phi[2]*dpsi[0],phi[2]*dpsi[2],phi[0]*dpsi[2],phi[1]*dpsi[0],phi[2]*dpsi[1],phi[1]*dpsi[2],phi[0]*dpsi[1],phi[1]*dpsi[1])))


def _quad9_newton(mesh: Mesh2D, deformed: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mesh.connectivity.shape != (1, 9):
        raise ValueError("quad9_newton requires exactly one Quad9 element.")
    ref, deformed = mesh.coords[mesh.connectivity[0]], deformed[mesh.connectivity[0]]
    rx, ry, valid = np.zeros_like(x), np.zeros_like(y), np.zeros(len(x), dtype=bool)
    for point, (qx, qy) in enumerate(zip(x, y, strict=True)):
        xi = eta = 0.0
        for _ in range(24):
            shape, dxi, deta = _shape_quad9(xi, eta)
            mapped = shape @ deformed
            jac = np.array(((dxi @ deformed[:, 0], deta @ deformed[:, 0]),
                            (dxi @ deformed[:, 1], deta @ deformed[:, 1])))
            residual = mapped - (qx, qy)
            if abs(np.linalg.det(jac)) < 1e-14: break
            step = np.linalg.solve(jac, residual); xi -= step[0]; eta -= step[1]
            if max(abs(step)) <= 1e-12 and max(abs(residual)) <= 1e-10: break
        else: continue
        if abs(xi) <= 1+1e-10 and abs(eta) <= 1+1e-10:
            shape, _, _ = _shape_quad9(xi, eta); mapped_ref = shape @ ref
            rx[point], ry[point], valid[point] = mapped_ref[0], mapped_ref[1], True
    return rx, ry, valid


def inverse_map(mesh: Mesh2D, displacement: DisplacementSeries, frame: int, x: np.ndarray, y: np.ndarray, mode: MappingMode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target-world points to reference coordinates for one mesh."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if frame == 0:
        return x.copy(), y.copy(), np.ones(len(x), dtype=bool)
    deformed = deformed_coords(mesh, displacement, frame)
    if mode is MappingMode.AFFINE: return _affine(mesh, deformed, x, y)
    if mode is MappingMode.QUAD9_NEWTON: return _quad9_newton(mesh, deformed, x, y)
    if mode is MappingMode.STRUCTURED_QUAD9:
        topology = build_structured_quad9_topology(mesh.coords, mesh.connectivity)
        return inverse_map_structured_quad9(x, y, deformed, mesh.coords, topology)
    if mode is MappingMode.VTK: return _vtk(mesh, deformed, x, y)
    raise ValueError(f"Unsupported mapping mode {mode!r}.")
