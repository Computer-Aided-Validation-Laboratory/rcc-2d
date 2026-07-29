"""Accurate 2D inverse mapping for one orthographic Quad9 element.

The public wrapper is deliberately narrow for now: the manufactured saddle is
one biquadratic quadrilateral.  Keeping the numerical kernel separate makes it
straightforward to add other element shape functions and a multi-cell locator
later without changing render code.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _inverse_quad9(
    query_x: np.ndarray,
    query_y: np.ndarray,
    deformed_coords: np.ndarray,
    reference_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = query_x.size
    x_reference = np.empty(count, dtype=np.float64)
    y_reference = np.empty(count, dtype=np.float64)
    valid = np.zeros(count, dtype=np.bool_)

    for point in range(count):
        xi = 0.0
        eta = 0.0
        converged = False
        for _iteration in range(24):
            phi0 = 0.5 * xi * (xi - 1.0)
            phi1 = 1.0 - xi * xi
            phi2 = 0.5 * xi * (xi + 1.0)
            psi0 = 0.5 * eta * (eta - 1.0)
            psi1 = 1.0 - eta * eta
            psi2 = 0.5 * eta * (eta + 1.0)
            dphi0 = xi - 0.5
            dphi1 = -2.0 * xi
            dphi2 = xi + 0.5
            dpsi0 = eta - 0.5
            dpsi1 = -2.0 * eta
            dpsi2 = eta + 0.5
            shape = np.array((
                phi0 * psi0, phi2 * psi0, phi2 * psi2, phi0 * psi2,
                phi1 * psi0, phi2 * psi1, phi1 * psi2, phi0 * psi1,
                phi1 * psi1,
            ))
            dxi = np.array((
                dphi0 * psi0, dphi2 * psi0, dphi2 * psi2, dphi0 * psi2,
                dphi1 * psi0, dphi2 * psi1, dphi1 * psi2, dphi0 * psi1,
                dphi1 * psi1,
            ))
            deta = np.array((
                phi0 * dpsi0, phi2 * dpsi0, phi2 * dpsi2, phi0 * dpsi2,
                phi1 * dpsi0, phi2 * dpsi1, phi1 * dpsi2, phi0 * dpsi1,
                phi1 * dpsi1,
            ))
            mapped_x = 0.0
            mapped_y = 0.0
            jac_x_xi = 0.0
            jac_x_eta = 0.0
            jac_y_xi = 0.0
            jac_y_eta = 0.0
            for node in range(9):
                node_x = deformed_coords[node, 0]
                node_y = deformed_coords[node, 1]
                mapped_x += shape[node] * node_x
                mapped_y += shape[node] * node_y
                jac_x_xi += dxi[node] * node_x
                jac_x_eta += deta[node] * node_x
                jac_y_xi += dxi[node] * node_y
                jac_y_eta += deta[node] * node_y
            residual_x = mapped_x - query_x[point]
            residual_y = mapped_y - query_y[point]
            determinant = jac_x_xi * jac_y_eta - jac_x_eta * jac_y_xi
            if abs(determinant) < 1.0e-14:
                break
            step_xi = (jac_y_eta * residual_x - jac_x_eta * residual_y) / determinant
            step_eta = (-jac_y_xi * residual_x + jac_x_xi * residual_y) / determinant
            xi -= step_xi
            eta -= step_eta
            if max(abs(step_xi), abs(step_eta)) <= 1.0e-12 and max(
                abs(residual_x), abs(residual_y)
            ) <= 1.0e-10:
                converged = True
                break
        if not converged or abs(xi) > 1.0 + 1.0e-10 or abs(eta) > 1.0 + 1.0e-10:
            x_reference[point] = 0.0
            y_reference[point] = 0.0
            continue
        # Re-evaluate the shape functions once at the converged parametric point.
        phi0 = 0.5 * xi * (xi - 1.0)
        phi1 = 1.0 - xi * xi
        phi2 = 0.5 * xi * (xi + 1.0)
        psi0 = 0.5 * eta * (eta - 1.0)
        psi1 = 1.0 - eta * eta
        psi2 = 0.5 * eta * (eta + 1.0)
        shape = np.array((
            phi0 * psi0, phi2 * psi0, phi2 * psi2, phi0 * psi2,
            phi1 * psi0, phi2 * psi1, phi1 * psi2, phi0 * psi1,
            phi1 * psi1,
        ))
        x_reference[point] = 0.0
        y_reference[point] = 0.0
        for node in range(9):
            x_reference[point] += shape[node] * reference_coords[node, 0]
            y_reference[point] += shape[node] * reference_coords[node, 1]
        valid[point] = True
    return x_reference, y_reference, valid


def inverse_map_quad9(
    query_x: np.ndarray,
    query_y: np.ndarray,
    deformed_coords: np.ndarray,
    reference_coords: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map deformed 2D points to reference coordinates through one Quad9 cell."""
    if connectivity.shape != (1, 9):
        raise NotImplementedError(
            "The Newton mapping currently supports exactly one Quad9 element; "
            f"received connectivity shape {connectivity.shape}."
        )
    node_ids = connectivity[0]
    if deformed_coords.shape[1] < 2 or reference_coords.shape[1] < 2:
        raise ValueError("Quad9 Newton mapping requires 2D coordinates.")
    return _inverse_quad9(
        np.ascontiguousarray(query_x, dtype=np.float64),
        np.ascontiguousarray(query_y, dtype=np.float64),
        np.ascontiguousarray(deformed_coords[node_ids, :2], dtype=np.float64),
        np.ascontiguousarray(reference_coords[node_ids, :2], dtype=np.float64),
    )
