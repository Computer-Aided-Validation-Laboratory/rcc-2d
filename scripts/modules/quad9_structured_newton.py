"""Fast inverse map for a structured, rectangular 2D Quad9 mesh.

This is deliberately separate from :mod:`quad9_newton`: the latter remains
the lean one-element path used by Exp1/2.  Here, a structured reference-grid
locator provides a good candidate cell and Numba performs the actual
isoparametric inverse, exactly as required for the graded finite-star mesh.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def build_structured_quad9_topology(coords: np.ndarray, connect: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_edges, y_edges, cell_connect[y, x, node])`` after validation."""
    if connect.ndim != 2 or connect.shape[1] != 9:
        raise ValueError("structured Quad9 mapping requires (n, 9) connectivity.")
    corners = coords[connect[:, :4], :2]
    x_edges = np.unique(np.round(corners[:, :, 0].ravel(), decimals=12))
    y_edges = np.unique(np.round(corners[:, :, 1].ravel(), decimals=12))
    x_edges.sort(); y_edges.sort()
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    if nx < 1 or ny < 1 or nx * ny != len(connect):
        raise ValueError("Quad9 mesh is not a complete rectangular element grid.")
    cells = np.full((ny, nx, 9), -1, dtype=np.int64)
    for row in connect:
        element = coords[row, :2]
        xmin, xmax = element[:4, 0].min(), element[:4, 0].max()
        ymin, ymax = element[:4, 1].min(), element[:4, 1].max()
        ix = int(np.searchsorted(x_edges, 0.5 * (xmin + xmax), side="right") - 1)
        iy = int(np.searchsorted(y_edges, 0.5 * (ymin + ymax), side="right") - 1)
        if not (0 <= ix < nx and 0 <= iy < ny) or cells[iy, ix, 0] >= 0:
            raise ValueError("Quad9 connectivity does not form a unique rectangular grid.")
        # Validate axis-aligned rectangular reference geometry and the Riley
        # corner/mid-edge/centre node convention before using the shortcut.
        expected = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                             ((xmin+xmax)/2, ymin), (xmax, (ymin+ymax)/2),
                             ((xmin+xmax)/2, ymax), (xmin, (ymin+ymax)/2),
                             ((xmin+xmax)/2, (ymin+ymax)/2)))
        if not np.allclose(element, expected, rtol=0.0, atol=1e-9):
            raise ValueError("structured Newton mapping requires rectangular Quad9 reference cells in Riley order.")
        cells[iy, ix] = row
    if np.any(cells < 0):
        raise ValueError("Quad9 rectangular grid has missing cells.")
    return np.ascontiguousarray(x_edges), np.ascontiguousarray(y_edges), np.ascontiguousarray(cells)


@njit(cache=True)
def _shape(xi: float, eta: float, shape: np.ndarray, dxi: np.ndarray, deta: np.ndarray) -> None:
    p0 = .5*xi*(xi-1.); p1 = 1.-xi*xi; p2 = .5*xi*(xi+1.)
    q0 = .5*eta*(eta-1.); q1 = 1.-eta*eta; q2 = .5*eta*(eta+1.)
    dp0 = xi-.5; dp1 = -2.*xi; dp2 = xi+.5
    dq0 = eta-.5; dq1 = -2.*eta; dq2 = eta+.5
    shape[0]=p0*q0; shape[1]=p2*q0; shape[2]=p2*q2; shape[3]=p0*q2
    shape[4]=p1*q0; shape[5]=p2*q1; shape[6]=p1*q2; shape[7]=p0*q1; shape[8]=p1*q1
    dxi[0]=dp0*q0; dxi[1]=dp2*q0; dxi[2]=dp2*q2; dxi[3]=dp0*q2
    dxi[4]=dp1*q0; dxi[5]=dp2*q1; dxi[6]=dp1*q2; dxi[7]=dp0*q1; dxi[8]=dp1*q1
    deta[0]=p0*dq0; deta[1]=p2*dq0; deta[2]=p2*dq2; deta[3]=p0*dq2
    deta[4]=p1*dq0; deta[5]=p2*dq1; deta[6]=p1*dq2; deta[7]=p0*dq1; deta[8]=p1*dq1


@njit(cache=True)
def _solve_cell(qx, qy, ix, iy, x_edges, y_edges, cells, deformed, reference, centre_seed):
    """Return reference x/y and a success flag for one candidate cell."""
    # Seed from the deformed corner box, not the reference box.  This matters
    # when an element's height is smaller than the local displacement: using
    # the reference coordinate can start Newton beyond a second Q2 root.
    n0=cells[iy,ix,0]; n1=cells[iy,ix,1]; n2=cells[iy,ix,2]; n3=cells[iy,ix,3]
    xmin=min(deformed[n0,0],deformed[n1,0],deformed[n2,0],deformed[n3,0])
    xmax=max(deformed[n0,0],deformed[n1,0],deformed[n2,0],deformed[n3,0])
    ymin=min(deformed[n0,1],deformed[n1,1],deformed[n2,1],deformed[n3,1])
    ymax=max(deformed[n0,1],deformed[n1,1],deformed[n2,1],deformed[n3,1])
    xi = 2.0 * (qx - xmin) / (xmax - xmin) - 1.0
    eta = 2.0 * (qy - ymin) / (ymax - ymin) - 1.0
    # A displaced point can lie beyond its *reference* cell bounds.  Starting
    # Newton outside [-1, 1] is unsafe for a biquadratic map: it can converge
    # to a second exterior root.  A bounded interior seed still exploits the
    # structured locator while converging to the physical in-cell root.
    xi = min(.75, max(-.75, xi))
    eta = min(.75, max(-.75, eta))
    if centre_seed:
        xi = 0.0
        eta = 0.0
    shape=np.empty(9); dxi=np.empty(9); deta=np.empty(9)
    converged = False
    for _ in range(48):
        _shape(xi, eta, shape, dxi, deta)
        fx=0.; fy=0.; j00=0.; j01=0.; j10=0.; j11=0.
        for node in range(9):
            n=cells[iy,ix,node]; x=deformed[n,0]; y=deformed[n,1]
            fx += shape[node]*x; fy += shape[node]*y
            j00 += dxi[node]*x; j01 += deta[node]*x
            j10 += dxi[node]*y; j11 += deta[node]*y
        rx=fx-qx; ry=fy-qy; determinant=j00*j11-j01*j10
        if abs(determinant) < 1.e-14: return 0., 0., False
        sx=(j11*rx-j01*ry)/determinant; sy=(-j10*rx+j00*ry)/determinant
        xi -= sx; eta -= sy
        if max(abs(sx), abs(sy)) < 1.e-11:
            converged=True; break
    if not converged or abs(xi) > 1.+1.e-9 or abs(eta) > 1.+1.e-9: return 0.,0.,False
    _shape(xi, eta, shape, dxi, deta)
    xr=0.; yr=0.
    for node in range(9):
        n=cells[iy,ix,node]; xr += shape[node]*reference[n,0]; yr += shape[node]*reference[n,1]
    return xr,yr,True


@njit(cache=True)
def _inverse(query_x, query_y, x_edges, y_edges, cells, deformed, reference, neighbour_radius, centre_seed):
    count=query_x.size; xr=np.empty(count); yr=np.empty(count); valid=np.zeros(count,dtype=np.bool_)
    nx=len(x_edges)-1; ny=len(y_edges)-1
    for point in range(count):
        qx=query_x[point]; qy=query_y[point]
        ix=np.searchsorted(x_edges,qx,side='right')-1; iy=np.searchsorted(y_edges,qy,side='right')-1
        if ix<0: ix=0
        if ix>=nx: ix=nx-1
        if iy<0: iy=0
        if iy>=ny: iy=ny-1
        found=False
        for radius in range(neighbour_radius+1):
            for jj in range(max(0,iy-radius),min(ny,iy+radius+1)):
                for ii in range(max(0,ix-radius),min(nx,ix+radius+1)):
                    # Avoid redoing the inner square on every radius.
                    if radius > 0 and ii > ix-radius and ii < ix+radius and jj > iy-radius and jj < iy+radius: continue
                    x,y,ok=_solve_cell(qx,qy,ii,jj,x_edges,y_edges,cells,deformed,reference,centre_seed)
                    if ok: xr[point]=x; yr[point]=y; valid[point]=True; found=True; break
                if found: break
            if found: break
        # A rare large deformation should remain correct, not merely fast.
        if not found:
            for jj in range(ny):
                for ii in range(nx):
                    x,y,ok=_solve_cell(qx,qy,ii,jj,x_edges,y_edges,cells,deformed,reference,centre_seed)
                    if ok: xr[point]=x; yr[point]=y; valid[point]=True; found=True; break
                if found: break
        if not found: xr[point]=0.; yr[point]=0.
    return xr,yr,valid


@njit(cache=True)
def _inverse_x_preserving(query_x, query_y, x_edges, y_edges, cells, deformed, reference):
    """Robust scalar fallback when the field leaves world X unchanged."""
    count=query_x.size; xr=np.empty(count); yr=np.empty(count); valid=np.zeros(count,dtype=np.bool_)
    nx=len(x_edges)-1; ny=len(y_edges)-1
    shape=np.empty(9); dxi=np.empty(9); deta=np.empty(9)
    for point in range(count):
        qx=query_x[point]; qy=query_y[point]
        ix=np.searchsorted(x_edges,qx,side='right')-1
        if ix<0: ix=0
        if ix>=nx: ix=nx-1
        xi=2.*(qx-x_edges[ix])/(x_edges[ix+1]-x_edges[ix])-1.
        found=False
        for iy in range(ny):
            eta=0.0
            for _ in range(48):
                _shape(xi,eta,shape,dxi,deta)
                value=0.; derivative=0.
                for node in range(9):
                    n=cells[iy,ix,node]; value+=shape[node]*deformed[n,1]; derivative+=deta[node]*deformed[n,1]
                if abs(derivative)<1.e-14: break
                step=(value-qy)/derivative; eta-=step
                if abs(step)<1.e-11: break
            if abs(eta)<=1.+1.e-9:
                _shape(xi,eta,shape,dxi,deta)
                residual=0.
                for node in range(9): residual+=shape[node]*deformed[cells[iy,ix,node],1]
                if abs(residual-qy)<1.e-8:
                    rx=0.; ry=0.
                    for node in range(9):
                        n=cells[iy,ix,node]; rx+=shape[node]*reference[n,0]; ry+=shape[node]*reference[n,1]
                    xr[point]=rx;yr[point]=ry;valid[point]=True;found=True;break
        if not found: xr[point]=0.;yr[point]=0.
    return xr,yr,valid


def inverse_map_structured_quad9(query_x, query_y, deformed_coords, reference_coords, topology, neighbour_radius: int = 2):
    """Invert a deformed structured Quad9 mesh using exact shape functions."""
    x_edges, y_edges, cells = topology
    query_x = np.ascontiguousarray(query_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_y, dtype=np.float64)
    deformed = np.ascontiguousarray(deformed_coords[:, :2])
    reference = np.ascontiguousarray(reference_coords[:, :2])
    xr, yr, valid = _inverse(query_x, query_y, x_edges, y_edges, cells, deformed, reference, neighbour_radius, False)
    # A centre-seeded retry covers rare high-curvature/cell-boundary Newton
    # basins without penalising the overwhelmingly common fast path.
    missing = ~valid
    if np.any(missing):
        rx, ry, ok = _inverse(query_x[missing], query_y[missing], x_edges, y_edges, cells, deformed, reference, neighbour_radius, True)
        xr[missing], yr[missing], valid[missing] = rx, ry, ok
    missing = ~valid
    if np.any(missing) and np.allclose(deformed[:, 0], reference[:, 0], rtol=0.0, atol=1e-12):
        rx, ry, ok = _inverse_x_preserving(query_x[missing], query_y[missing], x_edges, y_edges, cells, deformed, reference)
        xr[missing], yr[missing], valid[missing] = rx, ry, ok
    return xr, yr, valid
