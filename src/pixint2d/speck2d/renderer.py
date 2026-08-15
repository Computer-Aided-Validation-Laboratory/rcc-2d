"""Speck2D: deterministic additive disk and Gaussian speckle rendering."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from pixint2d.core import (AnalyticRule, Camera2D, DisplacementSeries, GaussRule,
                            Mesh2D, RectRule, RenderOptions, RenderResult)
from pixint2d.grid2d.renderer import _rule_points
from pixint2d.mapping import inverse_map


def _circle_box_area(x0: np.ndarray, y0: np.ndarray, width: float, height: float, radius: float) -> np.ndarray:
    def primitive(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -radius, radius)
        return .5*(x*np.sqrt(np.maximum(radius*radius-x*x, 0.0))+radius*radius*np.arcsin(x/radius))
    left, right, bottom, top = x0, x0+width, y0, y0+height
    lo, hi = np.maximum(left, -radius), np.minimum(right, radius); valid = hi > lo
    roots = []
    for edge in (bottom, top):
        root = np.sqrt(np.maximum(radius*radius-edge*edge, 0.0)); roots.extend((-root, root))
    points = np.stack([lo, hi, *[np.clip(root, lo, hi) for root in roots]], axis=1); points.sort(axis=1)
    area = np.zeros_like(x0)
    for index in range(points.shape[1]-1):
        start, end = points[:, index], points[:, index+1]; mid = .5*(start+end)
        half = np.sqrt(np.maximum(radius*radius-mid*mid, 0.0))
        upper, lower = half < top, -half > bottom
        integral = np.where(upper & lower, 2*(primitive(end)-primitive(start)),
                    np.where(upper, primitive(end)-primitive(start)-bottom*(end-start),
                    np.where(lower, top*(end-start)+primitive(end)-primitive(start), (top-bottom)*(end-start))))
        area += np.where((np.minimum(top, half)-np.maximum(bottom, -half)>0)&valid, integral, 0.0)
    return area


@dataclass(frozen=True)
class AdditiveSpeckles:
    """Finite seeded lattice of additive disks or continuous Gaussian blobs."""

    kind: str
    pitch: float
    diameter: float
    centres: np.ndarray
    grid_shape: tuple[int, int]
    lattice_origin: tuple[float, float]
    max_jitter: float
    intensity_mean: float = .5
    intensity_contrast: float = .4
    gaussian_edge_fraction: float = .1
    tail_sigmas: float = 6.0

    @property
    def radius(self) -> float: return .5*self.diameter
    @property
    def sigma(self) -> float: return self.radius/np.sqrt(-2*np.log(self.gaussian_edge_fraction))
    @property
    def support_radius(self) -> float: return self.radius if self.kind == "disk" else self.tail_sigmas*self.sigma

    @classmethod
    def jittered_lattice(cls, *, kind: str, speckle_diameter: float, black_area_fraction: float,
                         jitter_pdf: str, jitter: float, seed: int,
                         bounds: tuple[float, float, float, float], intensity_mean: float=.5,
                         intensity_contrast: float=.4, gaussian_edge_fraction: float=.1,
                         tail_sigmas: float=6.0) -> "AdditiveSpeckles":
        if kind not in {"disk", "gaussian"}: raise ValueError("kind must be 'disk' or 'gaussian'.")
        if jitter_pdf not in {"uniform", "gaussian"}: raise ValueError("Unknown jitter PDF.")
        pitch = speckle_diameter*np.sqrt(np.pi/(4*black_area_fraction)); radius=.5*speckle_diameter
        sigma = radius/np.sqrt(-2*np.log(gaussian_edge_fraction)); support = radius if kind == "disk" else tail_sigmas*sigma
        xmin,xmax,ymin,ymax=bounds; margin=support+4*jitter*pitch
        ix=np.arange(np.floor((xmin-margin)/pitch), np.ceil((xmax+margin)/pitch)+1)
        iy=np.arange(np.floor((ymin-margin)/pitch), np.ceil((ymax+margin)/pitch)+1)
        gx,gy=np.meshgrid(ix*pitch,iy*pitch); centres=np.column_stack((gx.ravel(),gy.ravel()))
        rng=np.random.default_rng(seed)
        offsets=(rng.uniform(-jitter,jitter,centres.shape) if jitter_pdf == "uniform" else rng.normal(0,jitter,centres.shape))
        centres += offsets*pitch
        return cls(kind,pitch,speckle_diameter,np.ascontiguousarray(centres),gx.shape,tuple(centres[0]),float(np.max(np.linalg.norm(offsets*pitch,axis=1))),intensity_mean,intensity_contrast,gaussian_edge_fraction,tail_sigmas)

    def coverage(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        shape = np.shape(x)
        x,y=np.asarray(x).ravel(),np.asarray(y).ravel(); ny,nx=self.grid_shape; centres=self.centres.reshape(ny,nx,2)
        base_x=np.rint((x-self.lattice_origin[0])/self.pitch).astype(int); base_y=np.rint((y-self.lattice_origin[1])/self.pitch).astype(int)
        reach=int(np.ceil((self.support_radius+self.max_jitter)/self.pitch)); result=np.zeros(len(x))
        for oy in range(-reach,reach+1):
            iy=base_y+oy; valid_y=(iy>=0)&(iy<ny)
            for ox in range(-reach,reach+1):
                ix=base_x+ox; valid=valid_y&(ix>=0)&(ix<nx)
                if not valid.any(): continue
                delta=np.column_stack((x[valid],y[valid]))-centres[iy[valid],ix[valid]]; r2=np.sum(delta*delta,axis=1)
                if self.kind == "disk": result[valid] += r2 <= self.radius*self.radius
                else:
                    inside=r2 <= self.support_radius*self.support_radius; result[np.flatnonzero(valid)[inside]] += np.exp(-.5*r2[inside]/self.sigma**2)
        return result.reshape(shape)

    def to_intensity(self, coverage: np.ndarray) -> np.ndarray:
        return np.clip(self.intensity_mean+self.intensity_contrast*(1-2*np.clip(coverage,0,1)),0,1)


class Speck2D:
    """Render an additive speckle field over one 2D FE mesh."""

    def __init__(self, mesh: Mesh2D, camera: Camera2D, pattern: AdditiveSpeckles, *, options: RenderOptions=RenderOptions()) -> None:
        self.mesh,self.camera,self.pattern,self.options=mesh,camera,pattern,options

    def _analytic_axis_aligned(self, displacement: DisplacementSeries, frame: int) -> RenderResult:
        if self.options.mapping.value != "affine": raise ValueError("Analytic Speck2D integration requires affine mapping.")
        # Manufactured rigid cases have an axis-aligned global inverse; reject
        # rotations/shears rather than silently applying an invalid box formula.
        rx,ry,_=inverse_map(self.mesh,displacement,frame,np.array((0.,1.,0.)),np.array((0.,0.,1.)),self.options.mapping)
        if abs(ry[1]-ry[0])>1e-10 or abs(rx[2]-rx[0])>1e-10 or abs((rx[1]-rx[0])-1)>1e-10 or abs((ry[2]-ry[0])-1)>1e-10:
            raise ValueError("Analytic disk/Gaussian rule currently requires rigid translation.")
        x0,y0=self.camera.pixel_origins(); sx,sy=self.camera.pixel_size; x0+=rx[0]; y0+=ry[0]
        ny,nx=self.pattern.grid_shape; centres=self.pattern.centres.reshape(ny,nx,2); base_x=np.rint((x0-self.pattern.lattice_origin[0])/self.pattern.pitch).astype(int); base_y=np.rint((y0-self.pattern.lattice_origin[1])/self.pattern.pitch).astype(int)
        reach=int(np.ceil((self.pattern.support_radius+self.pattern.max_jitter)/self.pattern.pitch)); coverage=np.zeros(len(x0))
        for oy in range(-reach,reach+1):
            iy=base_y+oy; valid_y=(iy>=0)&(iy<ny)
            for ox in range(-reach,reach+1):
                ix=base_x+ox; valid=valid_y&(ix>=0)&(ix<nx)
                if not valid.any(): continue
                centre=centres[iy[valid],ix[valid]]
                if self.pattern.kind == "disk": coverage[valid]+=_circle_box_area(x0[valid]-centre[:,0],y0[valid]-centre[:,1],sx,sy,self.pattern.radius)/(sx*sy)
                else:
                    scale=np.sqrt(2)*self.pattern.sigma; factor=self.pattern.sigma*np.sqrt(np.pi/2)
                    ixavg=factor*(erf((x0[valid]+sx-centre[:,0])/scale)-erf((x0[valid]-centre[:,0])/scale))
                    iyavg=factor*(erf((y0[valid]+sy-centre[:,1])/scale)-erf((y0[valid]-centre[:,1])/scale)); coverage[valid]+=ixavg*iyavg/(sx*sy)
        raw=coverage.reshape(self.camera.pixels[1],self.camera.pixels[0]); valid=np.ones_like(raw,dtype=bool)
        return RenderResult(np.flipud(self.pattern.to_intensity(raw)),np.flipud(raw),np.flipud(valid))

    def render(self, displacement: DisplacementSeries, *, frame: int=0, integration: RectRule|GaussRule|AnalyticRule=AnalyticRule()) -> RenderResult:
        if isinstance(integration,AnalyticRule): return self._analytic_axis_aligned(displacement,frame)
        x0,y0=self.camera.pixel_origins(); sx,sy=self.camera.pixel_size; ox,oy,weights=_rule_points(integration); n=len(weights); total=len(x0); chunk=max(1,self.options.max_points_per_chunk//n)
        def render_chunk(bounds: tuple[int,int]):
            begin,end=bounds; qx=(x0[begin:end,None]+sx*ox).ravel(); qy=(y0[begin:end,None]+sy*oy).ravel()
            rx,ry,valid=inverse_map(self.mesh,displacement,frame,qx,qy,self.options.mapping); values=self.pattern.coverage(rx,ry); values[~valid]=0
            return begin,values.reshape(end-begin,n)@weights,valid.reshape(end-begin,n).all(axis=1)
        bounds=[(start,min(start+chunk,total)) for start in range(0,total,chunk)]
        if self.options.workers == 1:
            results = [render_chunk(item) for item in bounds]
        else:
            with ThreadPoolExecutor(max_workers=self.options.workers) as executor:
                results = list(executor.map(render_chunk, bounds))
        raw,valid=np.empty(total),np.empty(total,dtype=bool)
        for begin,values,mask in results: raw[begin:begin+len(values)],valid[begin:begin+len(mask)]=values,mask
        raw=raw.reshape(self.camera.pixels[1],self.camera.pixels[0]); valid=valid.reshape(raw.shape)
        if self.options.psf is not None:
            psf=self.options.psf; raw=gaussian_filter(raw,psf.sigma_pixels,mode="constant",cval=0.,radius=round(psf.sigma_pixels*psf.support_sigmas))
        return RenderResult(np.flipud(self.pattern.to_intensity(raw)),np.flipud(raw),np.flipud(valid))

    def render_many(self, displacement: DisplacementSeries, frames: list[int], integration: RectRule|GaussRule|AnalyticRule) -> list[RenderResult]:
        return [self.render(displacement,frame=frame,integration=integration) for frame in frames]
