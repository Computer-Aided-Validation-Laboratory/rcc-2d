"""Grid2D: analytic eggbox rendering with exact or numerical pixel integrals."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter

from pixint2d.core import (AnalyticRule, Camera2D, DisplacementSeries, GaussRule,
                            Mesh2D, RectRule, RenderOptions, RenderResult)
from pixint2d.mapping import inverse_map


@dataclass(frozen=True)
class Eggbox:
    mean: float = 0.5
    contrast: float = 0.4
    period: tuple[float, float] = (5.0, 5.0)
    phase: tuple[float, float] = (0.0, 0.0)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        kx, ky = 2*np.pi/self.period[0], 2*np.pi/self.period[1]
        return self.mean + .5*self.contrast*(1+np.cos(kx*x+self.phase[0])) * (1+np.cos(ky*y+self.phase[1])) - self.contrast


def _rule_points(rule: RectRule | GaussRule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = rule.samples_per_axis
    if isinstance(rule, RectRule):
        points, weights = (np.arange(n)+.5)/n, np.full(n, 1/n)
    else:
        points, weights = np.polynomial.legendre.leggauss(n); points = .5*(points+1); weights = .5*weights
    xx, yy = np.meshgrid(points, points); ww = np.outer(weights, weights)
    return xx.ravel(), yy.ravel(), ww.ravel()


class Grid2D:
    """Render a physical-coordinate eggbox texture over one 2D FE mesh."""

    def __init__(self, mesh: Mesh2D, camera: Camera2D, texture: Eggbox, *, options: RenderOptions = RenderOptions()) -> None:
        self.mesh, self.camera, self.texture, self.options = mesh, camera, texture, options

    def _analytic(self, displacement: DisplacementSeries, frame: int) -> RenderResult:
        if self.options.mapping.value != "affine":
            raise ValueError("Exact Grid2D integration requires MappingMode.AFFINE.")
        x0, y0 = self.camera.pixel_origins(); sx, sy = self.camera.pixel_size
        cx, cy = x0+.5*sx, y0+.5*sy
        # Querying three points yields the global inverse affine relation without
        # exposing mapper internals or changing the row-vector convention.
        qx = np.array((0., 1., 0.)); qy = np.array((0., 0., 1.))
        rx, ry, valid = inverse_map(self.mesh, displacement, frame, qx, qy, self.options.mapping)
        if not valid.all(): raise RuntimeError("Affine inverse map is invalid.")
        ax, ay = rx[1]-rx[0], ry[1]-ry[0]
        bx, by = rx[2]-rx[0], ry[2]-ry[0]
        offx, offy = rx[0], ry[0]
        kx, ky = 2*np.pi/self.texture.period[0], 2*np.pi/self.texture.period[1]

        def average(wx: float, wy: float, phase: float) -> np.ndarray:
            factor = np.sinc(wx*sx/(2*np.pi))*np.sinc(wy*sy/(2*np.pi))
            return factor*np.cos(wx*cx + wy*cy + phase)

        cosx = average(kx*ax, kx*bx, kx*offx+self.texture.phase[0])
        cosy = average(ky*ay, ky*by, ky*offy+self.texture.phase[1])
        plus = average(kx*ax+ky*ay, kx*bx+ky*by, kx*offx+ky*offy+self.texture.phase[0]+self.texture.phase[1])
        minus = average(kx*ax-ky*ay, kx*bx-ky*by, kx*offx-ky*offy+self.texture.phase[0]-self.texture.phase[1])
        image = self.texture.mean-.5*self.texture.contrast + .5*self.texture.contrast*(cosx+cosy) + .25*self.texture.contrast*(plus+minus)
        image = image.reshape(self.camera.pixels[1], self.camera.pixels[0])
        valid_mask = np.ones(image.shape, dtype=bool)
        return RenderResult(np.flipud(image), np.flipud(image.copy()), np.flipud(valid_mask))

    def render(self, displacement: DisplacementSeries, *, frame: int = 0, integration: RectRule | GaussRule | AnalyticRule = AnalyticRule()) -> RenderResult:
        """Render one frame; images use standard top-row-first camera order."""
        if isinstance(integration, AnalyticRule): return self._analytic(displacement, frame)
        x0, y0 = self.camera.pixel_origins(); sx, sy = self.camera.pixel_size
        ox, oy, weights = _rule_points(integration)
        points_per_pixel = len(weights); total = len(x0)
        chunk = max(1, self.options.max_points_per_chunk // points_per_pixel)

        def render_chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
            begin, end = bounds; base_x, base_y = x0[begin:end], y0[begin:end]
            qx = (base_x[:, None] + sx*ox).ravel(); qy = (base_y[:, None] + sy*oy).ravel()
            rx, ry, valid = inverse_map(self.mesh, displacement, frame, qx, qy, self.options.mapping)
            values = self.texture.evaluate(rx, ry); values[~valid] = self.camera.background
            return begin, (values.reshape(end-begin, points_per_pixel) @ weights), valid.reshape(end-begin, points_per_pixel).all(axis=1)

        bounds = [(start, min(start+chunk, total)) for start in range(0, total, chunk)]
        if self.options.workers == 1:
            results = [render_chunk(item) for item in bounds]
        else:
            with ThreadPoolExecutor(max_workers=self.options.workers) as executor:
                results = list(executor.map(render_chunk, bounds))
        image, valid = np.empty(total), np.empty(total, dtype=bool)
        for begin, values, mask in results: image[begin:begin+len(values)], valid[begin:begin+len(mask)] = values, mask
        image = image.reshape(self.camera.pixels[1], self.camera.pixels[0]); valid = valid.reshape(image.shape)
        if self.options.psf is not None:
            psf = self.options.psf
            image = gaussian_filter(image, psf.sigma_pixels, mode="constant", cval=self.camera.background,
                                    radius=round(psf.sigma_pixels*psf.support_sigmas))
        return RenderResult(np.flipud(image), np.flipud(image.copy()), np.flipud(valid))

    def render_many(self, displacement: DisplacementSeries, frames: list[int], integration: RectRule | GaussRule | AnalyticRule) -> list[RenderResult]:
        return [self.render(displacement, frame=frame, integration=integration) for frame in frames]
