"""Public, experiment-independent data objects for PixInt2D."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


class MappingMode(StrEnum):
    """Inverse reference-map implementation used by a renderer."""

    AFFINE = "affine"
    VTK = "vtk"
    QUAD9_NEWTON = "quad9_newton"
    STRUCTURED_QUAD9 = "structured_quad9"


@dataclass(frozen=True)
class Camera2D:
    """Orthographic camera specified in physical 2D world coordinates."""

    pixels: tuple[int, int]
    roi_size: tuple[float, float]
    background: float = 0.5

    def __post_init__(self) -> None:
        if min(self.pixels) < 1 or min(self.roi_size) <= 0.0:
            raise ValueError("Camera pixels and ROI dimensions must be positive.")

    @property
    def pixel_size(self) -> tuple[float, float]:
        return self.roi_size[0] / self.pixels[0], self.roi_size[1] / self.pixels[1]

    def pixel_origins(self) -> tuple[np.ndarray, np.ndarray]:
        """Return +Y-up bottom-left origins for every camera pixel."""
        width, height = self.pixels
        sx, sy = self.pixel_size
        ids = np.arange(width * height)
        return (-self.roi_size[0] / 2 + (ids % width) * sx,
                -self.roi_size[1] / 2 + (ids // width) * sy)


@dataclass(frozen=True)
class Mesh2D:
    """One 2D FE mesh. Connectivity is zero-based and uses Riley/VTK order."""

    coords: np.ndarray
    connectivity: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float64)
        connect = np.asarray(self.connectivity, dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("coords must have shape (nodes, >=2).")
        if connect.ndim != 2 or connect.size == 0:
            raise ValueError("connectivity must have shape (elements, nodes_per_element).")
        if connect.min() < 0 or connect.max() >= len(coords):
            raise ValueError("connectivity contains an invalid node index.")
        object.__setattr__(self, "coords", np.ascontiguousarray(coords[:, :2]))
        object.__setattr__(self, "connectivity", np.ascontiguousarray(connect))

    @classmethod
    def from_csv(cls, directory: str | Path) -> "Mesh2D":
        directory = Path(directory)
        connectivity = np.loadtxt(directory / "connectivity.csv", delimiter=",", dtype=np.int64)
        if connectivity.ndim == 1:
            connectivity = connectivity[None, :]
        return cls(
            np.loadtxt(directory / "coords.csv", delimiter=","),
            connectivity,
        )


@dataclass(frozen=True)
class DisplacementSeries:
    """Nodal in-plane displacement fields with shape ``(nodes, frames)``."""

    ux: np.ndarray
    uy: np.ndarray

    def __post_init__(self) -> None:
        ux, uy = np.asarray(self.ux, dtype=np.float64), np.asarray(self.uy, dtype=np.float64)
        if ux.ndim == 1: ux = ux[:, None]
        if uy.ndim == 1: uy = uy[:, None]
        if ux.shape != uy.shape or ux.ndim != 2:
            raise ValueError("ux and uy must have equal shape (nodes, frames).")
        object.__setattr__(self, "ux", np.ascontiguousarray(ux))
        object.__setattr__(self, "uy", np.ascontiguousarray(uy))

    @classmethod
    def from_csv(cls, directory: str | Path) -> "DisplacementSeries":
        directory = Path(directory)
        return cls(np.loadtxt(directory / "field_disp_x.csv", delimiter=","),
                   np.loadtxt(directory / "field_disp_y.csv", delimiter=","))

    @property
    def frames(self) -> int:
        return self.ux.shape[1]


@dataclass(frozen=True)
class RectRule:
    samples_per_axis: int
    kind: str = "rect"

    def __post_init__(self) -> None:
        if self.samples_per_axis < 1: raise ValueError("samples_per_axis must be positive.")


@dataclass(frozen=True)
class GaussRule:
    points_per_axis: int
    kind: str = "gauss"

    def __post_init__(self) -> None:
        if self.points_per_axis < 1: raise ValueError("points_per_axis must be positive.")

    @property
    def samples_per_axis(self) -> int:
        return self.points_per_axis


@dataclass(frozen=True)
class AnalyticRule:
    """Request an exact pixel integral, where the renderer supports it."""

    kind: str = "analytic"


@dataclass(frozen=True)
class GaussianPSF:
    sigma_pixels: float = 1.0
    support_sigmas: float = 4.0

    def __post_init__(self) -> None:
        if self.sigma_pixels <= 0.0 or self.support_sigmas <= 0.0:
            raise ValueError("Gaussian PSF sigma/support must be positive.")


@dataclass(frozen=True)
class RenderOptions:
    """Operational renderer choices; physical inputs remain separate."""

    mapping: MappingMode = MappingMode.VTK
    workers: int = 1
    max_points_per_chunk: int = 500_000
    psf: GaussianPSF | None = None
    use_numba: bool = True

    def __post_init__(self) -> None:
        if self.workers < 1 or self.max_points_per_chunk < 1:
            raise ValueError("workers and max_points_per_chunk must be positive.")


@dataclass(frozen=True)
class RenderResult:
    """A normalised image plus raw values before additive postprocessing."""

    image: np.ndarray
    raw: np.ndarray
    valid_mask: np.ndarray


def quantise_image(image: np.ndarray, bits: int) -> np.ndarray:
    """Clamp a normalised image to [0, 1] and return camera code values."""
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16].")
    maximum = (1 << bits) - 1
    return np.rint(np.clip(image, 0.0, 1.0) * maximum).astype(
        np.uint8 if bits <= 8 else np.uint16
    )
