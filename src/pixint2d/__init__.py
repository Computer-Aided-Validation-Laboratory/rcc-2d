"""PixInt2D: analytic textures and pixel integrals on one 2D FE mesh."""

from .core import (AnalyticRule, Camera2D, DisplacementSeries, GaussRule,
                   GaussianPSF, MappingMode, Mesh2D, RectRule, RenderOptions,
                   RenderResult, quantise_image)
from .grid2d import Eggbox, Grid2D
from .speck2d import AdditiveSpeckles, Speck2D

__version__ = "0.1.0"
__all__ = [
    "AdditiveSpeckles", "AnalyticRule", "Camera2D", "DisplacementSeries",
    "Eggbox", "GaussRule", "GaussianPSF", "Grid2D", "MappingMode", "Mesh2D",
    "RectRule", "RenderOptions", "RenderResult", "Speck2D", "quantise_image",
]
