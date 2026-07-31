"""Shared render-family selection for every experiment."""
from __future__ import annotations

from exp0params_common import CUSTOM_RENDER_CASES, RILEY_RENDER_CASES


def custom_enabled(case: str) -> bool:
    return case in CUSTOM_RENDER_CASES


def riley_enabled(case: str) -> bool:
    return case in RILEY_RENDER_CASES


def uint_textures_enabled(*, psf: bool = False) -> bool:
    return riley_enabled("texuint_psf" if psf else "texuint")


def float_textures_enabled(*, psf: bool = False) -> bool:
    return riley_enabled("texfloat_psf" if psf else "texfloat")


def measurement_family(render_root: str) -> str:
    """Classify an Exp3 render root for DIC/Grid Method selection."""
    name = render_root.lower()
    if "texuint" in name or "texu" in name:
        return "texuint"
    if "texfloat" in name or "texf" in name:
        return "texfloat"
    if "riley_render_func" in name:
        return "func"
    if "gridint2d" in name or "speckint2d" in name or "grid2d" in name or "speck2d" in name:
        return "custom"
    raise ValueError(f"Cannot classify measurement render family: {render_root}")


def measurement_enabled(render_root: str, enabled_cases: tuple[str, ...]) -> bool:
    return measurement_family(render_root) in enabled_cases
