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
