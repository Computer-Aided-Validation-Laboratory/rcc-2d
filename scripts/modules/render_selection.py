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


def quantised_float_textures_enabled(*, psf: bool = False) -> bool:
    """Whether simulated b-bit-equivalent f64 input textures are enabled."""
    return riley_enabled("texfq_psf" if psf else "texfq")


def measurement_family(render_root: str) -> str:
    """Classify an Exp3 render root for DIC/Grid Method selection."""
    name = render_root.lower()
    if "texuint" in name or "texu" in name:
        return "texuint"
    if "texfq" in name:
        return "texfq"
    if "texfloat" in name or "texf" in name:
        return "texfloat"
    if "riley_render_func" in name:
        return "func"
    if "gridint2d" in name or "speckint2d" in name or "grid2d" in name or "speck2d" in name:
        return "custom"
    raise ValueError(f"Cannot classify measurement render family: {render_root}")


def measurement_enabled(render_root: str, enabled_cases: tuple[str, ...]) -> bool:
    return measurement_family(render_root) in enabled_cases


def analysis_enabled(render_root: str, pattern: str | None = None) -> bool:
    """Return whether an existing render is enabled for analysis.

    This applies the same Exp0 family switches used for rendering, including
    PSF variants, so stale output directories cannot silently re-enter an
    all-analysis run.
    """
    name = render_root.lower()
    psf = "_psf" in name
    if "riley" in name:
        if "func" in name:
            return riley_enabled("func_psf" if psf else "func")
        if "texfq" in name:
            return riley_enabled("texfq_psf" if psf else "texfq")
        if "texu" in name or "texuint" in name:
            return riley_enabled("texuint_psf" if psf else "texuint")
        if "texf" in name or "texfloat" in name:
            return riley_enabled("texfloat_psf" if psf else "texfloat")
        return False
    pattern_name = (pattern or "").lower()
    if pattern_name in {"eggb", "eggbox"}:
        return custom_enabled("eggbox_psf" if psf else "eggbox")
    if pattern_name in {"diskadd", "diskaddsat", "disk"}:
        return custom_enabled("disk_psf" if psf else "disk")
    if pattern_name in {"gaussadd", "gausscont", "gauss"}:
        return custom_enabled("gauss")
    return False
