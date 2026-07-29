"""Thin command-line entry points for the Experiment 3 render matrix."""
from __future__ import annotations

import os

from exp3_common import (
    bespoke_render, bit_depths, oversamples, riley_render, selected_cases,
    ssaa_levels,
)


def run(kind: str) -> None:
    """Run one resumable Exp3 render family selected by its wrapper script."""
    psf = kind.endswith("_psf")
    kind = kind.removesuffix("_psf")
    cases = selected_cases()
    if kind == "texgen_eggbox":
        from exp3_common import generate_texture
        for case in cases:
            for osamp in oversamples(): generate_texture(case, "eggbox", osamp)
        return
    if kind == "texgen_speckle":
        from exp3_common import generate_texture
        for case in cases:
            for pattern in ("diskaddsat", "gausscont"):
                for osamp in oversamples(): generate_texture(case, pattern, osamp)
        return
    if kind == "gridint2d_analytic":
        for case in cases:
            if "chirp" in case:
                print(f"  {case}: closed-form affine eggbox integration unavailable; skipping.")
                continue
            bespoke_render(case, "eggbox", "analytic", 1)
        return
    if kind == "gridint2d_ssaa":
        for case in cases:
            for ss in ssaa_levels(): bespoke_render(case, "eggbox", "ssaa", ss)
        return
    if kind == "speckint2d_analytic":
        for case in cases:
            if "chirp" in case:
                print(f"  {case}: closed-form affine speckle integration unavailable; skipping.")
                continue
            for pattern in ("diskaddsat", "gausscont"): bespoke_render(case, pattern, "analytic", 1)
        return
    if kind == "speckint2d_ssaa":
        for case in cases:
            for pattern in ("diskaddsat", "gausscont"):
                for ss in ssaa_levels(): bespoke_render(case, pattern, "ssaa", ss)
        return
    if kind == "speckint2d_disk":
        for case in cases:
            for ss in ssaa_levels(): bespoke_render(case, "diskaddsat", "ssaa", ss, psf=psf)
        return
    if kind == "riley_func_ssaa":
        for case in cases:
            for ss in ssaa_levels(): riley_render(case, "eggbox", "func", ss)
        return
    if kind in {"riley_texfloat_ssaa", "riley_texuint_ssaa"}:
        storage = "uint" if "texuint" in kind else "float"
        interps = os.environ.get("EXP3_TEX_INTERPOLATORS", "linear,cubic_catmull_rom").split(",")
        for case in cases:
            for pattern in ("eggbox", "diskaddsat", "gausscont"):
                for ss in ssaa_levels():
                    for osamp in oversamples():
                        for interp in interps: riley_render(case, pattern, "tex" + storage, ss, texture_os=osamp, interp=interp.strip(), storage=storage)
        return
    if kind in {"riley_texfloat_disk", "riley_texuint_disk"}:
        storage = "uint" if "texuint" in kind else "float"
        interps = os.environ.get("EXP3_TEX_INTERPOLATORS", "linear,cubic_catmull_rom").split(",")
        for case in cases:
            for ss in ssaa_levels():
                for osamp in oversamples():
                    for interp in interps:
                        riley_render(case, "diskaddsat", "tex" + storage, ss, texture_os=osamp, interp=interp.strip(), storage=storage, psf=psf)
        return
    raise ValueError(f"Unknown Exp3 render kind: {kind}")
