"""Thin command-line entry points for the Experiment 3 render matrix."""
from __future__ import annotations

import os

from modules.exp3_common import (
    bespoke_render, bit_depths, oversamples, riley_render, selected_cases,
    ssaa_levels,
)
from exp3params import RILEY_TEXTURE_SAMPLERS, TEX_INTERPOLATORS
from modules.render_selection import custom_enabled, float_textures_enabled, riley_enabled, uint_textures_enabled


def run(kind: str) -> None:
    """Run one resumable Exp3 render family selected by its wrapper script."""
    psf = kind.endswith("_psf")
    kind = kind.removesuffix("_psf")
    cases = selected_cases()
    if kind == "texgen_eggbox":
        if not float_textures_enabled() and not uint_textures_enabled():
            print("  Eggbox texture generation disabled by RILEY_RENDER_CASES; skipping.")
            return
        from modules.exp3_common import generate_texture
        for case in cases:
            for osamp in oversamples(): generate_texture(case, "eggbox", osamp)
        return
    if kind == "texgen_speckle":
        if not float_textures_enabled() and not uint_textures_enabled():
            print("  Speckle texture generation disabled by RILEY_RENDER_CASES; skipping.")
            return
        from modules.exp3_common import generate_texture
        for case in cases:
            for pattern in ("diskaddsat", "gausscont"):
                for osamp in oversamples(): generate_texture(case, pattern, osamp)
        return
    if kind == "gridint2d_analytic":
        if not custom_enabled("eggbox"):
            return
        for case in cases:
            if "chirp" in case:
                print(f"  {case}: closed-form affine eggbox integration unavailable; skipping.")
                continue
            bespoke_render(case, "eggbox", "analytic", 1)
        return
    if kind == "gridint2d_ssaa":
        if not custom_enabled("eggbox"):
            return
        for case in cases:
            for ss in ssaa_levels(): bespoke_render(case, "eggbox", "ssaa", ss)
        return
    if kind == "speckint2d_analytic":
        if not any(custom_enabled(pattern) for pattern in ("disk", "gauss")):
            return
        for case in cases:
            if "chirp" in case:
                print(f"  {case}: closed-form affine speckle integration unavailable; skipping.")
                continue
            # Disk--pixel overlap is exact only for rigid maps: affine motion
            # maps a camera square to a parallelogram.  The general affine
            # Gaussian CDF integral is mathematically exact but is not a
            # tractable 512² production reference, so affine/chirp use the
            # highest SSAA result instead.
            patterns = tuple(pattern for pattern, selection in (("diskaddsat", "disk"), ("gausscont", "gauss")) if custom_enabled(selection) and "rigid" in case)
            if "rigid" not in case:
                print(f"  {case}: analytic speckle integration is unavailable at Exp3 scale; SSAA is the reference path.")
            for pattern in patterns:
                bespoke_render(case, pattern, "analytic", 1)
        return
    if kind == "speckint2d_ssaa":
        for case in cases:
            for pattern, selection in (("diskaddsat", "disk"), ("gausscont", "gauss")):
                if not custom_enabled(selection):
                    continue
                for ss in ssaa_levels(): bespoke_render(case, pattern, "ssaa", ss)
        return
    if kind == "speckint2d_disk":
        if not custom_enabled("disk_psf"):
            return
        for case in cases:
            for ss in ssaa_levels(): bespoke_render(case, "diskaddsat", "ssaa", ss, psf=psf)
        return
    if kind == "riley_func_ssaa":
        if not riley_enabled("func"):
            return
        for case in cases:
            for ss in ssaa_levels(): riley_render(case, "eggbox", "func", ss)
        return
    if kind in {"riley_texfloat_ssaa", "riley_texuint_ssaa"}:
        storage = "uint" if "texuint" in kind else "float"
        if storage == "uint" and not uint_textures_enabled():
            return
        if storage == "float" and not float_textures_enabled():
            return
        interps = os.environ.get("EXP3_TEX_INTERPOLATORS", ",".join(TEX_INTERPOLATORS)).split(",")
        for case in cases:
            for pattern in ("eggbox", "diskaddsat", "gausscont"):
                for ss in ssaa_levels():
                    for osamp in oversamples():
                        for interp in interps:
                            if interp.strip() not in RILEY_TEXTURE_SAMPLERS:
                                raise ValueError(f"Unsupported Riley sampler {interp.strip()!r}; choose from {', '.join(RILEY_TEXTURE_SAMPLERS)}")
                            if storage == "uint":
                                for bits in bit_depths():
                                    riley_render(case, pattern, "tex" + storage, ss, texture_os=osamp, interp=interp.strip(), storage=storage, source_bits=bits)
                            else:
                                riley_render(case, pattern, "tex" + storage, ss, texture_os=osamp, interp=interp.strip(), storage=storage)
        return
    if kind in {"riley_texfloat_disk", "riley_texuint_disk"}:
        storage = "uint" if "texuint" in kind else "float"
        if storage == "uint" and not uint_textures_enabled(psf=True):
            return
        if storage == "float" and not float_textures_enabled(psf=True):
            return
        interps = os.environ.get("EXP3_TEX_INTERPOLATORS", ",".join(TEX_INTERPOLATORS)).split(",")
        for case in cases:
            for ss in ssaa_levels():
                for osamp in oversamples():
                    for interp in interps:
                        if interp.strip() not in RILEY_TEXTURE_SAMPLERS:
                            raise ValueError(f"Unsupported Riley sampler {interp.strip()!r}; choose from {', '.join(RILEY_TEXTURE_SAMPLERS)}")
                        if storage == "uint":
                            for bits in bit_depths():
                                riley_render(case, "diskaddsat", "tex" + storage, ss, texture_os=osamp, interp=interp.strip(), storage=storage, source_bits=bits, psf=psf)
                        else:
                            riley_render(case, "diskaddsat", "tex" + storage, ss, texture_os=osamp, interp=interp.strip(), storage=storage, psf=psf)
        return
    raise ValueError(f"Unknown Exp3 render kind: {kind}")
