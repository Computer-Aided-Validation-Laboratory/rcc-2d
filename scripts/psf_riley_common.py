"""Riley camera arguments shared by the explicit PSF render entry points."""

from __future__ import annotations

import os

import riley


def enabled() -> bool:
    return os.environ.get("RCC_ENABLE_PSF", "0") == "1"


def camera_kwargs(sigma_px: float, support_sigmas: float) -> dict[str, object]:
    """Return the isotropic separable Gaussian settings for ``riley.Camera``."""
    if not enabled():
        return {}
    return {
        "psf_type": riley.PsfType.gaussian,
        "psf_sigma_x": sigma_px,
        "psf_sigma_y": sigma_px,
        "psf_support_rad": sigma_px * support_sigmas,
        "psf_separable": 1,
    }
