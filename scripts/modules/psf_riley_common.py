"""Riley camera arguments shared by the explicit PSF render entry points."""

from __future__ import annotations

import os

import riley


def enabled() -> bool:
    return os.environ.get("RCC_ENABLE_PSF", "0") == "1"


def old_version_comparison() -> bool:
    """Return whether a deliberate pre-buffer-mode Riley comparison is running."""
    return os.environ.get("RCC_PSF_OLDVER", "0") == "1"


def output_name(name: str) -> str:
    """Keep deliberate old-Riley PSF renders separate from current results."""
    return f"{name}_oldver" if old_version_comparison() else name


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


def configure_raster_config(
    config: riley.RasterConfig, *, psf: bool | None = None,
    buffer_mode: str = "global_subpx_full",
) -> None:
    """Select an explicit Riley global sub-pixel buffer for PSF rasterisation.

    The PSF is evaluated after sub-pixel accumulation, so this mode avoids
    tile-boundary buffering artefacts and is the intended high-performance
    path in Riley's PSF implementation.  Fail loudly if an older Riley build
    is used instead of silently changing the PSF algorithm.
    """
    if psf is None:
        psf = enabled()
    if not psf:
        return
    if old_version_comparison():
        # The historical Riley build has no BufferMode.  Its default path is
        # intentionally retained for a like-for-like old/new comparison.
        return
    try:
        mode = getattr(riley.BufferMode, buffer_mode)
        config.buffer_mode = mode
    except AttributeError as error:
        raise RuntimeError(
            f"This PSF render requires Riley BufferMode.{buffer_mode}. "
            "Rebuild and reinstall the updated Riley binding into .venv."
        ) from error
