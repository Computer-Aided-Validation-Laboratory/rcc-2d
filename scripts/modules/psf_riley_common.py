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
    workers: int | None = None,
    sub_sample: int | None = None,
    global_subpx_tile_size: int | None = None,
    global_subpx_stripe_size: int | None = None,
) -> None:
    """Select an explicit Riley global sub-pixel buffer for PSF rasterisation.

    The PSF is evaluated after sub-pixel accumulation, so this mode avoids
    tile-boundary buffering artefacts and is the intended high-performance
    path in Riley's PSF implementation.  Fail loudly if an older Riley build
    is used instead of silently changing the PSF algorithm.
    """
    if workers is not None:
        worker_count = max(1, int(workers))
        # Riley's global-buffer paths read these two limits directly.  Set
        # both after construction so an API-default change cannot silently
        # reduce a PSF render to one raster worker.
        config.total_threads = worker_count
        config.max_raster_workers_per_job = worker_count
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

    if sub_sample is None:
        return
    if sub_sample < 1:
        raise ValueError(f"sub_sample must be positive, got {sub_sample}")

    def configured_size(name: str, supplied: int | None) -> int | None:
        if supplied is not None:
            return supplied
        value = os.environ.get(name)
        return int(value) if value else None

    def aligned_size(
        name: str,
        requested: int | None,
        minimum: int,
        maximum: int,
    ) -> int:
        # A zero/omitted setting means choose the smallest valid size at this
        # SSAA level.  Explicit values stay explicit so a typo cannot silently
        # alter the intended global-buffer memory/performance trade-off.
        size = requested if requested is not None else (
            (minimum + sub_sample - 1) // sub_sample * sub_sample
        )
        if size < minimum or size > maximum:
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}], got {size}",
            )
        if size % sub_sample:
            raise ValueError(
                f"{name}={size} subpx must be divisible by SSAA={sub_sample}",
            )
        return size

    tile_size = aligned_size(
        "RCC_GLOBAL_SUBPX_TILE_SIZE",
        configured_size("RCC_GLOBAL_SUBPX_TILE_SIZE", global_subpx_tile_size),
        config.global_subpx_tile_size_min,
        config.global_subpx_tile_size_max,
    )
    config.global_subpx_tile_size_override = tile_size

    if mode is riley.BufferMode.global_subpx_stripe:
        stripe_size = aligned_size(
            "RCC_GLOBAL_SUBPX_STRIPE_SIZE",
            configured_size("RCC_GLOBAL_SUBPX_STRIPE_SIZE", global_subpx_stripe_size),
            config.global_subpx_stripe_size_min,
            config.global_subpx_stripe_size_max,
        )
        config.global_subpx_stripe_size_override = stripe_size
