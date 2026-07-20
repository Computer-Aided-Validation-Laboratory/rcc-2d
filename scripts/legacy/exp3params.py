# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Shared configuration for the high-accuracy 32 by 32 Exp3 renders."""

from pathlib import Path
from typing import Final

import riley


OUTPUT_DIR: Path = Path("./out/exp3")
TEXTURE_OUTPUT_DIR: Path = Path("./out/exp3_analytic_textures")
CLEAR_DIR: bool = False
FORCE_RENDER_OVER: bool = False

# Final camera image.  Textures cover this image plus a four-pixel border on
# every side, i.e. 40 by 40 base texels before TEX_OVERSAMPLES is applied.
TARG_PX_X: int = 32
TARG_PX_Y: int = 32
TEX_PX_PAD: int = 4
BIT_DEPTHS: list[int] = [8, 12, 16]
TEX_OVERSAMPLES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
RILEY_SSAA_LEVELS: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NUM_PROCESSES: int = 8
# Bound inter-process payloads and temporary arrays during texture generation.
TEXGEN_MAX_TEXELS_PER_BATCH: int = 262_144

TEX_INTERPOLATORS: dict[str, riley.TextureSample] = {
    "nearest": riley.TextureSample.nearest,
    "linear": riley.TextureSample.linear,
    "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
    # "cubic_mitchell_netravali": riley.TextureSample.cubic_mitchell_netravali,
    "lanczos3": riley.TextureSample.lanczos3,
}

# Existing deformation meshes describe the same world-space ROI.  Exp3 simply
# projects them into a 32 by 32 camera rather than their source camera size.
DEFORMATION_CASES: list[str] = [
    "plate260_cam256_quad9_rigid",
    "plate260_cam256_quad9_affine",
]
ACTIVE_FRAMES: list[int] = list(range(11))

# Eggbox parameters.
P_PIXELS: float = 5.0
I0: float = 0.5
GAMMA: float = 0.4

# Additive-speckle parameters.
ANALYTIC_SPECKLE_TYPES: list[str] = ["diskaddsat", "gausscont"]
BLACK_AREA_FRACTIONS: list[float] = [0.6]
PERTURBATION_DISTRIBUTIONS: list[str] = ["uniform"]
PERTURBATION_FRACTIONS: list[float] = [0.25]
RANDOM_SEED: int = 3
PX_PER_SPECK: float = 5.0
GAUSSIAN_CUTOFF_SIGMAS: float = 4.0
GAUSSIAN_EQUIVALENT_DISK_EDGE_FRACTION: float = 0.4
GAUSSIAN_CONTINUOUS_TAIL_SIGMAS: float = 8.0

# Laptop memory model.  20 GiB is available to the render; reserve 2 GiB for
# Python, the OS, mesh/image bookkeeping, and allocator headroom.  Riley f64
# scaling scratch is measured at about 154 B per sub-pixel.  Lanczos3 has the
# widest active filter support, so the seven-sub-pixel (1 + 2*3) tile width is
# used for every filter.  The texture factor permits modest binding overhead.
RILEY_AVAILABLE_MEMORY_GIB: Final[float] = 20.0
RILEY_RESERVED_MEMORY_GIB: Final[float] = 2.0
RILEY_MEMORY_BUDGET_BYTES: Final[int] = int(RILEY_AVAILABLE_MEMORY_GIB * 2**30)
RILEY_RESERVED_MEMORY_BYTES: Final[int] = int(RILEY_RESERVED_MEMORY_GIB * 2**30)
RILEY_MAX_THREADS: Final[int] = 8
RILEY_SCRATCH_BYTES_PER_SUBPIXEL: Final[int] = 154
RILEY_MAX_FILTER_HALO_PX: Final[int] = 3
RILEY_TEXTURE_BINDING_OVERHEAD: Final[float] = 1.10
