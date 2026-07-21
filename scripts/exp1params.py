# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List, Tuple

import riley

CORES: int = 8
TEST_RUN: bool = True

if TEST_RUN:
    # SSAA levels to render with Riley
    SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
    TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]
    TEX_INTERPOLATORS: dict[str, riley.TextureSample] = {
        "nearest": riley.TextureSample.nearest,
        "linear": riley.TextureSample.linear,
        "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
        # "cubic_mitchell_netravali": riley.TextureSample.cubic_mitchell_netravali,
        # "lanczos3": riley.TextureSample.lanczos3,
    }
    # Integration methods and parameters
    INTEGRATION_METHODS: List[Tuple[str, int]] = [
        ("rect", 1),
        ("rect", 2),
        ("rect", 4),
        ("rect", 8),
        ("rect", 16),
        ("rect", 32),
        ("rect", 64),
        ("rect", 128),
        ("gauss", 2),
        ("gauss", 4),
        ("gauss", 8),
        ("gauss", 16),
        ("gauss", 32),
        ("gauss", 64),
        ("gauss", 128),
        ("analytic", 0),
    ]
    

else:
    # SSAA levels to render with Riley
    SSAA_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]    
    TEX_OVERSAMPLES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    TEX_INTERPOLATORS: dict[str, riley.TextureSample] = {
        "nearest": riley.TextureSample.nearest,
        "linear": riley.TextureSample.linear,
        "cubic_catmull_rom": riley.TextureSample.cubic_catmull_rom,
        # "cubic_mitchell_netravali": riley.TextureSample.cubic_mitchell_netravali,
        # "lanczos3": riley.TextureSample.lanczos3,
    }
    # Integration methods and parameters
    INTEGRATION_METHODS: List[Tuple[str, int]] = [
        ("rect", 1),
        ("rect", 2),
        ("rect", 4),
        ("rect", 8),
        ("rect", 16),
        ("rect", 32),
        ("rect", 64),
        ("rect", 128),
        ("rect", 256),
        ("rect", 512),
        ("rect", 1024), # TODO: check we have the RAM for this
        # ("mc", 16),
        # ("mc", 64),
        # ("mc", 256),
        # ("mc", 1024),
        ("gauss", 2),
        ("gauss", 4),
        ("gauss", 8),
        ("gauss", 16),
        ("gauss", 32),
        ("gauss", 64),
        ("gauss", 128),
        ("analytic", 0),
    ]
            

#-------------------------------------------------------------------------------
TARG_PX_X: int = 32
TARG_PX_Y: int = 32

def exp1_output_dir(name: str) -> Path:
    """Return a size-qualified Experiment 1 output directory."""
    return Path("./out") / f"{name}_im{TARG_PX_X}"

# Output directories for Exp 1.  The image-size suffix permits retaining
# results for several target sizes side by side.
OUTPUT_DIR: Path = exp1_output_dir("exp1_gridint2d_render_world")
TEXTURE_OUTPUT_DIR: Path = exp1_output_dir("exp1_analytic_textures")
CLEAR_DIR: bool = False
# Re-render existing outputs instead of skipping completed render frames.
FORCE_RENDER_OVER: bool = False

BACKGROUND: float = 0.5
TEX_PX_PAD: int = 4
BIT_DEPTHS: List[int] = [8, 12, 16]
NUM_PROCESSES: int = CORES
# Limit quadrature points held by each bespoke-renderer worker.  VTK mapping
# retains query, sampled-point and field arrays simultaneously, so it needs a
# much smaller cap than the affine corner-fit path on a 32 GiB workstation.
AFFINE_MAX_POINTS_PER_CHUNK: int = 10_000_000
VTK_MAX_POINTS_PER_CHUNK: int = 1_000_000
NEWTON_MAX_POINTS_PER_CHUNK: int = 1_000_000
# Riley uses one scratch tile per active raster worker.  For f64 builds,
# scalingpolicy uses about 154 B/sub-pixel, so per-worker scratch is
# 154 * ((tile_px + 2 * halo_px) * SSAA)^2 bytes.  With tile_size_min=1
# and no halo: SSAA 256/512/1024 uses about 9.6/38.5/154 MiB per worker.
# `RASTER_CHUNKS_PER_WORKER=4` schedules four work chunks, not four buffers.
RILEY_RASTER_THREADS: int = CORES


# Grid pattern parameters
P_PIXELS: float = 5.0
I0: float = 0.5
GAMMA: float = 0.4


# List of deformation cases to process (e.g. rigid, affine)
DEFORMATION_CASES: List[str] = [
    "plate260_cam256_quad9_rigid",
    "plate260_cam256_quad9_affine",
    "plate260_cam256_quad9_quadsaddle",
]

# Mapping from a deformed camera/image-plane point back to its reference
# coordinate.  ``affine`` fits one inverse affine map from the four pixel
# corners: it is exact for the rigid and global-affine manufactured fields,
# but is only an approximation for a general FE displacement.  ``newton`` is
# an accurate 2D inverse map for the current single Quad9 saddle; it remains
# explicit until other element-specific shape functions are added.
DEFORMATION_MAPPING_MODES: dict[str, str] = {
    "plate260_cam256_quad9_rigid": "affine",
    "plate260_cam256_quad9_affine": "affine",
    "plate260_cam256_quad9_quadsaddle": "newton",
}


def mapping_mode_for_case(case_name: str) -> str:
    """Return the explicitly configured reference-mapping mode for a case."""
    mode = DEFORMATION_MAPPING_MODES.get(case_name)
    if mode not in {"affine", "vtk", "newton"}:
        raise ValueError(f"No valid mapping mode configured for {case_name!r}.")
    return mode

# List of frames to generate and analyze (e.g. [0, 5])
ACTIVE_FRAMES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
