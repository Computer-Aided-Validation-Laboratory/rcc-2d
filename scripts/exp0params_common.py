"""Machine-wide run controls shared by every experiment."""

from enum import Enum


class RunMode(str, Enum):
    """Render-matrix selection.

    ``BIG`` is the strict set difference ``ALL - TEST`` so a fast test pass
    followed by BIG completes the full matrix without repeating test levels.
    """

    TEST = "test"
    ALL = "all"
    BIG = "big"

CORES: int = 8
RUN_MODE = RunMode.TEST
EXP12_TEST_SAMPLE_LEVELS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
FORCE_RENDER_OVER: bool = False
# Rebuild analysis outputs even when a completed-suite marker exists.  Set
# false after a full analysis pass to make the all-analysis launcher resume
# only incomplete suites.
FORCE_ANALYSIS_OVER: bool = True

# Maximum independent analysis scripts launched by ``expall_analysis.py``.
ANALYSIS_JOBS: int = CORES
# Grid Method processes one independent rendered image sequence per worker.
# Keep this separate from ``ANALYSIS_JOBS`` so an interactive Grid Method run
# can be throttled without changing the all-script analysis launcher.
GRIDMETHOD_JOBS: int = CORES
# Exp3 texgen distributes independent analytic row batches across this many
# processes.  The environment variable ``EXP3_TEXGEN_JOBS`` can temporarily
# reduce it without changing the shared workstation default.
TEXGEN_JOBS: int = CORES
NUM_PROCESSES: int = CORES
RILEY_RASTER_THREADS: int = CORES

# Render-family selection shared by Exp1--3 and their all-render launchers.
# Custom names select procedural/pixel-integration pattern families; Riley
# names select shader/storage families.  To re-enable digitised source-texture
# studies, add ``"texuint"`` (and, if wanted, ``"texuint_psf"``) below.
CUSTOM_RENDER_CASES: tuple[str, ...] = (
    "eggbox",
    # "eggbox_psf",
    "disk",
    "gauss",
    "disk_psf",
)
RILEY_RENDER_CASES: tuple[str, ...] = (
    "func",
    "texfloat",
    "func_psf",
    "texfloat_psf",
    # "texuint",       # Re-enable digitised texture source studies here.
    # "texuint_psf",   # Re-enable digitised PSF texture source studies here.
)

# Analysis-family selection for Exp3 image-measurement workflows.  These use
# the renderer/storage family, not the deformation case name: ``custom``
# covers gridint2d/speckint2d, while the remaining names identify Riley shader
# or texture paths.  Leave texuint disabled while its large source-texture
# render campaign is still running; add ``"texuint"`` later to process its
# already-rendered sequences.
DIC_CASES: tuple[str, ...] = (
    "custom",
    "texfloat",
    # "texuint",
)
GRIDMETHOD_CASES: tuple[str, ...] = (
    "custom",
    "func",
    "texfloat",
    # "texuint",
)
