"""Machine-wide run controls shared by every experiment."""

CORES: int = 8
TEST_RUN: bool = True
FORCE_RENDER_OVER: bool = False

# Maximum independent analysis scripts launched by ``expall_analysis.py``.
ANALYSIS_JOBS: int = CORES
NUM_PROCESSES: int = CORES
RILEY_RASTER_THREADS: int = CORES
