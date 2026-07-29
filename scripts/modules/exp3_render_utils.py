"""Experiment 3 camera, mesh and output helpers.

The render engine is intentionally in :mod:`exp3_common`; this module is the
experiment-local facade used by its entry points.
"""
from modules.exp3_common import (CASE_CAMERA_PIXELS, CASE_ROI_SIZES, load_case,
    selected_cases, selected_frames, texture_path)

__all__ = ["CASE_CAMERA_PIXELS", "CASE_ROI_SIZES", "load_case", "selected_cases", "selected_frames", "texture_path"]
