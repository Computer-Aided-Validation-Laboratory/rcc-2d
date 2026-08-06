"""Compact, validated persistence for Exp3 DIC fields."""
from __future__ import annotations

import os
from pathlib import Path

import re

import numpy as np


SCHEMA_VERSION = 1


def parse_config(config_name: str) -> tuple[str, str]:
    """Parse config_name into (base_config, suffix)."""
    # 1. Riley format: pattern_seed_sampler_osX_ssY_f
    match_riley = re.match(r"^(.+?)_os(\d+)_ss(\d+)_f$", config_name)
    if match_riley:
        base = match_riley.group(1)
        suffix = f"os{match_riley.group(2)}_ss{match_riley.group(3)}_f"
        return base, suffix

    # 2. SSAA format: pattern_seed_ssY_f
    match_ssaa = re.match(r"^(.+?)_ss(\d+)_f$", config_name)
    if match_ssaa:
        base = match_ssaa.group(1) + "_ssaa"
        suffix = f"ss{match_ssaa.group(2)}_f"
        return base, suffix

    # 3. Analytic format: pattern_seed_analytic_f
    if config_name.endswith("_analytic_f"):
        base = config_name[:-11] + "_analytic"
        suffix = "analytic_f"
        return base, suffix

    return config_name, ""


def reconstruct_config_name(base_config: str, suffix: str) -> str:
    """Reconstruct the original config name from base_config and suffix."""
    if suffix == "analytic_f":
        return base_config.replace("_analytic", "") + "_analytic_f"
    elif suffix.startswith("ss"):
        return base_config.replace("_ssaa", "") + "_" + suffix
    else:
        return base_config + "_" + suffix


def result_path(directory: Path, suffix: str, bit_depth: int, frame: int) -> Path:
    return directory / f"dic_{suffix}_b{bit_depth:02d}_frame{frame:02d}.npz"


def save_result(path: Path, result: object) -> None:
    """Atomically save the DIC fields retained for analysis and diagnostics."""
    save_arrays(
        path, result.ss_x, result.ss_y, result.u_px, result.v_px,
        result.converged, result.cost,
    )


def read_pyvale_binary(path: Path) -> dict[str, np.ndarray]:
    """Read only the fields retained from a PyVale ``.dic2d`` result.

    This mirrors PyVale's documented row layout without importing PyVale.  It
    keeps spawned figure workers independent of PyVale's optional Blender
    dependency, and avoids allocating diagnostic arrays that Exp3 does not
    retain.
    """
    raw = path.read_bytes()
    basic_size = 2 * 4 + 3 * 8 + 1 + 3 * 8 + 4
    row_size = next((size for size in (basic_size + params * 8 for params in (12, 6, 2, 0)) if len(raw) % size == 0), None)
    if row_size is None or not raw:
        raise ValueError(f"PyVale binary has incomplete rows: {path}")
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(-1, row_size)

    def extract(width: int, dtype: np.dtype, offset: int) -> np.ndarray:
        return np.frombuffer(rows[:, offset:offset + width].copy(), dtype=dtype)

    offset = 0
    x = extract(4, np.int32, offset); offset += 4
    y = extract(4, np.int32, offset); offset += 4
    u = extract(8, np.float64, offset); offset += 8
    v = extract(8, np.float64, offset); offset += 8
    offset += 8  # displacement magnitude: deliberately not retained
    converged = extract(1, np.uint8, offset).astype(bool); offset += 1
    cost = extract(8, np.float64, offset)
    x_values, y_values = np.unique(x), np.unique(y)
    shape = (len(y_values), len(x_values))
    if x.size != shape[0] * shape[1]:
        raise ValueError(f"PyVale binary does not describe a rectangular subset grid: {path}")
    x_index = np.searchsorted(x_values, x)
    y_index = np.searchsorted(y_values, y)
    fields: dict[str, np.ndarray] = {
        "ss_x": np.meshgrid(x_values, y_values)[0],
        "ss_y": np.meshgrid(x_values, y_values)[1],
        "u_px": np.full((1, *shape), np.nan, dtype=np.float64),
        "v_px": np.full((1, *shape), np.nan, dtype=np.float64),
        "converged": np.zeros((1, *shape), dtype=bool),
        "cost_zncc": np.full((1, *shape), np.nan, dtype=np.float64),
    }
    fields["u_px"][0, y_index, x_index] = u
    fields["v_px"][0, y_index, x_index] = v
    fields["converged"][0, y_index, x_index] = converged
    fields["cost_zncc"][0, y_index, x_index] = cost
    return fields


def save_arrays(
    path: Path,
    ss_x: np.ndarray,
    ss_y: np.ndarray,
    u_px: np.ndarray,
    v_px: np.ndarray,
    converged: np.ndarray,
    cost_zncc: np.ndarray,
) -> None:
    """Atomically save the retained arrays from either PyVale or a CSV parser."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.uint8),
        "ss_x": np.asarray(ss_x, dtype=np.int16),
        "ss_y": np.asarray(ss_y, dtype=np.int16),
        "u_px": np.asarray(u_px, dtype=np.float32),
        "v_px": np.asarray(v_px, dtype=np.float32),
        "converged": np.asarray(converged, dtype=bool),
        "cost_zncc": np.asarray(cost_zncc, dtype=np.float32),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    validate_result(path, arrays)


def load_result(path: Path) -> dict[str, np.ndarray]:
    """Load one compact result without allowing object deserialisation."""
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"schema_version", "ss_x", "ss_y", "u_px", "v_px", "converged", "cost_zncc"}
    missing = required - result.keys()
    if missing or int(result["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported or incomplete DIC NPZ: {path}")
    shape = result["u_px"].shape
    grid_shape = shape[1:] if len(shape) == 3 else shape
    if (
        len(shape) != 3
        or any(result[name].shape != shape for name in ("v_px", "converged", "cost_zncc"))
        or any(result[name].shape != grid_shape for name in ("ss_x", "ss_y"))
    ):
        raise ValueError(f"Inconsistent DIC NPZ array shapes: {path}")
    return result


def validate_result(path: Path, source: dict[str, np.ndarray] | None = None) -> None:
    """Verify an atomically written compact result before raw-file deletion."""
    loaded = load_result(path)
    if source is None:
        return
    for name in ("ss_x", "ss_y", "converged"):
        if not np.array_equal(loaded[name], source[name]):
            raise RuntimeError(f"DIC NPZ validation failed for {name}: {path}")
    for name in ("u_px", "v_px", "cost_zncc"):
        if not np.allclose(loaded[name], source[name], rtol=0.0, atol=2e-6, equal_nan=True):
            raise RuntimeError(f"DIC NPZ validation failed for {name}: {path}")
