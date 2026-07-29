"""Shared discovery, reference, plotting and memory helpers for Exp3 analysis."""
from __future__ import annotations

import gc
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, ScalarFormatter

from exp_common_analysis import release_batch

OUT = Path("out")
CASE_RE = re.compile(r"plate.*?(?:rigid|affine|chirp)$")
SS_RE = re.compile(r"_ss(\d+)")
OS_RE = re.compile(r"_os(\d+)")


@dataclass(frozen=True)
class Render:
    case: str
    root: str
    config: str
    directory: Path
    pattern: str
    ssaa: int
    oversamp: int
    interpolator: str
    analytic: bool


def pattern_of(config: str) -> str:
    if config.startswith("eggbox"):
        return "eggbox"
    if config.startswith("diskaddsat"):
        return "diskaddsat"
    if config.startswith("gausscont"):
        return "gausscont"
    return config.split("_")[0]


def parameter(config: str, regex: re.Pattern[str]) -> int:
    match = regex.search(config)
    return int(match.group(1)) if match else 0


def interpolator_of(config: str) -> str:
    for name in ("cubic_catmull_rom", "linear", "func"):
        if f"_{name}_" in config or config.startswith(f"eggbox_{name}_"):
            return name
    return "bespoke"


def title_lines(text: str, width: int = 54) -> str:
    """Break long configuration names at underscores for figure titles."""
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text.replace("_", " "), width=width, break_long_words=False))


def numeric_y_axis(axis, values: list[float] | np.ndarray, *, log_when_positive: bool = True) -> None:
    """Use readable numbered vertical ticks, including exact-zero metrics."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    positive = data[data > 0.0]
    if log_when_positive and positive.size and not np.any(data <= 0.0):
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        return
    limit = max(1.0, float(np.max(np.abs(data)))) if data.size else 1.0
    if limit <= 1.0:
        axis.set_ylim(-0.05, 1.05)
    else:
        axis.set_ylim(-0.05 * limit, 1.05 * limit)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axis.yaxis.set_major_formatter(formatter)


def image_frames(directory: Path) -> dict[int, Path]:
    paths = list(directory.glob("frame*.npy")) or list(directory.glob("image_c00_f*.npy"))
    result: dict[int, Path] = {}
    for path in paths:
        match = re.search(r"(?:frame|_f)(\d+)", path.stem)
        if match:
            result[int(match.group(1))] = path
    return result


def load_image(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float64)
    return value / 255.0 if value.size and np.nanmax(value) > 1.0 + 1e-8 else value


def discover_renders() -> list[Render]:
    found: list[Render] = []
    for directory in OUT.glob("exp3_*render*/*/*"):
        if not directory.is_dir() or not image_frames(directory):
            continue
        case, root = directory.parent.name, directory.parent.parent.name
        if not CASE_RE.fullmatch(case):
            continue
        config = directory.name
        found.append(Render(
            case, root, config, directory, pattern_of(config),
            parameter(config, SS_RE), parameter(config, OS_RE),
            interpolator_of(config), "_analytic_" in config,
        ))
    return found


def best_reference(items: list[Render]) -> tuple[Render | None, str]:
    """Analytic first; otherwise the highest bespoke SSAA, then any renderer."""
    analytic = [item for item in items if item.analytic]
    if analytic:
        return analytic[0], "Analytic reference"
    if not items:
        return None, "No reference"
    bespoke = [item for item in items if "gridint2d" in item.root or "speckint2d" in item.root]
    if bespoke:
        reference = max(bespoke, key=lambda item: item.ssaa)
        return reference, f"Highest bespoke SSAA reference: SSAA={reference.ssaa}"
    reference = max(items, key=lambda item: (item.ssaa, item.oversamp))
    return reference, f"Highest SSAA/OS: SSAA={reference.ssaa}, OS={reference.oversamp or 1}"


def release(*arrays: object) -> None:
    del arrays
    gc.collect()
    release_batch()
