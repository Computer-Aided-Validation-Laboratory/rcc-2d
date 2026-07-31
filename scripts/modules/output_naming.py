"""Canonical, portable names for experiment output trees.

The names deliberately keep the experiment and renderer readable, while
avoiding repeated camera dimensions and verbose parameter strings that exceed
common removable-drive path limits.  Full fixed pattern settings remain in
the experiment parameter files and render metadata, not in every path.
"""
from __future__ import annotations

import re
from pathlib import Path


_ROOT_REPLACEMENTS = (
    ("gridint2d", "grid2d"),
    ("speckint2d", "speck2d"),
    ("texfloat", "texf"),
    ("texuint", "texu"),
)
_TOKEN_REPLACEMENTS = (
    ("cubic_catmull_rom", "cubiccm"),
    ("cubiccatrom", "cubiccm"),
    ("catmull_rom", "cubiccm"),
    ("nearest", "near"),
    ("linear", "line"),
    ("quadsaddle", "qsadd"),
    ("affine", "aff"),
    ("rigid", "rig"),
    ("quad9", "q9"),
    ("eggbox", "eggb"),
    ("diskaddsat", "diskadd"),
    ("gausscont", "gaussadd"),
)


def root_name(name: str) -> str:
    """Return a readable root name without its redundant image-size suffix."""
    result = re.sub(r"_im\d+(?:x\d+)?(?=_|$)", "", name)
    for old, new in _ROOT_REPLACEMENTS:
        result = result.replace(old, new)
    for old, new in _TOKEN_REPLACEMENTS:
        result = result.replace(old, new)
    return result


def case_name(name: str) -> str:
    """Abbreviate a data-case label for an output directory only."""
    result = name
    result = re.sub(r"\bplate", "pt", result)
    for old, new in _TOKEN_REPLACEMENTS:
        result = result.replace(old, new)
    return result


def data_case_name(name: str) -> str:
    """Return the original data-directory label for a canonical output case."""
    result = name
    for old, new in (
        ("qsadd", "quadsaddle"), ("aff", "affine"), ("rig", "rigid"),
        ("q9", "quad9"),
    ):
        result = re.sub(rf"(?<=_){old}(?=_|$)", new, result)
    return re.sub(r"^pt", "plate", result)


def is_rigid_case(name: str) -> bool:
    return name.endswith("_rig") or name.endswith("_rigid")


def config_name(name: str) -> str:
    """Abbreviate a render configuration, omitting fixed speckle controls."""
    result = name
    # Additive patterns are deterministic for the experiment configuration;
    # retain only their seed in output names.
    result = re.sub(
        r"disk(?:addsat|add)_blackfrac[0-9.]+_[a-z]+_j[0-9.]+_seed(\d+)",
        r"diskadd_seed\1", result,
    )
    result = re.sub(
        r"gauss(?:cont|add)_blackfrac[0-9.]+_[a-z]+_j[0-9.]+_seed(\d+)",
        r"gaussadd_seed\1", result,
    )
    # Older Exp3 configurations use the same details but sometimes omit the
    # black-fraction token spelling.
    result = re.sub(r"diskaddsat(?:_[a-z]+_j[0-9.]+)?_seed(\d+)", r"diskadd_seed\1", result)
    result = re.sub(r"gausscont(?:_[a-z]+_j[0-9.]+)?_seed(\d+)", r"gaussadd_seed\1", result)
    for old, new in _TOKEN_REPLACEMENTS:
        result = result.replace(old, new)
    result = result.replace("_oversamp", "_os")
    result = result.replace("_param_", "_")
    result = result.replace("_int_", "_")
    return result


def filename(name: str) -> str:
    """Canonicalise tokens embedded in a filename without changing its schema."""
    return config_name(name)


def output_root(name: str) -> Path:
    return Path("out") / root_name(name)


def canonical_path_component(name: str, depth: int) -> str:
    """Map an existing component by its position under ``out``.

    depth 0 is the render/analysis root; depth 1 is the FE case; deeper
    components are render configs or ordinary filenames.
    """
    if depth == 0:
        return root_name(name)
    if depth == 1 and (name.startswith("plate") or name.startswith("pt")):
        return case_name(config_name(name))
    # Keep frame-file schemas stable: they are parsed by analysis, DIC and
    # Grid Method.  Texture source filenames do carry verbose pattern tags,
    # so shorten only those unambiguous ``tex_px...`` files.
    if Path(name).suffix.lower() in {".npy", ".npz", ".tif", ".tiff", ".png", ".csv", ".dic2d", ".sha256"}:
        return config_name(name) if name.startswith("tex_px") else name
    return config_name(name)
