#!/usr/bin/env python3
"""Promote Experiment 1 outputs from commit 9df8472 to the current layout.

Commit ``9df8472d3fc3c87a24c68477db312e483a1f5a97`` saved each floating
Riley render under a bit-labelled directory and saved its ``.npy`` image in
camera-code units.  Current render and analysis scripts instead use one
normalised f64 image per configuration (in an ``_f`` directory) and derive
all requested camera TIFF depths from it.  The old bespoke renderer similarly
stored one code-scaled NPY for every bit depth in the filename.

This tool performs no rendering.  It promotes every completed *or partial*
frame it finds, selecting the highest available historical bit depth for each
canonical float image.  A partial configuration remains incomplete, so the
normal renderer will resume it rather than incorrectly skipping it.

Run a review first, then apply and prune only after checking the summary:

    .venv/bin/python scripts/exp1_migration.py
    .venv/bin/python scripts/exp1_migration.py --apply --remove-source

``--remove-source`` only removes files which have been verified against their
new normalised target.  It is deliberately opt-in.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from exp1params import BIT_DEPTHS
from modules.render_outputs import camera_tiff_path, quantise_camera


HISTORICAL_COMMIT = "9df8472d3fc3c87a24c68477db312e483a1f5a97"

RILEY_DIRS = (
    "exp1_riley_render_func_uvs_im32",
    "exp1_riley_render_func_uvs_psf_im32",
    "exp1_riley_render_texfloat_im32",
    "exp1_riley_render_texfloat_psf_im32",
)
BESPOKE_DIRS = (
    "exp1_gridint2d_render_uvs_im32",
    "exp1_gridint2d_render_uvs_psf_im32",
)

FUNC_CONFIG = re.compile(r"ss(?P<ssaa>\d+)_b(?P<bits>\d+)$")
TEXFLOAT_CONFIG = re.compile(
    r"ss(?P<ssaa>\d+)_b(?P<bits>\d+)_oversamp(?P<oversamp>\d+)$"
)
BESPOKE_FILE = re.compile(
    r"(?P<prefix>targ_px\d+_int_.+_param_-?\d+(?:_psf)?)_b(?P<bits>\d+)_frame(?P<frame>\d+)\.(?P<suffix>npy|tiff)$"
)
RILEY_NPY = re.compile(r"image_c00_f(?P<frame>\d+)\.npy$")
RILEY_TIFF = re.compile(r"cam0_frame(?P<frame>\d+)_field0\.tiff$")


@dataclass(frozen=True)
class Source:
    path: Path
    bits: int
    kind: str  # ``npy`` is preferred over the historical TIFF.


@dataclass
class Target:
    path: Path
    sources: list[Source] = field(default_factory=list)


def _normalise(source: Source) -> np.ndarray:
    """Load a commit-9df8472 camera-code image as a normalised float array."""
    if source.kind == "npy":
        image = np.asarray(np.load(source.path, mmap_mode="r"), dtype=np.float64)
    else:
        with Image.open(source.path) as opened:
            image = np.asarray(opened, dtype=np.float64)
    return np.ascontiguousarray(image / float((1 << source.bits) - 1))


def _add(targets: dict[Path, Target], target: Path, source: Source) -> None:
    targets.setdefault(target, Target(target)).sources.append(source)


def _riley_targets(out_root: Path, directory_name: str) -> dict[Path, Target]:
    """Discover historical bit-labelled Riley function/float configurations."""
    root = out_root / directory_name
    targets: dict[Path, Target] = {}
    if not root.is_dir():
        return targets
    config_pattern = TEXFLOAT_CONFIG if "texfloat" in directory_name else FUNC_CONFIG
    for config_dir in root.rglob("*"):
        if not config_dir.is_dir():
            continue
        match = config_pattern.fullmatch(config_dir.name)
        if match is None:
            continue
        bits = int(match["bits"])
        if "texfloat" in directory_name:
            canonical_name = (
                f"ss{match['ssaa']}_oversamp{match['oversamp']}_f"
            )
        else:
            canonical_name = f"ss{match['ssaa']}_f"
        canonical_dir = config_dir.with_name(canonical_name)
        for source_path in config_dir.iterdir():
            npy_match = RILEY_NPY.fullmatch(source_path.name)
            tiff_match = RILEY_TIFF.fullmatch(source_path.name)
            if npy_match is not None:
                frame = npy_match["frame"]
                _add(targets, canonical_dir / f"image_c00_f{frame}.npy", Source(source_path, bits, "npy"))
            elif tiff_match is not None:
                frame = int(tiff_match["frame"])
                _add(targets, canonical_dir / f"image_c00_f{frame:02d}.npy", Source(source_path, bits, "tiff"))
    return targets


def _bespoke_targets(out_root: Path, directory_name: str) -> dict[Path, Target]:
    """Discover old per-bit bespoke NPY/TIFF files in their existing case dirs."""
    root = out_root / directory_name
    targets: dict[Path, Target] = {}
    if not root.is_dir():
        return targets
    for source_path in root.rglob("targ_px*_int_*_param_*_b*_frame*.*"):
        match = BESPOKE_FILE.fullmatch(source_path.name)
        if match is None:
            continue
        target = source_path.with_name(
            f"{match['prefix']}_frame{match['frame']}.npy"
        )
        _add(targets, target, Source(source_path, int(match["bits"]), match["suffix"]))
    return targets


def _merge(*collections: dict[Path, Target]) -> dict[Path, Target]:
    targets: dict[Path, Target] = {}
    for collection in collections:
        for path, target in collection.items():
            targets.setdefault(path, Target(path)).sources.extend(target.sources)
    return targets


def _best_source(sources: list[Source]) -> Source:
    """Prefer source NPY and then the highest camera precision."""
    return max(sources, key=lambda item: (item.bits, item.kind == "npy"))


def _verify(existing: Path, normalised: np.ndarray) -> None:
    actual = np.asarray(np.load(existing, mmap_mode="r"), dtype=np.float64)
    if actual.shape != normalised.shape or not np.allclose(actual, normalised, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Refusing to overwrite non-equivalent canonical output: {existing}"
        )


def _write_tiffs(target: Path, image: np.ndarray, depths: set[int]) -> int:
    written = 0
    for bits in sorted(depths):
        path = camera_tiff_path(target, bits)
        if not path.exists():
            Image.fromarray(quantise_camera(image, bits)).save(path)
            written += 1
    return written


def _remove_sources(sources: list[Source], stop_at: Path) -> int:
    removed = 0
    for source in sources:
        if source.path.exists():
            source.path.unlink()
            removed += 1
        parent = source.path.parent
        while parent != stop_at and parent.is_dir():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return removed


def migrate(out_root: Path, *, apply: bool, remove_source: bool) -> tuple[int, int, int]:
    collections = [
        *(_riley_targets(out_root, name) for name in RILEY_DIRS),
        *(_bespoke_targets(out_root, name) for name in BESPOKE_DIRS),
    ]
    targets = _merge(*collections)
    migrated = tiffs = removed = 0
    print(f"Historical commit: {HISTORICAL_COMMIT}")
    print(f"Discovered {len(targets):,} canonical frame targets under {out_root}.")

    for target in sorted(targets.values(), key=lambda item: str(item.path)):
        source = _best_source(target.sources)
        depths = set(BIT_DEPTHS) | {item.bits for item in target.sources}
        if not apply:
            print(f"would promote {source.path} -> {target.path}")
            continue
        image = _normalise(source)
        if target.path.exists():
            _verify(target.path, image)
        else:
            target.path.parent.mkdir(parents=True, exist_ok=True)
            np.save(target.path, image)
            migrated += 1
        tiffs += _write_tiffs(target.path, image, depths)
        if remove_source:
            # Re-read the target before destructive cleanup, so a failed disk
            # write can never discard the historical workstation result.
            _verify(target.path, image)
            removed += _remove_sources(target.sources, out_root)
        del image

    action = "Migrated" if apply else "Would migrate"
    print(f"{action} {migrated:,} float frames and wrote {tiffs:,} TIFFs.")
    if remove_source:
        print(f"Removed {removed:,} verified historical files.")
    return migrated, tiffs, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("out"), help="output root containing the historical Exp1 directories (default: out)")
    parser.add_argument("--apply", action="store_true", help="perform the migration; default is a dry-run")
    parser.add_argument("--remove-source", action="store_true", help="after verification, remove historical bit-labelled files (requires --apply)")
    args = parser.parse_args()
    if args.remove_source and not args.apply:
        parser.error("--remove-source requires --apply")
    migrate(args.out_root.resolve(), apply=args.apply, remove_source=args.remove_source)


if __name__ == "__main__":
    main()
