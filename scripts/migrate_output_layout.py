#!/usr/bin/env python3
"""Safely migrate ``out`` to the compact canonical output layout.

Moves are within one filesystem and therefore use atomic renames.  The script
plans all destinations before changing anything, rejects non-identical
collisions, and verifies file count and byte total afterwards.  It has no
legacy-path mode: once applied, scripts must use the canonical layout.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

from modules.output_naming import canonical_path_component, root_name


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def destination(root: Path, source: Path) -> Path:
    relative = source.relative_to(root)
    parts = relative.parts
    # ``z_texu`` was a temporary parking tree during the former layout
    # migration.  Its contents are real render data, so fold it into the
    # canonical root rather than leaving a hidden legacy hierarchy.
    if parts and parts[0] == "z_texu":
        parts = parts[1:]
    mapped: list[str] = []
    for depth, part in enumerate(parts):
        # DIC/Grid Method store the canonical render-root one level below the
        # abbreviated case directory.
        if depth == 2 and part.startswith("exp3_"):
            mapped.append(root_name(part))
        else:
            mapped.append(canonical_path_component(part, depth))
    return root.joinpath(*mapped)


def files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file()), key=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--apply", action="store_true", help="perform the verified migration")
    args = parser.parse_args()
    root = args.out.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    source_files = files(root)
    plan = [(source, destination(root, source)) for source in source_files]
    changed = [(source, target) for source, target in plan if source != target]
    by_target: dict[Path, list[Path]] = defaultdict(list)
    for source, target in plan:
        by_target[target].append(source)
    # Two old spellings can normalise to one name.  Identical duplicates are
    # safely coalesced later; preserve a genuinely different ancillary file
    # under a compact ``_prelayout`` suffix rather than overwrite it.
    resolved: list[tuple[Path, Path]] = []
    for target, sources in by_target.items():
        for number, source in enumerate(sources):
            if number == 0:
                resolved.append((source, target))
                continue
            first = sources[0]
            identical = source.stat().st_size == first.stat().st_size and digest(source) == digest(first)
            if identical:
                resolved.append((source, target))
                continue
            suffix = target.suffix
            alternative = target.with_name(f"{target.stem}_prelayout{number}{suffix}")
            while alternative.exists() or any(existing == alternative for _, existing in resolved):
                number += 1
                alternative = target.with_name(f"{target.stem}_prelayout{number}{suffix}")
            print(f"Preserving non-identical ancillary collision as {alternative.name}")
            resolved.append((source, alternative))
    plan = resolved
    changed = [(source, target) for source, target in plan if source != target]
    before_bytes = sum(path.stat().st_size for path in source_files)
    print(f"Output migration plan: {len(source_files):,} files; {len(changed):,} paths to rename; {before_bytes / 2**30:.2f} GiB")
    if not args.apply:
        print("Dry run only. Re-run with --apply to migrate.")
        return
    for index, (source, target) in enumerate(changed, 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if source.stat().st_size != target.stat().st_size or digest(source) != digest(target):
                raise RuntimeError(f"Refusing to overwrite non-identical file: {target}")
            source.unlink()
        else:
            os.replace(source, target)
        if index % 5000 == 0:
            print(f"  migrated {index:,}/{len(changed):,}", flush=True)
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    after_files = files(root)
    after_bytes = sum(path.stat().st_size for path in after_files)
    if len(after_files) != len(source_files) or after_bytes != before_bytes:
        raise RuntimeError(
            f"Migration verification failed: before={len(source_files):,}/{before_bytes}, "
            f"after={len(after_files):,}/{after_bytes}"
        )
    leftovers = [path for path in root.iterdir() if path.is_dir() and "_im" in path.name]
    if leftovers:
        raise RuntimeError(f"Legacy size-qualified roots remain: {leftovers[:5]}")
    print(f"Migration complete: {len(after_files):,} files, {after_bytes / 2**30:.2f} GiB; no size-qualified roots remain.")


if __name__ == "__main__":
    main()
