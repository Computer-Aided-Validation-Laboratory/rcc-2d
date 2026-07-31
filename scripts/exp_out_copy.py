#!/usr/bin/env python3
"""Safely import completed non-test Exp render/measurement outputs.

This copier is intentionally source-driven: a workstation export may contain
only a subset of the TEST_RUN=False matrix.  Every source output file absent
from the local output tree is copied with its directory structure preserved;
an existing destination file is never overwritten.
"""
from __future__ import annotations

import shutil
from pathlib import Path


# Change these two paths when importing output from another machine.
SOURCE_OUT_PATH = Path("/home/lloydf/temp/out")
DEST_OUT_PATH = Path("/home/lloydf/rcc-2d/out")


def is_requested_output_root(path: Path) -> bool:
    """Return whether a top-level output directory is a render/DIC/Grid result."""
    name = path.name
    if not name.startswith(("exp1_", "exp2_", "exp3_")):
        return False
    return (
        "_render" in name
        or name.endswith("_dic")
        or name.endswith("_gridmethod")
        or "_texture" in name
        or "_texgen_" in name
    )


def copy_output_tree(source_root: Path) -> tuple[int, int, int]:
    """Copy missing files from one output root, reporting each output unit.

    A unit is a directory holding files (normally one render configuration or
    one DIC/Grid result).  This gives concise progress while preserving nested
    case/configuration directories.
    """
    copied_files = copied_bytes = skipped_files = 0
    directories = [source_root, *sorted(path for path in source_root.rglob("*") if path.is_dir())]
    for source_dir in directories:
        files = sorted(path for path in source_dir.iterdir() if path.is_file())
        if not files:
            continue
        relative_dir = source_dir.relative_to(SOURCE_OUT_PATH)
        destination_dir = DEST_OUT_PATH / relative_dir
        missing = [path for path in files if not (destination_dir / path.name).exists()]
        if not missing:
            print(f"[SKIP] {relative_dir}: {len(files)} files already present")
            skipped_files += len(files)
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in missing:
            destination = destination_dir / source.name
            shutil.copy2(source, destination)
            copied_files += 1
            copied_bytes += source.stat().st_size
        retained = len(files) - len(missing)
        print(f"[COPY] {relative_dir}: copied {len(missing)} files" + (f", kept {retained}" if retained else ""))
        skipped_files += retained
    return copied_files, copied_bytes, skipped_files


def main() -> None:
    if not SOURCE_OUT_PATH.is_dir():
        raise FileNotFoundError(f"SOURCE_OUT_PATH does not exist: {SOURCE_OUT_PATH}")
    DEST_OUT_PATH.mkdir(parents=True, exist_ok=True)
    roots = sorted(path for path in SOURCE_OUT_PATH.iterdir() if path.is_dir() and is_requested_output_root(path))
    if not roots:
        raise FileNotFoundError("No Exp1--3 render, texture, DIC, or Grid Method roots found in SOURCE_OUT_PATH.")
    total_files = total_bytes = total_skipped = 0
    for source_root in roots:
        print(f"\n=== {source_root.name} ===")
        copied, copied_bytes, skipped = copy_output_tree(source_root)
        total_files += copied
        total_bytes += copied_bytes
        total_skipped += skipped
    print(
        f"\nCompleted: copied {total_files:,} files ({total_bytes / 2**30:.2f} GiB); "
        f"skipped {total_skipped:,} existing files."
    )


if __name__ == "__main__":
    main()
