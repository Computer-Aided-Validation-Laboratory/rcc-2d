#!/usr/bin/env python3
"""Safely migrate Exp3 DIC CSV results to compact analysis NPZ files.

Every conversion writes an atomic compressed NPZ, reopens and validates it,
then (with ``--remove-source``) deletes that one CSV.  This is safe on the
nearly full workstation filesystem because it never accumulates a second full
result tree; ``--jobs`` only controls the small number of conversions in
flight.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from modules.exp3_dic_data import (
    result_path, save_arrays, validate_result, parse_config,
)


def frame_number(path: Path) -> int:
    return int(path.name.split("_")[1].removeprefix("frame"))


def convert_one(source: Path, remove_source: bool) -> tuple[str, bool, bool]:
    """Convert one CSV independently, keeping peak disk use to one NPZ."""
    bit_depth = int(source.parent.name.removeprefix("b"))
    config_name = source.parent.parent.name
    render_root = source.parent.parent.parent.name
    case = source.parent.parent.parent.parent.name
    base_config, suffix = parse_config(config_name)
    target_dir = (
        source.parent.parent.parent.parent.parent
        / "exp3_dic" / case / render_root / base_config
    )
    target = result_path(target_dir, suffix, bit_depth, frame_number(source))
    if target.exists():
        validate_result(target)
        removed = False
        if remove_source:
            source.unlink()
            removed = True
        return str(source), False, removed
    # PyVale's generic importer constructs every diagnostic field.  Migration
    # only retains six CSV columns, so parse and reshape those directly.
    raw = np.loadtxt(source, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 5, 6))
    x_values, y_values = np.unique(raw[:, 0]), np.unique(raw[:, 1])
    shape = (len(y_values), len(x_values))
    if raw.shape[0] != shape[0] * shape[1]:
        raise ValueError(f"DIC CSV does not describe a rectangular subset grid: {source}")
    ss_x, ss_y = np.meshgrid(x_values, y_values)
    save_arrays(
        target, ss_x, ss_y,
        raw[:, 2].reshape((1, *shape)), raw[:, 3].reshape((1, *shape)),
        raw[:, 4].astype(bool).reshape((1, *shape)), raw[:, 5].reshape((1, *shape)),
    )
    del raw
    removed = False
    if remove_source:
        source.unlink()
        removed = True
    return str(source), True, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("out/exp3_dic"))
    parser.add_argument("--apply", action="store_true", help="perform conversion; default is dry-run")
    parser.add_argument("--remove-source", action="store_true", help="delete a CSV only after its NPZ validates (requires --apply)")
    parser.add_argument("--jobs", type=int, default=1, help="independent conversions in flight (default: 1, safest for a full disk)")
    parser.add_argument("--limit", type=int, default=0, help="convert at most this many files; useful for a smoke test")
    args = parser.parse_args()
    if args.remove_source and not args.apply:
        parser.error("--remove-source requires --apply")
    sources = sorted(args.root.rglob("dic_frame*_*.csv"))
    if args.limit:
        sources = sources[:args.limit]
    converted = removed = skipped = 0
    if not args.apply:
        for source in sources:
            bit_depth = int(source.parent.name.removeprefix("b"))
            config_name = source.parent.parent.name
            render_root = source.parent.parent.parent.name
            case = source.parent.parent.parent.parent.name
            base_config, suffix = parse_config(config_name)
            target_dir = args.root / case / render_root / base_config
            target = result_path(target_dir, suffix, bit_depth, frame_number(source))
            print(f"would convert {source} -> {target}")
    else:
        workers = max(1, args.jobs)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(convert_one, source, args.remove_source): source for source in sources}
            for future in as_completed(futures):
                source, did_convert, did_remove = future.result()
                converted += int(did_convert)
                skipped += int(not did_convert)
                removed += int(did_remove)
                print(f"{'converted' if did_convert else 'verified'} {Path(source).name}", flush=True)
    print(f"Converted {converted:,}; existing NPZ {skipped:,}; removed source CSV {removed:,}.")


if __name__ == "__main__":
    main()
