#!/usr/bin/env python3
"""Migrate existing nested Exp3 directories to flattened files."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modules.exp3_dic_data import parse_config


def migrate_exp3_dic() -> None:
    root = Path("out/exp3_dic")
    if not root.is_dir():
        print("out/exp3_dic directory not found; skipping.")
        return

    files_to_move: list[tuple[Path, Path]] = []
    
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        
        parts = path.relative_to(root).parts
        if len(parts) != 5:
            continue
            
        case, render_root, config_name, b_dir, filename = parts
        if not b_dir.startswith("b"):
            continue
            
        try:
            bit_depth = int(b_dir.removeprefix("b"))
        except ValueError:
            continue
            
        base_config, suffix = parse_config(config_name)
        target_dir = root / case / render_root / base_config
        
        if filename.startswith("dic_frame") and filename.endswith(".npz"):
            frame_str = filename.removeprefix("dic_frame").removesuffix(".npz")
            try:
                frame = int(frame_str)
            except ValueError:
                continue
            new_filename = f"dic_{suffix}_b{bit_depth:02d}_frame{frame:02d}.npz"
            
        elif filename.startswith("displacement_frame") and filename.endswith(".png"):
            frame_str = filename.removeprefix("displacement_frame").removesuffix(".png")
            try:
                frame = int(frame_str)
            except ValueError:
                continue
            new_filename = f"displacement_{suffix}_b{bit_depth:02d}_frame{frame:02d}.png"
            
        elif filename == "rigid_interpolation_bias.png":
            new_filename = f"rigid_interpolation_bias_{suffix}_b{bit_depth:02d}.png"
            
        elif filename == "rigid_bias_summary.csv":
            new_filename = f"rigid_bias_summary_{suffix}_b{bit_depth:02d}.csv"
            
        else:
            new_filename = f"{suffix}_b{bit_depth:02d}_{filename}"

        target_path = target_dir / new_filename
        files_to_move.append((path, target_path))

    moved_count = 0
    for src, dst in files_to_move:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if src.stat().st_size == dst.stat().st_size:
                src.unlink()
                continue
        shutil.move(src, dst)
        moved_count += 1
        
    print(f"Moved {moved_count} files in out/exp3_dic.")


def migrate_exp3_analysis_dic() -> None:
    root = Path("out/exp3_analysis_dic")
    if not root.is_dir():
        print("out/exp3_analysis_dic directory not found; skipping.")
        return
    
    files_to_move: list[tuple[Path, Path]] = []
    
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
            
        parts = path.relative_to(root).parts
        
        if len(parts) == 5:
            case, render_root, config_name, b_dir, filename = parts
            if not b_dir.startswith("b") or not filename.endswith("_difference.png"):
                continue
            try:
                bit_depth = int(b_dir.removeprefix("b"))
            except ValueError:
                continue
            base_config, suffix = parse_config(config_name)
            target_dir = root / case / render_root / base_config
            
            frame_str = filename.removeprefix("frame").removesuffix("_difference.png")
            try:
                frame = int(frame_str)
            except ValueError:
                continue
                
            new_filename = f"difference_{suffix}_b{bit_depth:02d}_frame{frame:02d}.png"
            target_path = target_dir / new_filename
            files_to_move.append((path, target_path))
            
        elif len(parts) == 4:
            case, series_name, b_dir, filename = parts
            if not b_dir.startswith("b") or not filename.endswith("_convergence.png"):
                continue
            try:
                bit_depth = int(b_dir.removeprefix("b"))
            except ValueError:
                continue
                
            if "_frame" in filename:
                pattern, rest = filename.split("_frame")
                frame_str = rest.split("_")[0]
                try:
                    frame = int(frame_str)
                except ValueError:
                    continue
                new_filename = f"{pattern}_convergence_b{bit_depth:02d}_frame{frame:02d}.png"
                target_dir = root / case / series_name
                target_path = target_dir / new_filename
                files_to_move.append((path, target_path))

    moved_count = 0
    for src, dst in files_to_move:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if src.stat().st_size == dst.stat().st_size:
                src.unlink()
                continue
        shutil.move(src, dst)
        moved_count += 1
        
    print(f"Moved {moved_count} files in out/exp3_analysis_dic.")


def cleanup_empty_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for dirname in dirnames:
            full_path = Path(dirpath) / dirname
            if full_path.is_dir() and not os.listdir(full_path):
                full_path.rmdir()


def main() -> None:
    migrate_exp3_dic()
    migrate_exp3_analysis_dic()
    cleanup_empty_dirs(Path("out/exp3_dic"))
    cleanup_empty_dirs(Path("out/exp3_analysis_dic"))
    print("Migration and clean up complete.")


if __name__ == "__main__":
    main()
