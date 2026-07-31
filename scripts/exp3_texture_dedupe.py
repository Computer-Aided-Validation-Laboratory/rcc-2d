#!/usr/bin/env python3
"""Deduplicate byte-identical Exp3 rigid/affine texture assets safely.

Only matching non-marker files are replaced by hard links to the rigid owner.
The logical affine paths and their per-case render-signature markers remain,
so resumable render jobs do not need to rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from exp3params import CASE_CAMERA_PIXELS, CASE_ROI_SIZES, DEFORMATION_CASES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="replace verified duplicates by hard links")
    args = parser.parse_args()
    owner = next(case for case in DEFORMATION_CASES if "rigid" in case)
    geometry = (CASE_CAMERA_PIXELS[owner], CASE_ROI_SIZES[owner])
    peers = [case for case in DEFORMATION_CASES if case != owner and (CASE_CAMERA_PIXELS[case], CASE_ROI_SIZES[case]) == geometry]
    roots = sorted(Path("out").glob("exp3_texgen_*"))
    candidates: list[tuple[Path, Path]] = []
    for root in roots:
        source_root, peer_root = root / owner.replace("plate", "pt").replace("quad9", "q9").replace("rigid", "rig"), None
        # Existing canonical output labels use the compact case spelling.
        for peer in peers:
            peer_label = peer.replace("plate", "pt").replace("quad9", "q9").replace("affine", "aff")
            peer_root = root / peer_label
            if not source_root.is_dir() or not peer_root.is_dir():
                continue
            for target in peer_root.rglob("*"):
                if not target.is_file() or target.suffix == ".sha256":
                    continue
                source = source_root / target.relative_to(peer_root)
                if source.is_file():
                    candidates.append((source, target))
    reclaim = 0
    verified: list[tuple[Path, Path]] = []
    for source, target in candidates:
        if source.stat().st_size != target.stat().st_size:
            raise RuntimeError(f"Texture size mismatch: {target}")
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Texture content mismatch: {target}")
        if source.stat().st_ino != target.stat().st_ino:
            reclaim += target.stat().st_size
            verified.append((source, target))
    print(f"Verified {len(verified)} duplicate texture assets; reclaimable {reclaim / 2**30:.2f} GiB")
    if not args.apply:
        print("Dry run only. Re-run with --apply to deduplicate.")
        return
    for source, target in verified:
        temporary = target.with_name(f".{target.name}.linktmp")
        os.link(source, temporary)
        if temporary.stat().st_ino != source.stat().st_ino:
            raise RuntimeError(f"Hard-link verification failed: {target}")
        os.replace(temporary, target)
    print("Deduplication complete; affine logical paths now hard-link to rigid texture assets.")


if __name__ == "__main__":
    main()
