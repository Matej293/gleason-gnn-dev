#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.eval.eval_utils import resolve_split_manifest_path, safe_read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fast class distribution audit over consensus_hard_mask.png files for "
            "the train split from the configured split manifest."
        )
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/deconver.yaml",
        help="Path to training config YAML.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    consensus_root = Path(str(cfg["consensus_root"]))
    manifest = safe_read_json(resolve_split_manifest_path(cfg))
    train_ids = set(str(x) for x in manifest.get("train_image_ids", []))
    if not train_ids:
        raise RuntimeError("No train_image_ids found in split manifest.")

    class_pixel_counts = np.zeros(4, dtype=np.int64)
    class_image_counts = np.zeros(4, dtype=np.int64)
    missing_masks = 0

    for image_id in sorted(train_ids):
        mask_path = consensus_root / image_id / "consensus_hard_mask.png"
        if not mask_path.exists():
            missing_masks += 1
            continue

        arr = np.array(Image.open(mask_path), dtype=np.int64)
        for c in range(4):
            n = int((arr == c).sum())
            class_pixel_counts[c] += n
            if n > 0:
                class_image_counts[c] += 1

    total = int(class_pixel_counts.sum())
    fractions = (class_pixel_counts / max(total, 1)).tolist()

    print(f"Config: {cfg_path}")
    print(f"Split manifest: {resolve_split_manifest_path(cfg)}")
    print(f"Train IDs in manifest: {len(train_ids)}")
    print(f"Missing hard masks: {missing_masks}")
    print("Class IDs: 0=benign, 1=G3, 2=G4, 3=G5")
    print(f"Pixel counts: {class_pixel_counts.tolist()}")
    print(f"Pixel fractions: {fractions}")
    print(f"Images containing class: {class_image_counts.tolist()}")


if __name__ == "__main__":
    main()
