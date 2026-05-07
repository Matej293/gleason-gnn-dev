#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from src.consensus_overlay_viz import save_gt_overlay_png, save_gt_panel_png
from src.gleason_consensus_dataset import GleasonConsensusDataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GT consensus visualization PNGs.")
    p.add_argument("--data-root", type=str, default="./data", help="Root with Train_imgs/Test_imgs.")
    p.add_argument("--consensus-root", type=str, default="./data/consensus", help="Consensus directory root.")
    p.add_argument(
        "--image-subdirs",
        type=str,
        nargs="+",
        default=["Train_imgs", "Test_imgs"],
        help="Image subdirectories to search.",
    )
    p.add_argument("--output-dir", type=str, default="./outputs/consensus_gt_viz", help="Output directory.")
    p.add_argument("--alpha", type=float, default=0.45, help="GT overlay alpha in [0,1].")
    p.add_argument("--max-cases", type=int, default=64, help="Max number of cases (0 = all).")
    p.add_argument("--seed", type=int, default=42, help="Random seed used with --random-sample.")
    p.add_argument("--random-sample", action="store_true", help="Use random subset when --max-cases > 0.")
    p.add_argument("--image-ids", type=str, nargs="*", default=None, help="Optional explicit image IDs.")
    p.add_argument("--train-only", action="store_true", help="Restrict to Train_imgs samples.")
    p.add_argument("--test-only", action="store_true", help="Restrict to Test_imgs samples.")
    return p.parse_args()


def _select_items(ds: GleasonConsensusDataset, args: argparse.Namespace) -> list[dict]:
    items = list(ds.items)

    if args.train_only and args.test_only:
        raise ValueError("--train-only and --test-only are mutually exclusive")
    if args.train_only:
        items = [x for x in items if str(x.get("image_subdir")) == "Train_imgs"]
    if args.test_only:
        items = [x for x in items if str(x.get("image_subdir")) == "Test_imgs"]

    if args.image_ids:
        keep = set(str(x) for x in args.image_ids)
        items = [x for x in items if str(x["image_id"]) in keep]

    max_cases = max(0, int(args.max_cases))
    if max_cases > 0 and len(items) > max_cases:
        if args.random_sample:
            rng = random.Random(int(args.seed))
            rng.shuffle(items)
            items = items[:max_cases]
        else:
            items = items[:max_cases]

    return items


def main() -> int:
    args = _parse_args()

    ds = GleasonConsensusDataset(
        data_root=args.data_root,
        consensus_root=args.consensus_root,
        image_subdirs=tuple(str(x) for x in args.image_subdirs),
        transform=None,
        load_qc_report=False,
    )

    selected = _select_items(ds, args)

    out_root = Path(args.output_dir).resolve()
    overlay_dir = out_root / "overlay"
    panel_dir = out_root / "panel"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    id_to_idx = {str(item["image_id"]): i for i, item in enumerate(ds.items)}

    written = 0
    for i, item in enumerate(selected, start=1):
        image_id = str(item["image_id"])
        sample = ds[id_to_idx[image_id]]

        stem = f"{i:04d}_{image_id}"
        save_gt_overlay_png(
            output_path=overlay_dir / f"{stem}.png",
            image=sample["image"],
            hard_mask=sample["hard_mask"],
            ignore_mask=sample["ignore_mask"],
            alpha=float(args.alpha),
        )
        save_gt_panel_png(
            output_path=panel_dir / f"{stem}.png",
            image=sample["image"],
            hard_mask=sample["hard_mask"],
            ignore_mask=sample["ignore_mask"],
            image_id=image_id,
            alpha=float(args.alpha),
        )
        written += 1

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(Path(args.data_root).resolve()),
        "consensus_root": str(Path(args.consensus_root).resolve()),
        "output_dir": str(out_root),
        "count_total_discovered": len(ds.items),
        "count_selected": len(selected),
        "count_written": written,
        "filters": {
            "train_only": bool(args.train_only),
            "test_only": bool(args.test_only),
            "image_ids": [str(x) for x in args.image_ids] if args.image_ids else None,
            "max_cases": int(args.max_cases),
            "random_sample": bool(args.random_sample),
            "seed": int(args.seed),
        },
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote {written} overlay PNGs to {overlay_dir}")
    print(f"Wrote {written} panel PNGs to {panel_dir}")
    print(f"Summary: {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
