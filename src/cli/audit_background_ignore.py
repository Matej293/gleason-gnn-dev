#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.gleason_consensus_dataset import (
    GleasonConsensusDataset,
    build_tissue_mask_from_image,
    clean_ignore_mask,
)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit background-ignore behavior used by consensus training.")
    p.add_argument("--data-root", type=str, default="data", help="Dataset root containing Train_imgs/Test_imgs.")
    p.add_argument("--consensus-root", type=str, default="data/consensus", help="Consensus root.")
    p.add_argument(
        "--out-json",
        type=str,
        default="outputs/background_ignore_audit.json",
        help="Path to write audit summary/details JSON.",
    )
    p.add_argument("--otsu-close-radius", type=int, default=3, help="Morphological closing radius for tissue mask.")
    p.add_argument(
        "--otsu-min-object-size",
        type=int,
        default=4096,
        help="Minimum object size setting for tissue-mask cleanup.",
    )
    p.add_argument(
        "--otsu-min-hole-size",
        type=int,
        default=4096,
        help="Minimum hole size setting for tissue-mask cleanup.",
    )
    p.add_argument("--top-k", type=int, default=20, help="Number of worst background-leakage cases to include.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset = GleasonConsensusDataset(
        data_root=args.data_root,
        consensus_root=args.consensus_root,
        transform=None,
        load_qc_report=False,
        enforce_background_ignore=True,
        otsu_close_radius=int(args.otsu_close_radius),
        otsu_min_object_size=int(args.otsu_min_object_size),
        otsu_min_hole_size=int(args.otsu_min_hole_size),
    )

    rows: list[dict] = []
    for item in tqdm(dataset.items, desc="Background-ignore audit", unit="case"):
        image = np.asarray(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
        ignore_stored = np.asarray(Image.open(item["ignore_path"]), dtype=np.uint8)

        tissue_mask = build_tissue_mask_from_image(
            image_rgb=image,
            close_radius=int(args.otsu_close_radius),
            min_object_size=int(args.otsu_min_object_size),
            min_hole_size=int(args.otsu_min_hole_size),
        )
        ignore_clean = clean_ignore_mask(
            ignore_mask=ignore_stored,
            tissue_mask=tissue_mask,
            enforce_background_ignore=True,
        )

        background = tissue_mask == 0
        tissue = tissue_mask == 1
        bg_total = int(background.sum())
        tissue_total = int(tissue.sum())

        bg_not_ignored = int(np.logical_and(background, ignore_clean == 0).sum())
        tissue_ignored = int(np.logical_and(tissue, ignore_clean > 0).sum())

        row = {
            "image_id": str(item["image_id"]),
            "bg_pixels": bg_total,
            "tissue_pixels": tissue_total,
            "bg_not_ignored_pixels": bg_not_ignored,
            "tissue_ignored_pixels": tissue_ignored,
            "bg_not_ignored_frac": float(bg_not_ignored / max(1, bg_total)),
            "tissue_ignored_frac": float(tissue_ignored / max(1, tissue_total)),
        }
        rows.append(row)

    bg_fracs = [float(r["bg_not_ignored_frac"]) for r in rows]
    tissue_fracs = [float(r["tissue_ignored_frac"]) for r in rows]
    top_k = max(1, int(args.top_k))
    worst_bg = sorted(rows, key=lambda x: float(x["bg_not_ignored_frac"]), reverse=True)[:top_k]

    summary = {
        "n_cases": int(len(rows)),
        "otsu_close_radius": int(args.otsu_close_radius),
        "otsu_min_object_size": int(args.otsu_min_object_size),
        "otsu_min_hole_size": int(args.otsu_min_hole_size),
        "bg_not_ignored_mean": _mean(bg_fracs),
        "bg_not_ignored_p95": _percentile(bg_fracs, 95),
        "bg_not_ignored_max": max(bg_fracs) if bg_fracs else None,
        "tissue_ignored_mean": _mean(tissue_fracs),
        "tissue_ignored_p95": _percentile(tissue_fracs, 95),
        "tissue_ignored_max": max(tissue_fracs) if tissue_fracs else None,
        "count_bg_leak_gt_0": int(sum(v > 0.0 for v in bg_fracs)),
        "count_bg_leak_gt_1pct": int(sum(v > 0.01 for v in bg_fracs)),
        "count_bg_leak_gt_5pct": int(sum(v > 0.05 for v in bg_fracs)),
        "worst_bg_not_ignored": worst_bg,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "cases": rows}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
