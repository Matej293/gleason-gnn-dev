#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.viz.consensus_overlay_viz import save_gt_overlay_png, save_gt_panel_png
from src.data.gleason_consensus_dataset import GleasonConsensusDataset

_PIL_RESAMPLING = getattr(Image, "Resampling", Image)


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
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for rendering/writing.")
    p.add_argument(
        "--max-long-side",
        type=int,
        default=1536,
        help="Downscale input before rendering (0 disables).",
    )
    p.add_argument("--overlay-format", type=str, default="webp", choices=["png", "jpg", "webp"])
    p.add_argument("--panel-format", type=str, default="jpg", choices=["png", "jpg", "webp"])
    p.add_argument("--png-compress-level", type=int, default=3, help="PNG compression level [0..9].")
    p.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality [1..95].")
    p.add_argument("--webp-quality", type=int, default=80, help="WEBP quality [1..100].")
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


def _maybe_resize(
    image: np.ndarray,
    hard: np.ndarray,
    ignore: np.ndarray,
    max_long_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_long_side <= 0:
        return image, hard, ignore
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_long_side:
        return image, hard, ignore
    scale = float(max_long_side) / float(longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    image_rs = np.asarray(Image.fromarray(image).resize((new_w, new_h), resample=_PIL_RESAMPLING.BILINEAR))
    hard_rs = np.asarray(Image.fromarray(hard).resize((new_w, new_h), resample=_PIL_RESAMPLING.NEAREST))
    ignore_rs = np.asarray(Image.fromarray(ignore).resize((new_w, new_h), resample=_PIL_RESAMPLING.NEAREST))
    return image_rs, hard_rs, ignore_rs


def _write_one_case(task: tuple[int, dict, str, str, float, int, bool, str, str, int, int, int]) -> int:
    (
        i,
        item,
        overlay_dir_s,
        panel_dir_s,
        alpha,
        compress_level,
        optimize,
        overlay_format,
        panel_format,
        jpeg_quality,
        webp_quality,
        max_long_side,
    ) = task
    image_id = str(item["image_id"])
    stem = f"{i:04d}_{image_id}"

    image = np.array(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
    hard = np.array(Image.open(item["hard_path"]), dtype=np.uint8)
    ignore = np.array(Image.open(item["ignore_path"]), dtype=np.uint8)
    image, hard, ignore = _maybe_resize(image=image, hard=hard, ignore=ignore, max_long_side=max_long_side)

    overlay_dir = Path(overlay_dir_s)
    panel_dir = Path(panel_dir_s)
    overlay_ext = "jpg" if overlay_format == "jpg" else overlay_format
    panel_ext = "jpg" if panel_format == "jpg" else panel_format
    overlay_quality = jpeg_quality if overlay_format == "jpg" else webp_quality
    panel_quality = jpeg_quality if panel_format == "jpg" else webp_quality

    save_gt_overlay_png(
        output_path=overlay_dir / f"{stem}.{overlay_ext}",
        image=image,
        hard_mask=hard,
        ignore_mask=ignore,
        alpha=alpha,
        image_format=overlay_format,
        compress_level=compress_level,
        optimize=optimize,
        quality=overlay_quality,
    )
    save_gt_panel_png(
        output_path=panel_dir / f"{stem}.{panel_ext}",
        image=image,
        hard_mask=hard,
        ignore_mask=ignore,
        image_id=image_id,
        alpha=alpha,
        image_format=panel_format,
        compress_level=compress_level,
        optimize=optimize,
        quality=panel_quality,
    )
    return 1


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

    workers = max(1, int(args.workers))
    png_compress_level = min(9, max(0, int(args.png_compress_level)))
    png_optimize = False
    jpeg_quality = min(95, max(1, int(args.jpeg_quality)))
    webp_quality = min(100, max(1, int(args.webp_quality)))
    max_long_side = max(0, int(args.max_long_side))
    overlay_format = str(args.overlay_format).lower()
    panel_format = str(args.panel_format).lower()
    tasks = [
        (
            i,
            item,
            str(overlay_dir),
            str(panel_dir),
            float(args.alpha),
            png_compress_level,
            png_optimize,
            overlay_format,
            panel_format,
            jpeg_quality,
            webp_quality,
            max_long_side,
        )
        for i, item in enumerate(selected, start=1)
    ]

    written = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_write_one_case, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating GT visualizations"):
            written += fut.result()

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
        "performance": {
            "workers": workers,
            "max_long_side": max_long_side,
            "overlay_format": overlay_format,
            "panel_format": panel_format,
            "png_compress_level": png_compress_level,
            "png_optimize": png_optimize,
            "jpeg_quality": jpeg_quality,
            "webp_quality": webp_quality,
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
