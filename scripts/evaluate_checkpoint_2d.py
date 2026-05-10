#!/usr/bin/env python3
"""
Evaluate a 2D Deconver checkpoint on Gleason consensus labels.

Usage:
  PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py \
      --run outputs/runs/<run_name>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from config_validation import validate_2d_deconver_config  # noqa: E402
from eval_utils import (  # noqa: E402
    collate_consensus_batch,
    compute_multiclass_metrics_from_pred,
    fmt_metric,
    json_float,
    load_test_indices_from_manifest,
    postprocess_predictions,
    resolve_split_manifest_path,
)
from gleason_consensus_dataset import GleasonConsensusDataset  # noqa: E402
from models import build_model  # noqa: E402
from utils import ensure_cuda_binary_compatibility, load_checkpoint  # noqa: E402
from visualization_2d import render_case_panel, save_case_panel  # noqa: E402
from wandb_logger import WandbLogger  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = [
    "macro_dice",
    "grade5_dice",
    "miou",
    "grade5_iou",
    "dice_benign",
    "dice_g3",
    "dice_g4",
    "dice_g5",
    "iou_benign",
    "iou_g3",
    "iou_g4",
    "iou_g5",
    "iou_tumor_vs_benign",
    "sensitivity",
    "precision",
    "ignored_pixel_fraction",
    "tumor_pixels_ignored_fraction",
]


def _aggregate_loo_from_qc_reports(
    dataset: GleasonConsensusDataset,
    subset_indices: list[int],
) -> dict[str, float]:
    vals: list[float] = []
    for idx in subset_indices:
        item = dataset.items[int(idx)]
        qc_path = Path(str(item.get("qc_path", "")))
        if not qc_path.exists():
            continue
        try:
            with qc_path.open("r", encoding="utf-8") as f:
                qc = json.load(f)
        except Exception:
            continue
        loo = qc.get("leave_one_out_agreement_per_pathologist", {})
        if not isinstance(loo, dict):
            continue
        for v in loo.values():
            if not isinstance(v, dict):
                continue
            d = v.get("dice_multiclass", None)
            if isinstance(d, (int, float)) and math.isfinite(float(d)):
                vals.append(float(d))
    if not vals:
        return {"mean_loo_dice_multiclass": float("nan"), "num_loo_entries": 0.0}
    return {
        "mean_loo_dice_multiclass": float(np.mean(vals)),
        "num_loo_entries": float(len(vals)),
    }


def _resolve_checkpoint(run_dir: Path, ckpt_arg: str | None) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")

    if ckpt_arg:
        direct = Path(ckpt_arg)
        if direct.exists():
            return direct.resolve()
        candidate = ckpt_dir / ckpt_arg
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_arg} (checked direct path and {ckpt_dir})"
        )

    best = ckpt_dir / "best.pt"
    if best.exists():
        return best.resolve()

    epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not epoch_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    return epoch_files[-1].resolve()


def _aggregate_per_case_metrics(
    per_case: list[dict[str, object]],
    prefix: str,
) -> dict[str, float]:
    sums = {k: 0.0 for k in METRIC_KEYS}
    counts = {k: 0 for k in METRIC_KEYS}
    for row in per_case:
        for k in METRIC_KEYS:
            v = row.get(f"{prefix}_{k}")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                sums[k] += float(v)
                counts[k] += 1
    return {k: (sums[k] / counts[k]) if counts[k] > 0 else float("nan") for k in METRIC_KEYS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Deconver 2D checkpoint on Gleason consensus test split.",
    )
    parser.add_argument(
        "--run",
        required=True,
        type=str,
        help="Run directory containing config.yaml and checkpoints/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path or filename inside run/checkpoints (default: best.pt or latest epoch).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path (default: <run>/evaluation_2d_summary.json).",
    )
    parser.add_argument("--save-viz", action="store_true", help="Save prediction panels as PNG.")
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Visualization output directory (default: <run>/eval_viz).",
    )
    parser.add_argument(
        "--viz-max-cases",
        type=int,
        default=64,
        help="Maximum number of cases to save as visualizations.",
    )
    parser.add_argument(
        "--viz-worst-k",
        type=int,
        default=0,
        help="If >0, save worst-K by per-case macro Dice (bounded by viz-max-cases).",
    )
    parser.add_argument(
        "--log-wandb-viz",
        action="store_true",
        help="Log up to 8 evaluation panels to W&B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    validate_2d_deconver_config(cfg, for_eval=True, require_paths=True)

    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
    logger.info("Using checkpoint: %s", ckpt_path)

    max_long_side = int(cfg.get("max_long_side", 0))
    deconver_strides = tuple(int(x) for x in cfg.get("deconver_strides", [1, 2, 2, 2]))
    resize_divisor = int(math.prod([s for s in deconver_strides if s > 1])) or 1

    dataset = GleasonConsensusDataset(
        data_root=cfg["data_root"],
        consensus_root=cfg["consensus_root"],
        image_subdirs=tuple(str(x) for x in cfg.get("image_subdirs", ["Train_imgs", "Test_imgs"])),
        transform=None,
        renormalize_probs=bool(cfg.get("renormalize_probs", True)),
        enforce_background_ignore=bool(cfg.get("enforce_background_ignore", True)),
        otsu_close_radius=int(cfg.get("otsu_close_radius", 3)),
        otsu_min_object_size=int(cfg.get("otsu_min_object_size", 4096)),
        otsu_min_hole_size=int(cfg.get("otsu_min_hole_size", 4096)),
        probs_eps=float(cfg.get("probs_eps", 1e-8)),
        load_qc_report=False,
        max_long_side=max_long_side or None,
        resize_divisor=resize_divisor,
    )

    split_manifest_path = resolve_split_manifest_path(cfg)
    test_indices = load_test_indices_from_manifest(dataset.items, split_manifest_path)
    test_ds = Subset(dataset, test_indices)

    loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("val_batch_size", cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_consensus_batch,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    ensure_cuda_binary_compatibility(device)

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    include_background_in_dice = bool(cfg.get("include_background_in_dice", False))
    post_min_comp = {
        1: int(cfg.get("post_min_component_size_g3", 0)),
        2: int(cfg.get("post_min_component_size_g4", 0)),
        3: int(cfg.get("post_min_component_size_g5", 0)),
    }

    per_case: list[dict[str, object]] = []
    viz_candidates: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            ignore_mask = batch["ignore_mask"].to(device, non_blocking=True)
            image_ids = [str(x) for x in batch["image_id"]]

            out = model(images)
            logits = out[0] if isinstance(out, list) else out
            pred = logits.argmax(dim=1)
            pred_post = postprocess_predictions(
                pred=pred,
                ignore_mask=ignore_mask,
                tissue_mask=batch.get("tissue_mask", None).to(device, non_blocking=True)
                if "tissue_mask" in batch
                else None,
                min_component_size_by_class=post_min_comp,
            )
            valid = ignore_mask == 0
            for i, image_id in enumerate(image_ids):
                sample_metrics_raw = compute_multiclass_metrics_from_pred(
                    pred=pred[i : i + 1],
                    hard_mask=hard_mask[i : i + 1],
                    ignore_mask=ignore_mask[i : i + 1],
                    include_background_in_dice=include_background_in_dice,
                )
                sample_metrics_post = compute_multiclass_metrics_from_pred(
                    pred=pred_post[i : i + 1],
                    hard_mask=hard_mask[i : i + 1],
                    ignore_mask=ignore_mask[i : i + 1],
                    include_background_in_dice=include_background_in_dice,
                )
                valid_pixels = int(valid[i].sum().item())
                pred_pos = int(((pred_post[i] > 0) & valid[i]).sum().item())
                gt_pos = int(((hard_mask[i] > 0) & valid[i]).sum().item())
                per_case.append(
                    {
                        "image_id": image_id,
                        "raw_macro_dice": json_float(sample_metrics_raw["macro_dice"]),
                        "raw_grade5_dice": json_float(sample_metrics_raw["grade5_dice"]),
                        "raw_miou": json_float(sample_metrics_raw["miou"]),
                        "raw_grade5_iou": json_float(sample_metrics_raw["grade5_iou"]),
                        "raw_dice_benign": json_float(sample_metrics_raw["dice_benign"]),
                        "raw_dice_g3": json_float(sample_metrics_raw["dice_g3"]),
                        "raw_dice_g4": json_float(sample_metrics_raw["dice_g4"]),
                        "raw_dice_g5": json_float(sample_metrics_raw["dice_g5"]),
                        "raw_iou_benign": json_float(sample_metrics_raw["iou_benign"]),
                        "raw_iou_g3": json_float(sample_metrics_raw["iou_g3"]),
                        "raw_iou_g4": json_float(sample_metrics_raw["iou_g4"]),
                        "raw_iou_g5": json_float(sample_metrics_raw["iou_g5"]),
                        "raw_iou_tumor_vs_benign": json_float(sample_metrics_raw["iou_tumor_vs_benign"]),
                        "raw_sensitivity": json_float(sample_metrics_raw["sensitivity"]),
                        "raw_precision": json_float(sample_metrics_raw["precision"]),
                        "raw_ignored_pixel_fraction": json_float(sample_metrics_raw["ignored_pixel_fraction"]),
                        "raw_tumor_pixels_ignored_fraction": json_float(sample_metrics_raw["tumor_pixels_ignored_fraction"]),
                        "post_macro_dice": json_float(sample_metrics_post["macro_dice"]),
                        "post_grade5_dice": json_float(sample_metrics_post["grade5_dice"]),
                        "post_miou": json_float(sample_metrics_post["miou"]),
                        "post_grade5_iou": json_float(sample_metrics_post["grade5_iou"]),
                        "post_dice_benign": json_float(sample_metrics_post["dice_benign"]),
                        "post_dice_g3": json_float(sample_metrics_post["dice_g3"]),
                        "post_dice_g4": json_float(sample_metrics_post["dice_g4"]),
                        "post_dice_g5": json_float(sample_metrics_post["dice_g5"]),
                        "post_iou_benign": json_float(sample_metrics_post["iou_benign"]),
                        "post_iou_g3": json_float(sample_metrics_post["iou_g3"]),
                        "post_iou_g4": json_float(sample_metrics_post["iou_g4"]),
                        "post_iou_g5": json_float(sample_metrics_post["iou_g5"]),
                        "post_iou_tumor_vs_benign": json_float(sample_metrics_post["iou_tumor_vs_benign"]),
                        "post_sensitivity": json_float(sample_metrics_post["sensitivity"]),
                        "post_precision": json_float(sample_metrics_post["precision"]),
                        "post_ignored_pixel_fraction": json_float(sample_metrics_post["ignored_pixel_fraction"]),
                        "post_tumor_pixels_ignored_fraction": json_float(sample_metrics_post["tumor_pixels_ignored_fraction"]),
                        "valid_pixels": valid_pixels,
                        "pred_positive_pixels": pred_pos,
                        "gt_positive_pixels": gt_pos,
                    }
                )
                viz_candidates.append(
                    {
                        "image_id": image_id,
                        "macro_dice": float(sample_metrics_post["macro_dice"]),
                        "grade5_dice": float(sample_metrics_post["grade5_dice"]),
                        "image": images[i].detach().cpu(),
                        "hard_mask": hard_mask[i].detach().cpu(),
                        "pred_mask": pred_post[i].detach().cpu(),
                        "ignore_mask": ignore_mask[i].detach().cpu(),
                    }
                )

    aggregate_raw = _aggregate_per_case_metrics(per_case=per_case, prefix="raw")
    aggregate_post = _aggregate_per_case_metrics(per_case=per_case, prefix="post")
    aggregate_raw["num_test_samples"] = float(len(test_indices))
    aggregate_post["num_test_samples"] = float(len(test_indices))
    if bool(cfg.get("eval_leave_one_rater_out", False)):
        loo = _aggregate_loo_from_qc_reports(dataset, test_indices)
        aggregate_raw.update(loo)
        aggregate_post.update(loo)

    logger.info(
        "Aggregate (raw) | macro_dice=%s miou=%s grade5_dice=%s grade5_iou=%s sens=%s prec=%s | test_samples=%d",
        fmt_metric(aggregate_raw["macro_dice"]),
        fmt_metric(aggregate_raw["miou"]),
        fmt_metric(aggregate_raw["grade5_dice"]),
        fmt_metric(aggregate_raw["grade5_iou"]),
        fmt_metric(aggregate_raw["sensitivity"]),
        fmt_metric(aggregate_raw["precision"]),
        len(test_indices),
    )
    logger.info(
        "Aggregate (post) | macro_dice=%s miou=%s grade5_dice=%s grade5_iou=%s sens=%s prec=%s | test_samples=%d",
        fmt_metric(aggregate_post["macro_dice"]),
        fmt_metric(aggregate_post["miou"]),
        fmt_metric(aggregate_post["grade5_dice"]),
        fmt_metric(aggregate_post["grade5_iou"]),
        fmt_metric(aggregate_post["sensitivity"]),
        fmt_metric(aggregate_post["precision"]),
        len(test_indices),
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "data_root": str(cfg["data_root"]),
        "consensus_root": str(cfg["consensus_root"]),
        "split_manifest_path": str(split_manifest_path),
        "aggregate": {k: json_float(v) for k, v in aggregate_post.items()},
        "aggregate_raw": {k: json_float(v) for k, v in aggregate_raw.items()},
        "aggregate_post": {k: json_float(v) for k, v in aggregate_post.items()},
        "per_case": per_case,
    }

    if args.save_viz:
        viz_dir = (
            Path(args.viz_dir).resolve()
            if args.viz_dir is not None
            else (run_dir / "eval_viz")
        )
        viz_max_cases = max(0, int(args.viz_max_cases))
        viz_worst_k = max(0, int(args.viz_worst_k))
        selected = viz_candidates[:viz_max_cases]
        if viz_worst_k > 0:
            finite = [x for x in viz_candidates if not math.isnan(float(x["macro_dice"]))]
            finite = sorted(finite, key=lambda x: float(x["macro_dice"]))
            selected = finite[: min(viz_max_cases, viz_worst_k)]

        wandb_logger = None
        if args.log_wandb_viz:
            wandb_logger = WandbLogger(cfg=cfg, run_dir=run_dir)
        wandb_images = []

        for idx, case in enumerate(selected, start=1):
            image_id = str(case["image_id"])
            image_path = viz_dir / f"{idx:03d}_{image_id}.png"
            save_case_panel(
                output_path=image_path,
                image=case["image"],
                gt_mask=case["hard_mask"],
                pred_mask=case["pred_mask"],
                ignore_mask=case["ignore_mask"],
                image_id=image_id,
                metrics={
                    "macro_dice": f"{float(case['macro_dice']):.4f}",
                    "grade5_dice": f"{float(case['grade5_dice']):.4f}",
                },
            )
            if wandb_logger is not None and wandb_logger.enabled and len(wandb_images) < 8:
                panel = render_case_panel(
                    image=case["image"],
                    gt_mask=case["hard_mask"],
                    pred_mask=case["pred_mask"],
                    ignore_mask=case["ignore_mask"],
                    image_id=image_id,
                    metrics={
                        "macro_dice": f"{float(case['macro_dice']):.4f}",
                        "grade5_dice": f"{float(case['grade5_dice']):.4f}",
                    },
                )
                wb = wandb_logger.make_image(
                    panel,
                    caption=(
                        f"{image_id} | macro_dice={float(case['macro_dice']):.4f} "
                        f"| grade5_dice={float(case['grade5_dice']):.4f}"
                    ),
                )
                if wb is not None:
                    wandb_images.append(wb)
        if args.log_wandb_viz and wandb_logger is not None:
            wandb_logger.log_images("eval/panels", wandb_images, step=0)
            wandb_logger.finish()
        logger.info("Saved %d evaluation visualization panels to %s", len(selected), viz_dir)

    out_path = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else run_dir / "evaluation_2d_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("Saved evaluation summary to %s", out_path)


if __name__ == "__main__":
    main()
