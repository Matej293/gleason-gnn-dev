#!/usr/bin/env python3
"""
Evaluate a Deconver checkpoint on Gleason consensus labels.

Usage:
  PYTHONPATH=. python scripts/evaluate_checkpoint.py \
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

from config import consensus_dataset_kwargs_from_config, load_config  # noqa: E402
from cli_utils import (  # noqa: E402
    require_existing_dir,
    require_existing_file,
    resolve_checkpoint_path,
    validate_non_negative_int,
)
from config_validation import validate_deconver_config  # noqa: E402
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
from model_outputs import extract_logits as _extract_logits  # noqa: E402
from utils import ensure_cuda_binary_compatibility, load_checkpoint  # noqa: E402
from visualization import render_case_panel, save_case_panel  # noqa: E402
from wandb_logger import WandbLogger  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = [
    "macro_dice",
    "macro_f1",
    "micro_f1",
    "cohen_kappa",
    "challenge_score",
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
        description="Evaluate a segmentation checkpoint on the configured Gleason consensus test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("I/O")
    io_group.add_argument(
        "--run",
        required=True,
        type=str,
        help="Run directory containing config.yaml and checkpoints/.",
    )
    io_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path or filename inside run/checkpoints. Default resolves best.pt then latest epoch.",
    )
    io_group.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Evaluation summary output path.",
    )

    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--save-viz",
        action="store_true",
        help="Save per-case prediction panels.",
    )
    viz_group.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Visualization output directory.",
    )
    viz_group.add_argument(
        "--viz-max-cases",
        type=int,
        default=-1,
        help="Max number of panels to save (<=0 means all).",
    )
    viz_group.add_argument(
        "--viz-worst-k",
        type=int,
        default=0,
        help="If >0, save worst-K by post macro Dice (bounded by viz-max-cases).",
    )

    wb_group = parser.add_argument_group("Weights & Biases")
    wb_group.add_argument(
        "--log-wandb-viz",
        action="store_true",
        help="Log evaluation panels to W&B.",
    )
    wb_group.add_argument(
        "--log-wandb-metrics",
        action="store_true",
        help="Log aggregate and per-case evaluation metrics to W&B.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.viz_worst_k = validate_non_negative_int(
        args.viz_worst_k,
        field_name="viz_worst_k",
    )

    run_dir = require_existing_dir(args.run, label="Run directory")
    cfg_path = require_existing_file(run_dir / "config.yaml", label="Run config")

    cfg = load_config(cfg_path)
    validate_deconver_config(cfg, for_eval=True, require_paths=True)

    ckpt_path = resolve_checkpoint_path(run_dir, args.checkpoint, prefer_best=True)

    dataset_kwargs = consensus_dataset_kwargs_from_config(cfg)
    dataset = GleasonConsensusDataset(**dataset_kwargs)

    split_manifest_path = resolve_split_manifest_path(cfg)
    logger.info(
        "Evaluation setup | run=%s checkpoint=%s split_manifest=%s output_json=%s",
        run_dir,
        ckpt_path,
        split_manifest_path,
        args.output_json if args.output_json is not None else (run_dir / "evaluation_summary.json"),
    )
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
            logits = _extract_logits(out)
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

    wandb_logger = None
    if args.log_wandb_viz or args.log_wandb_metrics:
        wandb_logger = WandbLogger(cfg=cfg, run_dir=run_dir)

    if args.log_wandb_metrics and wandb_logger is not None and wandb_logger.enabled:
        metrics_payload: dict[str, float] = {}
        for key, value in aggregate_raw.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics_payload[f"eval/raw/{key}"] = float(value)
        for key, value in aggregate_post.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics_payload[f"eval/post/{key}"] = float(value)
        if metrics_payload:
            wandb_logger.log_dict(metrics_payload, step=0)

        raw_rows: list[dict[str, float | str]] = []
        for row in per_case:
            image_id = str(row.get("image_id", ""))
            for key, value in row.items():
                if key == "image_id":
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    raw_rows.append(
                        {
                            "image_id": image_id,
                            "metric": str(key),
                            "value": float(value),
                        }
                    )
        if raw_rows:
            wb_table = wandb_logger.make_table(raw_rows)
            if wb_table is not None:
                wandb_logger.log_dict({"eval/per_case_metrics": wb_table}, step=0)

    if args.save_viz:
        viz_dir = (
            Path(args.viz_dir).resolve()
            if args.viz_dir is not None
            else (run_dir / "eval_viz")
        )
        viz_max_cases = int(args.viz_max_cases)
        viz_worst_k = max(0, int(args.viz_worst_k))
        selected = list(viz_candidates) if viz_max_cases <= 0 else viz_candidates[:viz_max_cases]
        if viz_worst_k > 0:
            finite = [x for x in viz_candidates if not math.isnan(float(x["macro_dice"]))]
            finite = sorted(finite, key=lambda x: float(x["macro_dice"]))
            limit = viz_worst_k if viz_max_cases <= 0 else min(viz_max_cases, viz_worst_k)
            selected = finite[:limit]

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
            if wandb_logger is not None and wandb_logger.enabled and args.log_wandb_viz:
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
            wandb_logger.log_images("eval/panels/all", wandb_images, step=0)
        logger.info("Saved %d evaluation visualization panels to %s", len(selected), viz_dir)

    if wandb_logger is not None and (args.log_wandb_viz or args.log_wandb_metrics):
        wandb_logger.finish()

    out_path = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else run_dir / "evaluation_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("Saved evaluation summary to %s", out_path)


if __name__ == "__main__":
    main()
