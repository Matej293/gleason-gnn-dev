"""
Training for Gleason consensus labels (4-class segmentation).

Usage:
    python -m src.cli.train --config configs/deconver.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import multiprocessing as mp
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from src.common.config import (
    consensus_dataset_kwargs_from_config,
    consensus_train_val_transforms_from_config,
    load_config,
    resolve_inference_mode,
    resolve_resized_sliding_window_overlap,
    resolve_resized_sliding_window_patch_size,
)
from src.common.config_validation import validate_amp_runtime, validate_deconver_config
from src.eval.metric_config import BOUNDARY_METRIC_KEYS, LEGACY_METRIC_TRACK_KEYS, resolve_metric_settings
from src.data.consensus_transforms import set_transform_random_state
from src.common.cli_utils import (
    ensure_output_dir,
    require_existing_file,
    require_non_empty_str,
    validate_experiment_name,
    validate_seed,
)
from src.eval.eval_utils import (
    build_confusion_matrix,
    collate_consensus_batch,
    compute_multiclass_metrics_from_pred,
    postprocess_predictions,
    resolve_split_manifest_path,
    safe_read_json,
)
from src.data.gleason_consensus_dataset import GleasonConsensusDataset
from src.models import build_model
from src.common.utils import (
    create_run_dir,
    ensure_cuda_binary_compatibility,
    load_checkpoint,
    load_pretrained_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    save_config_copy,
    save_latest_pointer,
    save_metadata,
)
from src.viz.visualization import render_case_panel, save_case_panel
from src.common.wandb_logger import WandbLogger
from src.trainers.interfaces import EpochStats, LossOutputs, RunContext, TrainBatch
from src.trainers import data_utils as _data_utils
from src.trainers import loss_utils as _loss_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VAL_METRIC_KEYS = LEGACY_METRIC_TRACK_KEYS



def _fmt(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset

    if not bool(getattr(dataset, "_transform_seed_sync", True)):
        return

    set_transform_random_state(getattr(dataset, "transform", None), seed=int(worker_seed))


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _log_cuda_memory(stage: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    alloc_gib = torch.cuda.memory_allocated(device) / (1024**3)
    reserv_gib = torch.cuda.memory_reserved(device) / (1024**3)
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserv_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
    logger.info(
        "%s | CUDA alloc=%.2f GiB reserved=%.2f GiB peak_alloc=%.2f GiB peak_reserved=%.2f GiB",
        stage,
        alloc_gib,
        reserv_gib,
        peak_alloc_gib,
        peak_reserv_gib,
    )


def _resolve_dataloader_context(
    cfg: dict,
    num_workers: int,
) -> tuple[int, mp.context.BaseContext | None, str]:
    """
    Resolve DataLoader multiprocessing context robustly.

    Python 3.14 changed POSIX default start method to ``forkserver`` which
    requires opening a listener socket. In restricted environments this can
    raise PermissionError during worker startup. We default to ``fork`` here
    and gracefully fall back to single-process loading if unavailable.
    """
    raw = str(cfg.get("dataloader_start_method", "fork")).strip().lower()
    method = raw or "fork"
    if num_workers <= 0:
        return 0, None, method

    if method in {"default", "auto", "none"}:
        return num_workers, None, method

    try:
        ctx = mp.get_context(method)
        return num_workers, ctx, method
    except Exception as exc:
        logger.warning(
            "DataLoader start method '%s' unavailable (%s). Falling back to num_workers=0.",
            method,
            exc,
        )
        return 0, None, method


def _infer_qc_flags(
    qc: dict,
    fail_keys: tuple[str, ...],
    suspicious_keys: tuple[str, ...],
) -> tuple[bool, bool]:
    fail = any(bool(qc.get(k, False)) for k in fail_keys)
    suspicious = any(bool(qc.get(k, False)) for k in suspicious_keys)
    return fail, suspicious


def _infer_qc_weight(
    qc: dict,
    base_downweight: float,
    per_rater_penalty: float,
    min_weight: float,
) -> float:
    weight = 1.0
    excluded = qc.get("excluded_raters", [])
    downweighted = qc.get("downweighted_raters", [])

    n_excluded = len(excluded) if isinstance(excluded, list) else 0
    n_downweighted = len(downweighted) if isinstance(downweighted, list) else 0

    if bool(qc.get("suspicious", False)):
        weight *= base_downweight

    if n_excluded > 0 or n_downweighted > 0:
        penalty_steps = n_excluded + n_downweighted
        weight *= max(0.0, 1.0 - (per_rater_penalty * penalty_steps))

    return float(max(min_weight, min(1.0, weight)))


def _case_flags_from_hard_mask(mask_path: Path) -> tuple[bool, bool, bool, bool]:
    with Image.open(mask_path) as img:
        arr = np.asarray(img, dtype=np.uint8)
    has_cancer = bool((arr > 0).any())
    has_g3 = bool((arr == 1).any())
    has_g4 = bool((arr == 2).any())
    has_grade5 = bool((arr == 3).any())
    return has_cancer, has_g3, has_g4, has_grade5


def _build_sample_metadata(dataset: GleasonConsensusDataset, cfg: dict) -> list[dict]:
    supervised_subdirs = tuple(
        str(s) for s in cfg.get("supervised_image_subdirs", ["Train_imgs"])
    )
    qc_policy = str(cfg.get("qc_policy", "warn_downweight")).strip().lower()
    fail_keys = tuple(
        str(s) for s in cfg.get("qc_fail_keys", ["hard_fail", "failed", "qc_failed"])
    )
    suspicious_keys = tuple(
        str(s) for s in cfg.get("qc_suspicious_keys", ["suspicious", "needs_review"])
    )
    qc_downweight = float(cfg.get("qc_downweight_factor", 0.7))
    qc_per_rater_penalty = float(cfg.get("qc_per_rater_penalty", 0.1))
    qc_min_weight = float(cfg.get("qc_min_weight", 0.2))

    rows: list[dict] = []
    n_out_of_scope = 0
    n_hard_fail = 0
    n_suspicious = 0

    for i, item in enumerate(dataset.items):
        image_subdir = str(item.get("image_subdir", ""))
        if image_subdir not in supervised_subdirs:
            n_out_of_scope += 1
            continue

        has_cancer, has_g3, has_g4, has_grade5 = _case_flags_from_hard_mask(
            Path(item["hard_path"])
        )
        qc = safe_read_json(Path(item["qc_path"]))
        qc_fail, qc_suspicious = _infer_qc_flags(
            qc, fail_keys=fail_keys, suspicious_keys=suspicious_keys
        )

        if qc_fail:
            n_hard_fail += 1
        if qc_suspicious:
            n_suspicious += 1

        qc_weight = 1.0
        if qc_policy in {"warn_downweight", "strict_skip"}:
            qc_weight = _infer_qc_weight(
                qc,
                base_downweight=qc_downweight,
                per_rater_penalty=qc_per_rater_penalty,
                min_weight=qc_min_weight,
            )

        if qc_policy == "strict_skip" and qc_fail:
            continue

        rows.append(
            {
                "dataset_index": i,
                "image_id": str(item["image_id"]),
                "image_subdir": image_subdir,
                "has_cancer": has_cancer,
                "has_g3": has_g3,
                "has_g4": has_g4,
                "has_grade5": has_grade5,
                "qc_fail": qc_fail,
                "qc_suspicious": qc_suspicious,
                "qc_weight": qc_weight,
            }
        )

    logger.info(
        "Consensus supervised pool: %d included | out_of_scope=%d | qc_hard_fail=%d | qc_suspicious=%d | policy=%s",
        len(rows),
        n_out_of_scope,
        n_hard_fail,
        n_suspicious,
        qc_policy,
    )

    if not rows:
        raise RuntimeError(
            "No supervised consensus samples available after QC/scope filtering."
        )

    return rows


def _split_class_presence_summary(rows: list[dict]) -> dict[str, int]:
    return {
        "n_images": int(len(rows)),
        "n_g3_pos_images": int(sum(1 for r in rows if bool(r.get("has_g3", False)))),
        "n_g4_pos_images": int(sum(1 for r in rows if bool(r.get("has_g4", False)))),
        "n_g5_pos_images": int(sum(1 for r in rows if bool(r.get("has_grade5", False)))),
    }


def _log_split_class_presence(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> None:
    train_s = _split_class_presence_summary(train_rows)
    val_s = _split_class_presence_summary(val_rows)
    test_s = _split_class_presence_summary(test_rows)
    logger.info(
        "Split class-presence by image | train=%s | val=%s | test=%s",
        train_s,
        val_s,
        test_s,
    )
    if int(val_s["n_g5_pos_images"]) < 2:
        logger.warning(
            "Validation split has low G5 support: %d G5-positive images (<2).",
            int(val_s["n_g5_pos_images"]),
        )


def _val_min_class_presence_from_cfg(cfg: dict) -> dict[str, int]:
    return {
        "n_g3_pos_images": int(cfg.get("val_min_g3_pos_images", 1)),
        "n_g4_pos_images": int(cfg.get("val_min_g4_pos_images", 1)),
        "n_g5_pos_images": int(cfg.get("val_min_g5_pos_images", 2)),
    }


def _val_presence_shortfalls(val_rows: list[dict], required: dict[str, int]) -> dict[str, tuple[int, int]]:
    summary = _split_class_presence_summary(val_rows)
    out: dict[str, tuple[int, int]] = {}
    for k, req in required.items():
        got = int(summary.get(k, 0))
        if got < int(req):
            out[k] = (got, int(req))
    return out


def _stratify_key(row: dict) -> str:
    return (
        f"cancer={int(bool(row['has_cancer']))}|grade5={int(bool(row['has_grade5']))}"
    )


def _split_two_way_stratified(
    rows: list[dict],
    right_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < right_fraction < 1.0:
        raise ValueError(f"right_fraction must be in (0, 1), got {right_fraction}")

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(_stratify_key(r), []).append(r)

    rng = np.random.default_rng(seed)
    left: list[dict] = []
    right: list[dict] = []

    for key in sorted(grouped):
        grp = grouped[key]
        rng.shuffle(grp)
        n = len(grp)
        if n == 1:
            left.extend(grp)
            continue

        n_right = int(round(n * right_fraction))
        n_right = max(1, min(n - 1, n_right))

        right.extend(grp[:n_right])
        left.extend(grp[n_right:])

    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def _build_split_rows(
    rows: list[dict],
    split_mode: str,
    seed: int,
    required_val_presence: dict[str, int] | None = None,
    max_attempts: int = 1,
) -> tuple[list[dict], list[dict], list[dict]]:
    mode = split_mode.strip().lower()
    if mode not in {"iter_80_20", "final_80_10_10"}:
        raise ValueError(
            "split_mode must be one of {'iter_80_20', 'final_80_10_10'}, "
            f"got {split_mode!r}"
        )

    attempts = max(1, int(max_attempts))
    required = required_val_presence or {}
    last_shortfalls: dict[str, tuple[int, int]] = {}
    for i in range(attempts):
        split_seed = seed + i
        if mode == "iter_80_20":
            train_rows, val_rows = _split_two_way_stratified(
                rows, right_fraction=0.2, seed=split_seed
            )
            test_rows = []
        else:
            train_rows, holdout_rows = _split_two_way_stratified(
                rows, right_fraction=0.2, seed=split_seed
            )
            val_rows, test_rows = _split_two_way_stratified(
                holdout_rows, right_fraction=0.5, seed=split_seed + 1
            )

        if not train_rows or not val_rows:
            raise RuntimeError(
                f"Split produced empty subset(s): train={len(train_rows)} val={len(val_rows)} "
                f"test={len(test_rows)}"
            )

        train_ids = {r["image_id"] for r in train_rows}
        val_ids = {r["image_id"] for r in val_rows}
        test_ids = {r["image_id"] for r in test_rows}

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise RuntimeError("Split overlap detected across train/val/test.")

        shortfalls = _val_presence_shortfalls(val_rows=val_rows, required=required)
        if not shortfalls:
            return train_rows, val_rows, test_rows
        last_shortfalls = shortfalls

    req_desc = ", ".join(
        f"{k}>={v}" for k, v in sorted(required.items())
    ) or "none"
    got_desc = ", ".join(
        f"{k}:{got}/{need}" for k, (got, need) in sorted(last_shortfalls.items())
    ) or "unknown"
    raise RuntimeError(
        "Failed to build split meeting validation class-presence minimums after "
        f"{attempts} attempts (required: {req_desc}; shortfalls: {got_desc})."
    )


def _loo_consensus_mean_from_rows(dataset: GleasonConsensusDataset, rows: list[dict]) -> float:
    vals: list[float] = []
    for r in rows:
        item = dataset.items[int(r["dataset_index"])]
        qc = safe_read_json(Path(item["qc_path"]))
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
        return float("nan")
    return float(np.mean(vals))


def _write_split_manifest(
    path: Path,
    split_mode: str,
    seed: int,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> None:
    def _summ(rows: list[dict]) -> dict[str, int]:
        return {
            "n": len(rows),
            "n_cancer": int(sum(1 for r in rows if r["has_cancer"])),
            "n_grade5": int(sum(1 for r in rows if r["has_grade5"])),
            "n_qc_suspicious": int(sum(1 for r in rows if r["qc_suspicious"])),
            "n_qc_fail": int(sum(1 for r in rows if r["qc_fail"])),
        }

    manifest = {
        "version": 1,
        "split_mode": split_mode,
        "seed": int(seed),
        "summary": {
            "train": _summ(train_rows),
            "val": _summ(val_rows),
            "test": _summ(test_rows),
        },
        "train_image_ids": [str(r["image_id"]) for r in train_rows],
        "val_image_ids": [str(r["image_id"]) for r in val_rows],
        "test_image_ids": [str(r["image_id"]) for r in test_rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _load_hard_case_weight_map(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("hard_case_weights_path does not exist: %s", p)
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read hard_case_weights_path=%s (%s)", p, exc)
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[str(k)] = max(0.0, float(v))
    return out


def _build_train_sampler(
    train_rows: list[dict],
    cfg: dict,
    seed: int,
) -> WeightedRandomSampler | None:
    tumor_factor = float(cfg.get("oversample_tumor_positive_factor", 1.0))
    grade5_factor = float(cfg.get("oversample_grade5_factor", 1.0))
    hard_factor = float(cfg.get("oversample_hard_case_factor", 1.0))
    hard_map = _load_hard_case_weight_map(
        str(cfg.get("hard_case_weights_path", "")).strip() or None
    )
    use_sampler = (
        (tumor_factor != 1.0)
        or (grade5_factor != 1.0)
        or (hard_factor != 1.0 and bool(hard_map))
    )
    if not use_sampler:
        return None

    weights: list[float] = []
    for row in train_rows:
        w = 1.0
        if bool(row.get("has_cancer", False)):
            w *= tumor_factor
        if bool(row.get("has_grade5", False)):
            w *= grade5_factor
        image_id = str(row.get("image_id", ""))
        hard_w = float(hard_map.get(image_id, 1.0))
        w *= (hard_factor * hard_w) if image_id in hard_map else 1.0
        weights.append(max(1e-8, float(w)))

    gen = torch.Generator()
    gen.manual_seed(seed)
    logger.info(
        "Using weighted sampler: tumor_factor=%.3f grade5_factor=%.3f hard_factor=%.3f hard_cases=%d",
        tumor_factor,
        grade5_factor,
        hard_factor,
        len(hard_map),
    )
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=gen,
    )


def _resolve_epoch_lambda_weights(
    cfg: dict,
    epoch: int,
    base_lambda_soft: float,
    base_lambda_dice: float,
) -> tuple[float, float]:
    if not bool(cfg.get("loss_schedule_enabled", False)):
        return base_lambda_soft, base_lambda_dice
    switch_epoch = max(1, int(cfg.get("loss_schedule_transition_epoch", 15)))
    warm_soft = float(cfg.get("lambda_soft_warmup", base_lambda_soft))
    warm_dice = float(cfg.get("lambda_dice_warmup", base_lambda_dice))
    final_soft = float(cfg.get("lambda_soft_final", base_lambda_soft))
    final_dice = float(cfg.get("lambda_dice_final", base_lambda_dice))
    if epoch <= switch_epoch:
        return warm_soft, warm_dice
    return final_soft, final_dice


def _resize_targets_for_logits(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial = logits.shape[2:]
    if hard_mask.shape[1:] == spatial:
        return hard_mask, soft_probs, ignore_mask

    hard_rs = (
        F.interpolate(
            hard_mask.unsqueeze(1).float(),
            size=spatial,
            mode="nearest",
        )
        .squeeze(1)
        .long()
    )
    ignore_rs = (
        F.interpolate(
            ignore_mask.unsqueeze(1).float(),
            size=spatial,
            mode="nearest",
        )
        .squeeze(1)
        .to(ignore_mask.dtype)
    )

    soft_rs = F.interpolate(
        soft_probs.float(),
        size=spatial,
        mode="bilinear",
        align_corners=False,
    )
    soft_rs = torch.nan_to_num(soft_rs, nan=0.0, posinf=1.0, neginf=0.0)
    soft_rs = torch.clamp(soft_rs, min=0.0)
    soft_sum = torch.clamp(soft_rs.sum(dim=1, keepdim=True), min=1e-8)
    soft_rs = soft_rs / soft_sum
    return hard_rs, soft_rs, ignore_rs


def _make_valid_mask(
    ignore_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
) -> torch.Tensor:
    valid = ignore_mask == 0
    if use_confidence_mask:
        conf = soft_probs.max(dim=1).values
        valid = valid & (conf >= confidence_threshold)
    return valid


def _soft_loss_map(
    logits: torch.Tensor,
    soft_probs: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=1)
    if loss_type == "kl":
        return F.kl_div(log_p, soft_probs.float(), reduction="none").sum(dim=1)
    # cross-entropy with soft targets
    return -(soft_probs.float() * log_p).sum(dim=1)


def _hard_dice_per_class(
    probs: torch.Tensor,
    hard_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    target = F.one_hot(
        hard_mask.long().clamp(0, num_classes - 1), num_classes=num_classes
    )
    target = target.permute(0, 3, 1, 2).float()
    valid = valid_mask.unsqueeze(1).float()

    p = probs * valid
    t = target * valid

    intersection = (p * t).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice


def _hard_dice_valid_class_mask(
    hard_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    target = F.one_hot(
        hard_mask.long().clamp(0, num_classes - 1), num_classes=num_classes
    ).permute(0, 3, 1, 2).float()
    valid = valid_mask.unsqueeze(1).float()
    per_class_target = (target * valid).sum(dim=(0, 2, 3))
    return per_class_target > 0.0


def _nanmean_tensor(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    vals = values[valid_mask]
    if vals.numel() == 0:
        return values.new_tensor(0.0)
    return vals.mean()


def _build_scale_loss_context(
    *,
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    require_finite_logits: bool = False,
    check_resized_finite: bool = False,
    include_probs: bool = True,
    include_hard_terms: bool = True,
) -> dict[str, torch.Tensor]:
    if require_finite_logits and not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits passed to loss.")

    hard_rs, soft_rs, ignore_rs = _resize_targets_for_logits(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
    )
    if check_resized_finite:
        if not torch.isfinite(soft_rs).all():
            raise FloatingPointError("Non-finite soft targets after resize.")
        if not torch.isfinite(hard_rs.float()).all():
            raise FloatingPointError("Non-finite hard targets after resize.")
        if not torch.isfinite(ignore_rs.float()).all():
            raise FloatingPointError("Non-finite ignore mask after resize.")

    valid_mask = _make_valid_mask(
        ignore_mask=ignore_rs,
        soft_probs=soft_rs,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
    )
    pixel_weight = sample_weights.view(-1, 1, 1)
    class_weights_4d = class_weights.view(1, -1, 1, 1)
    expected_cls_weight = (soft_rs * class_weights_4d).sum(dim=1)

    ctx: dict[str, torch.Tensor] = {
        "hard_rs": hard_rs,
        "soft_rs": soft_rs,
        "ignore_rs": ignore_rs,
        "valid_mask": valid_mask,
        "valid_float": valid_mask.float(),
        "pixel_weight": pixel_weight,
        "class_weights_4d": class_weights_4d,
        "expected_cls_weight": expected_cls_weight,
    }

    probs: torch.Tensor | None = None
    if include_probs or include_hard_terms:
        probs = F.softmax(logits.float(), dim=1)
        ctx["probs"] = probs

    if include_hard_terms:
        if probs is None:
            probs = F.softmax(logits.float(), dim=1)
            ctx["probs"] = probs
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(
            hard_rs.long().clamp(0, num_classes - 1), num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        valid = valid_mask.unsqueeze(1).float()
        probs_valid = probs * valid
        target_valid = target_one_hot * valid
        per_class_target = target_valid.sum(dim=(0, 2, 3))
        dice_intersection = (probs_valid * target_valid).sum(dim=(0, 2, 3))
        dice_denom = probs_valid.sum(dim=(0, 2, 3)) + per_class_target
        dice_per_class = (2.0 * dice_intersection + 1e-5) / (dice_denom + 1e-5)

        ctx["target_one_hot"] = target_one_hot
        ctx["probs_valid"] = probs_valid
        ctx["target_valid"] = target_valid
        ctx["dice_per_class"] = dice_per_class
        ctx["dice_valid_mask"] = per_class_target > 0.0
        ctx["hard_cls_weight"] = class_weights[hard_rs.long()].float()

    return ctx


def _single_scale_loss_from_context(
    *,
    logits: torch.Tensor,
    ctx: dict[str, torch.Tensor],
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    soft_map = _soft_loss_map(logits, ctx["soft_rs"], loss_type=soft_loss_type)
    soft_map = soft_map * ctx["expected_cls_weight"]

    soft_num = (soft_map * ctx["valid_float"] * ctx["pixel_weight"]).sum()
    soft_den = (ctx["valid_float"] * ctx["pixel_weight"]).sum().clamp_min(1e-8)
    soft_loss = soft_num / soft_den

    if loss_variant == "focal_dice":
        ce = F.cross_entropy(logits.float(), ctx["hard_rs"].long(), reduction="none")
        pt = (ctx["probs"] * ctx["target_one_hot"]).sum(dim=1).clamp(1e-6, 1.0)
        focal_gamma = 2.0
        focal_map = ((1.0 - pt) ** focal_gamma) * ce
        focal_num = (
            focal_map * ctx["hard_cls_weight"] * ctx["valid_float"] * ctx["pixel_weight"]
        ).sum()
        focal_den = (
            ctx["hard_cls_weight"] * ctx["valid_float"] * ctx["pixel_weight"]
        ).sum().clamp_min(1e-8)
        soft_loss = focal_num / focal_den

    dice_c = ctx["dice_per_class"]
    dice_valid_mask = ctx["dice_valid_mask"]
    if include_background_in_dice:
        dice_used = dice_c
        dice_valid_used = dice_valid_mask
    else:
        dice_used = dice_c[1:]
        dice_valid_used = dice_valid_mask[1:]

    if loss_variant == "tversky_dice":
        fp = (ctx["probs_valid"] * (1.0 - ctx["target_valid"])).sum(dim=(0, 2, 3))
        fn = ((1.0 - ctx["probs_valid"]) * ctx["target_valid"]).sum(dim=(0, 2, 3))
        tp = (ctx["probs_valid"] * ctx["target_valid"]).sum(dim=(0, 2, 3))
        alpha = 0.3
        beta = 0.7
        tversky = (tp + 1e-5) / (tp + (alpha * fp) + (beta * fn) + 1e-5)
        tversky_used = tversky if include_background_in_dice else tversky[1:]
        if exclude_absent_classes_in_dice_loss:
            hard_dice_loss = 1.0 - _nanmean_tensor(tversky_used, dice_valid_used)
        else:
            hard_dice_loss = 1.0 - tversky_used.mean()
    else:
        if exclude_absent_classes_in_dice_loss:
            hard_dice_loss = 1.0 - _nanmean_tensor(dice_used, dice_valid_used)
        else:
            hard_dice_loss = 1.0 - dice_used.mean()

    total = (lambda_soft * soft_loss) + (lambda_dice * hard_dice_loss)

    with torch.no_grad():
        stats = {
            "soft_loss": float(soft_loss.detach().cpu().item()),
            "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
            "valid_fraction": float(ctx["valid_float"].mean().detach().cpu().item()),
        }
    return total, stats


def _single_scale_loss(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    ctx = _build_scale_loss_context(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=sample_weights,
        class_weights=class_weights,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        require_finite_logits=True,
        check_resized_finite=True,
        include_probs=True,
        include_hard_terms=True,
    )
    return _single_scale_loss_from_context(
        logits=logits,
        ctx=ctx,
        soft_loss_type=soft_loss_type,
        loss_variant=loss_variant,
        lambda_soft=lambda_soft,
        lambda_dice=lambda_dice,
        include_background_in_dice=include_background_in_dice,
        exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
    )


def _consensus_loss(
    outputs: list[torch.Tensor] | torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(outputs, torch.Tensor):
        return _single_scale_loss(
            logits=outputs,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )

    raw = [1.0 / (2**i) for i in range(len(outputs))]
    total_w = sum(raw)
    weights = [w / total_w for w in raw]

    total_loss = torch.zeros((), device=outputs[0].device, dtype=torch.float32)
    soft_acc = 0.0
    dice_acc = 0.0
    valid_acc = 0.0

    for out, w in zip(outputs, weights):
        l, stats = _single_scale_loss(
            logits=out,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        total_loss = total_loss + (w * l)
        soft_acc += w * stats["soft_loss"]
        dice_acc += w * stats["hard_dice_loss"]
        valid_acc += w * stats["valid_fraction"]

    return total_loss, {
        "soft_loss": soft_acc,
        "hard_dice_loss": dice_acc,
        "valid_fraction": valid_acc,
    }


def _gleason_ce_loss_from_context(
    *,
    logits: torch.Tensor,
    ctx: dict[str, torch.Tensor],
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = ctx["hard_rs"].clone()
    target[ctx["ignore_rs"] != 0] = 255
    loss = F.cross_entropy(logits, target, ignore_index=255)

    dice_c = ctx["dice_per_class"]
    dice_valid_mask = ctx["dice_valid_mask"]
    if include_background_in_dice:
        dice_used = dice_c
        dice_valid_used = dice_valid_mask
    else:
        dice_used = dice_c[1:]
        dice_valid_used = dice_valid_mask[1:]
    if exclude_absent_classes_in_dice_loss:
        hard_dice_loss = 1.0 - _nanmean_tensor(dice_used, dice_valid_used)
    else:
        hard_dice_loss = 1.0 - dice_used.mean()

    valid_fraction = float((target != 255).float().mean().detach().cpu().item())
    stats = {
        "soft_loss": float(loss.detach().cpu().item()),
        "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
        "valid_fraction": valid_fraction,
    }
    return loss, stats


def _gleason_ce_loss(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    ctx = _build_scale_loss_context(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=torch.ones((logits.shape[0],), device=logits.device, dtype=torch.float32),
        class_weights=torch.ones((logits.shape[1],), device=logits.device, dtype=torch.float32),
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        include_probs=True,
        include_hard_terms=True,
    )
    return _gleason_ce_loss_from_context(
        logits=logits,
        ctx=ctx,
        include_background_in_dice=include_background_in_dice,
        exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
    )


def _soft_target_term_loss_from_context(
    *,
    logits: torch.Tensor,
    ctx: dict[str, torch.Tensor],
    soft_loss_type: str,
) -> torch.Tensor:
    soft_map = _soft_loss_map(logits, ctx["soft_rs"], loss_type=soft_loss_type)
    soft_map = soft_map * ctx["expected_cls_weight"]
    soft_num = (soft_map * ctx["valid_float"] * ctx["pixel_weight"]).sum()
    soft_den = (ctx["valid_float"] * ctx["pixel_weight"]).sum().clamp_min(1e-8)
    return soft_num / soft_den


def _soft_target_term_loss(
    logits: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
) -> torch.Tensor:
    ctx = _build_scale_loss_context(
        logits=logits,
        hard_mask=ignore_mask.new_zeros(ignore_mask.shape),
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=sample_weights,
        class_weights=class_weights,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        include_probs=False,
        include_hard_terms=False,
    )
    return _soft_target_term_loss_from_context(
        logits=logits,
        ctx=ctx,
        soft_loss_type=soft_loss_type,
    )


def _compute_training_loss(
    *,
    logits: torch.Tensor,
    scales: list[torch.Tensor],
    aux: torch.Tensor | None,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
    model_name: str,
    pspnet_loss_mode: str,
    pspnet_aux_weight: float,
    pspnet_soft_weight: float,
    pspnet_soft_term: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    def _build_ctx_for_scale(
        scale_logits: torch.Tensor,
        *,
        include_hard_terms: bool = True,
        require_finite: bool = False,
        check_resized_finite: bool = False,
    ) -> dict[str, torch.Tensor]:
        return _build_scale_loss_context(
            logits=scale_logits,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            require_finite_logits=require_finite,
            check_resized_finite=check_resized_finite,
            include_probs=include_hard_terms,
            include_hard_terms=include_hard_terms,
        )

    if model_name == "pspnet" and pspnet_loss_mode == "gleason_ce":
        ctx = _build_ctx_for_scale(logits, include_hard_terms=True)
        loss, stats = _gleason_ce_loss_from_context(
            logits=logits,
            ctx=ctx,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        if aux is not None:
            aux_ctx = _build_ctx_for_scale(aux, include_hard_terms=True)
            aux_loss, _ = _gleason_ce_loss_from_context(
                logits=aux,
                ctx=aux_ctx,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            loss = loss + (float(pspnet_aux_weight) * aux_loss)
        return loss, stats

    if model_name == "pspnet" and pspnet_loss_mode == "gleason_ce_soft":
        ctx = _build_ctx_for_scale(logits, include_hard_terms=True)
        loss, stats = _gleason_ce_loss_from_context(
            logits=logits,
            ctx=ctx,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        if aux is not None:
            aux_ctx = _build_ctx_for_scale(aux, include_hard_terms=True)
            aux_loss, _ = _gleason_ce_loss_from_context(
                logits=aux,
                ctx=aux_ctx,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            loss = loss + (float(pspnet_aux_weight) * aux_loss)

        soft_ctx = _build_ctx_for_scale(logits, include_hard_terms=False)
        soft_term = _soft_target_term_loss_from_context(
            logits=logits,
            ctx=soft_ctx,
            soft_loss_type=pspnet_soft_term,
        )
        loss = loss + (float(pspnet_soft_weight) * soft_term)
        stats["soft_loss"] = float(soft_term.detach().cpu().item())
        return loss, stats

    if len(scales) > 1:
        raw = [1.0 / (2**i) for i in range(len(scales))]
        total_w = sum(raw)
        weights = [w / total_w for w in raw]

        total_loss = torch.zeros((), device=scales[0].device, dtype=torch.float32)
        soft_acc = 0.0
        dice_acc = 0.0
        valid_acc = 0.0
        for scale_logits, w in zip(scales, weights):
            ctx = _build_ctx_for_scale(
                scale_logits,
                include_hard_terms=True,
                require_finite=True,
                check_resized_finite=True,
            )
            scale_loss, scale_stats = _single_scale_loss_from_context(
                logits=scale_logits,
                ctx=ctx,
                soft_loss_type=soft_loss_type,
                loss_variant=loss_variant,
                lambda_soft=lambda_soft,
                lambda_dice=lambda_dice,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            total_loss = total_loss + (w * scale_loss)
            soft_acc += w * scale_stats["soft_loss"]
            dice_acc += w * scale_stats["hard_dice_loss"]
            valid_acc += w * scale_stats["valid_fraction"]
        loss = total_loss
        stats = {
            "soft_loss": soft_acc,
            "hard_dice_loss": dice_acc,
            "valid_fraction": valid_acc,
        }
    else:
        main_ctx = _build_ctx_for_scale(
            logits,
            include_hard_terms=True,
            require_finite=True,
            check_resized_finite=True,
        )
        loss, stats = _single_scale_loss_from_context(
            logits=logits,
            ctx=main_ctx,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )

    if model_name == "pspnet" and aux is not None:
        aux = aux.clamp(-15.0, 15.0)
        aux_ctx = _build_ctx_for_scale(
            aux,
            include_hard_terms=True,
            require_finite=True,
            check_resized_finite=True,
        )
        aux_loss, _ = _single_scale_loss_from_context(
            logits=aux,
            ctx=aux_ctx,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        loss = loss + (float(pspnet_aux_weight) * aux_loss)
    return loss, stats


def _parse_model_outputs(
    out: object,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
    if isinstance(out, torch.Tensor):
        return out, [out], None
    if isinstance(out, dict):
        main = out.get("out")
        if not isinstance(main, torch.Tensor):
            raise TypeError("Model output dict must include Tensor at key 'out'.")
        aux = out.get("aux")
        if aux is not None and not isinstance(aux, torch.Tensor):
            raise TypeError("Model output dict key 'aux' must be a Tensor when present.")
        scales = [main]
        if isinstance(aux, torch.Tensor):
            scales.append(aux)
        return main, scales, aux
    if isinstance(out, (list, tuple)) and out:
        if not all(isinstance(x, torch.Tensor) for x in out):
            raise TypeError("Model output sequences must contain tensors only.")
        main = out[0]
        return main, list(out), None
    raise TypeError(f"Unsupported model output type: {type(out)!r}")


def _forward_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    pad_multiple: int = 32,
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected images shape [B,C,H,W], got {tuple(images.shape)}")

    h, w = int(images.shape[-2]), int(images.shape[-1])
    multiple = max(1, int(pad_multiple))
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h > 0 or pad_w > 0:
        images = F.pad(images, (0, pad_w, 0, pad_h), mode="replicate")

    out = model(images)
    logits, _, _ = _parse_model_outputs(out)

    if pad_h > 0 or pad_w > 0:
        logits = logits[..., :h, :w]
    return logits


def _infer_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    inference_mode: str,
    resized_sliding_window_patch_size: tuple[int, int],
    resized_sliding_window_overlap: float,
) -> torch.Tensor:
    mode = str(inference_mode).strip().lower()
    if mode == "resized_full":
        return _forward_logits(model, images)
    if mode == "resized_sliding_window":
        from monai.inferers import sliding_window_inference

        def _predictor(window: torch.Tensor) -> torch.Tensor:
            return _forward_logits(model, window)

        return sliding_window_inference(
            inputs=images,
            roi_size=resized_sliding_window_patch_size,
            sw_batch_size=1,
            predictor=_predictor,
            overlap=resized_sliding_window_overlap,
        )
    raise ValueError(
        "Unsupported inference_mode: "
        f"{inference_mode!r}. Expected 'resized_full' or 'resized_sliding_window'."
    )


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    image_weight_map: dict[str, float],
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    min_component_size_by_class: dict[int, int],
    inference_mode: str,
    resized_sliding_window_patch_size: tuple[int, int],
    resized_sliding_window_overlap: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    model_name: str,
    pspnet_loss_mode: str,
    pspnet_soft_weight: float,
    pspnet_soft_term: str,
    pspnet_aux_weight: float,
    metric_track_keys: tuple[str, ...],
    include_boundary_metrics: bool,
    boundary_metric_cfg: dict[str, object],
    enable_channels_last: bool = True,
) -> dict[str, float]:
    model.eval()

    metric_keys = tuple(metric_track_keys)
    num_metrics = len(metric_keys)
    sums_raw = np.zeros(num_metrics, dtype=np.float64)
    counts_raw = np.zeros(num_metrics, dtype=np.int64)
    raw_vals = np.empty(num_metrics, dtype=np.float64)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    n_batches = 0

    supports_channels_last = bool(
        enable_channels_last
        and device.type == "cuda"
        and any(isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) for m in model.modules())
    )

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False, unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            if supports_channels_last and images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            soft_probs = batch["soft_probs"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            ignore_mask = batch["ignore_mask"].to(device, non_blocking=True)
            image_ids = [str(x) for x in batch["image_id"]]
            sample_w = torch.tensor(
                [float(image_weight_map.get(i, 1.0)) for i in image_ids],
                device=device,
                dtype=torch.float32,
            )

            with torch.autocast(
                device_type=autocast_device, dtype=amp_dtype, enabled=use_amp
            ):
                logits = _infer_logits(
                    model=model,
                    images=images,
                    inference_mode=inference_mode,
                    resized_sliding_window_patch_size=resized_sliding_window_patch_size,
                    resized_sliding_window_overlap=resized_sliding_window_overlap,
                )
                logits = logits.clamp(-15.0, 15.0)
                scales = [logits]
                aux = None
                try:
                    loss, _ = _compute_training_loss(
                        logits=logits,
                        scales=scales,
                        aux=aux,
                        hard_mask=hard_mask,
                        soft_probs=soft_probs,
                        ignore_mask=ignore_mask,
                        sample_weights=sample_w,
                        class_weights=class_weights,
                        use_confidence_mask=use_confidence_mask,
                        confidence_threshold=confidence_threshold,
                        soft_loss_type=soft_loss_type,
                        loss_variant=loss_variant,
                        lambda_soft=lambda_soft,
                        lambda_dice=lambda_dice,
                        include_background_in_dice=include_background_in_dice,
                        exclude_absent_classes_in_dice_loss=False,
                        model_name=model_name,
                        pspnet_loss_mode=pspnet_loss_mode,
                        pspnet_soft_weight=pspnet_soft_weight,
                        pspnet_soft_term=pspnet_soft_term,
                        pspnet_aux_weight=pspnet_aux_weight,
                    )
                except FloatingPointError:
                    loss = torch.tensor(float("nan"), device=device)

            if not torch.isfinite(loss):
                logger.warning(
                    "Non-finite validation loss under AMP. Retrying batch in FP32."
                )
                with torch.autocast(
                    device_type=autocast_device,
                    enabled=False,
                ):
                    logits = _infer_logits(
                        model=model,
                        images=images.float(),
                        inference_mode=inference_mode,
                        resized_sliding_window_patch_size=resized_sliding_window_patch_size,
                        resized_sliding_window_overlap=resized_sliding_window_overlap,
                    )
                    logits = logits.float().clamp(-15.0, 15.0)
                    scales = [logits]
                    aux = None
                    try:
                        loss, _ = _compute_training_loss(
                            logits=logits,
                            scales=scales,
                            aux=aux,
                            hard_mask=hard_mask,
                            soft_probs=soft_probs,
                            ignore_mask=ignore_mask,
                            sample_weights=sample_w,
                            class_weights=class_weights,
                            use_confidence_mask=use_confidence_mask,
                            confidence_threshold=confidence_threshold,
                            soft_loss_type=soft_loss_type,
                            loss_variant=loss_variant,
                            lambda_soft=lambda_soft,
                            lambda_dice=lambda_dice,
                            include_background_in_dice=include_background_in_dice,
                            exclude_absent_classes_in_dice_loss=False,
                            model_name=model_name,
                            pspnet_loss_mode=pspnet_loss_mode,
                            pspnet_soft_weight=pspnet_soft_weight,
                            pspnet_soft_term=pspnet_soft_term,
                            pspnet_aux_weight=pspnet_aux_weight,
                        )
                    except FloatingPointError:
                        loss = torch.tensor(float("nan"), device=device)

                if not torch.isfinite(loss):
                    logger.warning(
                        "Validation batch remained non-finite in FP32; skipping batch."
                    )
                    continue

            hard_rs, _, ignore_rs = _resize_targets_for_logits(
                logits=logits,
                hard_mask=hard_mask,
                soft_probs=soft_probs,
                ignore_mask=ignore_mask,
            )
            valid = ignore_rs == 0
            valid_n = int(valid.sum().item())
            if valid_n > 0:
                hard_valid_support = torch.bincount(
                    hard_rs[valid].long().clamp(0, 3).reshape(-1),
                    minlength=4,
                )
            else:
                hard_valid_support = torch.zeros((4,), dtype=torch.long, device=hard_rs.device)
            hard_valid_support = hard_valid_support.to(dtype=torch.float64)

            ignored_fraction = float((~valid).float().mean().item())
            tumor_pixels = hard_rs > 0
            tumor_ignored_den = float(tumor_pixels.sum().item())
            tumor_ignored_num = float((tumor_pixels & (~valid)).sum().item())
            tumor_ignored_fraction = (
                (tumor_ignored_num / tumor_ignored_den) if tumor_ignored_den > 0 else float("nan")
            )

            pred_raw = logits.argmax(dim=1)
            raw_conf = build_confusion_matrix(
                pred=pred_raw,
                hard_mask=hard_rs,
                valid=valid,
                num_classes=4,
                valid_n=valid_n,
            )

            m_raw = compute_multiclass_metrics_from_pred(
                pred=pred_raw,
                hard_mask=hard_rs,
                ignore_mask=ignore_rs,
                include_background_in_dice=include_background_in_dice,
                include_boundary_metrics=include_boundary_metrics,
                boundary_metric_cfg=boundary_metric_cfg,
                valid_mask=valid,
                valid_n=valid_n,
                hard_valid_support=hard_valid_support,
                ignored_pixel_fraction=ignored_fraction,
                tumor_pixels_ignored_fraction=tumor_ignored_fraction,
                confusion_matrix=raw_conf,
            )

            for idx, key in enumerate(metric_keys):
                raw_vals[idx] = float(m_raw.get(key, float("nan")))
            raw_ok = ~np.isnan(raw_vals)
            sums_raw[raw_ok] += raw_vals[raw_ok]
            counts_raw[raw_ok] += 1

            loss_sum += loss.detach().to(dtype=torch.float64)
            n_batches += 1

    out_metrics = {"val_loss": float((loss_sum / max(1, n_batches)).item())}
    for idx, key in enumerate(metric_keys):
        out_metrics[f"val_raw/{key}"] = (
            float(sums_raw[idx] / counts_raw[idx]) if counts_raw[idx] > 0 else float("nan")
        )
    return out_metrics


def validate_with_oom_retry(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    image_weight_map: dict[str, float],
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    min_component_size_by_class: dict[int, int],
    inference_mode: str,
    resized_sliding_window_patch_size: tuple[int, int],
    resized_sliding_window_overlap: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    model_name: str,
    pspnet_loss_mode: str,
    pspnet_soft_weight: float,
    pspnet_soft_term: str,
    pspnet_aux_weight: float,
    metric_track_keys: tuple[str, ...],
    include_boundary_metrics: bool,
    boundary_metric_cfg: dict[str, object],
    enable_channels_last: bool = True,
) -> dict[str, float]:
    try:
        return validate(
            model=model,
            loader=loader,
            device=device,
            class_weights=class_weights,
            image_weight_map=image_weight_map,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            min_component_size_by_class=min_component_size_by_class,
            inference_mode=inference_mode,
            resized_sliding_window_patch_size=resized_sliding_window_patch_size,
            resized_sliding_window_overlap=resized_sliding_window_overlap,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            model_name=model_name,
            pspnet_loss_mode=pspnet_loss_mode,
            pspnet_soft_weight=pspnet_soft_weight,
            pspnet_soft_term=pspnet_soft_term,
            pspnet_aux_weight=pspnet_aux_weight,
            metric_track_keys=metric_track_keys,
            include_boundary_metrics=include_boundary_metrics,
            boundary_metric_cfg=boundary_metric_cfg,
            enable_channels_last=enable_channels_last,
        )
    except RuntimeError as exc:
        if device.type == "cuda" and _is_cuda_oom(exc):
            logger.warning(
                "CUDA OOM during validation. Clearing cache and retrying once."
            )
            gc.collect()
            torch.cuda.empty_cache()
            return validate(
                model=model,
                loader=loader,
                device=device,
                class_weights=class_weights,
                image_weight_map=image_weight_map,
                use_confidence_mask=use_confidence_mask,
                confidence_threshold=confidence_threshold,
                soft_loss_type=soft_loss_type,
                loss_variant=loss_variant,
                lambda_soft=lambda_soft,
                lambda_dice=lambda_dice,
                include_background_in_dice=include_background_in_dice,
                min_component_size_by_class=min_component_size_by_class,
                inference_mode=inference_mode,
                resized_sliding_window_patch_size=resized_sliding_window_patch_size,
                resized_sliding_window_overlap=resized_sliding_window_overlap,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                model_name=model_name,
                pspnet_loss_mode=pspnet_loss_mode,
                pspnet_soft_weight=pspnet_soft_weight,
                pspnet_soft_term=pspnet_soft_term,
                pspnet_aux_weight=pspnet_aux_weight,
                metric_track_keys=metric_track_keys,
                include_boundary_metrics=include_boundary_metrics,
                boundary_metric_cfg=boundary_metric_cfg,
                enable_channels_last=enable_channels_last,
            )
        raise


def _pick_fixed_val_viz_ids(
    val_rows: list[dict],
    seed: int,
    num_cases: int,
) -> list[str]:
    ids = sorted({str(r["image_id"]) for r in val_rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[: max(0, int(num_cases))]


def _post_min_component_sizes_from_cfg(cfg: dict) -> dict[int, int]:
    return {
        1: int(cfg.get("post_min_component_size_g3", 0)),
        2: int(cfg.get("post_min_component_size_g4", 0)),
        3: int(cfg.get("post_min_component_size_g5", 0)),
    }


def _resolve_training_best_checkpoint_source(best_checkpoint_source: str) -> str:
    source = str(best_checkpoint_source).strip().lower()
    if source != "raw":
        raise ValueError(
            "Training epoch validation is raw-only. Set metrics.best_checkpoint_source='raw'."
        )
    return source


def _run_validation_visualizations(

    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    selected_ids: set[str],
    run_dir: Path,
    output_subdir: str,
    epoch: int,
    include_background_in_dice: bool,
    min_component_size_by_class: dict[int, int],
    inference_mode: str,
    resized_sliding_window_patch_size: tuple[int, int],
    resized_sliding_window_overlap: float,
    prediction_source: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    wandb_logger: WandbLogger,
    wandb_enabled: bool,
) -> int:
    if not selected_ids:
        return 0

    output_dir = run_dir / output_subdir / f"epoch_{epoch:04d}"
    saved = 0
    wandb_images = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image_ids = [str(x) for x in batch["image_id"]]
            keep_idx = [i for i, image_id in enumerate(image_ids) if image_id in selected_ids]
            if not keep_idx:
                continue
            images = batch["image"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            ignore_mask = batch["ignore_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                logits = _infer_logits(
                    model=model,
                    images=images,
                    inference_mode=inference_mode,
                    resized_sliding_window_patch_size=resized_sliding_window_patch_size,
                    resized_sliding_window_overlap=resized_sliding_window_overlap,
                )
            logits = logits.clamp(-15.0, 15.0)
            pred_raw = logits.argmax(dim=1)
            hard_rs, _, ignore_rs = _resize_targets_for_logits(
                logits=logits,
                hard_mask=hard_mask,
                soft_probs=batch["soft_probs"].to(device, non_blocking=True),
                ignore_mask=ignore_mask,
            )
            tissue_rs = None
            if "tissue_mask" in batch:
                tissue_mask = batch["tissue_mask"].to(device, non_blocking=True)
                tissue_rs = F.interpolate(
                    tissue_mask.unsqueeze(1).float(),
                    size=(logits.shape[-2], logits.shape[-1]),
                    mode="nearest",
                ).squeeze(1).to(dtype=tissue_mask.dtype)
            pred_post = postprocess_predictions(
                pred=pred_raw,
                ignore_mask=ignore_rs,
                tissue_mask=tissue_rs,
                min_component_size_by_class=min_component_size_by_class,
            )
            for i in keep_idx:
                image_id = image_ids[i]
                streams = [("post", pred_post)] if prediction_source == "post" else [("raw", pred_raw)]
                if prediction_source == "both":
                    streams = [("raw", pred_raw), ("post", pred_post)]

                for suffix, pred_stream in streams:
                    sample_metrics = compute_multiclass_metrics_from_pred(
                        pred=pred_stream[i : i + 1],
                        hard_mask=hard_rs[i : i + 1],
                        ignore_mask=ignore_rs[i : i + 1],
                        include_background_in_dice=include_background_in_dice,
                        include_boundary_metrics=False,
                        boundary_metric_cfg=None,
                    )
                    save_path = output_dir / f"{saved + 1:03d}_{image_id}_{suffix}.png"
                    save_case_panel(
                        output_path=save_path,
                        image=images[i].detach().cpu(),
                        gt_mask=hard_rs[i].detach().cpu(),
                        pred_mask=pred_stream[i].detach().cpu(),
                        ignore_mask=ignore_rs[i].detach().cpu(),
                        image_id=f"{image_id} [{suffix}]",
                        metrics={
                            "macro_dice": f"{sample_metrics['macro_dice']:.4f}",
                            "grade5_dice": f"{sample_metrics['grade5_dice']:.4f}",
                        },
                    )
                    if wandb_enabled and len(wandb_images) < 8:
                        panel = render_case_panel(
                            image=images[i].detach().cpu(),
                            gt_mask=hard_rs[i].detach().cpu(),
                            pred_mask=pred_stream[i].detach().cpu(),
                            ignore_mask=ignore_rs[i].detach().cpu(),
                            image_id=f"{image_id} [{suffix}]",
                            metrics={
                                "macro_dice": f"{sample_metrics['macro_dice']:.4f}",
                                "grade5_dice": f"{sample_metrics['grade5_dice']:.4f}",
                            },
                        )
                        wb = wandb_logger.make_image(
                            panel,
                            caption=(
                                f"{image_id} [{suffix}] | epoch={epoch} | "
                                f"macro_dice={sample_metrics['macro_dice']:.4f} | "
                                f"grade5_dice={sample_metrics['grade5_dice']:.4f}"
                            ),
                        )
                        if wb is not None:
                            wandb_images.append(wb)
                    saved += 1
    if wandb_enabled and wandb_images:
        wandb_logger.log_images("val/panels", wandb_images, step=epoch)
    return saved


# Internal refactor compatibility shims: keep public helper names stable while
# routing logic through extracted modules.
_seed_worker = _data_utils.seed_worker
_resolve_dataloader_context = _data_utils.resolve_dataloader_context
_infer_qc_flags = _data_utils.infer_qc_flags
_infer_qc_weight = _data_utils.infer_qc_weight
_case_flags_from_hard_mask = _data_utils.case_flags_from_hard_mask
_build_sample_metadata = _data_utils.build_sample_metadata
_split_class_presence_summary = _data_utils.split_class_presence_summary
_log_split_class_presence = _data_utils.log_split_class_presence
_val_min_class_presence_from_cfg = _data_utils.val_min_class_presence_from_cfg
_val_presence_shortfalls = _data_utils.val_presence_shortfalls
_split_two_way_stratified = _data_utils.split_two_way_stratified
_build_split_rows = _data_utils.build_split_rows
_loo_consensus_mean_from_rows = _data_utils.loo_consensus_mean_from_rows
_write_split_manifest = _data_utils.write_split_manifest
_load_hard_case_weight_map = _data_utils.load_hard_case_weight_map
_build_train_sampler = _data_utils.build_train_sampler
_pick_fixed_val_viz_ids = _data_utils.pick_fixed_val_viz_ids
_post_min_component_sizes_from_cfg = _data_utils.post_min_component_sizes_from_cfg
_resolve_training_best_checkpoint_source = _data_utils.resolve_training_best_checkpoint_source

_resolve_epoch_lambda_weights = _loss_utils.resolve_epoch_lambda_weights
_resize_targets_for_logits = _loss_utils.resize_targets_for_logits
_make_valid_mask = _loss_utils.make_valid_mask
_soft_loss_map = _loss_utils.soft_loss_map
_hard_dice_per_class = _loss_utils.hard_dice_per_class
_hard_dice_valid_class_mask = _loss_utils.hard_dice_valid_class_mask
_nanmean_tensor = _loss_utils.nanmean_tensor
_build_scale_loss_context = _loss_utils.build_scale_loss_context
_single_scale_loss_from_context = _loss_utils.single_scale_loss_from_context
_single_scale_loss = _loss_utils.single_scale_loss
_consensus_loss = _loss_utils.consensus_loss
_gleason_ce_loss_from_context = _loss_utils.gleason_ce_loss_from_context
_gleason_ce_loss = _loss_utils.gleason_ce_loss
_soft_target_term_loss_from_context = _loss_utils.soft_target_term_loss_from_context
_soft_target_term_loss = _loss_utils.soft_target_term_loss
_compute_training_loss = _loss_utils.compute_training_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train segmentation model on Gleason consensus labels.",
    )
    io_group = parser.add_argument_group("I/O")
    io_group.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    io_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Checkpoint path to resume from (CLI takes precedence over config resume_checkpoint).",
    )
    io_group.add_argument(
        "--pretrained",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Checkpoint path for warm-start model weights (CLI takes precedence over config pretrained_checkpoint). Ignored when resume is used.",
    )

    split_group = parser.add_argument_group("Split Control")
    split_group.add_argument(
        "--new-split-manifest",
        action="store_true",
        help="Regenerate train/val/test split manifest before this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = require_existing_file(args.config, label="Config file")
    cfg = load_config(config_path)
    validate_deconver_config(cfg, for_eval=False, require_paths=True)

    cfg["experiment_name"] = validate_experiment_name(cfg.get("experiment_name", ""))
    cfg["base_output_dir"] = require_non_empty_str(
        cfg.get("base_output_dir", ""),
        field_name="base_output_dir",
    )
    cfg["random_seed"] = validate_seed(cfg.get("random_seed", 42), field_name="random_seed")

    resume_from_cli: str | None = None
    if args.resume is not None:
        resume_from_cli = str(
            require_existing_file(args.resume, label="Resume checkpoint")
        )

    pretrained_from_cli: str | None = None
    if args.pretrained is not None:
        pretrained_from_cli = str(
            require_existing_file(args.pretrained, label="Pretrained checkpoint")
        )

    cfg_model = str(cfg.get("model", "deconver")).lower()
    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(
            f"train requires spatial_dims=2, got {spatial_dims}"
        )

    # Requested class mapping: 0=benign, 1=G3, 2=G4, 3=G5.
    out_channels = int(cfg.get("out_channels", 4))
    if out_channels != 4:
        raise ValueError(
            f"This consensus trainer requires out_channels=4, got {out_channels}."
        )

    seed = int(cfg["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(cfg.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda"
        else "cpu"
    )
    if cfg.get("device", "cuda") == "cuda" and device.type != "cuda":
        logger.warning("CUDA requested in config but unavailable; falling back to CPU.")
    ensure_cuda_binary_compatibility(device)

    base_output_dir = ensure_output_dir(
        cfg["base_output_dir"],
        label="base_output_dir",
    )
    logger.info(
        "Starting training | config=%s model=%s experiment=%s seed=%d split_mode=%s output_root=%s",
        config_path,
        cfg_model,
        cfg["experiment_name"],
        seed,
        str(cfg.get("split_mode", "iter_80_20")),
        base_output_dir,
    )

    run_dir = create_run_dir(str(base_output_dir), str(cfg["experiment_name"]))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    run_context = RunContext(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        split_manifest_copy_path=run_dir / "train_val_split_manifest.json",
        summary_path=run_dir / "training_summary.json",
    )

    save_metadata(run_dir, cfg)
    save_config_copy(run_dir, cfg)
    save_latest_pointer(str(base_output_dir), run_dir)

    train_transform, val_transform = consensus_train_val_transforms_from_config(cfg)

    seed_sync = bool(cfg.get("transforms_seed_sync", True))
    if seed_sync:
        set_transform_random_state(train_transform, seed=seed)
        set_transform_random_state(val_transform, seed=seed)

    resize_short_side = int(cfg.get("resize_short_side", 1024))
    inference_resize_short_side = int(cfg.get("inference_resize_short_side", 1024))
    train_crop_enabled = bool(cfg.get("train_crop_enabled", True))
    train_crop_size = tuple(int(x) for x in cfg.get("train_crop_size", [800, 800]))
    inference_mode = resolve_inference_mode(cfg)
    resized_sliding_window_patch_size = resolve_resized_sliding_window_patch_size(cfg)
    resized_sliding_window_overlap = resolve_resized_sliding_window_overlap(cfg)

    logger.info(
        "Resized pipeline | train_resize_short=%d train_crop_enabled=%s train_crop_size=(%d,%d) "
        "infer_resize_short=%d inference_mode=%s sw_patch=(%d,%d) sw_overlap=%.2f seed_sync=%s",
        resize_short_side,
        train_crop_enabled,
        train_crop_size[0],
        train_crop_size[1],
        inference_resize_short_side,
        inference_mode,
        resized_sliding_window_patch_size[0],
        resized_sliding_window_patch_size[1],
        resized_sliding_window_overlap,
        seed_sync,
    )

    split_dataset = GleasonConsensusDataset(
        **consensus_dataset_kwargs_from_config(cfg, transform=None)
    )
    setattr(split_dataset, "_transform_seed_sync", seed_sync)

    all_rows = _build_sample_metadata(dataset=split_dataset, cfg=cfg)
    enforce_val_presence = bool(cfg.get("enforce_val_class_presence", True))
    val_presence_required = _val_min_class_presence_from_cfg(cfg)
    split_search_attempts = max(1, int(cfg.get("split_search_max_attempts", 100)))

    split_mode = str(cfg.get("split_mode", "iter_80_20"))
    split_manifest_path = resolve_split_manifest_path(cfg)

    if args.new_split_manifest or not split_manifest_path.exists():
        train_rows, val_rows, test_rows = _build_split_rows(
            rows=all_rows,
            split_mode=split_mode,
            seed=seed,
            required_val_presence=(val_presence_required if enforce_val_presence else {}),
            max_attempts=split_search_attempts,
        )
        _write_split_manifest(
            path=split_manifest_path,
            split_mode=split_mode,
            seed=seed,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
        )
        logger.info("Wrote split manifest to %s", split_manifest_path)
    else:
        manifest = safe_read_json(split_manifest_path)
        train_ids = set(str(x) for x in manifest.get("train_image_ids", []))
        val_ids = set(str(x) for x in manifest.get("val_image_ids", []))
        test_ids = set(str(x) for x in manifest.get("test_image_ids", []))

        def _pick(ids: set[str]) -> list[dict]:
            return [r for r in all_rows if str(r["image_id"]) in ids]

        train_rows = _pick(train_ids)
        val_rows = _pick(val_ids)
        test_rows = _pick(test_ids)

        if not train_rows or not val_rows:
            raise RuntimeError(
                "Existing split manifest does not match discovered samples. "
                "Pass --new-split-manifest to regenerate."
            )
        if enforce_val_presence:
            shortfalls = _val_presence_shortfalls(
                val_rows=val_rows,
                required=val_presence_required,
            )
            if shortfalls:
                detail = ", ".join(
                    f"{k}:{got}/{need}" for k, (got, need) in sorted(shortfalls.items())
                )
                raise RuntimeError(
                    "Existing split manifest failed validation class-presence minimums: "
                    f"{detail}. Pass --new-split-manifest to regenerate."
                )
        logger.info("Using existing split manifest at %s", split_manifest_path)

    split_manifest_copy_path = run_context.split_manifest_copy_path
    shutil.copy2(split_manifest_path, split_manifest_copy_path)

    train_indices = [int(r["dataset_index"]) for r in train_rows]
    val_indices = [int(r["dataset_index"]) for r in val_rows]
    test_indices = [int(r["dataset_index"]) for r in test_rows]

    train_dataset = GleasonConsensusDataset(
        **consensus_dataset_kwargs_from_config(cfg, transform=train_transform)
    )
    val_dataset = GleasonConsensusDataset(
        **consensus_dataset_kwargs_from_config(cfg, transform=val_transform)
    )
    test_dataset = val_dataset

    setattr(train_dataset, "_transform_seed_sync", seed_sync)
    setattr(val_dataset, "_transform_seed_sync", seed_sync)

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    test_ds = Subset(test_dataset, test_indices) if test_indices else None

    image_weight_map = {
        str(r["image_id"]): float(r.get("qc_weight", 1.0)) for r in all_rows
    }

    train_sample_rows = list(train_rows)

    num_workers_cfg = int(cfg.get("num_workers", 0))
    num_workers, mp_context, start_method = _resolve_dataloader_context(
        cfg=cfg,
        num_workers=num_workers_cfg,
    )
    use_persistent = num_workers > 0
    if num_workers != num_workers_cfg:
        logger.warning(
            "Reduced num_workers from %d to %d due to DataLoader context constraints.",
            num_workers_cfg,
            num_workers,
        )
    logger.info(
        "DataLoader workers=%d (requested=%d), start_method=%s",
        num_workers,
        num_workers_cfg,
        start_method,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_sampler = _build_train_sampler(train_rows=train_sample_rows, cfg=cfg, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        worker_init_fn=_seed_worker,
        multiprocessing_context=mp_context if use_persistent else None,
        collate_fn=collate_consensus_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_batch_size", 4)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
        multiprocessing_context=mp_context if use_persistent else None,
        collate_fn=collate_consensus_batch,
    )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg.get("val_batch_size", 4)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
            multiprocessing_context=mp_context if use_persistent else None,
            collate_fn=collate_consensus_batch,
        )

    logger.info(
        "Split mode=%s | train_images=%d | train_samples=%d | val_images=%d | test_images=%d",
        split_mode,
        len(train_rows),
        len(train_ds),
        len(val_ds),
        len(test_ds) if test_ds is not None else 0,
    )
    _log_split_class_presence(train_rows=train_rows, val_rows=val_rows, test_rows=test_rows)
    if bool(cfg.get("eval_leave_one_rater_out", False)):
        loo_train = _loo_consensus_mean_from_rows(split_dataset, train_rows)
        loo_val = _loo_consensus_mean_from_rows(split_dataset, val_rows)
        loo_test = _loo_consensus_mean_from_rows(split_dataset, test_rows) if test_rows else float("nan")
        logger.info(
            "LOO-consensus diagnostics | train=%s val=%s test=%s",
            _fmt(loo_train),
            _fmt(loo_val),
            _fmt(loo_test),
        )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s | trainable parameters: %s", cfg_model, f"{n_params:,}")

    compiled_model: torch.nn.Module = model
    if cfg.get("use_compile", False):
        cc_major, cc_minor = (
            torch.cuda.get_device_capability(device)
            if device.type == "cuda"
            else (0, 0)
        )
        if device.type == "cuda" and (cc_major, cc_minor) < (7, 5):
            logger.warning(
                "Disabling torch.compile: GPU is sm_%d%d (requires sm_75+ for stable Triton support).",
                cc_major,
                cc_minor,
            )
        else:
            logger.info("Compiling model with torch.compile …")
            compiled_model = torch.compile(model)  # type: ignore[assignment]
    model = compiled_model

    lr = float(cfg.get("learning_rate", 2e-4))
    wd = float(cfg.get("weight_decay", 1e-5))
    optimizer_name = str(cfg.get("optimizer", "adamw")).strip().lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}. Use 'adamw' or 'sgd'.")

    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
    epochs = int(cfg.get("epochs", 100))
    lr_schedule = str(cfg.get("lr_schedule", "cosine")).strip().lower()
    scheduler_step_per_batch = False
    if lr_schedule == "poly":
        poly_power = float(cfg.get("poly_power", 0.9))
        total_steps = max(1, len(train_loader) * epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1.0 - min(step, total_steps) / total_steps) ** poly_power,
        )
        scheduler_step_per_batch = True
    else:
        cosine_t_max = max(1, epochs - warmup_epochs)
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_t_max,
                eta_min=lr * 1e-2,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=lr * 1e-2,
            )

    class_weights_cfg = cfg.get("class_loss_weights", cfg.get("class_weights", None))
    if class_weights_cfg is not None:
        if not isinstance(class_weights_cfg, list) or len(class_weights_cfg) != 4:
            raise ValueError("class_weights must be a list of 4 floats [w0,w1,w2,w3].")
        class_weights = torch.tensor(
            [float(x) for x in class_weights_cfg], dtype=torch.float32, device=device
        )
    else:
        grade5_boost = float(cfg.get("grade5_boost", 1.0))
        class_weights = torch.tensor(
            [1.0, 1.0, 1.0, grade5_boost], dtype=torch.float32, device=device
        )

    lambda_soft = float(cfg.get("lambda_soft", 1.0))
    lambda_dice = float(cfg.get("lambda_dice", 1.0))
    soft_loss_type = str(cfg.get("soft_label_loss", "ce")).strip().lower()
    if soft_loss_type not in {"ce", "kl"}:
        raise ValueError(
            f"soft_label_loss must be 'ce' or 'kl', got {soft_loss_type!r}"
        )
    loss_variant = str(cfg.get("loss_variant", "soft_dice")).strip().lower()
    if loss_variant not in {"soft_dice", "focal_dice", "tversky_dice"}:
        raise ValueError(
            "loss_variant must be one of {'soft_dice','focal_dice','tversky_dice'}, "
            f"got {loss_variant!r}"
        )

    use_confidence_mask = bool(cfg.get("use_confidence_mask", False))
    confidence_threshold = float(cfg.get("confidence_threshold", 0.6))
    include_background_in_dice = bool(cfg.get("include_background_in_dice", False))
    exclude_absent_classes_in_dice_loss = bool(
        cfg.get("exclude_absent_classes_in_dice_loss", False)
    )
    post_min_comp = _post_min_component_sizes_from_cfg(cfg)

    amp_dtype_str = str(cfg.get("amp_dtype", "fp16")).lower()
    amp_dtype = validate_amp_runtime(cfg, device)
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    val_enable_channels_last = bool(cfg.get("val_enable_channels_last", True))

    use_fp16 = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # type: ignore[attr-defined]

    best_metric = float("-inf")
    best_val_macro_dice = float("nan")
    start_epoch = 1

    es_patience = int(cfg.get("early_stopping_patience", 30))
    es_min_delta = float(cfg.get("early_stopping_min_delta", 0.0005))
    es_enabled = es_patience > 0
    es_counter = 0

    resume_path = resume_from_cli
    if resume_path is None and cfg.get("resume_checkpoint"):
        resume_path = str(
            require_existing_file(
                cfg["resume_checkpoint"],
                label="Config resume_checkpoint",
            )
        )

    pretrained_path: str | None = None
    if resume_path is None:
        pretrained_candidate = pretrained_from_cli
        if pretrained_candidate is None and cfg.get("pretrained_checkpoint"):
            pretrained_candidate = str(
                require_existing_file(
                    cfg["pretrained_checkpoint"],
                    label="Config pretrained_checkpoint",
                )
            )
        pretrained_path = pretrained_candidate
    elif pretrained_from_cli is not None or cfg.get("pretrained_checkpoint"):
        logger.warning(
            "Ignoring pretrained checkpoint because resume checkpoint is set."
        )

    if pretrained_path:
        pretrain_info = load_pretrained_checkpoint(
            path=pretrained_path,
            model=model,
            device=device,
        )
        skipped_shape_keys = pretrain_info["skipped_shape_mismatch_keys"]
        skipped_unexpected_keys = pretrain_info["skipped_unexpected_keys"]
        logger.info(
            "Warm-start loaded %d/%d tensors from %s (shape_mismatch=%d, unexpected=%d)",
            pretrain_info["loaded_count"],
            pretrain_info["target_param_count"],
            pretrained_path,
            len(skipped_shape_keys),
            len(skipped_unexpected_keys),
        )
        if skipped_shape_keys:
            preview = ", ".join(str(k) for k in skipped_shape_keys[:12])
            logger.info(
                "Warm-start skipped shape-mismatch keys (first %d): %s",
                min(12, len(skipped_shape_keys)),
                preview,
            )

    wandb_logger = WandbLogger(cfg=cfg, run_dir=run_dir, resume_checkpoint=resume_path)
    if resume_path:
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_challenge_score", float("-inf")))
        best_val_d = ckpt.get("best_val_dice", float("nan"))
        best_val_macro_dice = (
            float(best_val_d) if best_val_d is not None else float("nan")
        )
        logger.info(
            "Resuming from epoch %d (best_metric=%s, best_val_macro_dice=%s) -> epoch %d",
            ckpt["epoch"],
            _fmt(best_metric),
            _fmt(best_val_macro_dice),
            start_epoch,
        )

    logger.info("Device: %s", device)
    logger.info("AMP (%s): %s", amp_dtype_str.upper(), use_amp)
    if str(cfg.get("model", "")).strip().lower() == "deconver":
        fp32_islands = bool(cfg.get("deconver_fp32_islands", False))
        fp32_scope = str(cfg.get("deconver_fp32_scope", "update_only")).strip().lower()
        logger.info(
            "Deconver FP32 islands: %s (scope=%s)",
            fp32_islands,
            fp32_scope,
        )
    logger.info("torch.compile: %s", cfg.get("use_compile", False))
    logger.info("Experiment: %s", cfg["experiment_name"])
    logger.info("Run directory: %s", run_dir)
    if cfg_model == "pspnet":
        logger.info(
            "Loss setup (pspnet): mode=%s aux_weight=%.3f soft_term=%s soft_weight=%.3f | confidence_mask=%s(th=%.2f)",
            str(cfg.get("pspnet_loss_mode", "consensus")).strip().lower(),
            float(cfg.get("pspnet_aux_weight", 0.5)),
            str(cfg.get("pspnet_soft_term", "ce")).strip().lower(),
            float(cfg.get("pspnet_soft_weight", 0.2)),
            use_confidence_mask,
            confidence_threshold,
        )
    else:
        logger.info(
            "Loss setup (%s): variant=%s soft=%s (lambda=%.3f), hard_dice(lambda=%.3f), confidence_mask=%s(th=%.2f)",
            cfg_model,
            loss_variant,
            soft_loss_type,
            lambda_soft,
            lambda_dice,
            use_confidence_mask,
            confidence_threshold,
        )
    logger.info(
        "Class weights: %s", [float(x) for x in class_weights.detach().cpu().tolist()]
    )
    logger.info(
        "Optimizer/schedule: %s / %s",
        optimizer_name,
        lr_schedule,
    )
    if cfg_model != "pspnet":
        logger.info(
            "Loss schedule: enabled=%s switch_epoch=%d warmup(soft=%.3f,dice=%.3f) final(soft=%.3f,dice=%.3f)",
            bool(cfg.get("loss_schedule_enabled", False)),
            int(cfg.get("loss_schedule_transition_epoch", 15)),
            float(cfg.get("lambda_soft_warmup", lambda_soft)),
            float(cfg.get("lambda_dice_warmup", lambda_dice)),
            float(cfg.get("lambda_soft_final", lambda_soft)),
            float(cfg.get("lambda_dice_final", lambda_dice)),
        )

    max_nan_batches = 10
    nan_batch_count = 0

    val_every = max(1, int(cfg.get("val_every", 1)))
    val_start_epoch = max(1, int(cfg.get("val_start_epoch", 1)))
    keep_last_n = int(cfg.get("keep_last_checkpoints", 3))

    metric_settings = resolve_metric_settings(cfg)
    val_metric_keys = tuple(
        key for key in metric_settings.track_keys if key not in set(BOUNDARY_METRIC_KEYS)
    )
    include_boundary_metrics = False
    boundary_metric_cfg: dict[str, object] = {
        "hausdorff_variant": metric_settings.boundary.hausdorff_variant,
        "hausdorff_percentile": float(metric_settings.boundary.hausdorff_percentile),
        "include_background": bool(metric_settings.boundary.include_background),
        "symmetric_asd": bool(metric_settings.boundary.symmetric_asd),
    }
    best_ckpt_metric_name = str(metric_settings.best_checkpoint_metric).strip() or "challenge_score"
    if best_ckpt_metric_name in set(BOUNDARY_METRIC_KEYS):
        logger.warning(
            "metrics.best_checkpoint_metric=%s is a boundary metric, but epoch validation boundary metrics are disabled. Falling back to challenge_score.",
            best_ckpt_metric_name,
        )
        best_ckpt_metric_name = "challenge_score"
    if best_ckpt_metric_name != "challenge_score":
        logger.warning(
            "metrics.best_checkpoint_metric=%s requested, but checkpoint selection currently uses challenge_score. Falling back to challenge_score.",
            best_ckpt_metric_name,
        )
        best_ckpt_metric_name = "challenge_score"
    logger.info("Epoch validation boundary metrics: disabled (offline evaluation keeps them enabled).")
    best_ckpt_metric_source = _resolve_training_best_checkpoint_source(
        metric_settings.best_checkpoint_source
    )
    nan_recovery_log_every = max(1, int(cfg.get("nan_recovery_log_every", 1)))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    non_finite_stats = {"soft_loss": float("nan"), "hard_dice_loss": float("nan"), "valid_fraction": float("nan")}
    viz_enabled = bool(cfg.get("viz_enabled", True))
    viz_every_n_epochs = max(1, int(cfg.get("viz_every_n_epochs", 5)))
    viz_num_cases = max(0, int(cfg.get("viz_num_cases", 8)))
    viz_output_subdir = str(cfg.get("viz_output_subdir", "val_viz")).strip() or "val_viz"
    viz_log_wandb = bool(cfg.get("viz_log_wandb", True))
    viz_prediction_source = str(cfg.get("viz_prediction_source", "post")).strip().lower()
    if viz_prediction_source not in {"raw", "post", "both"}:
        raise ValueError(
            f"viz_prediction_source must be one of ['raw', 'post', 'both'], got {viz_prediction_source!r}"
        )
    fixed_val_viz_ids = _pick_fixed_val_viz_ids(
        val_rows=val_rows,
        seed=seed,
        num_cases=viz_num_cases,
    )
    logger.info(
        "Validation viz: enabled=%s every=%d epochs cases=%d wandb=%s pred_source=%s",
        viz_enabled,
        viz_every_n_epochs,
        len(fixed_val_viz_ids),
        viz_log_wandb,
        viz_prediction_source,
    )
    logger.info(
        "Best-checkpoint metric=%s | source=%s",
        best_ckpt_metric_name,
        best_ckpt_metric_source,
    )
    pspnet_aux_weight = float(cfg.get("pspnet_aux_weight", 0.5))
    pspnet_loss_mode = str(cfg.get("pspnet_loss_mode", "consensus")).strip().lower()
    if pspnet_loss_mode not in {"consensus", "gleason_ce", "gleason_ce_soft"}:
        raise ValueError(
            "pspnet_loss_mode must be one of {'consensus','gleason_ce','gleason_ce_soft'}, "
            f"got {pspnet_loss_mode!r}"
        )
    pspnet_soft_term = str(cfg.get("pspnet_soft_term", "ce")).strip().lower()
    if pspnet_soft_term not in {"ce", "kl"}:
        raise ValueError(
            f"pspnet_soft_term must be one of {{'ce','kl'}}, got {pspnet_soft_term!r}"
        )
    pspnet_soft_weight = float(cfg.get("pspnet_soft_weight", 0.2))
    if pspnet_soft_weight < 0.0:
        raise ValueError(f"pspnet_soft_weight must be >= 0, got {pspnet_soft_weight}")
    if cfg_model != "pspnet":
        pspnet_loss_mode = "consensus"

    def _forward_train_batch_loss(
        batch_images: torch.Tensor,
        *,
        batch_hard_mask: torch.Tensor,
        batch_soft_probs: torch.Tensor,
        batch_ignore_mask: torch.Tensor,
        batch_sample_w: torch.Tensor,
        epoch_soft_weight: float,
        epoch_dice_weight: float,
        force_fp32: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        autocast_enabled = use_amp and not force_fp32
        with torch.autocast(
            device_type=autocast_device,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            model_input = batch_images.float() if force_fp32 else batch_images
            out_local = model(model_input)
            logits_local, scales_local, aux_local = _parse_model_outputs(out_local)
            if force_fp32:
                logits_local = logits_local.float().clamp(-15.0, 15.0)
                scales_local = [s.float().clamp(-15.0, 15.0) for s in scales_local]
            else:
                logits_local = logits_local.clamp(-15.0, 15.0)
                scales_local = [s.clamp(-15.0, 15.0) for s in scales_local]
            return _compute_training_loss(
                logits=logits_local,
                scales=scales_local,
                aux=aux_local,
                hard_mask=batch_hard_mask,
                soft_probs=batch_soft_probs,
                ignore_mask=batch_ignore_mask,
                sample_weights=batch_sample_w,
                class_weights=class_weights,
                use_confidence_mask=use_confidence_mask,
                confidence_threshold=confidence_threshold,
                soft_loss_type=soft_loss_type,
                loss_variant=loss_variant,
                lambda_soft=epoch_soft_weight,
                lambda_dice=epoch_dice_weight,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
                model_name=cfg_model,
                pspnet_loss_mode=pspnet_loss_mode,
                pspnet_soft_weight=pspnet_soft_weight,
                pspnet_soft_term=pspnet_soft_term,
                pspnet_aux_weight=pspnet_aux_weight,
            )

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch"):
        epoch_lambda_soft, epoch_lambda_dice = _resolve_epoch_lambda_weights(
            cfg=cfg,
            epoch=epoch,
            base_lambda_soft=lambda_soft,
            base_lambda_dice=lambda_dice,
        )
        model.train()
        epoch_stats = EpochStats()

        batch_bar = tqdm(
            train_loader, desc=f"Train {epoch}/{epochs}", leave=False, unit="batch"
        )
        for step, batch in enumerate(batch_bar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            soft_probs = batch["soft_probs"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            ignore_mask = batch["ignore_mask"].to(device, non_blocking=True)
            image_ids = [str(x) for x in batch["image_id"]]
            sample_w = torch.tensor(
                [float(image_weight_map.get(i, 1.0)) for i in image_ids],
                device=device,
                dtype=torch.float32,
            )
            train_batch = TrainBatch(
                images=images,
                soft_probs=soft_probs,
                hard_mask=hard_mask,
                ignore_mask=ignore_mask,
                image_ids=image_ids,
                sample_weights=sample_w,
            )

            optimizer.zero_grad(set_to_none=True)

            recovered_with_fp32 = False
            try:
                loss, stats = _forward_train_batch_loss(
                    train_batch.images,
                    batch_hard_mask=train_batch.hard_mask,
                    batch_soft_probs=train_batch.soft_probs,
                    batch_ignore_mask=train_batch.ignore_mask,
                    batch_sample_w=train_batch.sample_weights,
                    epoch_soft_weight=epoch_lambda_soft,
                    epoch_dice_weight=epoch_lambda_dice,
                    force_fp32=False,
                )
            except FloatingPointError:
                loss = torch.tensor(float("nan"), device=device)
                stats = non_finite_stats

            if not torch.isfinite(loss):
                loss_value = float(loss.detach().cpu().item())
                if (nan_batch_count % nan_recovery_log_every) == 0:
                    logger.warning(
                        "Non-finite loss at epoch %d, batch %d under AMP: %s. Retrying in FP32.",
                        epoch,
                        step,
                        loss_value,
                    )
                optimizer.zero_grad(set_to_none=True)
                del loss, stats

                try:
                    loss, stats = _forward_train_batch_loss(
                        images,
                        batch_hard_mask=hard_mask,
                        batch_soft_probs=soft_probs,
                        batch_ignore_mask=ignore_mask,
                        batch_sample_w=sample_w,
                        epoch_soft_weight=epoch_lambda_soft,
                        epoch_dice_weight=epoch_lambda_dice,
                        force_fp32=True,
                    )
                except FloatingPointError:
                    loss = torch.tensor(float("nan"), device=device)
                    stats = non_finite_stats

                if not torch.isfinite(loss):
                    nan_batch_count += 1
                    logger.warning(
                        "Non-finite loss persisted in FP32 at epoch %d, batch %d (consecutive=%d/%d). Skipping batch.",
                        epoch,
                        step,
                        nan_batch_count,
                        max_nan_batches,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    del images, soft_probs, hard_mask, ignore_mask, sample_w, loss
                    if nan_batch_count >= max_nan_batches:
                        raise FloatingPointError(
                            f"Non-finite loss for {max_nan_batches} consecutive batches "
                            f"(epoch={epoch}, batch={step})."
                        )
                    continue

                recovered_with_fp32 = True
                if (nan_batch_count % nan_recovery_log_every) == 0:
                    logger.warning(
                        "Recovered non-finite batch using FP32 at epoch %d, batch %d.",
                        epoch,
                        step,
                    )

            nan_batch_count = 0

            if recovered_with_fp32:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer_stepped = True
            else:
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= prev_scale

            if optimizer_stepped:
                epoch_stats.mark_optimizer_step()
            if scheduler_step_per_batch and optimizer_stepped:
                scheduler.step()

            loss_out = LossOutputs(
                loss=loss,
                soft_loss=float(stats["soft_loss"]),
                hard_dice_loss=float(stats["hard_dice_loss"]),
                valid_fraction=float(stats["valid_fraction"]),
            )
            loss_item = float(loss_out.loss.detach().cpu().item())
            epoch_stats.add(
                loss=loss_item,
                soft_loss=loss_out.soft_loss,
                hard_dice_loss=loss_out.hard_dice_loss,
                valid_fraction=loss_out.valid_fraction,
            )
            batch_bar.set_postfix(
                loss=f"{loss_item:.4f}",
                soft=f"{loss_out.soft_loss:.4f}",
                dice=f"{loss_out.hard_dice_loss:.4f}",
            )

            del images, soft_probs, hard_mask, ignore_mask, sample_w, train_batch, loss

        if not scheduler_step_per_batch and epoch_stats.optimizer_steps > 0:
            scheduler.step()

        epoch_avg = epoch_stats.averages(len(train_loader))
        avg_loss = epoch_avg["loss"]
        avg_soft = epoch_avg["soft_loss"]
        avg_hard = epoch_avg["hard_dice_loss"]
        avg_valid = epoch_avg["valid_fraction"]
        current_lr = scheduler.get_last_lr()[0]

        wandb_logger.log_epoch(
            {
                "train/loss": avg_loss,
                "train/soft_loss": avg_soft,
                "train/hard_dice_loss": avg_hard,
                "train/valid_pixel_fraction": avg_valid,
                "train/lr": current_lr,
                "train/lambda_soft": epoch_lambda_soft,
                "train/lambda_dice": epoch_lambda_dice,
            },
            step=epoch,
        )
        logger.info(
            "Epoch %d/%d | loss=%.4f soft=%.4f hard_dice=%.4f valid_px=%.3f lr=%.2e lambda_soft=%.3f lambda_dice=%.3f",
            epoch,
            epochs,
            avg_loss,
            avg_soft,
            avg_hard,
            avg_valid,
            current_lr,
            epoch_lambda_soft,
            epoch_lambda_dice,
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            str(checkpoint_dir / f"epoch_{epoch:04d}.pt"),
            scheduler=scheduler,
            scaler=scaler,
            best_val_dice=best_val_macro_dice,
            best_challenge_score=best_metric,
            last_hd95=float("nan"),
        )
        rotate_checkpoints(checkpoint_dir, keep_last_n)

        if device.type == "cuda":
            _log_cuda_memory(f"Epoch {epoch} after train", device)

        run_val = (epoch >= val_start_epoch) and (
            (epoch % val_every == 0) or (epoch == epochs)
        )
        if not run_val:
            logger.info(
                "Epoch %d/%d | validation skipped (val_every=%d)",
                epoch,
                epochs,
                val_every,
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        val_metrics = validate_with_oom_retry(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            image_weight_map=image_weight_map,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=epoch_lambda_soft,
            lambda_dice=epoch_lambda_dice,
            include_background_in_dice=include_background_in_dice,
            min_component_size_by_class=post_min_comp,
            inference_mode=inference_mode,
            resized_sliding_window_patch_size=resized_sliding_window_patch_size,
            resized_sliding_window_overlap=resized_sliding_window_overlap,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            model_name=cfg_model,
            pspnet_loss_mode=pspnet_loss_mode,
            pspnet_soft_weight=pspnet_soft_weight,
            pspnet_soft_term=pspnet_soft_term,
            pspnet_aux_weight=pspnet_aux_weight,
            metric_track_keys=val_metric_keys,
            include_boundary_metrics=include_boundary_metrics,
            boundary_metric_cfg=boundary_metric_cfg,
            enable_channels_last=val_enable_channels_last,
        )

        val_log: dict[str, float] = {"val/loss": val_metrics["val_loss"]}
        for k in val_metric_keys:
            val_log[f"val_raw/{k}"] = val_metrics[f"val_raw/{k}"]
        wandb_logger.log_epoch(val_log, step=epoch)

        logger.info(
            "Epoch %d/%d | val_loss=%s | raw_macro=%s | raw_sens=%s",
            epoch,
            epochs,
            _fmt(val_metrics["val_loss"]),
            _fmt(val_metrics["val_raw/macro_dice"]),
            _fmt(val_metrics["val_raw/sensitivity"]),
        )

        if device.type == "cuda":
            _log_cuda_memory(f"Epoch {epoch} after val", device)

        do_viz = viz_enabled and (len(fixed_val_viz_ids) > 0) and (epoch % viz_every_n_epochs == 0)
        if do_viz:
            num_saved = _run_validation_visualizations(
                model=model,
                loader=val_loader,
                device=device,
                selected_ids=set(fixed_val_viz_ids),
                run_dir=run_dir,
                output_subdir=viz_output_subdir,
                epoch=epoch,
                include_background_in_dice=include_background_in_dice,
                min_component_size_by_class=post_min_comp,
                inference_mode=inference_mode,
                resized_sliding_window_patch_size=resized_sliding_window_patch_size,
                resized_sliding_window_overlap=resized_sliding_window_overlap,
                prediction_source=viz_prediction_source,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                wandb_logger=wandb_logger,
                wandb_enabled=(viz_log_wandb and wandb_logger.enabled),
            )
            logger.info(
                "Epoch %d/%d | saved %d validation visualization panels to %s",
                epoch,
                epochs,
                num_saved,
                run_dir / viz_output_subdir / f"epoch_{epoch:04d}",
            )

        selected_metric = val_metrics[f"val_raw/{best_ckpt_metric_name}"]
        selected_macro = val_metrics["val_raw/macro_dice"]

        if not math.isnan(selected_metric):
            wandb_logger.log_epoch(
                {f"val/{best_ckpt_metric_name}": selected_metric},
                step=epoch,
            )
            logger.info(
                "Epoch %d/%d | selected_%s(%s)=%.4f (best=%s)",
                epoch,
                epochs,
                best_ckpt_metric_name,
                best_ckpt_metric_source,
                selected_metric,
                _fmt(best_metric),
            )

        if (
            not math.isnan(selected_metric)
            and selected_metric > best_metric + es_min_delta
        ):
            best_metric = selected_metric
            best_val_macro_dice = selected_macro
            save_checkpoint(
                model,
                optimizer,
                epoch,
                str(checkpoint_dir / "best.pt"),
                scheduler=scheduler,
                scaler=scaler,
                best_val_dice=best_val_macro_dice,
                best_challenge_score=best_metric,
                last_hd95=float("nan"),
            )
            logger.info(
                "New best model at epoch %d (%s %s=%.4f, val_macro_dice=%s) -> %s",
                epoch,
                best_ckpt_metric_source,
                best_ckpt_metric_name,
                best_metric,
                _fmt(best_val_macro_dice),
                checkpoint_dir / "best.pt",
            )
            es_counter = 0
        else:
            if es_enabled and not math.isnan(selected_metric):
                es_counter += 1
                logger.info("Early stopping counter: %d / %d", es_counter, es_patience)

        if es_enabled and es_counter >= es_patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d, min_delta=%.4f).",
                epoch,
                es_patience,
                es_min_delta,
            )
            break

    epoch_ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    best_ckpt = checkpoint_dir / "best.pt"
    training_summary = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "num_epoch_checkpoints": int(len(epoch_ckpts)),
        "best_checkpoint_exists": bool(best_ckpt.exists()),
        "best_checkpoint": str(best_ckpt if best_ckpt.exists() else ""),
        "latest_epoch_checkpoint": str(epoch_ckpts[-1]) if epoch_ckpts else "",
        "best_challenge_score": None if math.isnan(best_metric) else float(best_metric),
        "best_val_macro_dice": None
        if math.isnan(best_val_macro_dice)
        else float(best_val_macro_dice),
        "split_manifest_path": str(split_manifest_copy_path),
        "resize_short_side": int(resize_short_side),
        "train_crop_enabled": bool(train_crop_enabled),
        "train_crop_size": [int(train_crop_size[0]), int(train_crop_size[1])],
        "inference_resize_short_side": int(inference_resize_short_side),
        "inference_mode": str(inference_mode),
        "resized_sliding_window_patch_size": [
            int(resized_sliding_window_patch_size[0]),
            int(resized_sliding_window_patch_size[1]),
        ],
        "resized_sliding_window_overlap": float(resized_sliding_window_overlap),
        "train_sample_count": int(len(train_ds)),
        "val_sample_count": int(len(val_ds)),
        "test_sample_count": int(len(test_ds) if test_ds is not None else 0),
    }
    summary_path = run_context.summary_path
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
        f.write("\n")
    wandb_logger.set_summary(training_summary)
    wandb_logger.finish()

    logger.info("Training complete.")
    logger.info("Best challenge score: %s", _fmt(best_metric))
    logger.info("Best validation macro Dice: %s", _fmt(best_val_macro_dice))
    logger.info("Artifacts saved to: %s", run_dir)
    logger.info("Training summary: %s", summary_path)

    if not epoch_ckpts:
        logger.warning(
            "No epoch checkpoints were written to %s. Evaluation requires at least one .pt file.",
            checkpoint_dir,
        )
    elif not best_ckpt.exists():
        logger.warning(
            "No best checkpoint at %s. Eval will fall back to latest epoch checkpoint: %s",
            best_ckpt,
            epoch_ckpts[-1],
        )
    else:
        logger.info("Best checkpoint available: %s", best_ckpt)

    if test_loader is not None and len(test_loader) > 0:
        logger.info(
            "Test split present (%d samples). Supervised metrics are valid only if this split is from labeled Train_imgs.",
            len(test_loader.dataset),
        )


if __name__ == "__main__":
    main()
