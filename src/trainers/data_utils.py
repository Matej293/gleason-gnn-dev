from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset, WeightedRandomSampler

from src.data.consensus_transforms import set_transform_random_state
from src.eval.eval_utils import safe_read_json
from src.data.gleason_consensus_dataset import GleasonConsensusDataset

logger = logging.getLogger(__name__)


def seed_worker(worker_id: int) -> None:
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


def resolve_dataloader_context(
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


def infer_qc_flags(
    qc: dict,
    fail_keys: tuple[str, ...],
    suspicious_keys: tuple[str, ...],
) -> tuple[bool, bool]:
    fail = any(bool(qc.get(k, False)) for k in fail_keys)
    suspicious = any(bool(qc.get(k, False)) for k in suspicious_keys)
    return fail, suspicious


def infer_qc_weight(
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


def case_flags_from_hard_mask(mask_path: Path) -> tuple[bool, bool, bool, bool]:
    with Image.open(mask_path) as img:
        arr = np.asarray(img, dtype=np.uint8)
    has_cancer = bool((arr > 0).any())
    has_g3 = bool((arr == 1).any())
    has_g4 = bool((arr == 2).any())
    has_grade5 = bool((arr == 3).any())
    return has_cancer, has_g3, has_g4, has_grade5


def build_sample_metadata(dataset: GleasonConsensusDataset, cfg: dict) -> list[dict]:
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

        has_cancer, has_g3, has_g4, has_grade5 = case_flags_from_hard_mask(
            Path(item["hard_path"])
        )
        qc = safe_read_json(Path(item["qc_path"]))
        qc_fail, qc_suspicious = infer_qc_flags(
            qc, fail_keys=fail_keys, suspicious_keys=suspicious_keys
        )

        if qc_fail:
            n_hard_fail += 1
        if qc_suspicious:
            n_suspicious += 1

        qc_weight = 1.0
        if qc_policy in {"warn_downweight", "strict_skip"}:
            qc_weight = infer_qc_weight(
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


def split_class_presence_summary(rows: list[dict]) -> dict[str, int]:
    return {
        "n_images": int(len(rows)),
        "n_g3_pos_images": int(sum(1 for r in rows if bool(r.get("has_g3", False)))),
        "n_g4_pos_images": int(sum(1 for r in rows if bool(r.get("has_g4", False)))),
        "n_g5_pos_images": int(sum(1 for r in rows if bool(r.get("has_grade5", False)))),
    }


def log_split_class_presence(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> None:
    train_s = split_class_presence_summary(train_rows)
    val_s = split_class_presence_summary(val_rows)
    test_s = split_class_presence_summary(test_rows)
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


def val_min_class_presence_from_cfg(cfg: dict) -> dict[str, int]:
    return {
        "n_g3_pos_images": int(cfg.get("val_min_g3_pos_images", 1)),
        "n_g4_pos_images": int(cfg.get("val_min_g4_pos_images", 1)),
        "n_g5_pos_images": int(cfg.get("val_min_g5_pos_images", 2)),
    }


def val_presence_shortfalls(val_rows: list[dict], required: dict[str, int]) -> dict[str, tuple[int, int]]:
    summary = split_class_presence_summary(val_rows)
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


def split_two_way_stratified(
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


def build_split_rows(
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
            train_rows, val_rows = split_two_way_stratified(
                rows, right_fraction=0.2, seed=split_seed
            )
            test_rows = []
        else:
            train_rows, holdout_rows = split_two_way_stratified(
                rows, right_fraction=0.2, seed=split_seed
            )
            val_rows, test_rows = split_two_way_stratified(
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

        shortfalls = val_presence_shortfalls(val_rows=val_rows, required=required)
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


def loo_consensus_mean_from_rows(dataset: GleasonConsensusDataset, rows: list[dict]) -> float:
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


def write_split_manifest(
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


def load_hard_case_weight_map(path: str | None) -> dict[str, float]:
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


def build_train_sampler(
    train_rows: list[dict],
    cfg: dict,
    seed: int,
) -> WeightedRandomSampler | None:
    tumor_factor = float(cfg.get("oversample_tumor_positive_factor", 1.0))
    grade5_factor = float(cfg.get("oversample_grade5_factor", 1.0))
    hard_factor = float(cfg.get("oversample_hard_case_factor", 1.0))
    hard_map = load_hard_case_weight_map(
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


def pick_fixed_val_viz_ids(
    val_rows: list[dict],
    seed: int,
    num_cases: int,
) -> list[str]:
    ids = sorted({str(r["image_id"]) for r in val_rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[: max(0, int(num_cases))]


def post_min_component_sizes_from_cfg(cfg: dict) -> dict[int, int]:
    return {
        1: int(cfg.get("post_min_component_size_g3", 0)),
        2: int(cfg.get("post_min_component_size_g4", 0)),
        3: int(cfg.get("post_min_component_size_g5", 0)),
    }


def resolve_training_best_checkpoint_source(best_checkpoint_source: str) -> str:
    source = str(best_checkpoint_source).strip().lower()
    if source != "raw":
        raise ValueError(
            "Training epoch validation is raw-only. Set metrics.best_checkpoint_source='raw'."
        )
    return source
