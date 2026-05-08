from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label


def fmt_metric(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def json_float(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


def safe_read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def resolve_split_manifest_path(cfg: dict) -> Path:
    split_manifest_cfg = str(cfg.get("split_manifest_path", "")).strip()
    return (
        Path(split_manifest_cfg)
        if split_manifest_cfg
        else Path(cfg["base_output_dir"]).parent / "splits" / "gleason_consensus_split.json"
    )


def pad_to_hw(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
    value: float | int,
) -> torch.Tensor:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=value)


def collate_consensus_batch(batch: list[dict]) -> dict:
    if not batch:
        raise RuntimeError("Empty batch in collate function.")

    max_h = max(int(s["image"].shape[-2]) for s in batch)
    max_w = max(int(s["image"].shape[-1]) for s in batch)

    images: list[torch.Tensor] = []
    soft_probs: list[torch.Tensor] = []
    hard_masks: list[torch.Tensor] = []
    ignore_masks: list[torch.Tensor] = []
    tissue_masks: list[torch.Tensor] = []
    image_ids: list[str] = []

    for s in batch:
        images.append(pad_to_hw(s["image"], max_h, max_w, value=0.0))
        hard_masks.append(pad_to_hw(s["hard_mask"], max_h, max_w, value=0))
        ignore_masks.append(pad_to_hw(s["ignore_mask"], max_h, max_w, value=1))
        image_ids.append(str(s["image_id"]))
        if "soft_probs" in s:
            soft_probs.append(pad_to_hw(s["soft_probs"], max_h, max_w, value=0.0))
        if "tissue_mask" in s:
            tissue_masks.append(pad_to_hw(s["tissue_mask"], max_h, max_w, value=0))

    out = {
        "image": torch.stack(images, dim=0),
        "hard_mask": torch.stack(hard_masks, dim=0),
        "ignore_mask": torch.stack(ignore_masks, dim=0),
        "image_id": image_ids,
    }
    if soft_probs:
        out["soft_probs"] = torch.stack(soft_probs, dim=0)
    if tissue_masks:
        out["tissue_mask"] = torch.stack(tissue_masks, dim=0)
    return out


def compute_multiclass_metrics(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    include_background_in_dice: bool,
) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    return compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard_mask,
        ignore_mask=ignore_mask,
        include_background_in_dice=include_background_in_dice,
    )


def compute_multiclass_metrics_from_pred(
    pred: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    include_background_in_dice: bool,
) -> dict[str, float]:
    valid = ignore_mask == 0

    num_classes = 4
    dice_values: list[float] = []
    iou_values: list[float] = []
    for c in range(num_classes):
        p = (pred == c) & valid
        t = (hard_mask == c) & valid
        denom = p.sum().item() + t.sum().item()
        union = (p | t).sum().item()
        if denom == 0:
            dice_values.append(float("nan"))
        else:
            inter = (p & t).sum().item()
            dice_values.append((2.0 * inter + 1e-5) / (denom + 1e-5))
        if union == 0:
            iou_values.append(float("nan"))
        else:
            inter = (p & t).sum().item()
            iou_values.append((inter + 1e-5) / (union + 1e-5))

    start = 0 if include_background_in_dice else 1
    used = [d for d in dice_values[start:] if not math.isnan(d)]
    used_iou = [x for x in iou_values[start:] if not math.isnan(x)]
    macro_dice = float(sum(used) / len(used)) if used else float("nan")
    miou = float(sum(used_iou) / len(used_iou)) if used_iou else float("nan")
    grade5_dice = dice_values[3] if len(dice_values) > 3 else float("nan")
    grade5_iou = iou_values[3] if len(iou_values) > 3 else float("nan")

    p_pos = (pred > 0) & valid
    t_pos = (hard_mask > 0) & valid
    tp = float((p_pos & t_pos).sum().item())
    fn = float((~p_pos & t_pos).sum().item())
    fp = float((p_pos & ~t_pos).sum().item())

    sens = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else float("nan")
    prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else float("nan")
    iou_tumor = (tp + 1e-6) / (tp + fp + fn + 1e-6) if (tp + fp + fn) > 0 else float("nan")

    ignored_fraction = float((~valid).float().mean().item())
    tumor_pixels = (hard_mask > 0)
    tumor_ignored_den = float(tumor_pixels.sum().item())
    tumor_ignored_num = float((tumor_pixels & (~valid)).sum().item())
    tumor_ignored_fraction = (
        (tumor_ignored_num / tumor_ignored_den) if tumor_ignored_den > 0 else float("nan")
    )

    return {
        "macro_dice": macro_dice,
        "grade5_dice": grade5_dice,
        "miou": miou,
        "grade5_iou": grade5_iou,
        "dice_benign": dice_values[0] if len(dice_values) > 0 else float("nan"),
        "dice_g3": dice_values[1] if len(dice_values) > 1 else float("nan"),
        "dice_g4": dice_values[2] if len(dice_values) > 2 else float("nan"),
        "dice_g5": dice_values[3] if len(dice_values) > 3 else float("nan"),
        "iou_benign": iou_values[0] if len(iou_values) > 0 else float("nan"),
        "iou_g3": iou_values[1] if len(iou_values) > 1 else float("nan"),
        "iou_g4": iou_values[2] if len(iou_values) > 2 else float("nan"),
        "iou_g5": iou_values[3] if len(iou_values) > 3 else float("nan"),
        "iou_tumor_vs_benign": iou_tumor,
        "sensitivity": sens,
        "precision": prec,
        "ignored_pixel_fraction": ignored_fraction,
        "tumor_pixels_ignored_fraction": tumor_ignored_fraction,
    }


def postprocess_predictions(
    pred: torch.Tensor,
    ignore_mask: torch.Tensor,
    tissue_mask: torch.Tensor | None = None,
    min_component_size_by_class: dict[int, int] | None = None,
) -> torch.Tensor:
    """
    Memory-neutral label-space postprocessing:
    - Force benign outside valid tissue/ignore.
    - Remove tiny isolated components for lesion classes.
    """
    out = pred.clone().long()
    if out.ndim != 3:
        raise ValueError(f"pred must have shape [B,H,W], got {tuple(out.shape)}")
    if ignore_mask.shape != out.shape:
        raise ValueError("ignore_mask and pred must have the same shape.")
    if tissue_mask is not None and tissue_mask.shape != out.shape:
        raise ValueError("tissue_mask and pred must have the same shape when provided.")

    valid = (ignore_mask == 0)
    if tissue_mask is not None:
        valid = valid & (tissue_mask > 0)
    out[~valid] = 0

    if not min_component_size_by_class:
        return out

    out_np = out.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    for b in range(out_np.shape[0]):
        sample = out_np[b]
        sample_valid = valid_np[b]
        for cls, min_size in min_component_size_by_class.items():
            cls_id = int(cls)
            min_sz = int(min_size)
            if cls_id <= 0 or min_sz <= 1:
                continue
            mask = (sample == cls_id) & sample_valid
            if not mask.any():
                continue
            comps = label(mask, connectivity=2)
            ids, counts = np.unique(comps, return_counts=True)
            keep_ids = {int(i) for i, c in zip(ids, counts) if int(i) != 0 and int(c) >= min_sz}
            cleaned = np.isin(comps, list(keep_ids))
            sample[np.logical_and(mask, ~cleaned)] = 0
        out_np[b] = sample

    return torch.from_numpy(out_np).to(device=out.device, dtype=out.dtype)


def load_test_indices_from_manifest(dataset_items: list[dict], split_manifest_path: Path) -> list[int]:
    manifest = safe_read_json(split_manifest_path)
    test_ids = set(str(x) for x in manifest.get("test_image_ids", []))
    if not test_ids:
        raise RuntimeError(f"No test_image_ids in split manifest: {split_manifest_path}")

    indices = [
        i for i, item in enumerate(dataset_items)
        if str(item.get("image_id", "")) in test_ids
    ]
    if not indices:
        raise RuntimeError("No dataset items matched test_image_ids from split manifest.")
    return indices


__all__ = [
    "collate_consensus_batch",
    "compute_multiclass_metrics",
    "compute_multiclass_metrics_from_pred",
    "fmt_metric",
    "json_float",
    "load_test_indices_from_manifest",
    "pad_to_hw",
    "postprocess_predictions",
    "resolve_split_manifest_path",
    "safe_read_json",
]
