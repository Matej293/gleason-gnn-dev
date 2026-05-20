from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from scipy.ndimage import binary_dilation, binary_fill_holes
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
    orig_hws: list[tuple[int, int]] = []
    resized_hws: list[tuple[int, int]] = []

    for s in batch:
        images.append(pad_to_hw(s["image"], max_h, max_w, value=0.0))
        hard_masks.append(pad_to_hw(s["hard_mask"], max_h, max_w, value=0))
        ignore_masks.append(pad_to_hw(s["ignore_mask"], max_h, max_w, value=255))
        image_ids.append(str(s["image_id"]))
        if "_orig_hw" in s:
            oh, ow = s["_orig_hw"]
            orig_hws.append((int(oh), int(ow)))
        if "_resized_hw" in s:
            rh, rw = s["_resized_hw"]
            resized_hws.append((int(rh), int(rw)))
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
    if len(orig_hws) == len(batch):
        out["orig_hw"] = orig_hws
    if len(resized_hws) == len(batch):
        out["resized_hw"] = resized_hws
    return out


def _nanmean(values: list[float]) -> float:
    finite_vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite_vals) / len(finite_vals)) if finite_vals else float("nan")


def _nanmean_tensor(values: torch.Tensor) -> float:
    if values.ndim != 1:
        values = values.view(-1)
    vals = [float(x) for x in values.detach().cpu().tolist() if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _compute_boundary_metrics(
    pred: torch.Tensor,
    hard_mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_classes: int,
    include_background: bool,
    hausdorff_percentile: float,
    symmetric_asd: bool,
) -> dict[str, float]:
    pred_valid = pred.long().clone()
    hard_valid = hard_mask.long().clone()
    pred_valid[~valid] = 0
    hard_valid[~valid] = 0

    pred_oh = F.one_hot(pred_valid.clamp(0, num_classes - 1), num_classes=num_classes)
    true_oh = F.one_hot(hard_valid.clamp(0, num_classes - 1), num_classes=num_classes)
    pred_oh = pred_oh.permute(0, 3, 1, 2).float()
    true_oh = true_oh.permute(0, 3, 1, 2).float()

    hd95_metric = HausdorffDistanceMetric(
        include_background=include_background,
        percentile=hausdorff_percentile,
        reduction="none",
    )
    asd_metric = SurfaceDistanceMetric(
        include_background=include_background,
        symmetric=symmetric_asd,
        reduction="none",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"monai\..*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"monai\..*")
        hd95_vals = hd95_metric(pred_oh, true_oh).detach().cpu()
        asd_vals = asd_metric(pred_oh, true_oh).detach().cpu()

    if hd95_vals.ndim == 1:
        hd95_vals = hd95_vals.unsqueeze(0)
    if asd_vals.ndim == 1:
        asd_vals = asd_vals.unsqueeze(0)

    class_to_channel = {1: 1, 2: 2, 3: 3} if include_background else {1: 0, 2: 1, 3: 2}

    out: dict[str, float] = {}
    hd95_lesion: list[float] = []
    asd_lesion: list[float] = []
    for class_id, suffix in ((1, "g3"), (2, "g4"), (3, "g5")):
        ch = class_to_channel[class_id]

        hd95_c = float("nan")
        if ch < hd95_vals.shape[1]:
            hd95_c = _nanmean_tensor(hd95_vals[:, ch])
        out[f"hd95_{suffix}"] = hd95_c
        hd95_lesion.append(hd95_c)

        asd_c = float("nan")
        if ch < asd_vals.shape[1]:
            asd_c = _nanmean_tensor(asd_vals[:, ch])
        out[f"asd_{suffix}"] = asd_c
        asd_lesion.append(asd_c)

    out["hd95_mean"] = _nanmean(hd95_lesion)
    out["asd_mean"] = _nanmean(asd_lesion)
    return out


def compute_multiclass_metrics(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    include_background_in_dice: bool,
    include_boundary_metrics: bool = True,
    boundary_metric_cfg: Mapping[str, object] | None = None,
) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    return compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard_mask,
        ignore_mask=ignore_mask,
        include_background_in_dice=include_background_in_dice,
        include_boundary_metrics=include_boundary_metrics,
        boundary_metric_cfg=boundary_metric_cfg,
    )




def build_confusion_matrix(
    pred: torch.Tensor,
    hard_mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_classes: int,
    valid_n: int | None = None,
) -> torch.Tensor:
    if valid_n is None:
        valid_n = int(valid.sum().item())
    if valid_n <= 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred.device)

    pred_v = pred[valid].long().clamp(0, num_classes - 1).reshape(-1)
    hard_v = hard_mask[valid].long().clamp(0, num_classes - 1).reshape(-1)
    flat_idx = (hard_v * num_classes) + pred_v
    flat_counts = torch.bincount(flat_idx, minlength=num_classes * num_classes)
    return flat_counts.reshape(num_classes, num_classes).to(dtype=torch.float64)


def _compute_metrics_from_confusion(
    conf: torch.Tensor,
    *,
    include_background_in_dice: bool,
    valid_n: int,
    hard_valid_support: torch.Tensor,
    ignored_pixel_fraction: float,
    tumor_pixels_ignored_fraction: float,
) -> dict[str, float]:
    num_classes = int(conf.shape[0])
    diag = conf.diag()
    row = conf.sum(dim=1)
    col = conf.sum(dim=0)
    tp_per_class = diag
    fp_per_class = col - diag
    fn_per_class = row - diag

    denom = row + col
    union = row + col - diag
    dice_t = torch.full((num_classes,), float("nan"), dtype=torch.float64, device=conf.device)
    iou_t = torch.full((num_classes,), float("nan"), dtype=torch.float64, device=conf.device)
    dice_valid = denom > 0
    iou_valid = union > 0
    dice_t[dice_valid] = (2.0 * diag[dice_valid] + 1e-5) / (denom[dice_valid] + 1e-5)
    iou_t[iou_valid] = (diag[iou_valid] + 1e-5) / (union[iou_valid] + 1e-5)
    dice_values = [float(x) for x in dice_t.tolist()]
    iou_values = [float(x) for x in iou_t.tolist()]

    start = 0 if include_background_in_dice else 1
    used = [d for d in dice_values[start:] if not math.isnan(d)]
    used_iou = [x for x in iou_values[start:] if not math.isnan(x)]
    macro_dice = float(sum(used) / len(used)) if used else float("nan")
    weighted_num = 0.0
    weighted_den = 0.0
    hard_valid_support = hard_valid_support.to(device=conf.device, dtype=torch.float64)
    for c in range(start, num_classes):
        support = int(hard_valid_support[c].item())
        dice_c = dice_values[c]
        if support > 0 and not math.isnan(dice_c):
            weighted_num += float(support) * float(dice_c)
            weighted_den += float(support)
    weighted_macro_dice = (
        float(weighted_num / weighted_den) if weighted_den > 0.0 else float("nan")
    )

    macro_f1 = macro_dice
    tp_sum = float(tp_per_class.sum().item())
    fp_sum = float(fp_per_class.sum().item())
    fn_sum = float(fn_per_class.sum().item())
    micro_f1 = (
        (2.0 * tp_sum) / ((2.0 * tp_sum) + fp_sum + fn_sum + 1e-8)
        if valid_n > 0
        else float("nan")
    )

    if valid_n > 0:
        po = float(diag.sum().item() / float(valid_n))
        pe = float((row * col).sum().item() / float(valid_n * valid_n))
        denom_kappa = 1.0 - pe
        cohen_kappa = (
            float((po - pe) / denom_kappa) if abs(denom_kappa) > 1e-12 else float("nan")
        )
    else:
        cohen_kappa = float("nan")

    if any(math.isnan(v) for v in (cohen_kappa, macro_f1, micro_f1)):
        challenge_score = float("nan")
    else:
        challenge_score = float(cohen_kappa + ((macro_f1 + micro_f1) / 2.0))

    miou = float(sum(used_iou) / len(used_iou)) if used_iou else float("nan")
    grade5_dice = dice_values[3] if len(dice_values) > 3 else float("nan")
    grade5_iou = iou_values[3] if len(iou_values) > 3 else float("nan")

    tp = float(conf[1:, 1:].sum().item())
    fn = float(conf[1:, 0].sum().item())
    fp = float(conf[0, 1:].sum().item())

    sens = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else float("nan")
    prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else float("nan")
    iou_tumor = (tp + 1e-6) / (tp + fp + fn + 1e-6) if (tp + fp + fn) > 0 else float("nan")

    return {
        "macro_dice": macro_dice,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "cohen_kappa": cohen_kappa,
        "challenge_score": challenge_score,
        "weighted_macro_dice": weighted_macro_dice,
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
        "ignored_pixel_fraction": ignored_pixel_fraction,
        "tumor_pixels_ignored_fraction": tumor_pixels_ignored_fraction,
    }

def compute_multiclass_metrics_from_pred(
    pred: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    include_background_in_dice: bool,
    include_boundary_metrics: bool = True,
    boundary_metric_cfg: Mapping[str, object] | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    valid_n: int | None = None,
    hard_valid_support: torch.Tensor | None = None,
    ignored_pixel_fraction: float | None = None,
    tumor_pixels_ignored_fraction: float | None = None,
    confusion_matrix: torch.Tensor | None = None,
) -> dict[str, float]:
    valid = valid_mask if valid_mask is not None else (ignore_mask == 0)
    if valid_n is None:
        valid_n = int(valid.sum().item())

    num_classes = 4
    conf = (
        confusion_matrix.to(device=pred.device, dtype=torch.float64)
        if confusion_matrix is not None
        else build_confusion_matrix(
            pred=pred,
            hard_mask=hard_mask,
            valid=valid,
            num_classes=num_classes,
            valid_n=valid_n,
        )
    )
    if conf.shape != (num_classes, num_classes):
        raise ValueError(
            "confusion_matrix must have shape "
            f"({num_classes}, {num_classes}), got {tuple(conf.shape)}"
        )
    row = conf.sum(dim=1)

    if ignored_pixel_fraction is None:
        ignored_pixel_fraction = float((~valid).float().mean().item())
    if tumor_pixels_ignored_fraction is None:
        tumor_pixels = hard_mask > 0
        tumor_ignored_den = float(tumor_pixels.sum().item())
        tumor_ignored_num = float((tumor_pixels & (~valid)).sum().item())
        tumor_pixels_ignored_fraction = (
            (tumor_ignored_num / tumor_ignored_den) if tumor_ignored_den > 0 else float("nan")
        )

    if hard_valid_support is None:
        hard_valid_support = row
    metrics = _compute_metrics_from_confusion(
        conf=conf,
        include_background_in_dice=include_background_in_dice,
        valid_n=valid_n,
        hard_valid_support=hard_valid_support,
        ignored_pixel_fraction=ignored_pixel_fraction,
        tumor_pixels_ignored_fraction=tumor_pixels_ignored_fraction,
    )

    if include_boundary_metrics:
        boundary_cfg = boundary_metric_cfg if isinstance(boundary_metric_cfg, Mapping) else {}
        hausdorff_variant = str(boundary_cfg.get("hausdorff_variant", "hd95")).strip().lower()
        if hausdorff_variant != "hd95":
            hausdorff_variant = "hd95"
        hausdorff_percentile = float(boundary_cfg.get("hausdorff_percentile", 95.0))
        if hausdorff_percentile <= 0.0 or hausdorff_percentile > 100.0:
            hausdorff_percentile = 95.0
        include_background = bool(boundary_cfg.get("include_background", False))
        symmetric_asd = bool(boundary_cfg.get("symmetric_asd", True))

        if valid_n > 0:
            metrics.update(
                _compute_boundary_metrics(
                    pred=pred,
                    hard_mask=hard_mask,
                    valid=valid,
                    num_classes=num_classes,
                    include_background=include_background,
                    hausdorff_percentile=hausdorff_percentile,
                    symmetric_asd=symmetric_asd,
                )
            )
        else:
            metrics.update(
                {
                    "hd95_mean": float("nan"),
                    "hd95_g3": float("nan"),
                    "hd95_g4": float("nan"),
                    "hd95_g5": float("nan"),
                    "asd_mean": float("nan"),
                    "asd_g3": float("nan"),
                    "asd_g4": float("nan"),
                    "asd_g5": float("nan"),
                }
            )

    return metrics


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

    valid = ignore_mask == 0
    if tissue_mask is not None:
        valid = valid & (tissue_mask > 0)
    out[~valid] = 0

    if not min_component_size_by_class:
        return out

    active_classes = [
        (int(cls), int(min_size))
        for cls, min_size in min_component_size_by_class.items()
        if int(cls) > 0 and int(min_size) > 1
    ]

    def _fill_internal_tumor_holes(sample: np.ndarray, sample_valid: np.ndarray) -> np.ndarray:
        tumor = (sample > 0) & sample_valid
        if not tumor.any():
            return sample
        holes = np.logical_and(binary_fill_holes(tumor), ~tumor) & sample_valid
        if not holes.any():
            return sample
        hole_comps = label(holes, connectivity=2)
        hole_ids = np.unique(hole_comps)
        for hid in hole_ids:
            hid_int = int(hid)
            if hid_int == 0:
                continue
            hole = hole_comps == hid_int
            ring = np.logical_and(binary_dilation(hole), ~hole)
            neighbor_classes = sample[np.logical_and(ring, sample > 0)]
            if neighbor_classes.size == 0:
                continue
            counts = np.bincount(neighbor_classes.astype(np.int64), minlength=4)
            if counts.shape[0] <= 1:
                continue
            fill_cls = int(np.argmax(counts[1:]) + 1)
            sample[hole] = fill_cls
        return sample

    tumor_present = ((out > 0) & valid).flatten(1).any(dim=1)
    if not bool(tumor_present.any().item()):
        return out

    for b in torch.nonzero(tumor_present, as_tuple=False).view(-1).tolist():
        sample = out[b].detach().cpu().numpy().copy()
        sample_valid = valid[b].detach().cpu().numpy()
        present_counts = np.bincount(sample[sample_valid].reshape(-1), minlength=4)

        for cls_id, min_sz in active_classes:
            if cls_id >= present_counts.shape[0] or int(present_counts[cls_id]) == 0:
                continue
            mask = (sample == cls_id) & sample_valid
            comps = label(mask, connectivity=2)
            comp_counts = np.bincount(comps.reshape(-1))
            keep = comp_counts >= min_sz
            keep[0] = False
            sample[np.logical_and(mask, ~keep[comps])] = 0

        sample = _fill_internal_tumor_holes(sample, sample_valid)
        out[b] = torch.from_numpy(sample).to(device=out.device, dtype=out.dtype)

    return out


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
    "build_confusion_matrix",
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
