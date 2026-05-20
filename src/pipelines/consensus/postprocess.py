from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_opening, label


@dataclass
class PostConfig:
    epsilon: float = 1e-6
    conf_threshold_3_raters: float = 0.50
    conf_threshold_6_raters: float = 0.60
    ignore_conf_threshold_loose: float = 0.30
    ignore_conf_threshold_strict: float = 0.50
    grade5_floor: float = 0.08
    boundary_dilate_px: int = 1
    apply_boundary_penalty: bool = True
    edge_smooth_open_px: int = 0
    edge_smooth_close_px: int = 1
    remove_small_islands_px: int = 64
    fill_small_holes_px: int = 64


def _torch_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def normalize_probs(probs_raw: np.ndarray, epsilon: float = 1e-6, use_gpu: bool = False) -> np.ndarray:
    probs_raw = np.nan_to_num(probs_raw, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    if use_gpu and _torch_available():
        import torch

        t = torch.as_tensor(probs_raw, dtype=torch.float32, device="cuda")
        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
        t = torch.clamp(t, epsilon, 1.0 - epsilon)
        s = t.sum(dim=0, keepdim=True)
        zeroish = s <= epsilon
        t = t / torch.clamp(s, min=epsilon)
        if torch.any(zeroish):
            t[:, zeroish.squeeze(0)] = 1.0 / t.shape[0]
        t = torch.nan_to_num(t, nan=1.0 / t.shape[0], posinf=1.0, neginf=0.0)
        t = t / torch.clamp(t.sum(dim=0, keepdim=True), min=epsilon)
        return t.detach().cpu().numpy().astype(np.float32)

    probs = np.clip(probs_raw, epsilon, 1.0 - epsilon)
    denom = probs.sum(axis=0, keepdims=True)
    zeroish = denom <= epsilon
    probs = probs / np.maximum(denom, epsilon)
    if np.any(zeroish):
        probs[:, zeroish.squeeze(0)] = 1.0 / probs.shape[0]
    probs = np.nan_to_num(probs, nan=1.0 / probs.shape[0], posinf=1.0, neginf=0.0)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), epsilon)
    return probs.astype(np.float32)


def apply_grade5_safeguard(
    probs: np.ndarray,
    masks: list[np.ndarray],
    reliable_flags: list[bool],
    grade5_floor: float,
) -> np.ndarray:
    out = probs.copy()
    n = len(masks)
    if n == 0:
        return out

    g5_votes = np.zeros_like(masks[0], dtype=np.float32)
    for i, m in enumerate(masks):
        if reliable_flags[i]:
            g5_votes += (m == 3).astype(np.float32)
    g5_frac = g5_votes / max(1, int(sum(reliable_flags)))

    threshold = 0.33 if n <= 3 else 0.25
    preserve = g5_frac >= threshold
    if not np.any(preserve):
        return out

    out[3, preserve] = np.maximum(out[3, preserve], grade5_floor)
    out /= np.maximum(out.sum(axis=0, keepdims=True), 1e-8)
    return out


def confidence_uncertainty_maps(probs: np.ndarray, use_gpu: bool = False) -> tuple[np.ndarray, np.ndarray]:
    probs = np.nan_to_num(probs, nan=1.0 / max(1, probs.shape[0]), posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-8)
    if use_gpu and _torch_available():
        import torch

        t = torch.as_tensor(probs, dtype=torch.float32, device="cuda")
        t = torch.nan_to_num(t, nan=1.0 / t.shape[0], posinf=1.0, neginf=0.0)
        t = t / torch.clamp(t.sum(dim=0, keepdim=True), min=1e-8)
        ent = -(t * torch.log(torch.clamp(t, min=1e-8))).sum(dim=0)
        ent = ent / np.log(probs.shape[0])
        unc = torch.clamp(ent, 0.0, 1.0)
        conf = 1.0 - unc
        return conf.detach().cpu().numpy().astype(np.float32), unc.detach().cpu().numpy().astype(np.float32)

    ent = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=0)
    unc = np.clip(ent / np.log(probs.shape[0]), 0.0, 1.0).astype(np.float32)
    conf = (1.0 - unc).astype(np.float32)
    return conf, unc


def make_ignore_mask(confidence: np.ndarray, n_raters: int, strict: bool = False) -> np.ndarray:
    if strict:
        th = 0.50
    else:
        th = 0.30 if n_raters <= 3 else 0.40
    return (confidence < th).astype(np.uint8)


def make_ignore_mask_with_threshold(confidence: np.ndarray, threshold: float) -> np.ndarray:
    return (confidence < float(threshold)).astype(np.uint8)


def boundary_disagreement_penalty(
    hard_mask: np.ndarray,
    masks: list[np.ndarray],
    confidence: np.ndarray,
    dilate_px: int,
    agreement_clip_min: float = 0.75,
) -> np.ndarray:
    if len(masks) <= 1:
        return confidence

    boundary_union = np.zeros_like(hard_mask, dtype=bool)
    for m in masks:
        b = np.zeros_like(m, dtype=bool)
        b[:, 1:] |= m[:, 1:] != m[:, :-1]
        b[1:, :] |= m[1:, :] != m[:-1, :]
        boundary_union |= binary_dilation(b, iterations=dilate_px)

    stack = np.stack(masks, axis=0)
    agreement = (stack == hard_mask[None, ...]).mean(axis=0)
    penalized = confidence.copy()
    penalized[boundary_union] *= np.clip(agreement[boundary_union], float(agreement_clip_min), 1.0)
    return penalized.astype(np.float32)


def _remove_small_components(mask: np.ndarray, min_size: int, keep_large: bool) -> np.ndarray:
    if min_size <= 0:
        return mask
    comp, n_comp = label(mask)
    if n_comp <= 0:
        return mask
    counts = np.bincount(comp.ravel())
    out = mask.copy()
    for comp_id in range(1, len(counts)):
        if counts[comp_id] < min_size:
            out[comp == comp_id] = keep_large
    return out


def refine_hard_mask_classes(
    hard: np.ndarray,
    num_classes: int,
    edge_smooth_open_px: int = 0,
    edge_smooth_close_px: int = 1,
    remove_small_islands_px: int = 64,
    fill_small_holes_px: int = 64,
) -> np.ndarray:
    out = hard.copy().astype(np.uint8, copy=False)
    for cls in range(1, num_classes):
        cls_mask = hard == cls
        if edge_smooth_open_px > 0:
            cls_mask = binary_opening(cls_mask, iterations=edge_smooth_open_px)
        if edge_smooth_close_px > 0:
            cls_mask = binary_closing(cls_mask, iterations=edge_smooth_close_px)
        cls_mask = _remove_small_components(cls_mask, int(remove_small_islands_px), keep_large=False)
        if fill_small_holes_px > 0:
            holes = np.logical_and(binary_fill_holes(cls_mask), ~cls_mask)
            holes = _remove_small_components(holes, int(fill_small_holes_px), keep_large=False)
            cls_mask = np.logical_or(cls_mask, holes)
        out[out == cls] = 0
        out[cls_mask] = np.uint8(cls)
    return out


def boundary_length_4conn(mask: np.ndarray) -> int:
    b = np.zeros_like(mask, dtype=bool)
    b[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    b[1:, :] |= mask[1:, :] != mask[:-1, :]
    return int(b.sum())


def count_small_components_by_class(mask: np.ndarray, num_classes: int, max_size: int) -> int:
    if max_size <= 0:
        return 0
    total = 0
    for cls in range(1, num_classes):
        comp, n_comp = label(mask == cls)
        if n_comp <= 0:
            continue
        counts = np.bincount(comp.ravel())
        total += int(np.sum((counts[1:] > 0) & (counts[1:] < int(max_size))))
    return total


def choose_hard_mask(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=0).astype(np.uint8)


def gpu_info() -> dict[str, Any]:
    out = {"enabled": False, "backend": None, "device": None}
    try:
        import torch

        if torch.cuda.is_available():
            out["enabled"] = True
            out["backend"] = "torch"
            out["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out
