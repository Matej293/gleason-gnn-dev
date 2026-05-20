from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StapleConfig:
    max_iterations: int = 100
    convergence_threshold: float = 1e-5


def run_binary_staple(binary_masks: list[np.ndarray], config: StapleConfig) -> np.ndarray:
    # Imported lazily so non-STAPLE code can still run if SimpleITK is absent.
    import SimpleITK as sitk

    imgs = []
    for m in binary_masks:
        img = sitk.GetImageFromArray(m.astype(np.uint8))
        imgs.append(img)

    filt = sitk.STAPLEImageFilter()
    filt.SetMaximumIterations(config.max_iterations)
    filt.SetConfidenceWeight(1.0)
    filt.SetForegroundValue(1)

    out = filt.Execute(imgs)
    arr = sitk.GetArrayFromImage(out).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def run_multiclass_one_vs_rest_staple(masks: list[np.ndarray], num_classes: int, config: StapleConfig) -> np.ndarray:
    if not masks:
        raise ValueError("masks must not be empty")

    h, w = masks[0].shape
    probs = []
    stack = np.stack(masks, axis=0)
    for c in range(num_classes):
        # Fast path: if no rater marks this class anywhere, STAPLE is unnecessary.
        if not np.any(stack == c):
            probs.append(np.zeros((h, w), dtype=np.float32))
            continue
        binaries = [(m == c).astype(np.uint8) for m in masks]
        probs.append(run_binary_staple(binaries, config))
    return np.stack(probs, axis=0).astype(np.float32)


def run_multiclass_weighted_vote(
    masks: list[np.ndarray],
    weights: list[float],
    num_classes: int,
) -> np.ndarray:
    if not masks:
        raise ValueError("masks must not be empty")
    if len(masks) != len(weights):
        raise ValueError("masks and weights must have the same length")

    h, w = masks[0].shape
    weighted = np.zeros((num_classes, h, w), dtype=np.float32)
    total_weight = float(sum(max(0.0, float(w)) for w in weights))
    if total_weight <= 0.0:
        raise ValueError("sum(weights) must be > 0")

    for m, w in zip(masks, weights):
        wv = max(0.0, float(w))
        if wv <= 0.0:
            continue
        for c in range(num_classes):
            weighted[c] += wv * (m == c).astype(np.float32)

    weighted /= total_weight
    return weighted.astype(np.float32)
