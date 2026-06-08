from __future__ import annotations

import numpy as np


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
