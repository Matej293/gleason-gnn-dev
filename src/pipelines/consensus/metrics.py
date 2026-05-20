from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    denom = a.sum(dtype=np.float64) + b.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def dice_class(a: np.ndarray, b: np.ndarray, cls: int) -> float | None:
    a_bin = a == cls
    b_bin = b == cls
    if not (a_bin.any() or b_bin.any()):
        return None
    return dice_binary(a_bin, b_bin)


def multiclass_dice(a: np.ndarray, b: np.ndarray, num_classes: int) -> float:
    vals = []
    for c in range(num_classes):
        d = dice_class(a, b, c)
        if d is not None:
            vals.append(d)
    return float(np.mean(vals)) if vals else 1.0


def pairwise_agreement(masks: dict[str, np.ndarray], num_classes: int) -> list[dict[str, Any]]:
    out = []
    for r1, r2 in combinations(sorted(masks), 2):
        m1, m2 = masks[r1], masks[r2]
        item = {
            "rater_a": r1,
            "rater_b": r2,
            "dice_multiclass": multiclass_dice(m1, m2, num_classes),
            "dice_cancer_vs_benign": dice_binary(m1 > 0, m2 > 0),
            "dice_per_class": {},
        }
        for c in range(num_classes):
            d = dice_class(m1, m2, c)
            if d is not None:
                item["dice_per_class"][str(c)] = d
        out.append(item)
    return out


def majority_vote(masks: list[np.ndarray], num_classes: int) -> np.ndarray:
    stack = np.stack(masks, axis=0)
    h, w = stack.shape[1:]
    counts = np.zeros((num_classes, h, w), dtype=np.int32)
    for c in range(num_classes):
        counts[c] = (stack == c).sum(axis=0)
    return np.argmax(counts, axis=0).astype(np.uint8)


def leave_one_out_agreement(masks: dict[str, np.ndarray], num_classes: int) -> dict[str, dict[str, Any]]:
    raters = sorted(masks)
    out: dict[str, dict[str, Any]] = {}
    for r in raters:
        others = [masks[o] for o in raters if o != r]
        if not others:
            out[r] = {
                "dice_multiclass": None,
                "dice_cancer_vs_benign": None,
                "dice_per_class": {},
            }
            continue
        loo = majority_vote(others, num_classes)
        mine = masks[r]
        item: dict[str, Any] = {
            "dice_multiclass": multiclass_dice(mine, loo, num_classes),
            "dice_cancer_vs_benign": dice_binary(mine > 0, loo > 0),
            "dice_per_class": {},
        }
        for c in range(num_classes):
            d = dice_class(mine, loo, c)
            if d is not None:
                item["dice_per_class"][str(c)] = d
        out[r] = item
    return out
