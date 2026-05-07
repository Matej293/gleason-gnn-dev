from __future__ import annotations

import torch

from src.eval_utils import compute_multiclass_metrics


def test_multiclass_metrics_keys_and_ranges():
    logits = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    logits[:, 1, 2:6, 2:6] = 5.0

    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    hard[:, 2:6, 2:6] = 1

    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)

    m = compute_multiclass_metrics(logits, hard, ignore, include_background_in_dice=False)
    for k in (
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
    ):
        assert k in m
    assert 0.0 <= m["macro_dice"] <= 1.0
    assert 0.0 <= m["miou"] <= 1.0
    assert 0.0 <= m["sensitivity"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["iou_tumor_vs_benign"] <= 1.0
    assert 0.0 <= m["ignored_pixel_fraction"] <= 1.0
