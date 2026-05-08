from __future__ import annotations

import math

import torch

from src.eval_utils import (
    compute_multiclass_metrics,
    compute_multiclass_metrics_from_pred,
    postprocess_predictions,
)


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


def test_postprocess_predictions_forces_benign_outside_valid_and_removes_tiny_components():
    pred = torch.zeros((1, 12, 12), dtype=torch.long)
    pred[:, 0:2, 0:2] = 1
    pred[:, 4:10, 4:10] = 1
    ignore = torch.zeros((1, 12, 12), dtype=torch.uint8)
    ignore[:, :3, :] = 1
    tissue = torch.ones((1, 12, 12), dtype=torch.uint8)
    tissue[:, :, :2] = 0

    out = postprocess_predictions(
        pred=pred,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class={1: 8},
    )
    assert int(out[:, :3, :].sum().item()) == 0
    assert int(out[:, :, :2].sum().item()) == 0
    assert int((out == 1).sum().item()) == 36


def test_metrics_from_pred_matches_logits_path():
    logits = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    logits[:, 2, 1:7, 1:7] = 7.0
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    hard[:, 1:7, 1:7] = 2
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)

    m_logits = compute_multiclass_metrics(logits, hard, ignore, include_background_in_dice=False)
    pred = logits.argmax(dim=1)
    m_pred = compute_multiclass_metrics_from_pred(pred, hard, ignore, include_background_in_dice=False)
    assert abs(float(m_logits["macro_dice"]) - float(m_pred["macro_dice"])) < 1e-8
    assert abs(float(m_logits["miou"]) - float(m_pred["miou"])) < 1e-8


def test_absent_class_metrics_return_nan_and_are_excluded_from_macro():
    pred = torch.zeros((1, 8, 8), dtype=torch.long)
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)
    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    assert math.isnan(float(m["dice_g3"]))
    assert math.isnan(float(m["dice_g4"]))
    assert math.isnan(float(m["dice_g5"]))
    assert math.isnan(float(m["macro_dice"]))


def test_postprocess_improves_metrics_when_raw_predicts_outside_tissue():
    pred_raw = torch.zeros((1, 8, 8), dtype=torch.long)
    pred_raw[:, :3, :3] = 1  # false positive region outside tissue
    pred_raw[:, 5:7, 5:7] = 1  # true positive region
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    hard[:, 5:7, 5:7] = 1
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)
    tissue = torch.ones((1, 8, 8), dtype=torch.uint8)
    tissue[:, :3, :3] = 0

    m_raw = compute_multiclass_metrics_from_pred(
        pred=pred_raw,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    pred_post = postprocess_predictions(
        pred=pred_raw,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class={},
    )
    m_post = compute_multiclass_metrics_from_pred(
        pred=pred_post,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )

    assert float(m_post["dice_benign"]) > float(m_raw["dice_benign"])
    assert float(m_post["iou_tumor_vs_benign"]) > float(m_raw["iou_tumor_vs_benign"])
