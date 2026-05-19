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
        "macro_f1",
        "micro_f1",
        "cohen_kappa",
        "challenge_score",
        "weighted_macro_dice",
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
        "hd95_mean",
        "hd95_g3",
        "hd95_g4",
        "hd95_g5",
        "asd_mean",
        "asd_g3",
        "asd_g4",
        "asd_g5",
    ):
        assert k in m
    assert 0.0 <= m["macro_dice"] <= 1.0
    assert 0.0 <= m["miou"] <= 1.0
    assert 0.0 <= m["sensitivity"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["iou_tumor_vs_benign"] <= 1.0
    assert 0.0 <= m["ignored_pixel_fraction"] <= 1.0
    assert -1.0 <= m["cohen_kappa"] <= 1.0


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
    assert abs(float(m_logits["weighted_macro_dice"]) - float(m_pred["weighted_macro_dice"])) < 1e-8
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
    assert math.isnan(float(m["hd95_g3"]))
    assert math.isnan(float(m["hd95_g4"]))
    assert math.isnan(float(m["hd95_g5"]))
    assert math.isnan(float(m["hd95_mean"]))
    assert math.isnan(float(m["asd_g3"]))
    assert math.isnan(float(m["asd_g4"]))
    assert math.isnan(float(m["asd_g5"]))
    assert math.isnan(float(m["asd_mean"]))


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


def test_postprocess_fills_internal_holes_created_by_component_pruning():
    pred = torch.ones((1, 16, 16), dtype=torch.long)
    pred[:, 6:8, 6:8] = 2
    ignore = torch.zeros((1, 16, 16), dtype=torch.uint8)

    out = postprocess_predictions(
        pred=pred,
        ignore_mask=ignore,
        tissue_mask=None,
        min_component_size_by_class={2: 16},
    )

    assert int((out == 0).sum().item()) == 0
    assert int((out == 2).sum().item()) == 0
    assert int((out == 1).sum().item()) == int(out.numel())


def test_weighted_macro_matches_macro_on_balanced_support():
    pred = torch.zeros((1, 4, 4), dtype=torch.long)
    hard = torch.zeros((1, 4, 4), dtype=torch.long)
    ignore = torch.zeros((1, 4, 4), dtype=torch.uint8)

    hard[:, 0:2, 0:2] = 1
    hard[:, 0:2, 2:4] = 2
    pred[:, 0:2, 0:2] = 1
    pred[:, 0:2, 2:4] = 2

    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    assert abs(float(m["weighted_macro_dice"]) - float(m["macro_dice"])) < 1e-8


def test_weighted_macro_reflects_imbalanced_support():
    pred = torch.zeros((1, 6, 6), dtype=torch.long)
    hard = torch.zeros((1, 6, 6), dtype=torch.long)
    ignore = torch.zeros((1, 6, 6), dtype=torch.uint8)

    hard[:, 0:4, :] = 1  # 24 pixels
    hard[:, 4:6, 0:2] = 2  # 4 pixels
    pred[:, 0:3, :] = 1  # misses one row of class-1 region
    pred[:, 4:6, :] = 2  # overpredicts class-2 region

    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    macro = float(m["macro_dice"])
    weighted = float(m["weighted_macro_dice"])
    assert 0.0 <= macro <= 1.0
    assert 0.0 <= weighted <= 1.0
    assert weighted != macro


def test_challenge_score_matches_formula_identity():
    pred = torch.zeros((1, 8, 8), dtype=torch.long)
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)

    hard[:, 1:5, 1:5] = 1
    pred[:, 2:6, 2:6] = 1

    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    expected = float(m["cohen_kappa"] + ((m["macro_f1"] + m["micro_f1"]) / 2.0))
    assert abs(float(m["challenge_score"]) - expected) < 1e-8


def test_boundary_metrics_improve_with_better_overlap():
    hard = torch.zeros((1, 16, 16), dtype=torch.long)
    hard[:, 4:12, 4:12] = 1
    ignore = torch.zeros((1, 16, 16), dtype=torch.uint8)

    pred_perfect = hard.clone()
    pred_shifted = torch.zeros_like(hard)
    pred_shifted[:, 6:14, 6:14] = 1

    m_perfect = compute_multiclass_metrics_from_pred(
        pred=pred_perfect,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    m_shifted = compute_multiclass_metrics_from_pred(
        pred=pred_shifted,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )

    assert float(m_perfect["hd95_g3"]) <= float(m_shifted["hd95_g3"])
    assert float(m_perfect["asd_g3"]) <= float(m_shifted["asd_g3"])
    assert float(m_perfect["hd95_mean"]) <= float(m_shifted["hd95_mean"])
    assert float(m_perfect["asd_mean"]) <= float(m_shifted["asd_mean"])
