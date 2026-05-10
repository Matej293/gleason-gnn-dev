from __future__ import annotations

import math

from src.train_deconver_2d import _composite_from_metrics, _selected_stream_metrics


def test_best_ckpt_metric_source_changes_selected_composite_stream():
    val_metrics = {
        "val_raw/macro_dice": 0.40,
        "val_raw/weighted_macro_dice": 0.50,
        "val_raw/dice_g5": 0.10,
        "val_raw/sensitivity": 0.20,
        "val_post/macro_dice": 0.70,
        "val_post/weighted_macro_dice": 0.65,
        "val_post/dice_g5": 0.60,
        "val_post/sensitivity": 0.90,
    }
    for stream in ("raw", "post"):
        for key in (
            "grade5_dice",
            "miou",
            "grade5_iou",
            "dice_benign",
            "dice_g3",
            "dice_g4",
            "iou_benign",
            "iou_g3",
            "iou_g4",
            "iou_g5",
            "iou_tumor_vs_benign",
            "precision",
            "ignored_pixel_fraction",
            "tumor_pixels_ignored_fraction",
        ):
            val_metrics[f"val_{stream}/{key}"] = 0.0

    raw_comp = _composite_from_metrics(
        _selected_stream_metrics(val_metrics, "raw"),
        w_macro=0.45,
        w_weighted_macro=0.15,
        w_dice_g5=0.30,
        w_sens=0.10,
    )
    post_comp = _composite_from_metrics(
        _selected_stream_metrics(val_metrics, "post"),
        w_macro=0.45,
        w_weighted_macro=0.15,
        w_dice_g5=0.30,
        w_sens=0.10,
    )

    assert math.isclose(raw_comp, (0.45 * 0.40) + (0.15 * 0.50) + (0.30 * 0.10) + (0.10 * 0.20))
    assert math.isclose(post_comp, (0.45 * 0.70) + (0.15 * 0.65) + (0.30 * 0.60) + (0.10 * 0.90))
    assert post_comp > raw_comp
