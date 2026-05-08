from __future__ import annotations

from src.train_deconver_2d import _composite_from_metrics, _selected_stream_metrics


def test_best_ckpt_metric_source_changes_selected_composite_stream():
    val_metrics = {
        "val_raw/macro_dice": 0.40,
        "val_raw/sensitivity": 0.20,
        "val_post/macro_dice": 0.70,
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
            "dice_g5",
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

    raw_comp = _composite_from_metrics(_selected_stream_metrics(val_metrics, "raw"), 0.75, 0.25)
    post_comp = _composite_from_metrics(_selected_stream_metrics(val_metrics, "post"), 0.75, 0.25)

    assert raw_comp == (0.75 * 0.40) + (0.25 * 0.20)
    assert post_comp == (0.75 * 0.70) + (0.25 * 0.90)
    assert post_comp > raw_comp
