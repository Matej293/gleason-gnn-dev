from __future__ import annotations

from scripts.create_seg_eval_report_pdf import _resolve_metric_rows


def test_metric_rows_use_tracked_metric_order_and_append_supplementals() -> None:
    summary_a = {
        "tracked_metric_keys": ["macro_dice", "hd95_mean", "asd_mean"],
        "aggregate": {
            "macro_dice": 0.71,
            "hd95_mean": 3.4,
            "asd_mean": 1.2,
            "num_test_samples": 24.0,
        },
    }
    summary_b = {
        "tracked_metric_keys": ["macro_dice", "hd95_mean", "asd_mean"],
        "aggregate": {
            "macro_dice": 0.73,
            "hd95_mean": 3.0,
            "asd_mean": 1.0,
            "num_test_samples": 24.0,
        },
    }

    rows = _resolve_metric_rows(summary_a, summary_b)
    assert rows == ["macro_dice", "hd95_mean", "asd_mean", "num_test_samples"]


def test_metric_rows_fallback_to_numeric_intersection_for_legacy_summaries() -> None:
    summary_a = {
        "aggregate": {
            "macro_dice": 0.61,
            "hd95_mean": 4.2,
            "text_value": "n/a",
            "num_test_samples": 12.0,
            "mean_loo_dice_multiclass": 0.55,
        }
    }
    summary_b = {
        "aggregate": {
            "macro_dice": 0.63,
            "hd95_mean": 3.8,
            "other_metric": 0.99,
            "num_test_samples": 12.0,
            "mean_loo_dice_multiclass": 0.58,
        }
    }

    rows = _resolve_metric_rows(summary_a, summary_b)
    assert rows == [
        "hd95_mean",
        "macro_dice",
        "num_test_samples",
        "mean_loo_dice_multiclass",
    ]


def test_metric_rows_append_loo_counter_only_when_present_in_both() -> None:
    summary_a = {
        "tracked_metric_keys": ["macro_dice", "hd95_mean"],
        "aggregate": {
            "macro_dice": 0.70,
            "hd95_mean": 3.2,
            "num_loo_entries": 36.0,
        },
    }
    summary_b = {
        "tracked_metric_keys": ["macro_dice", "hd95_mean"],
        "aggregate": {
            "macro_dice": 0.72,
            "hd95_mean": 3.0,
        },
    }

    rows = _resolve_metric_rows(summary_a, summary_b)
    assert rows == ["macro_dice", "hd95_mean"]
