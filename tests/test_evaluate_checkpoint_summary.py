from __future__ import annotations

from pathlib import Path

from src.cli.evaluate_checkpoint import (
    _build_evaluation_summary,
    _effective_metric_keys,
    _finalize_aggregate_metrics,
)
from src.eval.metric_config import BOUNDARY_METRIC_KEYS, resolve_metric_settings


def _base_cfg() -> dict[str, object]:
    return {
        "data_root": "data",
        "consensus_root": "data/consensus",
    }


def test_eval_summary_tracks_resolved_metrics_and_includes_boundary_keys() -> None:
    cfg = _base_cfg()
    cfg["metrics"] = {
        "track_keys": ["macro_dice", "challenge_score", "hd95_mean", "asd_mean"],
        "include_boundary_metrics": True,
    }
    settings = resolve_metric_settings(cfg)

    aggregate_raw = _finalize_aggregate_metrics(
        {"macro_dice": 0.71, "challenge_score": 0.82, "num_test_samples": 12.0},
        tracked_metric_keys=tuple(settings.track_keys),
    )
    aggregate_post = _finalize_aggregate_metrics(
        {"macro_dice": 0.73, "challenge_score": 0.85, "num_test_samples": 12.0},
        tracked_metric_keys=tuple(settings.track_keys),
    )

    summary = _build_evaluation_summary(
        run_dir=Path("outputs/runs/run_a"),
        checkpoint_path=Path("outputs/runs/run_a/checkpoints/best.pt"),
        cfg=cfg,
        split_manifest_path=Path("outputs/splits/gleason_consensus_split.json"),
        tracked_metric_keys=tuple(settings.track_keys),
        include_boundary_metrics=settings.include_boundary_metrics,
        aggregate_raw=aggregate_raw,
        aggregate_post=aggregate_post,
        per_case=[],
    )

    assert summary["tracked_metric_keys"] == list(settings.track_keys)
    assert summary["include_boundary_metrics"] is True

    for key in settings.track_keys:
        assert key in summary["aggregate_raw"]
        assert key in summary["aggregate_post"]

    for key in BOUNDARY_METRIC_KEYS:
        assert key in settings.track_keys


def test_eval_summary_excludes_boundary_keys_when_disabled() -> None:
    cfg = _base_cfg()
    cfg["metrics"] = {
        "track_keys": ["macro_dice", "challenge_score", "hd95_mean", "asd_mean"],
        "include_boundary_metrics": False,
    }
    settings = resolve_metric_settings(cfg)

    aggregate_raw = _finalize_aggregate_metrics(
        {"macro_dice": 0.61, "challenge_score": 0.72, "num_test_samples": 4.0},
        tracked_metric_keys=tuple(settings.track_keys),
    )
    aggregate_post = _finalize_aggregate_metrics(
        {"macro_dice": 0.64, "challenge_score": 0.75, "num_test_samples": 4.0},
        tracked_metric_keys=tuple(settings.track_keys),
    )

    summary = _build_evaluation_summary(
        run_dir=Path("outputs/runs/run_b"),
        checkpoint_path=Path("outputs/runs/run_b/checkpoints/best.pt"),
        cfg=cfg,
        split_manifest_path=Path("outputs/splits/gleason_consensus_split.json"),
        tracked_metric_keys=tuple(settings.track_keys),
        include_boundary_metrics=settings.include_boundary_metrics,
        aggregate_raw=aggregate_raw,
        aggregate_post=aggregate_post,
        per_case=[],
    )

    assert summary["tracked_metric_keys"] == list(settings.track_keys)
    assert summary["include_boundary_metrics"] is False

    for key in BOUNDARY_METRIC_KEYS:
        assert key not in settings.track_keys
        assert key not in summary["aggregate_raw"]
        assert key not in summary["aggregate_post"]

    for key in settings.track_keys:
        assert key in summary["aggregate_raw"]
        assert key in summary["aggregate_post"]



def test_effective_metric_keys_appends_boundary_when_enabled() -> None:
    base_keys = ("macro_dice", "challenge_score")

    keys_with_boundary = _effective_metric_keys(
        base_keys,
        include_boundary_metrics=True,
    )
    assert keys_with_boundary[: len(base_keys)] == base_keys
    for key in BOUNDARY_METRIC_KEYS:
        assert key in keys_with_boundary


def test_effective_metric_keys_preserves_disabled_behavior() -> None:
    base_keys = ("macro_dice", "challenge_score")
    keys = _effective_metric_keys(base_keys, include_boundary_metrics=False)
    assert keys == base_keys
