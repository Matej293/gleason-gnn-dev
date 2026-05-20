from __future__ import annotations

from src.eval.metric_config import BOUNDARY_METRIC_KEYS, LEGACY_METRIC_TRACK_KEYS, resolve_metric_settings


def test_resolve_metric_settings_defaults_for_legacy_config() -> None:
    cfg = {"best_ckpt_metric_source": "post"}
    resolved = resolve_metric_settings(cfg)

    for key in LEGACY_METRIC_TRACK_KEYS:
        assert key in resolved.track_keys
    for key in BOUNDARY_METRIC_KEYS:
        assert key in resolved.track_keys
    assert resolved.best_checkpoint_metric == "challenge_score"
    assert resolved.best_checkpoint_source == "post"


def test_resolve_metric_settings_disables_boundary_keys_when_configured() -> None:
    cfg = {
        "metrics": {
            "track_keys": ["macro_dice", "challenge_score", "hd95_mean", "asd_mean"],
            "include_boundary_metrics": False,
        }
    }
    resolved = resolve_metric_settings(cfg)
    assert "macro_dice" in resolved.track_keys
    assert "challenge_score" in resolved.track_keys
    assert "hd95_mean" not in resolved.track_keys
    assert "asd_mean" not in resolved.track_keys


def test_resolve_metric_settings_falls_back_on_invalid_track_keys() -> None:
    cfg = {
        "metrics": {
            "track_keys": ["macro_dice", "unknown_key"],
        }
    }
    resolved = resolve_metric_settings(cfg)
    for key in LEGACY_METRIC_TRACK_KEYS:
        assert key in resolved.track_keys
