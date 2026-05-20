from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

LEGACY_METRIC_TRACK_KEYS = (
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
)

BOUNDARY_METRIC_KEYS = (
    "hd95_mean",
    "hd95_g3",
    "hd95_g4",
    "hd95_g5",
    "asd_mean",
    "asd_g3",
    "asd_g4",
    "asd_g5",
)

SUPPORTED_METRIC_KEYS = LEGACY_METRIC_TRACK_KEYS + BOUNDARY_METRIC_KEYS
SUPPORTED_METRIC_KEY_SET = frozenset(SUPPORTED_METRIC_KEYS)

SUPPORTED_HAUSDORFF_VARIANTS = frozenset({"hd95"})
BEST_CHECKPOINT_SOURCES = frozenset({"raw", "post"})

DEFAULT_BEST_CHECKPOINT_METRIC = "challenge_score"
DEFAULT_BEST_CHECKPOINT_SOURCE = "raw"
DEFAULT_HAUSDORFF_VARIANT = "hd95"
DEFAULT_HAUSDORFF_PERCENTILE = 95.0


@dataclass(frozen=True)
class BoundaryMetricSettings:
    hausdorff_variant: str
    hausdorff_percentile: float
    include_background: bool
    symmetric_asd: bool


@dataclass(frozen=True)
class MetricSettings:
    track_keys: tuple[str, ...]
    best_checkpoint_metric: str
    best_checkpoint_source: str
    include_boundary_metrics: bool
    boundary: BoundaryMetricSettings


def _dedupe_preserve_order(keys: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        out.append(key)
        seen.add(key)
    return tuple(out)


def _normalize_track_keys(raw_keys: Any) -> tuple[str, ...] | None:
    if raw_keys is None:
        return None
    if not isinstance(raw_keys, (list, tuple)):
        return tuple()
    keys: list[str] = []
    for value in raw_keys:
        key = str(value).strip()
        if key:
            keys.append(key)
    return tuple(keys)


def _resolve_boundary_settings(metrics_cfg: Mapping[str, Any]) -> BoundaryMetricSettings:
    boundary_raw = metrics_cfg.get("boundary", {})
    boundary_cfg = boundary_raw if isinstance(boundary_raw, Mapping) else {}

    variant = str(
        boundary_cfg.get("hausdorff_variant", DEFAULT_HAUSDORFF_VARIANT)
    ).strip().lower()
    if variant not in SUPPORTED_HAUSDORFF_VARIANTS:
        variant = DEFAULT_HAUSDORFF_VARIANT

    percentile = float(
        boundary_cfg.get("hausdorff_percentile", DEFAULT_HAUSDORFF_PERCENTILE)
    )
    if percentile <= 0.0 or percentile > 100.0:
        percentile = DEFAULT_HAUSDORFF_PERCENTILE

    include_background = bool(boundary_cfg.get("include_background", False))
    symmetric_asd = bool(boundary_cfg.get("symmetric_asd", True))
    return BoundaryMetricSettings(
        hausdorff_variant=variant,
        hausdorff_percentile=percentile,
        include_background=include_background,
        symmetric_asd=symmetric_asd,
    )


def resolve_metric_settings(cfg: Mapping[str, Any]) -> MetricSettings:
    metrics_raw = cfg.get("metrics", {})
    metrics_cfg = metrics_raw if isinstance(metrics_raw, Mapping) else {}

    include_boundary_metrics = bool(metrics_cfg.get("include_boundary_metrics", True))
    boundary = _resolve_boundary_settings(metrics_cfg)

    configured_keys = _normalize_track_keys(metrics_cfg.get("track_keys"))
    if configured_keys and all(k in SUPPORTED_METRIC_KEY_SET for k in configured_keys):
        track_keys = configured_keys
    else:
        track_keys = LEGACY_METRIC_TRACK_KEYS

    if include_boundary_metrics:
        track_keys = tuple(track_keys) + tuple(
            key for key in BOUNDARY_METRIC_KEYS if key not in set(track_keys)
        )
    else:
        track_keys = tuple(k for k in track_keys if k not in BOUNDARY_METRIC_KEYS)
    track_keys = _dedupe_preserve_order(track_keys)

    best_metric = str(
        metrics_cfg.get("best_checkpoint_metric", DEFAULT_BEST_CHECKPOINT_METRIC)
    ).strip()
    if best_metric not in SUPPORTED_METRIC_KEY_SET:
        best_metric = DEFAULT_BEST_CHECKPOINT_METRIC

    legacy_source = cfg.get("best_ckpt_metric_source", DEFAULT_BEST_CHECKPOINT_SOURCE)
    best_source = str(
        metrics_cfg.get("best_checkpoint_source", legacy_source)
    ).strip().lower()
    if best_source not in BEST_CHECKPOINT_SOURCES:
        best_source = DEFAULT_BEST_CHECKPOINT_SOURCE

    return MetricSettings(
        track_keys=track_keys,
        best_checkpoint_metric=best_metric,
        best_checkpoint_source=best_source,
        include_boundary_metrics=include_boundary_metrics,
        boundary=boundary,
    )


def validate_metrics_config(cfg: Mapping[str, Any]) -> None:
    metrics_raw = cfg.get("metrics", None)
    if metrics_raw is None:
        return
    if not isinstance(metrics_raw, Mapping):
        raise ValueError("metrics must be a mapping when provided.")

    if "include_boundary_metrics" in metrics_raw and not isinstance(
        metrics_raw["include_boundary_metrics"], bool
    ):
        raise ValueError("metrics.include_boundary_metrics must be a boolean.")

    track_keys_raw = metrics_raw.get("track_keys")
    if track_keys_raw is not None:
        if not isinstance(track_keys_raw, (list, tuple)) or len(track_keys_raw) == 0:
            raise ValueError("metrics.track_keys must be a non-empty list/tuple when provided.")
        normalized = _normalize_track_keys(track_keys_raw) or tuple()
        if len(normalized) != len(track_keys_raw):
            raise ValueError("metrics.track_keys cannot contain empty values.")
        unsupported = sorted(set(normalized) - SUPPORTED_METRIC_KEY_SET)
        if unsupported:
            raise ValueError(
                f"metrics.track_keys contains unsupported keys: {unsupported}. "
                f"Supported keys: {list(SUPPORTED_METRIC_KEYS)}"
            )

    best_metric_raw = metrics_raw.get("best_checkpoint_metric", DEFAULT_BEST_CHECKPOINT_METRIC)
    best_metric = str(best_metric_raw).strip()
    if not best_metric:
        raise ValueError("metrics.best_checkpoint_metric must be a non-empty string.")
    if best_metric not in SUPPORTED_METRIC_KEY_SET:
        raise ValueError(
            f"metrics.best_checkpoint_metric must be one of {list(SUPPORTED_METRIC_KEYS)}, "
            f"got {best_metric!r}"
        )

    best_source = str(
        metrics_raw.get("best_checkpoint_source", cfg.get("best_ckpt_metric_source", "raw"))
    ).strip().lower()
    if best_source not in BEST_CHECKPOINT_SOURCES:
        raise ValueError(
            "metrics.best_checkpoint_source must be one of ['raw', 'post'], "
            f"got {best_source!r}"
        )

    boundary_raw = metrics_raw.get("boundary", None)
    if boundary_raw is not None and not isinstance(boundary_raw, Mapping):
        raise ValueError("metrics.boundary must be a mapping when provided.")
    if isinstance(boundary_raw, Mapping):
        if "hausdorff_variant" in boundary_raw:
            variant = str(boundary_raw["hausdorff_variant"]).strip().lower()
            if variant not in SUPPORTED_HAUSDORFF_VARIANTS:
                raise ValueError(
                    "metrics.boundary.hausdorff_variant must be one of "
                    f"{sorted(SUPPORTED_HAUSDORFF_VARIANTS)}, got {variant!r}"
                )
        if "hausdorff_percentile" in boundary_raw:
            percentile = float(boundary_raw["hausdorff_percentile"])
            if percentile <= 0.0 or percentile > 100.0:
                raise ValueError(
                    "metrics.boundary.hausdorff_percentile must be in (0, 100], "
                    f"got {percentile}"
                )
        if "include_background" in boundary_raw and not isinstance(
            boundary_raw["include_background"], bool
        ):
            raise ValueError("metrics.boundary.include_background must be a boolean.")
        if "symmetric_asd" in boundary_raw and not isinstance(
            boundary_raw["symmetric_asd"], bool
        ):
            raise ValueError("metrics.boundary.symmetric_asd must be a boolean.")

    resolved = resolve_metric_settings(cfg)
    if (
        resolved.best_checkpoint_metric == "challenge_score"
        and "challenge_score" not in resolved.track_keys
    ):
        raise ValueError(
            "metrics.track_keys must contain 'challenge_score' when "
            "metrics.best_checkpoint_metric is 'challenge_score'."
        )


__all__ = [
    "BEST_CHECKPOINT_SOURCES",
    "BOUNDARY_METRIC_KEYS",
    "BoundaryMetricSettings",
    "DEFAULT_BEST_CHECKPOINT_METRIC",
    "DEFAULT_BEST_CHECKPOINT_SOURCE",
    "DEFAULT_HAUSDORFF_PERCENTILE",
    "DEFAULT_HAUSDORFF_VARIANT",
    "LEGACY_METRIC_TRACK_KEYS",
    "MetricSettings",
    "SUPPORTED_HAUSDORFF_VARIANTS",
    "SUPPORTED_METRIC_KEYS",
    "SUPPORTED_METRIC_KEY_SET",
    "resolve_metric_settings",
    "validate_metrics_config",
]
