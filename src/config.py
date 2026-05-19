from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config at {config_path} must be a mapping/object, got {type(cfg).__name__}."
        )
    return cfg


def _require_cfg_key(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key!r}")
    return cfg[key]


def resolve_patch_size(cfg: dict[str, Any]) -> tuple[int, int]:
    raw = _require_cfg_key(cfg, "patch_size")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("patch_size must be a 2-item list/tuple [H, W].")
    patch_h = int(raw[0])
    patch_w = int(raw[1])
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size entries must be > 0, got [{patch_h}, {patch_w}]")
    return patch_h, patch_w


def resolve_patch_overlap(cfg: dict[str, Any]) -> float:
    overlap = float(cfg.get("patch_overlap", 0.5))
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"patch_overlap must be in [0.0, 1.0), got {overlap}")
    return overlap


def consensus_dataset_kwargs_from_config(
    cfg: dict[str, Any],
    *,
    transform: Callable | None = None,
) -> dict[str, Any]:
    raw_image_subdirs = _require_cfg_key(cfg, "image_subdirs")
    if not isinstance(raw_image_subdirs, (list, tuple)) or not raw_image_subdirs:
        raise ValueError("image_subdirs must be a non-empty list/tuple.")

    return {
        "data_root": str(cfg.get("data_root", "./data")),
        "consensus_root": str(cfg.get("consensus_root", "./data/consensus")),
        "image_subdirs": tuple(str(x) for x in raw_image_subdirs),
        "transform": transform,
        "renormalize_probs": bool(cfg.get("renormalize_probs", True)),
        "enforce_background_ignore": bool(cfg.get("enforce_background_ignore", True)),
        "otsu_close_radius": int(cfg.get("otsu_close_radius", 3)),
        "otsu_min_object_size": int(cfg.get("otsu_min_object_size", 4096)),
        "otsu_min_hole_size": int(cfg.get("otsu_min_hole_size", 4096)),
        "probs_eps": float(cfg.get("probs_eps", 1e-8)),
        "load_qc_report": False,
    }


def consensus_train_val_transforms_from_config(
    cfg: dict[str, Any],
) -> tuple[Callable | None, Callable | None]:
    from src.consensus_transforms import build_consensus_train_val_transforms

    return build_consensus_train_val_transforms(cfg)
