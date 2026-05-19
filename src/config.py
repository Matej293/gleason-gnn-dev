from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import yaml

DEFAULT_IMAGE_SUBDIRS: tuple[str, ...] = ("Train_imgs", "Test_imgs")
DEFAULT_DECONVER_STRIDES: tuple[int, ...] = (1, 2, 2, 2)


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


def resolve_resize_divisor(cfg: dict[str, Any]) -> int:
    model_name = str(cfg.get("model", "deconver")).strip().lower()
    if model_name == "deconver":
        deconver_strides = tuple(
            int(x) for x in cfg.get("deconver_strides", DEFAULT_DECONVER_STRIDES)
        )
        return int(math.prod([s for s in deconver_strides if s > 1])) or 1
    return 8


def consensus_dataset_kwargs_from_config(
    cfg: dict[str, Any],
    *,
    transform: Callable | None = None,
) -> dict[str, Any]:
    max_long_side = int(cfg.get("max_long_side", 0))
    return {
        "data_root": str(cfg.get("data_root", "./data")),
        "consensus_root": str(cfg.get("consensus_root", "./data/consensus")),
        "image_subdirs": tuple(
            str(x) for x in cfg.get("image_subdirs", DEFAULT_IMAGE_SUBDIRS)
        ),
        "transform": transform,
        "renormalize_probs": bool(cfg.get("renormalize_probs", True)),
        "enforce_background_ignore": bool(cfg.get("enforce_background_ignore", True)),
        "otsu_close_radius": int(cfg.get("otsu_close_radius", 3)),
        "otsu_min_object_size": int(cfg.get("otsu_min_object_size", 4096)),
        "otsu_min_hole_size": int(cfg.get("otsu_min_hole_size", 4096)),
        "probs_eps": float(cfg.get("probs_eps", 1e-8)),
        "load_qc_report": False,
        "max_long_side": max_long_side or None,
        "resize_divisor": resolve_resize_divisor(cfg),
    }


def consensus_train_val_transforms_from_config(
    cfg: dict[str, Any],
) -> tuple[Callable | None, Callable | None]:
    from src.consensus_transforms import build_consensus_train_val_transforms

    return build_consensus_train_val_transforms(cfg)
