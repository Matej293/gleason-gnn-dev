from __future__ import annotations

from typing import Any, Callable

import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    Lambdad,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandSpatialCropd,
)

SAMPLE_KEYS: tuple[str, ...] = (
    "image",
    "soft_probs",
    "hard_mask",
    "ignore_mask",
    "tissue_mask",
)
MASK_KEYS: tuple[str, ...] = ("hard_mask", "ignore_mask", "tissue_mask")
SPATIAL_MODES: tuple[str, ...] = (
    "bilinear",
    "bilinear",
    "nearest",
    "nearest",
    "nearest",
)
SUPPORTED_PROFILES: tuple[str, ...] = ("light", "medium", "strong")
SUPPORTED_PROB_KEYS: tuple[str, ...] = (
    "flip_h",
    "flip_v",
    "rotate90",
    "affine",
    "crop",
    "scale_intensity",
    "adjust_contrast",
    "gaussian_noise",
)
_PROFILE_PROBS: dict[str, dict[str, float]] = {
    "light": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.20,
        "affine": 0.15,
        "crop": 0.00,
        "scale_intensity": 0.15,
        "adjust_contrast": 0.10,
        "gaussian_noise": 0.10,
    },
    "medium": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.30,
        "affine": 0.25,
        "crop": 0.00,
        "scale_intensity": 0.20,
        "adjust_contrast": 0.15,
        "gaussian_noise": 0.15,
    },
    "strong": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.40,
        "affine": 0.35,
        "crop": 0.00,
        "scale_intensity": 0.25,
        "adjust_contrast": 0.20,
        "gaussian_noise": 0.20,
    },
}


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)


def _add_channel_if_missing(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3 and x.shape[0] == 1:
        return x
    raise ValueError(f"Expected [H,W] or [1,H,W] mask tensor, got shape={tuple(x.shape)}")


def _drop_singleton_channel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected [1,H,W] or [H,W] mask tensor, got shape={tuple(x.shape)}")


def _normalize_soft_probs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 3 or x.shape[0] != 4:
        raise ValueError(f"Expected soft_probs shape [4,H,W], got {tuple(x.shape)}")

    probs = _to_float_tensor(x)
    probs = torch.clamp(probs, min=0.0)
    probs_sum = probs.sum(dim=0, keepdim=True)
    valid = probs_sum >= float(eps)
    out = torch.zeros_like(probs, dtype=torch.float32)
    out = torch.where(valid, probs / torch.clamp(probs_sum, min=float(eps)), out)
    zero_mask = ~valid[0]
    if zero_mask.any():
        out[0, zero_mask] = 1.0
    return out


def _finalize_image(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(_to_float_tensor(x), min=0.0, max=1.0).to(torch.float32)


def _finalize_hard_mask(x: torch.Tensor) -> torch.Tensor:
    x = _drop_singleton_channel(x)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.round(x).clamp_(0.0, 3.0).to(torch.int64)


def _finalize_binary_mask(x: torch.Tensor) -> torch.Tensor:
    x = _drop_singleton_channel(x)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return (x >= 0.5).to(torch.uint8)


def _resolve_patch_size(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("transforms_patch_size must be a 2-item list/tuple [H, W].")
    h = int(raw[0])
    w = int(raw[1])
    if h <= 0 or w <= 0:
        raise ValueError("transforms_patch_size entries must be positive.")
    return h, w


def _resolve_profile_probs(
    profile: str,
    overrides: dict[str, Any] | None,
) -> dict[str, float]:
    if profile not in _PROFILE_PROBS:
        raise ValueError(
            f"Unsupported transforms_profile={profile!r}. Supported: {SUPPORTED_PROFILES}."
        )
    resolved = dict(_PROFILE_PROBS[profile])
    if overrides is None:
        return resolved
    if not isinstance(overrides, dict):
        raise ValueError("transforms_prob must be a mapping of op_name -> probability.")
    for key, value in overrides.items():
        if key not in resolved:
            raise ValueError(
                f"Unsupported transforms_prob key {key!r}. Supported: {SUPPORTED_PROB_KEYS}."
            )
        p = float(value)
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"transforms_prob[{key!r}] must be in [0,1], got {p}."
            )
        resolved[key] = p
    return resolved


def build_consensus_train_transform(cfg: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    if not bool(cfg.get("transforms_enabled", False)):
        return None

    profile = str(cfg.get("transforms_profile", "light")).strip().lower()
    prob_overrides = cfg.get("transforms_prob", None)
    probs = _resolve_profile_probs(profile=profile, overrides=prob_overrides)
    patch_size = _resolve_patch_size(cfg.get("transforms_patch_size", None))

    transforms: list[Callable] = [
        EnsureTyped(keys=SAMPLE_KEYS, dtype=torch.float32, track_meta=False),
        Lambdad(keys=MASK_KEYS, func=_add_channel_if_missing),
        RandFlipd(keys=SAMPLE_KEYS, prob=probs["flip_h"], spatial_axis=1),
        RandFlipd(keys=SAMPLE_KEYS, prob=probs["flip_v"], spatial_axis=0),
        RandRotate90d(keys=SAMPLE_KEYS, prob=probs["rotate90"], max_k=3),
    ]

    if probs["affine"] > 0.0:
        transforms.append(
            RandAffined(
                keys=SAMPLE_KEYS,
                prob=probs["affine"],
                rotate_range=(0.12,),
                translate_range=(32, 32),
                scale_range=(0.08, 0.08),
                mode=SPATIAL_MODES,
                padding_mode=("zeros",) * len(SAMPLE_KEYS),
            )
        )
    if probs["crop"] > 0.0:
        if patch_size is None:
            raise ValueError(
                "transforms_prob['crop'] > 0 requires transforms_patch_size=[H,W]."
            )
        transforms.append(
            RandSpatialCropd(
                keys=SAMPLE_KEYS,
                roi_size=patch_size,
                random_center=True,
                random_size=False,
            )
        )

    transforms.extend(
        [
            RandScaleIntensityd(
                keys=("image",),
                factors=0.10,
                prob=probs["scale_intensity"],
            ),
            RandAdjustContrastd(
                keys=("image",),
                prob=probs["adjust_contrast"],
                gamma=(0.85, 1.15),
            ),
            RandGaussianNoised(
                keys=("image",),
                prob=probs["gaussian_noise"],
                mean=0.0,
                std=0.03,
            ),
            Lambdad(keys=("image",), func=_finalize_image),
            Lambdad(keys=("soft_probs",), func=_normalize_soft_probs),
            Lambdad(keys=("hard_mask",), func=_finalize_hard_mask),
            Lambdad(keys=("ignore_mask",), func=_finalize_binary_mask),
            Lambdad(keys=("tissue_mask",), func=_finalize_binary_mask),
        ]
    )

    return Compose(transforms)


def build_consensus_train_val_transforms(
    cfg: dict[str, Any],
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]] | None, Callable[[dict[str, Any]], dict[str, Any]] | None]:
    return build_consensus_train_transform(cfg), None


def set_transform_random_state(transform: Callable | None, seed: int) -> None:
    if transform is None:
        return
    setter = getattr(transform, "set_random_state", None)
    if callable(setter):
        setter(seed=int(seed))


__all__ = [
    "build_consensus_train_transform",
    "build_consensus_train_val_transforms",
    "set_transform_random_state",
]
