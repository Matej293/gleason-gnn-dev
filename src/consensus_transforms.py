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


def _require_cfg_key(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required transform config key: {key!r}")
    return cfg[key]


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


def _resolve_numeric_sequence(raw: Any, *, key: str, expected_len: int) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_len:
        raise ValueError(f"{key} must be a {expected_len}-item list/tuple.")
    return tuple(float(x) for x in raw)


def _resolve_profile_probs(
    profile: str,
    profiles: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, float]:
    if profile not in profiles:
        raise ValueError(
            f"Unsupported transforms_profile={profile!r}. Supported: {sorted(profiles.keys())}."
        )

    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"transforms_profiles[{profile!r}] must be a mapping.")

    resolved: dict[str, float] = {}
    for key in SUPPORTED_PROB_KEYS:
        if key not in selected:
            raise ValueError(
                f"transforms_profiles[{profile!r}] is missing probability key {key!r}."
            )
        p = float(selected[key])
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"transforms_profiles[{profile!r}][{key!r}] must be in [0,1], got {p}."
            )
        resolved[key] = p

    extra = set(selected.keys()) - set(SUPPORTED_PROB_KEYS)
    if extra:
        raise ValueError(
            f"transforms_profiles[{profile!r}] has unsupported keys: {sorted(extra)}"
        )

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
            raise ValueError(f"transforms_prob[{key!r}] must be in [0,1], got {p}.")
        resolved[key] = p
    return resolved


def build_consensus_train_transform(cfg: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    if not bool(cfg.get("transforms_enabled", False)):
        return None

    profile = str(_require_cfg_key(cfg, "transforms_profile")).strip().lower()
    profiles = _require_cfg_key(cfg, "transforms_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("transforms_profiles must be a mapping of profile_name -> probability map.")
    if any(name not in profiles for name in SUPPORTED_PROFILES):
        missing = [name for name in SUPPORTED_PROFILES if name not in profiles]
        raise ValueError(f"transforms_profiles missing required profiles: {missing}")

    prob_overrides = cfg.get("transforms_prob", None)
    probs = _resolve_profile_probs(
        profile=profile,
        profiles=profiles,
        overrides=prob_overrides,
    )
    patch_size = _resolve_patch_size(_require_cfg_key(cfg, "transforms_patch_size"))

    affine_rotate_range = _resolve_numeric_sequence(
        _require_cfg_key(cfg, "transforms_affine_rotate_range"),
        key="transforms_affine_rotate_range",
        expected_len=1,
    )
    affine_translate_range = _resolve_numeric_sequence(
        _require_cfg_key(cfg, "transforms_affine_translate_range"),
        key="transforms_affine_translate_range",
        expected_len=2,
    )
    affine_scale_range = _resolve_numeric_sequence(
        _require_cfg_key(cfg, "transforms_affine_scale_range"),
        key="transforms_affine_scale_range",
        expected_len=2,
    )
    scale_intensity_factors = float(_require_cfg_key(cfg, "transforms_scale_intensity_factors"))
    adjust_contrast_gamma = _resolve_numeric_sequence(
        _require_cfg_key(cfg, "transforms_adjust_contrast_gamma"),
        key="transforms_adjust_contrast_gamma",
        expected_len=2,
    )
    gaussian_noise_mean = float(_require_cfg_key(cfg, "transforms_gaussian_noise_mean"))
    gaussian_noise_std = float(_require_cfg_key(cfg, "transforms_gaussian_noise_std"))

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
                rotate_range=affine_rotate_range,
                translate_range=affine_translate_range,
                scale_range=affine_scale_range,
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
                factors=scale_intensity_factors,
                prob=probs["scale_intensity"],
            ),
            RandAdjustContrastd(
                keys=("image",),
                prob=probs["adjust_contrast"],
                gamma=adjust_contrast_gamma,
            ),
            RandGaussianNoised(
                keys=("image",),
                prob=probs["gaussian_noise"],
                mean=gaussian_noise_mean,
                std=gaussian_noise_std,
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
