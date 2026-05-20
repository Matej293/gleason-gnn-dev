from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureTyped,
    Lambdad,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
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
IGNORE_LABEL = 255

SUPPORTED_PROFILES: tuple[str, ...] = ("light", "medium", "strong")
SUPPORTED_PROB_KEYS: tuple[str, ...] = (
    "flip_h",
    "flip_v",
    "rotate90",
    "affine",
    "scale_intensity",
    "adjust_contrast",
    "gaussian_noise",
    "gaussian_smooth",
    "shift_intensity",
)

DEFAULT_TRANSFORM_PROFILES: dict[str, dict[str, float]] = {
    "light": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.20,
        "affine": 0.15,
        "scale_intensity": 0.15,
        "adjust_contrast": 0.10,
        "gaussian_noise": 0.10,
        "gaussian_smooth": 0.05,
        "shift_intensity": 0.05,
    },
    "medium": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.30,
        "affine": 0.25,
        "scale_intensity": 0.20,
        "adjust_contrast": 0.15,
        "gaussian_noise": 0.15,
        "gaussian_smooth": 0.10,
        "shift_intensity": 0.10,
    },
    "strong": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.40,
        "affine": 0.35,
        "scale_intensity": 0.25,
        "adjust_contrast": 0.20,
        "gaussian_noise": 0.20,
        "gaussian_smooth": 0.15,
        "shift_intensity": 0.15,
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


def _finalize_ignore_mask(x: torch.Tensor) -> torch.Tensor:
    x = _drop_singleton_channel(x)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return torch.where(x >= 0.5, IGNORE_LABEL, 0).to(torch.uint8)


def _finalize_binary_mask(x: torch.Tensor) -> torch.Tensor:
    x = _drop_singleton_channel(x)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return (x >= 0.5).to(torch.uint8)


def _require_cfg_key(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required transform config key: {key!r}")
    return cfg[key]


def _resolve_hw_pair(raw: Any, *, key: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{key} must be a 2-item list/tuple [H, W].")
    h = int(raw[0])
    w = int(raw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{key} entries must be positive.")
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


class _ShortSideResizeDict:
    def __init__(
        self,
        *,
        short_side: int,
        random_scale_enabled: bool,
        random_scale_min: float,
        random_scale_max: float,
    ) -> None:
        self.short_side = int(short_side)
        self.random_scale_enabled = bool(random_scale_enabled)
        self.random_scale_min = float(random_scale_min)
        self.random_scale_max = float(random_scale_max)
        self._rng = np.random.RandomState()

    def set_random_state(self, seed: int | None = None) -> _ShortSideResizeDict:
        if seed is not None:
            self._rng = np.random.RandomState(int(seed))
        return self

    def _resolve_short_side(self) -> int:
        if not self.random_scale_enabled:
            return self.short_side
        scale = float(self._rng.uniform(self.random_scale_min, self.random_scale_max))
        return max(1, int(round(self.short_side * scale)))

    @staticmethod
    def _resize_tensor(x: torch.Tensor, *, size: tuple[int, int], mode: str) -> torch.Tensor:
        if x.ndim == 2:
            x4 = x.unsqueeze(0).unsqueeze(0).float()
            squeezed = True
        elif x.ndim == 3:
            x4 = x.unsqueeze(0).float()
            squeezed = False
        else:
            raise ValueError(f"Expected [H,W] or [C,H,W], got shape={tuple(x.shape)}")

        kwargs: dict[str, object] = {}
        if mode != "nearest":
            kwargs["align_corners"] = False
        out = F.interpolate(x4, size=size, mode=mode, **kwargs)
        out = out.squeeze(0)
        if squeezed:
            out = out.squeeze(0)
        return out

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError("Expected sample['image'] to be a tensor with shape [C,H,W].")

        h, w = int(image.shape[-2]), int(image.shape[-1])
        short_side = self._resolve_short_side()
        scale = float(short_side) / float(max(1, min(h, w)))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        target_size = (new_h, new_w)

        sample["_orig_hw"] = (h, w)
        sample["_resized_hw"] = target_size

        sample["image"] = self._resize_tensor(sample["image"], size=target_size, mode="bilinear")
        sample["soft_probs"] = self._resize_tensor(sample["soft_probs"], size=target_size, mode="bilinear")
        sample["hard_mask"] = self._resize_tensor(sample["hard_mask"], size=target_size, mode="nearest")
        sample["ignore_mask"] = self._resize_tensor(sample["ignore_mask"], size=target_size, mode="nearest")
        sample["tissue_mask"] = self._resize_tensor(sample["tissue_mask"], size=target_size, mode="nearest")
        return sample


class _PadToMinSizeDict:
    def __init__(
        self,
        *,
        min_size: tuple[int, int],
    ) -> None:
        self.min_h = int(min_size[0])
        self.min_w = int(min_size[1])

    @staticmethod
    def _pad_tensor(x: torch.Tensor, *, target_h: int, target_w: int, value: float) -> torch.Tensor:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=float(value))

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample["image"] = self._pad_tensor(sample["image"], target_h=self.min_h, target_w=self.min_w, value=0.0)
        sample["soft_probs"] = self._pad_tensor(sample["soft_probs"], target_h=self.min_h, target_w=self.min_w, value=0.0)
        sample["hard_mask"] = self._pad_tensor(sample["hard_mask"], target_h=self.min_h, target_w=self.min_w, value=0.0)
        sample["ignore_mask"] = self._pad_tensor(sample["ignore_mask"], target_h=self.min_h, target_w=self.min_w, value=float(IGNORE_LABEL))
        sample["tissue_mask"] = self._pad_tensor(sample["tissue_mask"], target_h=self.min_h, target_w=self.min_w, value=0.0)
        return sample


def _build_spatial_preprocess(
    cfg: dict[str, Any],
    *,
    train: bool,
) -> list[Callable[[dict[str, Any]], dict[str, Any]]]:
    resize_short_side_key = "resize_short_side" if train else "inference_resize_short_side"
    resize_short_side = int(_require_cfg_key(cfg, resize_short_side_key))
    if resize_short_side <= 0:
        raise ValueError(f"{resize_short_side_key} must be > 0.")

    random_scale_enabled = bool(cfg.get("train_resize_random_scale_enabled", False)) if train else False
    random_scale_min = float(cfg.get("train_resize_random_scale_min", 1.0))
    random_scale_max = float(cfg.get("train_resize_random_scale_max", 1.0))
    if random_scale_enabled and random_scale_min <= 0.0:
        raise ValueError("train_resize_random_scale_min must be > 0 when random scale is enabled.")
    if random_scale_enabled and random_scale_max < random_scale_min:
        raise ValueError(
            "train_resize_random_scale_max must be >= train_resize_random_scale_min when random scale is enabled."
        )

    preprocess: list[Callable[[dict[str, Any]], dict[str, Any]]] = [
        _ShortSideResizeDict(
            short_side=resize_short_side,
            random_scale_enabled=random_scale_enabled,
            random_scale_min=random_scale_min,
            random_scale_max=random_scale_max,
        )
    ]

    if train and bool(cfg.get("train_crop_enabled", True)):
        crop_h, crop_w = _resolve_hw_pair(_require_cfg_key(cfg, "train_crop_size"), key="train_crop_size")
        preprocess.extend(
            [
                _PadToMinSizeDict(min_size=(crop_h, crop_w)),
                RandSpatialCropd(
                    keys=SAMPLE_KEYS,
                    roi_size=(crop_h, crop_w),
                    random_center=True,
                    random_size=False,
                ),
            ]
        )

    return preprocess


def _build_train_augmentations(cfg: dict[str, Any]) -> list[Callable]:
    if not bool(cfg.get("transforms_enabled", False)):
        return []

    profile = str(cfg.get("transforms_profile", "light")).strip().lower()
    profiles = cfg.get("transforms_profiles", DEFAULT_TRANSFORM_PROFILES)
    if not isinstance(profiles, dict):
        raise ValueError("transforms_profiles must be a mapping of profile_name -> probability map.")

    normalized_profiles = dict(DEFAULT_TRANSFORM_PROFILES)
    normalized_profiles.update(profiles)

    prob_overrides = cfg.get("transforms_prob", None)
    probs = _resolve_profile_probs(
        profile=profile,
        profiles=normalized_profiles,
        overrides=prob_overrides,
    )

    affine_rotate_range = _resolve_numeric_sequence(
        cfg.get("transforms_affine_rotate_range", [0.12]),
        key="transforms_affine_rotate_range",
        expected_len=1,
    )
    affine_translate_range = _resolve_numeric_sequence(
        cfg.get("transforms_affine_translate_range", [32, 32]),
        key="transforms_affine_translate_range",
        expected_len=2,
    )
    affine_scale_range = _resolve_numeric_sequence(
        cfg.get("transforms_affine_scale_range", [0.08, 0.08]),
        key="transforms_affine_scale_range",
        expected_len=2,
    )
    scale_intensity_factors = float(cfg.get("transforms_scale_intensity_factors", 0.10))
    adjust_contrast_gamma = _resolve_numeric_sequence(
        cfg.get("transforms_adjust_contrast_gamma", [0.85, 1.15]),
        key="transforms_adjust_contrast_gamma",
        expected_len=2,
    )
    gaussian_noise_mean = float(cfg.get("transforms_gaussian_noise_mean", 0.0))
    gaussian_noise_std = float(cfg.get("transforms_gaussian_noise_std", 0.03))
    gaussian_smooth_sigma_x = _resolve_numeric_sequence(
        cfg.get("transforms_gaussian_smooth_sigma_x", [0.25, 1.00]),
        key="transforms_gaussian_smooth_sigma_x",
        expected_len=2,
    )
    gaussian_smooth_sigma_y = _resolve_numeric_sequence(
        cfg.get("transforms_gaussian_smooth_sigma_y", [0.25, 1.00]),
        key="transforms_gaussian_smooth_sigma_y",
        expected_len=2,
    )
    shift_intensity_offsets = _resolve_numeric_sequence(
        cfg.get("transforms_shift_intensity_offsets", [-0.08, 0.08]),
        key="transforms_shift_intensity_offsets",
        expected_len=2,
    )

    if gaussian_smooth_sigma_x[0] < 0.0 or gaussian_smooth_sigma_x[1] < 0.0:
        raise ValueError("transforms_gaussian_smooth_sigma_x entries must be >= 0.")
    if gaussian_smooth_sigma_x[1] < gaussian_smooth_sigma_x[0]:
        raise ValueError("transforms_gaussian_smooth_sigma_x must satisfy [min, max] with max >= min.")
    if gaussian_smooth_sigma_y[0] < 0.0 or gaussian_smooth_sigma_y[1] < 0.0:
        raise ValueError("transforms_gaussian_smooth_sigma_y entries must be >= 0.")
    if gaussian_smooth_sigma_y[1] < gaussian_smooth_sigma_y[0]:
        raise ValueError("transforms_gaussian_smooth_sigma_y must satisfy [min, max] with max >= min.")
    if shift_intensity_offsets[1] < shift_intensity_offsets[0]:
        raise ValueError("transforms_shift_intensity_offsets must satisfy [min, max] with max >= min.")

    out: list[Callable] = [
        RandFlipd(keys=SAMPLE_KEYS, prob=probs["flip_h"], spatial_axis=1),
        RandFlipd(keys=SAMPLE_KEYS, prob=probs["flip_v"], spatial_axis=0),
        RandRotate90d(keys=SAMPLE_KEYS, prob=probs["rotate90"], max_k=3),
    ]

    if probs["affine"] > 0.0:
        out.append(
            RandAffined(
                keys=SAMPLE_KEYS,
                prob=probs["affine"],
                rotate_range=affine_rotate_range,
                translate_range=affine_translate_range,
                scale_range=affine_scale_range,
                mode=("bilinear", "bilinear", "nearest", "nearest", "nearest"),
                padding_mode=("zeros",) * len(SAMPLE_KEYS),
            )
        )

    out.extend(
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
            RandGaussianSmoothd(
                keys=("image",),
                prob=probs["gaussian_smooth"],
                sigma_x=gaussian_smooth_sigma_x,
                sigma_y=gaussian_smooth_sigma_y,
            ),
            RandShiftIntensityd(
                keys=("image",),
                prob=probs["shift_intensity"],
                offsets=shift_intensity_offsets,
            ),
        ]
    )
    return out


def _build_common_finalize() -> list[Callable]:
    return [
        Lambdad(keys=("image",), func=_finalize_image),
        Lambdad(keys=("soft_probs",), func=_normalize_soft_probs),
        Lambdad(keys=("hard_mask",), func=_finalize_hard_mask),
        Lambdad(keys=("ignore_mask",), func=_finalize_ignore_mask),
        Lambdad(keys=("tissue_mask",), func=_finalize_binary_mask),
    ]


def build_consensus_train_transform(cfg: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    transforms: list[Callable] = [
        EnsureTyped(keys=SAMPLE_KEYS, dtype=torch.float32, track_meta=False),
        Lambdad(keys=MASK_KEYS, func=_add_channel_if_missing),
    ]
    transforms.extend(_build_spatial_preprocess(cfg, train=True))
    transforms.extend(_build_train_augmentations(cfg))
    transforms.extend(_build_common_finalize())
    return Compose(transforms)


def build_consensus_val_transform(cfg: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    transforms: list[Callable] = [
        EnsureTyped(keys=SAMPLE_KEYS, dtype=torch.float32, track_meta=False),
        Lambdad(keys=MASK_KEYS, func=_add_channel_if_missing),
    ]
    transforms.extend(_build_spatial_preprocess(cfg, train=False))
    transforms.extend(_build_common_finalize())
    return Compose(transforms)


def build_consensus_train_val_transforms(
    cfg: dict[str, Any],
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
    return build_consensus_train_transform(cfg), build_consensus_val_transform(cfg)


def set_transform_random_state(transform: Callable | None, seed: int) -> None:
    if transform is None:
        return
    setter = getattr(transform, "set_random_state", None)
    if callable(setter):
        setter(seed=int(seed))


__all__ = [
    "build_consensus_train_transform",
    "build_consensus_val_transform",
    "build_consensus_train_val_transforms",
    "set_transform_random_state",
]
