from __future__ import annotations

from pathlib import Path

import torch

try:
    from src.eval.metric_config import validate_metrics_config
except Exception:  # pragma: no cover - fallback for scripts importing src modules directly.
    from metric_config import validate_metrics_config


_TRANSFORM_PROB_KEYS = {
    "flip_h",
    "flip_v",
    "rotate90",
    "affine",
    "scale_intensity",
    "adjust_contrast",
    "gaussian_noise",
    "gaussian_smooth",
    "shift_intensity",
}

_REMOVED_NATIVE_KEYS = (
    "patch_size",
    "patch_overlap",
    "patch_tissue_filter_enabled",
    "patch_min_tissue_fraction",
    "transforms_patch_size",
)


def _require_keys(cfg: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def _validate_transform_profile_probs(
    *,
    profile_name: str,
    probs: dict,
) -> None:
    if not isinstance(probs, dict):
        raise ValueError(f"transforms_profiles[{profile_name!r}] must be a mapping.")

    missing = sorted(_TRANSFORM_PROB_KEYS - set(probs.keys()))
    extra = sorted(set(probs.keys()) - _TRANSFORM_PROB_KEYS)
    if missing:
        raise ValueError(
            f"transforms_profiles[{profile_name!r}] missing keys: {missing}"
        )
    if extra:
        raise ValueError(
            f"transforms_profiles[{profile_name!r}] has unsupported keys: {extra}"
        )

    for key in sorted(_TRANSFORM_PROB_KEYS):
        p = float(probs[key])
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"transforms_profiles[{profile_name!r}][{key!r}] must be in [0,1], got {p}"
            )


def _validate_fixed_len_numeric_sequence(
    raw: object,
    *,
    key: str,
    expected_len: int,
) -> None:
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_len:
        raise ValueError(f"{key} must be a {expected_len}-item list/tuple.")
    for value in raw:
        float(value)


def _validate_hw_pair(cfg: dict, key: str) -> tuple[int, int]:
    raw = cfg.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{key} must be a 2-item list/tuple [H, W].")
    h = int(raw[0])
    w = int(raw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{key} entries must be > 0, got [{h}, {w}]")
    return h, w


def _validate_resized_schema(cfg: dict) -> None:
    for key in _REMOVED_NATIVE_KEYS:
        if key in cfg:
            raise ValueError(
                f"Config key '{key}' was removed in the resized-only migration. "
                "Use resized pipeline keys instead: "
                "resize_short_side, train_crop_enabled, train_crop_size, "
                "train_resize_random_scale_enabled, inference_resize_short_side, "
                "inference_mode, resized_sliding_window_patch_size, resized_sliding_window_overlap."
            )

    resize_short_side = int(cfg.get("resize_short_side", 0))
    if resize_short_side <= 0:
        raise ValueError(f"resize_short_side must be > 0, got {resize_short_side}")

    if not isinstance(cfg.get("train_crop_enabled", None), bool):
        raise ValueError("train_crop_enabled must be a boolean.")
    _validate_hw_pair(cfg, "train_crop_size")

    if not isinstance(cfg.get("train_resize_random_scale_enabled", None), bool):
        raise ValueError("train_resize_random_scale_enabled must be a boolean.")
    if bool(cfg.get("train_resize_random_scale_enabled", False)):
        if "train_resize_random_scale_min" not in cfg or "train_resize_random_scale_max" not in cfg:
            raise ValueError(
                "train_resize_random_scale_enabled=true requires train_resize_random_scale_min and train_resize_random_scale_max."
            )
        random_scale_min = float(cfg["train_resize_random_scale_min"])
        random_scale_max = float(cfg["train_resize_random_scale_max"])
        if random_scale_min <= 0.0:
            raise ValueError(
                f"train_resize_random_scale_min must be > 0, got {random_scale_min}"
            )
        if random_scale_max < random_scale_min:
            raise ValueError(
                "train_resize_random_scale_max must be >= train_resize_random_scale_min."
            )

    inference_resize_short_side = int(cfg.get("inference_resize_short_side", 0))
    if inference_resize_short_side <= 0:
        raise ValueError(
            f"inference_resize_short_side must be > 0, got {inference_resize_short_side}"
        )

    inference_mode = str(cfg.get("inference_mode", "")).strip().lower()
    if inference_mode not in {"resized_full", "resized_sliding_window"}:
        raise ValueError(
            "inference_mode must be one of ['resized_full', 'resized_sliding_window'], "
            f"got {inference_mode!r}"
        )

    _validate_hw_pair(cfg, "resized_sliding_window_patch_size")

    resized_sw_overlap = float(cfg.get("resized_sliding_window_overlap", -1.0))
    if resized_sw_overlap < 0.0 or resized_sw_overlap >= 1.0:
        raise ValueError(
            "resized_sliding_window_overlap must be in [0.0, 1.0), "
            f"got {resized_sw_overlap}"
        )


def validate_deconver_config(
    cfg: dict,
    *,
    for_eval: bool,
    require_paths: bool = True,
) -> None:
    _require_keys(
        cfg,
        [
            "model",
            "spatial_dims",
            "out_channels",
            "data_root",
            "consensus_root",
            "base_output_dir",
            "image_subdirs",
            "resize_short_side",
            "train_crop_enabled",
            "train_crop_size",
            "train_resize_random_scale_enabled",
            "inference_resize_short_side",
            "inference_mode",
            "resized_sliding_window_patch_size",
            "resized_sliding_window_overlap",
        ],
    )

    _validate_resized_schema(cfg)

    model_name = str(cfg.get("model", "")).strip().lower()
    if model_name not in {"deconver", "pspnet"}:
        raise ValueError(
            f"Expected model in ['deconver', 'pspnet'], got {model_name!r}"
        )

    image_subdirs = cfg.get("image_subdirs")
    if not isinstance(image_subdirs, (list, tuple)) or not image_subdirs:
        raise ValueError("image_subdirs must be a non-empty list/tuple.")

    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(f"Expected spatial_dims=2, got {spatial_dims}")

    out_channels = int(cfg.get("out_channels", 4))
    if out_channels != 4:
        raise ValueError(f"Expected out_channels=4 for Gleason consensus, got {out_channels}")

    input_channels = int(cfg.get("input_channels", 3))
    if input_channels <= 0:
        raise ValueError(f"input_channels must be > 0, got {input_channels}")

    if model_name == "deconver":
        deconver_strides = cfg.get("deconver_strides")
        if not isinstance(deconver_strides, (list, tuple)) or not deconver_strides:
            raise ValueError("deconver_strides must be a non-empty list/tuple for model='deconver'.")
        if any(int(x) <= 0 for x in deconver_strides):
            raise ValueError("deconver_strides entries must all be > 0.")

        encoder_depth = cfg.get("deconver_encoder_depth", [1, 1, 1, 1])
        if not isinstance(encoder_depth, (list, tuple)) or not encoder_depth:
            raise ValueError("deconver_encoder_depth must be a non-empty list/tuple for model='deconver'.")
        if any(int(x) <= 0 for x in encoder_depth):
            raise ValueError("deconver_encoder_depth entries must all be > 0.")

        encoder_width = cfg.get("deconver_encoder_width", [64, 128, 256, 512])
        if not isinstance(encoder_width, (list, tuple)) or not encoder_width:
            raise ValueError("deconver_encoder_width must be a non-empty list/tuple for model='deconver'.")
        if any(int(x) <= 0 for x in encoder_width):
            raise ValueError("deconver_encoder_width entries must all be > 0.")

        if len(encoder_width) != len(encoder_depth):
            raise ValueError(
                "deconver_encoder_width length must match deconver_encoder_depth length. "
                f"Got {len(encoder_width)} vs {len(encoder_depth)}."
            )
        if len(deconver_strides) != len(encoder_depth):
            raise ValueError(
                "deconver_strides length must match deconver_encoder_depth length. "
                f"Got {len(deconver_strides)} vs {len(encoder_depth)}."
            )

        decoder_depth = cfg.get("deconver_decoder_depth", None)
        if decoder_depth is not None:
            if not isinstance(decoder_depth, (list, tuple)) or not decoder_depth:
                raise ValueError("deconver_decoder_depth must be a non-empty list/tuple when provided.")
            if any(int(x) <= 0 for x in decoder_depth):
                raise ValueError("deconver_decoder_depth entries must all be > 0.")
            if len(decoder_depth) != max(0, len(encoder_depth) - 1):
                raise ValueError(
                    "deconver_decoder_depth length must be len(deconver_encoder_depth) - 1. "
                    f"Got {len(decoder_depth)} vs {max(0, len(encoder_depth) - 1)}."
                )

    if model_name == "pspnet":
        if input_channels != 3:
            raise ValueError(
                f"pspnet requires input_channels=3, got {input_channels}"
            )
        pspnet_loss_mode = str(cfg.get("pspnet_loss_mode", "gleason_ce_soft")).strip().lower()
        if pspnet_loss_mode != "gleason_ce_soft":
            raise ValueError(
                "pspnet_loss_mode must be 'gleason_ce_soft', "
                f"got {pspnet_loss_mode!r}"
            )
        pspnet_soft_term = str(cfg.get("pspnet_soft_term", "ce")).strip().lower()
        if pspnet_soft_term != "ce":
            raise ValueError(
                "pspnet_soft_term must be 'ce', "
                f"got {pspnet_soft_term!r}"
            )
        pspnet_soft_weight = float(cfg.get("pspnet_soft_weight", 0.2))
        if pspnet_soft_weight < 0.0:
            raise ValueError(
                f"pspnet_soft_weight must be >= 0, got {pspnet_soft_weight}"
            )
        aux_weight = float(cfg.get("pspnet_aux_weight", 0.5))
        if not 0.0 <= aux_weight <= 1.0:
            raise ValueError(
                f"pspnet_aux_weight must be in [0, 1], got {aux_weight}"
            )
        encoder_name = str(cfg.get("pspnet_encoder_name", "resnet101")).strip().lower()
        if not encoder_name:
            raise ValueError("pspnet_encoder_name must be a non-empty string")
        encoder_weights_raw = cfg.get("pspnet_encoder_weights", None)
        if encoder_weights_raw is not None:
            encoder_weights = str(encoder_weights_raw).strip().lower()
            if encoder_weights not in {"imagenet", "none"}:
                raise ValueError(
                    "pspnet_encoder_weights must be one of ['imagenet', 'none'] when provided, "
                    f"got {encoder_weights!r}"
                )

    if model_name == "deconver":
        soft_label_loss = str(cfg.get("soft_label_loss", "ce")).strip().lower()
        if soft_label_loss != "ce":
            raise ValueError(f"soft_label_loss must be 'ce', got {soft_label_loss!r}")
        loss_variant = str(cfg.get("loss_variant", "soft_dice")).strip().lower()
        if loss_variant != "soft_dice":
            raise ValueError(f"loss_variant must be 'soft_dice', got {loss_variant!r}")

    confidence_threshold = float(cfg.get("confidence_threshold", 0.6))
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
    class_loss_weights = cfg.get("class_loss_weights", None)
    if class_loss_weights is not None:
        if not isinstance(class_loss_weights, list) or len(class_loss_weights) != 4:
            raise ValueError("class_loss_weights must be a list of 4 floats [w0,w1,w2,w3].")
        if any(float(x) <= 0.0 for x in class_loss_weights):
            raise ValueError("class_loss_weights entries must all be > 0.")

    for k in ["post_min_component_size_g3", "post_min_component_size_g4", "post_min_component_size_g5"]:
        v = int(cfg.get(k, 0))
        if v < 0:
            raise ValueError(f"{k} must be >= 0, got {v}")
    best_ckpt_metric_source = str(cfg.get("best_ckpt_metric_source", "post")).strip().lower()
    if best_ckpt_metric_source not in {"raw", "post"}:
        raise ValueError(
            f"best_ckpt_metric_source must be one of ['raw', 'post'], got {best_ckpt_metric_source!r}"
        )
    for k in ["val_min_g3_pos_images", "val_min_g4_pos_images", "val_min_g5_pos_images"]:
        v = int(cfg.get(k, 0))
        if v < 0:
            raise ValueError(f"{k} must be >= 0, got {v}")
    split_search_max_attempts = int(cfg.get("split_search_max_attempts", 100))
    if split_search_max_attempts < 1:
        raise ValueError(
            f"split_search_max_attempts must be >= 1, got {split_search_max_attempts}"
        )

    validate_metrics_config(cfg)

    transforms_profiles = cfg.get("transforms_profiles", None)
    if transforms_profiles is not None:
        if not isinstance(transforms_profiles, dict):
            raise ValueError("transforms_profiles must be a mapping of profile_name -> probability map.")
        for profile_name, probs in transforms_profiles.items():
            _validate_transform_profile_probs(profile_name=str(profile_name), probs=probs)

    transforms_profile = str(cfg.get("transforms_profile", "light")).strip().lower()
    if transforms_profiles is not None and transforms_profile not in transforms_profiles:
        raise ValueError(
            "transforms_profile must match one of transforms_profiles keys, "
            f"got {transforms_profile!r}"
        )

    transforms_prob = cfg.get("transforms_prob", None)
    if transforms_prob is not None and not isinstance(transforms_prob, dict):
        raise ValueError("transforms_prob must be a mapping of op_name -> probability.")
    if isinstance(transforms_prob, dict):
        for key, value in transforms_prob.items():
            if key not in _TRANSFORM_PROB_KEYS:
                raise ValueError(
                    f"Unsupported transforms_prob key {key!r}. Supported: {sorted(_TRANSFORM_PROB_KEYS)}"
                )
            p = float(value)
            if p < 0.0 or p > 1.0:
                raise ValueError(
                    f"transforms_prob[{key!r}] must be in [0,1], got {p}"
                )

    if "transforms_affine_rotate_range" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_affine_rotate_range"),
            key="transforms_affine_rotate_range",
            expected_len=1,
        )
    if "transforms_affine_translate_range" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_affine_translate_range"),
            key="transforms_affine_translate_range",
            expected_len=2,
        )
    if "transforms_affine_scale_range" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_affine_scale_range"),
            key="transforms_affine_scale_range",
            expected_len=2,
        )
    if "transforms_adjust_contrast_gamma" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_adjust_contrast_gamma"),
            key="transforms_adjust_contrast_gamma",
            expected_len=2,
        )
    if "transforms_gaussian_smooth_sigma_x" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_gaussian_smooth_sigma_x"),
            key="transforms_gaussian_smooth_sigma_x",
            expected_len=2,
        )
    if "transforms_gaussian_smooth_sigma_y" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_gaussian_smooth_sigma_y"),
            key="transforms_gaussian_smooth_sigma_y",
            expected_len=2,
        )
    if "transforms_shift_intensity_offsets" in cfg:
        _validate_fixed_len_numeric_sequence(
            cfg.get("transforms_shift_intensity_offsets"),
            key="transforms_shift_intensity_offsets",
            expected_len=2,
        )

    if "transforms_scale_intensity_factors" in cfg:
        float(cfg.get("transforms_scale_intensity_factors"))
    if "transforms_gaussian_noise_mean" in cfg:
        float(cfg.get("transforms_gaussian_noise_mean"))
    if "transforms_gaussian_noise_std" in cfg:
        noise_std = float(cfg.get("transforms_gaussian_noise_std"))
        if noise_std < 0.0:
            raise ValueError(f"transforms_gaussian_noise_std must be >= 0, got {noise_std}")

    if "transforms_gaussian_smooth_sigma_x" in cfg:
        sigma_x_min, sigma_x_max = (
            float(cfg["transforms_gaussian_smooth_sigma_x"][0]),
            float(cfg["transforms_gaussian_smooth_sigma_x"][1]),
        )
        if sigma_x_min < 0.0 or sigma_x_max < 0.0:
            raise ValueError(
                f"transforms_gaussian_smooth_sigma_x entries must be >= 0, got [{sigma_x_min}, {sigma_x_max}]"
            )
        if sigma_x_max < sigma_x_min:
            raise ValueError(
                "transforms_gaussian_smooth_sigma_x must satisfy [min, max] with max >= min."
            )

    if "transforms_gaussian_smooth_sigma_y" in cfg:
        sigma_y_min, sigma_y_max = (
            float(cfg["transforms_gaussian_smooth_sigma_y"][0]),
            float(cfg["transforms_gaussian_smooth_sigma_y"][1]),
        )
        if sigma_y_min < 0.0 or sigma_y_max < 0.0:
            raise ValueError(
                f"transforms_gaussian_smooth_sigma_y entries must be >= 0, got [{sigma_y_min}, {sigma_y_max}]"
            )
        if sigma_y_max < sigma_y_min:
            raise ValueError(
                "transforms_gaussian_smooth_sigma_y must satisfy [min, max] with max >= min."
            )

    if "transforms_shift_intensity_offsets" in cfg:
        shift_min, shift_max = (
            float(cfg["transforms_shift_intensity_offsets"][0]),
            float(cfg["transforms_shift_intensity_offsets"][1]),
        )
        if shift_max < shift_min:
            raise ValueError(
                "transforms_shift_intensity_offsets must satisfy [min, max] with max >= min."
            )

    amp_dtype_str = str(cfg.get("amp_dtype", "fp16")).strip().lower()
    if amp_dtype_str not in {"fp16", "bf16"}:
        raise ValueError(f"amp_dtype must be one of ['fp16', 'bf16'], got {amp_dtype_str!r}")

    wandb_mode = str(cfg.get("wandb_mode", "online")).strip().lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            f"wandb_mode must be one of ['online', 'offline', 'disabled'], got {wandb_mode!r}"
        )

    wandb_enabled = bool(cfg.get("wandb_enabled", True))
    if wandb_enabled and wandb_mode != "disabled":
        wandb_project = str(cfg.get("wandb_project", "")).strip()
        if not wandb_project:
            raise ValueError("wandb_project must be provided when W&B is enabled.")

    split_mode = str(cfg.get("split_mode", "iter_80_20")).strip().lower()
    if split_mode not in {"iter_80_20", "final_80_10_10"}:
        raise ValueError(
            f"split_mode must be one of ['iter_80_20', 'final_80_10_10'], got {split_mode!r}"
        )

    batch_size = int(cfg.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    val_batch_size = int(cfg.get("val_batch_size", batch_size))
    if val_batch_size <= 0:
        raise ValueError(f"val_batch_size must be > 0, got {val_batch_size}")

    num_workers = int(cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    random_seed = int(cfg.get("random_seed", 42))
    if random_seed < 0:
        raise ValueError(f"random_seed must be >= 0, got {random_seed}")

    exp_name = str(cfg.get("experiment_name", "")).strip()
    if exp_name:
        if "/" in exp_name or "\\" in exp_name:
            raise ValueError("experiment_name must not contain path separators ('/' or '\\').")
        if exp_name in {".", ".."}:
            raise ValueError("experiment_name cannot be '.' or '..'.")

    base_output_dir = str(cfg.get("base_output_dir", "")).strip()
    if not base_output_dir:
        raise ValueError("base_output_dir must be a non-empty string.")
    if require_paths:
        data_root = Path(str(cfg["data_root"]))
        consensus_root = Path(str(cfg["consensus_root"]))
        if not data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {data_root}")
        if not consensus_root.exists():
            raise FileNotFoundError(f"consensus_root does not exist: {consensus_root}")

    if for_eval:
        split_manifest_cfg = str(cfg.get("split_manifest_path", "")).strip()
        if not split_manifest_cfg:
            default_manifest = Path(str(cfg["base_output_dir"])).parent / "splits" / "gleason_consensus_split.json"
            if not default_manifest.exists():
                raise FileNotFoundError(
                    "Split manifest missing. Expected config split_manifest_path or default path: "
                    f"{default_manifest}"
                )


def validate_amp_runtime(cfg: dict, device: torch.device) -> torch.dtype:
    amp_dtype_str = str(cfg.get("amp_dtype", "fp16")).strip().lower()
    dtype_map: dict[str, torch.dtype] = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    amp_dtype = dtype_map[amp_dtype_str]

    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    if use_amp and amp_dtype == torch.bfloat16:
        bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
        bf16_supported = bool(bf16_supported_fn()) if callable(bf16_supported_fn) else False
        if not bf16_supported:
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_index)
            raise RuntimeError(
                "amp_dtype=bf16 requested, but detected device does not support BF16 autocast: "
                f"{gpu_name}. Set `amp_dtype: fp16` for older GPUs."
            )

    return amp_dtype


__all__ = [
    "validate_deconver_config",
    "validate_amp_runtime",
]
