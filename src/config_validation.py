from __future__ import annotations

from pathlib import Path

import torch


def _require_keys(cfg: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


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
        ],
    )

    model_name = str(cfg.get("model", "")).strip().lower()
    if model_name not in {"deconver", "unet_lite", "pspnet_gleason"}:
        raise ValueError(
            f"Expected model in ['deconver', 'unet_lite', 'pspnet_gleason'], got {model_name!r}"
        )

    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(f"Expected spatial_dims=2, got {spatial_dims}")

    out_channels = int(cfg.get("out_channels", 4))
    if out_channels != 4:
        raise ValueError(f"Expected out_channels=4 for Gleason consensus, got {out_channels}")

    input_channels = int(cfg.get("input_channels", 3))
    if input_channels <= 0:
        raise ValueError(f"input_channels must be > 0, got {input_channels}")
    if model_name == "unet_lite":
        base_channels = int(cfg.get("unet_lite_base_channels", 32))
        if base_channels <= 0:
            raise ValueError(
                f"unet_lite_base_channels must be > 0, got {base_channels}"
            )
    if model_name == "pspnet_gleason":
        if input_channels != 3:
            raise ValueError(
                f"pspnet_gleason requires input_channels=3, got {input_channels}"
            )
        pspnet_loss_mode = str(cfg.get("pspnet_loss_mode", "consensus")).strip().lower()
        if pspnet_loss_mode not in {"consensus", "gleason_ce", "gleason_ce_soft"}:
            raise ValueError(
                "pspnet_loss_mode must be one of ['consensus', 'gleason_ce', 'gleason_ce_soft'], "
                f"got {pspnet_loss_mode!r}"
            )
        pspnet_soft_term = str(cfg.get("pspnet_soft_term", "ce")).strip().lower()
        if pspnet_soft_term not in {"ce", "kl"}:
            raise ValueError(
                "pspnet_soft_term must be one of ['ce', 'kl'], "
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

    soft_label_loss = str(cfg.get("soft_label_loss", "ce")).strip().lower()
    if soft_label_loss not in {"ce", "kl"}:
        raise ValueError(f"soft_label_loss must be 'ce' or 'kl', got {soft_label_loss!r}")
    loss_variant = str(cfg.get("loss_variant", "soft_dice")).strip().lower()
    if loss_variant not in {"soft_dice", "focal_dice", "tversky_dice"}:
        raise ValueError(
            "loss_variant must be one of ['soft_dice','focal_dice','tversky_dice'], "
            f"got {loss_variant!r}"
        )

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

    transforms_enabled = bool(cfg.get("transforms_enabled", False))
    transforms_profile = str(cfg.get("transforms_profile", "light")).strip().lower()
    if transforms_profile not in {"light", "medium", "strong"}:
        raise ValueError(
            "transforms_profile must be one of ['light', 'medium', 'strong'], "
            f"got {transforms_profile!r}"
        )

    transforms_patch = cfg.get("transforms_patch_size", None)
    if transforms_patch is not None:
        if not isinstance(transforms_patch, (list, tuple)) or len(transforms_patch) != 2:
            raise ValueError("transforms_patch_size must be a 2-item list/tuple [H, W].")
        patch_h = int(transforms_patch[0])
        patch_w = int(transforms_patch[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"transforms_patch_size entries must be > 0, got [{patch_h}, {patch_w}]"
            )

    transforms_prob = cfg.get("transforms_prob", None)
    if transforms_prob is not None:
        if not isinstance(transforms_prob, dict):
            raise ValueError("transforms_prob must be a mapping of op_name -> probability.")
        allowed_prob_keys = {
            "flip_h",
            "flip_v",
            "rotate90",
            "affine",
            "crop",
            "scale_intensity",
            "adjust_contrast",
            "gaussian_noise",
        }
        for key, value in transforms_prob.items():
            if key not in allowed_prob_keys:
                raise ValueError(
                    f"Unsupported transforms_prob key {key!r}. Supported: {sorted(allowed_prob_keys)}"
                )
            p = float(value)
            if p < 0.0 or p > 1.0:
                raise ValueError(
                    f"transforms_prob[{key!r}] must be in [0,1], got {p}"
                )

    if transforms_enabled:
        crop_p = float(transforms_prob.get("crop", 0.0)) if isinstance(transforms_prob, dict) else 0.0
        if crop_p > 0.0 and transforms_patch is None:
            raise ValueError(
                "transforms_prob['crop'] > 0 requires transforms_patch_size=[H, W]."
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
