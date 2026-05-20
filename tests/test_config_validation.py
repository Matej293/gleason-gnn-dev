from __future__ import annotations

import pytest

from src.common.config_validation import validate_deconver_config

_DEFAULT_TRANSFORM_PROFILES = {
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


def _base_cfg() -> dict:
    return {
        "model": "deconver",
        "spatial_dims": 2,
        "out_channels": 4,
        "data_root": "data",
        "consensus_root": "data/consensus",
        "base_output_dir": "outputs/runs",
        "image_subdirs": ["Train_imgs"],
        "resize_short_side": 1024,
        "train_crop_enabled": True,
        "train_crop_size": [800, 800],
        "train_resize_random_scale_enabled": False,
        "train_resize_random_scale_min": 0.9,
        "train_resize_random_scale_max": 1.1,
        "inference_resize_short_side": 1024,
        "inference_mode": "resized_full",
        "resized_sliding_window_patch_size": [800, 800],
        "resized_sliding_window_overlap": 0.25,
        "deconver_strides": [1, 2, 2, 2],
        "input_channels": 3,
        "soft_label_loss": "ce",
        "loss_variant": "soft_dice",
        "wandb_enabled": False,
        "transforms_enabled": False,
        "transforms_profile": "light",
        "transforms_seed_sync": True,
        "transforms_profiles": _DEFAULT_TRANSFORM_PROFILES,
        "transforms_prob": {},
        "transforms_affine_rotate_range": [0.12],
        "transforms_affine_translate_range": [32, 32],
        "transforms_affine_scale_range": [0.08, 0.08],
        "transforms_scale_intensity_factors": 0.10,
        "transforms_adjust_contrast_gamma": [0.85, 1.15],
        "transforms_gaussian_noise_mean": 0.0,
        "transforms_gaussian_noise_std": 0.03,
        "transforms_gaussian_smooth_sigma_x": [0.25, 1.00],
        "transforms_gaussian_smooth_sigma_y": [0.25, 1.00],
        "transforms_shift_intensity_offsets": [-0.08, 0.08],
    }


def test_validate_rejects_legacy_pspnet_name() -> None:
    cfg = _base_cfg()
    cfg["model"] = "legacy_pspnet"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_accepts_valid_aux_weight() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_aux_weight"] = 0.5
    validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_rejects_invalid_aux_weight() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_aux_weight"] = 1.5
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_rejects_invalid_input_channels() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["input_channels"] = 1
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_rejects_invalid_encoder_weights() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_encoder_weights"] = "random_init"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_rejects_invalid_loss_mode() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_loss_mode"] = "invalid"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_transforms_accepts_valid_profile_and_probs() -> None:
    cfg = _base_cfg()
    cfg["transforms_enabled"] = True
    cfg["transforms_profile"] = "light"
    cfg["transforms_prob"] = {
        "flip_h": 0.5,
        "flip_v": 0.5,
        "rotate90": 0.2,
        "affine": 0.1,
        "scale_intensity": 0.2,
        "adjust_contrast": 0.1,
        "gaussian_noise": 0.1,
        "gaussian_smooth": 0.1,
        "shift_intensity": 0.1,
    }
    validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_transforms_rejects_unknown_prob_key() -> None:
    cfg = _base_cfg()
    cfg["transforms_prob"] = {"unknown_op": 0.3}
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_rejects_removed_native_keys_with_migration_message() -> None:
    cfg = _base_cfg()
    cfg["patch_size"] = [512, 512]
    with pytest.raises(ValueError, match="removed in the resized-only migration"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_rejects_invalid_inference_mode() -> None:
    cfg = _base_cfg()
    cfg["inference_mode"] = "native"
    with pytest.raises(ValueError, match="inference_mode must be one of"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_rejects_missing_random_scale_bounds_when_enabled() -> None:
    cfg = _base_cfg()
    cfg["train_resize_random_scale_enabled"] = True
    cfg.pop("train_resize_random_scale_min")
    with pytest.raises(ValueError, match="requires train_resize_random_scale_min"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_metrics_block_accepts_supported_keys() -> None:
    cfg = _base_cfg()
    cfg["metrics"] = {
        "track_keys": ["macro_dice", "challenge_score", "weighted_macro_dice", "hd95_mean", "asd_mean"],
        "best_checkpoint_metric": "challenge_score",
        "best_checkpoint_source": "raw",
        "include_boundary_metrics": True,
        "boundary": {
            "hausdorff_variant": "hd95",
            "hausdorff_percentile": 95,
            "include_background": False,
            "symmetric_asd": True,
        },
    }
    validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_metrics_block_rejects_unsupported_track_key() -> None:
    cfg = _base_cfg()
    cfg["metrics"] = {
        "track_keys": ["macro_dice", "not_a_metric"],
        "best_checkpoint_metric": "challenge_score",
    }
    with pytest.raises(ValueError, match="metrics.track_keys contains unsupported keys"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_metrics_block_rejects_missing_challenge_score_for_challenge_selection() -> None:
    cfg = _base_cfg()
    cfg["metrics"] = {
        "track_keys": ["macro_dice", "weighted_macro_dice"],
        "best_checkpoint_metric": "challenge_score",
    }
    with pytest.raises(ValueError, match="must contain 'challenge_score'"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_metrics_block_rejects_invalid_boundary_percentile() -> None:
    cfg = _base_cfg()
    cfg["metrics"] = {
        "track_keys": ["macro_dice", "challenge_score"],
        "best_checkpoint_metric": "challenge_score",
        "boundary": {
            "hausdorff_percentile": 120,
        },
    }
    with pytest.raises(ValueError, match="hausdorff_percentile"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)
