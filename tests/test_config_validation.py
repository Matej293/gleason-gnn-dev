from __future__ import annotations

import pytest

from src.config_validation import validate_deconver_config

_DEFAULT_TRANSFORM_PROFILES = {
    "light": {
        "flip_h": 0.50,
        "flip_v": 0.50,
        "rotate90": 0.20,
        "affine": 0.15,
        "crop": 0.00,
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
        "crop": 0.00,
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
        "crop": 0.00,
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
        "patch_size": [512, 512],
        "patch_overlap": 0.5,
        "deconver_strides": [1, 2, 2, 2],
        "input_channels": 3,
        "soft_label_loss": "ce",
        "loss_variant": "soft_dice",
        "wandb_enabled": False,
        "transforms_enabled": False,
        "transforms_profile": "light",
        "transforms_seed_sync": True,
        "transforms_patch_size": None,
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

def test_validate_pspnet_accepts_gleason_ce_loss_mode() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_loss_mode"] = "gleason_ce"
    validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_pspnet_accepts_gleason_ce_soft_loss_mode() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_loss_mode"] = "gleason_ce_soft"
    cfg["pspnet_soft_term"] = "ce"
    cfg["pspnet_soft_weight"] = 0.2
    validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_pspnet_rejects_invalid_soft_term() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_soft_term"] = "bad"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_pspnet_rejects_negative_soft_weight() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet"
    cfg["pspnet_soft_weight"] = -0.1
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
        "crop": 0.0,
        "scale_intensity": 0.2,
        "adjust_contrast": 0.1,
        "gaussian_noise": 0.1,
        "gaussian_smooth": 0.1,
        "shift_intensity": 0.1,
    }
    validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_transforms_rejects_invalid_profile() -> None:
    cfg = _base_cfg()
    cfg["transforms_profile"] = "extreme"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_transforms_rejects_unknown_prob_key() -> None:
    cfg = _base_cfg()
    cfg["transforms_prob"] = {"unknown_op": 0.3}
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_transforms_rejects_crop_without_patch_size() -> None:
    cfg = _base_cfg()
    cfg["transforms_enabled"] = True
    cfg["transforms_prob"] = {"crop": 0.1}
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_transforms_rejects_negative_gaussian_smooth_sigma() -> None:
    cfg = _base_cfg()
    cfg["transforms_gaussian_smooth_sigma_x"] = [-0.1, 1.0]
    with pytest.raises(ValueError, match="transforms_gaussian_smooth_sigma_x entries must be >= 0"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_transforms_rejects_inverted_shift_offsets() -> None:
    cfg = _base_cfg()
    cfg["transforms_shift_intensity_offsets"] = [0.1, -0.1]
    with pytest.raises(
        ValueError,
        match=r"transforms_shift_intensity_offsets must satisfy \[min, max\] with max >= min",
    ):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_transforms_rejects_missing_shift_offsets_key() -> None:
    cfg = _base_cfg()
    cfg.pop("transforms_shift_intensity_offsets")
    with pytest.raises(ValueError, match="Missing required config keys"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_rejects_invalid_patch_size() -> None:
    cfg = _base_cfg()
    cfg["patch_size"] = [512]
    with pytest.raises(ValueError, match="patch_size must be a 2-item"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_rejects_invalid_patch_overlap() -> None:
    cfg = _base_cfg()
    cfg["patch_overlap"] = 1.0
    with pytest.raises(ValueError, match=r"patch_overlap must be in \[0\.0, 1\.0\)"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_patch_tissue_filter_defaults_when_keys_omitted() -> None:
    cfg = _base_cfg()
    cfg.pop("patch_tissue_filter_enabled", None)
    cfg.pop("patch_min_tissue_fraction", None)
    validate_deconver_config(cfg, for_eval=False, require_paths=False)

def test_validate_rejects_invalid_patch_min_tissue_fraction() -> None:
    cfg = _base_cfg()
    cfg["patch_min_tissue_fraction"] = -0.01
    with pytest.raises(ValueError, match=r"patch_min_tissue_fraction must be in \[0\.0, 1\.0\]"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)

    cfg = _base_cfg()
    cfg["patch_min_tissue_fraction"] = 1.01
    with pytest.raises(ValueError, match=r"patch_min_tissue_fraction must be in \[0\.0, 1\.0\]"):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)
