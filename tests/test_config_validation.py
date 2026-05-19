from __future__ import annotations

import pytest

from src.config_validation import validate_deconver_config


def _base_cfg() -> dict:
    return {
        "model": "deconver",
        "spatial_dims": 2,
        "out_channels": 4,
        "data_root": "data",
        "consensus_root": "data/consensus",
        "base_output_dir": "outputs/runs",
        "input_channels": 3,
        "soft_label_loss": "ce",
        "loss_variant": "soft_dice",
        "wandb_enabled": False,
    }


def test_validate_pspnet_gleason_accepts_valid_aux_weight() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_aux_weight"] = 0.5
    validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_rejects_invalid_aux_weight() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_aux_weight"] = 1.5
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_rejects_invalid_input_channels() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["input_channels"] = 1
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_rejects_invalid_encoder_weights() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_encoder_weights"] = "random_init"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_rejects_invalid_loss_mode() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_loss_mode"] = "invalid"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_accepts_gleason_ce_loss_mode() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_loss_mode"] = "gleason_ce"
    validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_accepts_gleason_ce_soft_loss_mode() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_loss_mode"] = "gleason_ce_soft"
    cfg["pspnet_soft_term"] = "ce"
    cfg["pspnet_soft_weight"] = 0.2
    validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_rejects_invalid_soft_term() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
    cfg["pspnet_soft_term"] = "bad"
    with pytest.raises(ValueError):
        validate_deconver_config(cfg, for_eval=False, require_paths=False)


def test_validate_pspnet_gleason_rejects_negative_soft_weight() -> None:
    cfg = _base_cfg()
    cfg["model"] = "pspnet_gleason"
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
