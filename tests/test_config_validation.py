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
