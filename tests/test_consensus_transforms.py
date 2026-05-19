from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import (
    consensus_dataset_kwargs_from_config,
    consensus_train_val_transforms_from_config,
)
from src.consensus_transforms import build_consensus_train_transform
from src.eval_utils import collate_consensus_batch
from src.gleason_consensus_dataset import GleasonConsensusDataset
from src.train_deconver import _consensus_loss


def _make_toy_dataset(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    (data_root / "Train_imgs").mkdir(parents=True)
    case_dir = consensus_root / "case001"
    case_dir.mkdir(parents=True)

    h, w = 48, 64
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    image[..., 1] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    image[..., 2] = 120
    Image.fromarray(image).save(data_root / "Train_imgs" / "case001.jpg")

    hard = np.zeros((h, w), dtype=np.uint8)
    hard[8:24, 8:28] = 1
    hard[12:30, 30:46] = 2
    hard[30:40, 42:58] = 3
    Image.fromarray(hard).save(case_dir / "consensus_hard_mask.png")

    ignore = np.zeros((h, w), dtype=np.uint8)
    ignore[:4, :] = 1
    ignore[:, :3] = 1
    Image.fromarray(ignore).save(case_dir / "ignore_mask.png")

    probs = np.zeros((4, h, w), dtype=np.float32)
    probs[0, ...] = 1.0
    probs[1, hard == 1] = 1.0
    probs[2, hard == 2] = 1.0
    probs[3, hard == 3] = 1.0
    probs[0, hard > 0] = 0.0
    np.savez(case_dir / "consensus_probs_compact.npz", probs=probs)

    return data_root, consensus_root


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


def _enabled_cfg() -> dict:
    return {
        "transforms_enabled": True,
        "transforms_profile": "light",
        "transforms_patch_size": None,
        "transforms_profiles": _DEFAULT_TRANSFORM_PROFILES,
        "transforms_prob": {
            "flip_h": 1.0,
            "flip_v": 1.0,
            "rotate90": 1.0,
            "affine": 0.0,
            "crop": 0.0,
            "scale_intensity": 0.0,
            "adjust_contrast": 0.0,
            "gaussian_noise": 0.0,
            "gaussian_smooth": 0.0,
            "shift_intensity": 0.0,
        },
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


def _find_transform_by_name(transform: object, name: str) -> object:
    ops = getattr(transform, "transforms", None)
    if not isinstance(ops, (list, tuple)):
        raise AssertionError("Expected MONAI Compose with transforms list.")
    for op in ops:
        if op.__class__.__name__ == name:
            return op
    raise AssertionError(f"Transform {name!r} not found in pipeline.")


def test_dataset_transform_keeps_contract(tmp_path: Path) -> None:
    data_root, consensus_root = _make_toy_dataset(tmp_path)
    transform = build_consensus_train_transform(_enabled_cfg())
    ds = GleasonConsensusDataset(
        data_root=data_root,
        consensus_root=consensus_root,
        transform=transform,
    )

    sample = ds[0]
    assert {"image", "soft_probs", "hard_mask", "ignore_mask", "tissue_mask", "image_id"}.issubset(sample.keys())
    assert sample["image"].dtype == torch.float32
    assert sample["soft_probs"].dtype == torch.float32
    assert sample["hard_mask"].dtype == torch.int64
    assert sample["ignore_mask"].dtype == torch.uint8
    assert sample["tissue_mask"].dtype == torch.uint8

    h, w = sample["hard_mask"].shape
    assert sample["image"].shape == (3, h, w)
    assert sample["soft_probs"].shape == (4, h, w)
    assert sample["ignore_mask"].shape == (h, w)
    assert sample["tissue_mask"].shape == (h, w)


def test_soft_probs_remain_valid_after_transform(tmp_path: Path) -> None:
    data_root, consensus_root = _make_toy_dataset(tmp_path)
    ds = GleasonConsensusDataset(
        data_root=data_root,
        consensus_root=consensus_root,
        transform=build_consensus_train_transform(_enabled_cfg()),
    )

    probs = ds[0]["soft_probs"]
    assert torch.isfinite(probs).all()
    assert float(probs.min().item()) >= 0.0
    sums = probs.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_ignore_and_tissue_masks_remain_binary_and_aligned(tmp_path: Path) -> None:
    data_root, consensus_root = _make_toy_dataset(tmp_path)
    ds = GleasonConsensusDataset(
        data_root=data_root,
        consensus_root=consensus_root,
        transform=build_consensus_train_transform(_enabled_cfg()),
    )

    sample = ds[0]
    hard = sample["hard_mask"]
    ignore = sample["ignore_mask"]
    tissue = sample["tissue_mask"]

    assert ignore.shape == hard.shape
    assert tissue.shape == hard.shape
    assert set(torch.unique(ignore).tolist()).issubset({0, 1})
    assert set(torch.unique(tissue).tolist()).issubset({0, 1})


def test_transform_enabled_loss_smoke(tmp_path: Path) -> None:
    data_root, consensus_root = _make_toy_dataset(tmp_path)
    ds = GleasonConsensusDataset(
        data_root=data_root,
        consensus_root=consensus_root,
        transform=build_consensus_train_transform(_enabled_cfg()),
    )

    batch = collate_consensus_batch([ds[0]])
    b, h, w = batch["hard_mask"].shape
    logits = torch.randn((b, 4, h, w), dtype=torch.float32)

    loss, stats = _consensus_loss(
        outputs=logits,
        hard_mask=batch["hard_mask"],
        soft_probs=batch["soft_probs"],
        ignore_mask=batch["ignore_mask"],
        sample_weights=torch.ones((b,), dtype=torch.float32),
        class_weights=torch.ones((4,), dtype=torch.float32),
        use_confidence_mask=False,
        confidence_threshold=0.0,
        soft_loss_type="ce",
        loss_variant="soft_dice",
        lambda_soft=1.0,
        lambda_dice=1.0,
        include_background_in_dice=False,
        exclude_absent_classes_in_dice_loss=True,
    )

    assert torch.isfinite(loss)
    assert 0.0 <= stats["valid_fraction"] <= 1.0


def test_transforms_disabled_keeps_interface_noop(tmp_path: Path) -> None:
    data_root, consensus_root = _make_toy_dataset(tmp_path)
    cfg = {
        "model": "deconver",
        "data_root": str(data_root),
        "consensus_root": str(consensus_root),
        "image_subdirs": ["Train_imgs"],
        "deconver_strides": [1, 2, 2, 2],
        "transforms_enabled": False,
        "renormalize_probs": True,
        "enforce_background_ignore": True,
    }

    train_t, val_t = consensus_train_val_transforms_from_config(cfg)
    assert train_t is None
    assert val_t is None

    ds_a = GleasonConsensusDataset(**consensus_dataset_kwargs_from_config(cfg))
    ds_b = GleasonConsensusDataset(**consensus_dataset_kwargs_from_config(cfg, transform=train_t))

    a = ds_a[0]
    b = ds_b[0]
    for key in ["image", "soft_probs", "hard_mask", "ignore_mask", "tissue_mask"]:
        assert torch.equal(a[key], b[key])


def test_new_augmentation_ops_use_profile_probs_when_not_overridden() -> None:
    cfg = _enabled_cfg()
    cfg["transforms_prob"] = {}
    transform = build_consensus_train_transform(cfg)
    assert transform is not None

    smooth = _find_transform_by_name(transform, "RandGaussianSmoothd")
    shift = _find_transform_by_name(transform, "RandShiftIntensityd")
    assert float(smooth.prob) == pytest.approx(0.05)
    assert float(shift.prob) == pytest.approx(0.05)


def test_new_augmentation_ops_respect_prob_overrides() -> None:
    cfg = _enabled_cfg()
    cfg["transforms_prob"]["gaussian_smooth"] = 1.0
    cfg["transforms_prob"]["shift_intensity"] = 1.0
    transform = build_consensus_train_transform(cfg)
    assert transform is not None

    smooth = _find_transform_by_name(transform, "RandGaussianSmoothd")
    shift = _find_transform_by_name(transform, "RandShiftIntensityd")
    assert float(smooth.prob) == pytest.approx(1.0)
    assert float(shift.prob) == pytest.approx(1.0)


def test_new_augmentation_ops_reject_invalid_sigma_range() -> None:
    cfg = _enabled_cfg()
    cfg["transforms_gaussian_smooth_sigma_x"] = [-0.1, 1.0]
    with pytest.raises(ValueError, match="transforms_gaussian_smooth_sigma_x entries must be >= 0"):
        build_consensus_train_transform(cfg)


def test_new_augmentation_ops_reject_inverted_shift_offsets() -> None:
    cfg = _enabled_cfg()
    cfg["transforms_shift_intensity_offsets"] = [0.1, -0.1]
    with pytest.raises(
        ValueError,
        match=r"transforms_shift_intensity_offsets must satisfy \[min, max\] with max >= min",
    ):
        build_consensus_train_transform(cfg)

