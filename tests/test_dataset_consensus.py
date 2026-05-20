from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.gleason_consensus_dataset import GleasonConsensusDataset, SlidingWindowPatchDataset

def _write_case(
    *,
    data_root,
    consensus_root,
    case_id: str,
    height: int,
    width: int,
) -> None:
    (data_root / "Train_imgs").mkdir(parents=True, exist_ok=True)
    case_dir = consensus_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[..., 0] = np.arange(height, dtype=np.uint8)[:, None]
    img[..., 1] = np.arange(width, dtype=np.uint8)[None, :]
    img[..., 2] = 127
    Image.fromarray(img).save(data_root / "Train_imgs" / f"{case_id}.jpg")

    hard = np.zeros((height, width), dtype=np.uint8)
    hard[height // 4 : height // 2, width // 4 : width // 2] = 2
    Image.fromarray(hard).save(case_dir / "consensus_hard_mask.png")

    ignore = np.zeros((height, width), dtype=np.uint8)
    Image.fromarray(ignore).save(case_dir / "ignore_mask.png")

    probs = np.zeros((4, height, width), dtype=np.float32)
    probs[0, ...] = 1.0
    probs[2, hard == 2] = 1.0
    probs[0, hard == 2] = 0.0
    np.savez(case_dir / "consensus_probs_compact.npz", probs=probs)

def test_dataset_discovery_and_shapes(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=32,
        width=32,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)
    sample = ds[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["soft_probs"].shape == (4, 32, 32)
    assert sample["hard_mask"].shape == (32, 32)
    assert sample["ignore_mask"].shape == (32, 32)

def test_dataset_preserves_original_resolution_no_downscale(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=700,
        width=650,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)
    sample = ds[0]
    assert sample["image"].shape == (3, 700, 650)
    assert sample["soft_probs"].shape == (4, 700, 650)
    assert sample["hard_mask"].shape == (700, 650)
    assert sample["ignore_mask"].shape == (700, 650)

def test_sliding_window_patch_dataset_uses_512_with_50pct_overlap(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=700,
        width=650,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)
    patch_ds = SlidingWindowPatchDataset(
        base_dataset=ds,
        source_indices=[0],
        patch_size=(512, 512),
        overlap=0.5,
    )

    assert patch_ds.scan_interval == (256, 256)
    assert len(patch_ds) == 4

    sample = patch_ds[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["soft_probs"].shape == (4, 512, 512)
    assert sample["hard_mask"].shape == (512, 512)
    assert sample["ignore_mask"].shape == (512, 512)

def test_sliding_window_patch_dataset_small_image_returns_single_patch(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=300,
        width=300,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)
    patch_ds = SlidingWindowPatchDataset(
        base_dataset=ds,
        source_indices=[0],
        patch_size=(512, 512),
        overlap=0.5,
    )

    assert len(patch_ds) == 1
    sample = patch_ds[0]
    assert sample["image"].shape == (3, 300, 300)
    assert sample["hard_mask"].shape == (300, 300)

def test_sliding_window_patch_dataset_filters_non_tissue_windows(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=64,
        width=64,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)

    tissue = np.zeros((64, 64), dtype=np.uint8)
    tissue[32:, 32:] = 1

    def _inject_tissue(sample: dict) -> dict:
        out = dict(sample)
        out["tissue_mask"] = torch.from_numpy(tissue.copy())
        return out

    ds.transform = _inject_tissue

    patch_ds_zero = SlidingWindowPatchDataset(
        base_dataset=ds,
        source_indices=[0],
        patch_size=(32, 32),
        overlap=0.5,
        patch_tissue_filter_enabled=True,
        patch_min_tissue_fraction=0.0,
    )
    assert patch_ds_zero.total_candidate_patches == 9
    assert patch_ds_zero.kept_patches == 4
    assert patch_ds_zero.skipped_patches == 5
    assert len(patch_ds_zero) == 4

    patch_ds_nonzero = SlidingWindowPatchDataset(
        base_dataset=ds,
        source_indices=[0],
        patch_size=(32, 32),
        overlap=0.5,
        patch_tissue_filter_enabled=True,
        patch_min_tissue_fraction=0.4,
    )
    assert patch_ds_nonzero.total_candidate_patches == 9
    assert patch_ds_nonzero.kept_patches == 3
    assert patch_ds_nonzero.skipped_patches == 6
    assert len(patch_ds_nonzero) == 3

    patch_ds_disabled = SlidingWindowPatchDataset(
        base_dataset=ds,
        source_indices=[0],
        patch_size=(32, 32),
        overlap=0.5,
        patch_tissue_filter_enabled=False,
        patch_min_tissue_fraction=0.4,
    )
    assert patch_ds_disabled.total_candidate_patches == 9
    assert patch_ds_disabled.kept_patches == 9
    assert patch_ds_disabled.skipped_patches == 0
    assert len(patch_ds_disabled) == 9

    sample = patch_ds_nonzero[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["hard_mask"].shape == (32, 32)

def test_sliding_window_patch_dataset_rejects_invalid_patch_min_tissue_fraction(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=64,
        width=64,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)

    with pytest.raises(ValueError, match="patch_min_tissue_fraction"):
        SlidingWindowPatchDataset(
            base_dataset=ds,
            source_indices=[0],
            patch_size=(32, 32),
            overlap=0.5,
            patch_min_tissue_fraction=-0.01,
        )

    with pytest.raises(ValueError, match="patch_min_tissue_fraction"):
        SlidingWindowPatchDataset(
            base_dataset=ds,
            source_indices=[0],
            patch_size=(32, 32),
            overlap=0.5,
            patch_min_tissue_fraction=1.01,
        )

def test_patch_index_build_skips_prob_loading_without_base_transform(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    _write_case(
        data_root=data_root,
        consensus_root=consensus_root,
        case_id="case001",
        height=64,
        width=64,
    )

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)

    import src.gleason_consensus_dataset as dataset_mod

    monkeypatch.setattr(
        dataset_mod,
        "build_tissue_mask_from_image",
        lambda image_rgb, close_radius=3, min_object_size=4096, min_hole_size=4096: np.ones(
            image_rgb.shape[:2],
            dtype=np.uint8,
        ),
    )

    def _fail_load_probs(self, path, image_id):
        raise AssertionError("_load_probs should not be called during patch-index precompute")

    monkeypatch.setattr(GleasonConsensusDataset, "_load_probs", _fail_load_probs)

    patch_ds = SlidingWindowPatchDataset(
        base_dataset=ds,
        source_indices=[0],
        patch_size=(32, 32),
        overlap=0.5,
        patch_tissue_filter_enabled=True,
        patch_min_tissue_fraction=0.0,
    )

    assert len(patch_ds) > 0

