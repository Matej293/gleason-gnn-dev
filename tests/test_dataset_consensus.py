from __future__ import annotations

import numpy as np
from PIL import Image

from src.gleason_consensus_dataset import GleasonConsensusDataset


def test_dataset_discovery_and_shapes(tmp_path):
    data_root = tmp_path / "data"
    consensus_root = tmp_path / "consensus"
    (data_root / "Train_imgs").mkdir(parents=True)
    case_dir = consensus_root / "case001"
    case_dir.mkdir(parents=True)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = np.arange(32, dtype=np.uint8)[:, None]
    Image.fromarray(img).save(data_root / "Train_imgs" / "case001.jpg")

    hard = np.zeros((32, 32), dtype=np.uint8)
    hard[8:16, 8:16] = 2
    Image.fromarray(hard).save(case_dir / "consensus_hard_mask.png")

    ignore = np.zeros((32, 32), dtype=np.uint8)
    Image.fromarray(ignore).save(case_dir / "ignore_mask.png")

    probs = np.zeros((4, 32, 32), dtype=np.float32)
    probs[0, ...] = 1.0
    probs[2, 8:16, 8:16] = 1.0
    probs[0, 8:16, 8:16] = 0.0
    np.savez(case_dir / "consensus_probs_compact.npz", probs=probs)

    ds = GleasonConsensusDataset(data_root=data_root, consensus_root=consensus_root)
    sample = ds[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["soft_probs"].shape == (4, 32, 32)
    assert sample["hard_mask"].shape == (32, 32)
    assert sample["ignore_mask"].shape == (32, 32)
