from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_generate_consensus_gt_viz_script_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    consensus_root = data_root / "consensus"
    (data_root / "Train_imgs").mkdir(parents=True)
    case_dir = consensus_root / "case001"
    case_dir.mkdir(parents=True)

    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[..., 0] = 120
    Image.fromarray(img).save(data_root / "Train_imgs" / "case001.jpg")

    hard = np.zeros((24, 24), dtype=np.uint8)
    hard[5:19, 5:19] = 2
    Image.fromarray(hard).save(case_dir / "consensus_hard_mask.png")

    ignore = np.zeros((24, 24), dtype=np.uint8)
    ignore[:3, :] = 1
    Image.fromarray(ignore).save(case_dir / "ignore_mask.png")

    probs = np.zeros((4, 24, 24), dtype=np.float32)
    probs[0] = 1.0
    np.savez(case_dir / "consensus_probs_compact.npz", probs=probs)

    out_dir = tmp_path / "viz_out"
    cmd = [
        sys.executable,
        "-m", "src.cli.generate_consensus_gt_viz",
        "--data-root",
        str(data_root),
        "--consensus-root",
        str(consensus_root),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    overlay = out_dir / "overlay" / "0001_case001.webp"
    panel = out_dir / "panel" / "0001_case001.jpg"
    summary = out_dir / "summary.json"

    assert overlay.exists()
    assert panel.exists()
    assert summary.exists()

    with summary.open("r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["count_written"] == 1
    assert report["count_selected"] == 1
