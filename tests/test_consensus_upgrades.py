from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.consensus_builder.pipeline import ConsensusConfig, ConsensusMaskBuilder
from src.consensus_builder.postprocess import (
    boundary_disagreement_penalty,
    make_ignore_mask_with_threshold,
)


def _save_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _load_probs(path: Path) -> np.ndarray:
    d = np.load(path)
    return d["probs"]


def test_weighted_fusion_changes_output_when_downweighted_rater_perturbed() -> None:
    sem_a = np.zeros((8, 8), dtype=np.uint8)
    sem_b = np.zeros((8, 8), dtype=np.uint8)
    sem_b[:, 4:] = 3

    cfg = ConsensusConfig(consensus_fusion_mode="weighted", strict_ignore=False)
    builder = ConsensusMaskBuilder(cfg)

    hard_hi, probs_hi, _, _ = builder._run_consensus(
        sem_maps={"p1": sem_a, "p2": sem_b},
        statuses={"p1": "keep", "p2": "down_weight"},
        weights={"p1": 1.0, "p2": 0.8},
    )
    hard_lo, probs_lo, _, _ = builder._run_consensus(
        sem_maps={"p1": sem_a, "p2": sem_b},
        statuses={"p1": "keep", "p2": "down_weight"},
        weights={"p1": 1.0, "p2": 0.05},
    )

    assert probs_hi.shape == probs_lo.shape == (4, 8, 8)
    assert np.mean(probs_hi[3]) > np.mean(probs_lo[3])
    assert int((hard_hi == 3).sum()) >= int((hard_lo == 3).sum())


def test_ignore_threshold_and_boundary_penalty_are_deterministic() -> None:
    conf = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    ignore1 = make_ignore_mask_with_threshold(conf, threshold=0.5)
    ignore2 = make_ignore_mask_with_threshold(conf, threshold=0.5)
    assert np.array_equal(ignore1, ignore2)

    hard = np.zeros((4, 4), dtype=np.uint8)
    m1 = hard.copy()
    m2 = hard.copy()
    m2[:, 2:] = 1
    penalized1 = boundary_disagreement_penalty(hard, [m1, m2], conf, dilate_px=2)
    penalized2 = boundary_disagreement_penalty(hard, [m1, m2], conf, dilate_px=2)
    assert np.allclose(penalized1, penalized2)


def test_single_rater_fallback_has_non_empty_supervised_region() -> None:
    sem = np.zeros((8, 8), dtype=np.uint8)
    sem[2:6, 2:6] = 2

    cfg = ConsensusConfig(single_rater_ignore_policy="confidence_mask", ignore_threshold_loose=0.96)
    builder = ConsensusMaskBuilder(cfg)

    _, probs, ignore, summary = builder._run_consensus(
        sem_maps={"p1": sem},
        statuses={"p1": "keep"},
        weights={"p1": 1.0},
    )

    assert summary["effective_fusion_mode"] == "single_rater_fallback"
    assert probs.shape == (4, 8, 8)
    assert int((ignore == 0).sum()) > 0


def test_end_to_end_consensus_build_writes_artifacts_and_qc_fields(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_root = data_root / "consensus"
    map1 = data_root / "Maps1_T" / "case001_classimg_nonconvex.png"
    map2 = data_root / "Maps2_T" / "case001_classimg_nonconvex.png"
    img = data_root / "Train_imgs" / "case001.png"

    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m2[:, 5:] = 2
    _save_mask(map1, m1)
    _save_mask(map2, m2)
    _save_mask(img, np.zeros((10, 10, 3), dtype=np.uint8))

    cfg = ConsensusConfig(
        dataset_root=str(data_root),
        output_root=str(out_root),
        raters=["p1", "p2"],
        consensus_fusion_mode="weighted",
    )
    builder = ConsensusMaskBuilder(cfg)
    res = builder.process_all()

    assert res["metadata"]["num_success"] == 1
    case_dir = out_root / "case001"
    assert (case_dir / "consensus_hard_mask.png").exists()
    assert (case_dir / "consensus_probs_compact.npz").exists()
    assert (case_dir / "ignore_mask.png").exists()
    assert (case_dir / "qc_report.json").exists()

    probs = _load_probs(case_dir / "consensus_probs_compact.npz")
    assert probs.shape == (4, 10, 10)

    with (case_dir / "qc_report.json").open("r", encoding="utf-8") as f:
        qc = json.load(f)
    assert "consensus_fusion" in qc
    assert "final_thresholds_used" in qc
    assert "weights_per_pathologist" in qc
