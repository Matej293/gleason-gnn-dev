from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gnn.baselines import seg_only_predict
from src.gnn.data import feature_index_map, load_graph_splits
from src.gnn.metrics import aggregate_case_metrics, json_safe
from src.gnn.train import TrainConfig, run_training


def _write_graph(path: Path, n: int = 4, feat_dim: int = 22) -> None:
    path.mkdir(parents=True, exist_ok=True)
    x = np.zeros((n, feat_dim), dtype=np.float32)
    x[:, 9:13] = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
            [0.05, 0.05, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    edge_index = np.array([[0, 1, 2, 3, 1, 2], [1, 0, 3, 2, 2, 1]], dtype=np.int64)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    train_mask = np.array([1, 1, 1, 1], dtype=np.uint8)
    node_ids = np.arange(n, dtype=np.int64)
    np.savez_compressed(path / "graph_data.npz", node_ids=node_ids, x=x, edge_index=edge_index, y=y, train_mask=train_mask)


def _make_graph_root(tmp_path: Path, feat_dim: int = 22) -> Path:
    root = tmp_path / "graphs"
    for split in ("train", "val", "test"):
        _write_graph(root / split / f"{split}_case0", feat_dim=feat_dim)
        _write_graph(root / split / f"{split}_case1", feat_dim=feat_dim)
    return root


def test_loader_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "graphs" / "train" / "case0"
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "graph_data.npz", x=np.zeros((3, 14), dtype=np.float32))
    (tmp_path / "graphs" / "val" / "v").mkdir(parents=True, exist_ok=True)
    (tmp_path / "graphs" / "test" / "t").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp_path / "graphs" / "val" / "v" / "graph_data.npz", node_ids=np.arange(3), x=np.zeros((3, 14), dtype=np.float32), edge_index=np.zeros((2, 0), dtype=np.int64), y=np.zeros((3,), dtype=np.int64), train_mask=np.ones((3,), dtype=np.uint8))
    np.savez_compressed(tmp_path / "graphs" / "test" / "t" / "graph_data.npz", node_ids=np.arange(3), x=np.zeros((3, 14), dtype=np.float32), edge_index=np.zeros((2, 0), dtype=np.int64), y=np.zeros((3,), dtype=np.int64), train_mask=np.ones((3,), dtype=np.uint8))
    with pytest.raises(ValueError, match="Missing key"):
        load_graph_splits(tmp_path / "graphs")


def test_seg_only_predict_uses_expected_feature_slice() -> None:
    x = torch.zeros((2, 22), dtype=torch.float32)
    x[0, 9:13] = torch.tensor([0.1, 0.7, 0.1, 0.1])
    x[1, 9:13] = torch.tensor([0.2, 0.1, 0.6, 0.1])
    pred = seg_only_predict(x)
    assert pred.tolist() == [1, 2]


def test_metric_aggregation_toy_case() -> None:
    from src.gnn.metrics import CaseEval

    cases = [CaseEval("a", np.array([0, 1, 2, 3]), np.array([0, 1, 2, 0]))]
    _, per_case_mean, cm, _ = aggregate_case_metrics(cases)
    assert cm.shape == (4, 4)
    assert per_case_mean["macro_f1"] < 1.0


def test_train_mlp_end_to_end(tmp_path: Path) -> None:
    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="mlp_test", cfg=TrainConfig(model="mlp", epochs=1, patience=1, seed=7, normalize_features=True), graphs_root=str(root))
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "metrics_summary.json").exists()
    assert (run_dir / "run_config.json").exists()
    payload = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert "split_metrics" in payload and "train" in payload["split_metrics"]


def test_train_graphsage_end_to_end_or_skip(tmp_path: Path) -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pytest.skip("torch_geometric not installed")

    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="sage_test", cfg=TrainConfig(model="graphsage", epochs=1, patience=1, seed=7), graphs_root=str(root))
    assert (run_dir / "best.pt").exists()


def test_train_gcn_end_to_end_or_skip(tmp_path: Path) -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pytest.skip("torch_geometric not installed")

    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="gcn_test", cfg=TrainConfig(model="gcn", epochs=1, patience=1, seed=7), graphs_root=str(root))
    assert (run_dir / "best.pt").exists()


def test_train_gat_end_to_end_or_skip(tmp_path: Path) -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pytest.skip("torch_geometric not installed")

    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="gat_test", cfg=TrainConfig(model="gat", epochs=1, patience=1, seed=7), graphs_root=str(root))
    assert (run_dir / "best.pt").exists()


def test_json_safe_nan_to_null() -> None:
    payload = {"a": float("nan"), "b": [1.0, float("inf")]}
    out = json_safe(payload)
    assert out["a"] is None
    assert out["b"][1] is None


def test_feature_index_map_supports_legacy_and_new() -> None:
    assert "seg_probs_mean" in feature_index_map(14)
    assert "seg_probs_std" in feature_index_map(22)


def test_run_config_persists_focal_and_regularization_knobs(tmp_path: Path) -> None:
    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    cfg = TrainConfig(
        model="mlp",
        epochs=1,
        patience=1,
        seed=7,
        normalize_features=True,
        loss="focal",
        focal_gamma=1.5,
        hidden_dim=96,
        dropout=0.25,
        feature_dropout=0.05,
        edge_dropout=0.1,
        lr=3e-4,
        weight_decay=5e-4,
    )
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="cfg_persist", cfg=cfg, graphs_root=str(root))
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert payload["loss"] == "focal"
    assert payload["focal_gamma"] == pytest.approx(1.5)
    assert payload["hidden_dim"] == 96
    assert payload["dropout"] == pytest.approx(0.25)
    assert payload["feature_dropout"] == pytest.approx(0.05)
    assert payload["edge_dropout"] == pytest.approx(0.1)
    assert payload["lr"] == pytest.approx(3e-4)
    assert payload["weight_decay"] == pytest.approx(5e-4)
