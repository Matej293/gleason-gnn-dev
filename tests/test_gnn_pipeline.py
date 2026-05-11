from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gnn.baselines import seg_only_predict
from src.gnn.data import feature_index_map, load_graph_splits
from src.gnn.metrics import aggregate_case_metrics, json_safe
from src.gnn.models import NodeMLP
from src.gnn.train import TrainConfig, apply_class_mask_to_logits, run_training, train_supported_classes


def _write_graph(path: Path, n: int = 4, feat_dim: int = 22, y: np.ndarray | None = None) -> None:
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
    labels = np.array([0, 1, 2, 3], dtype=np.int64) if y is None else y.astype(np.int64)
    train_mask = np.array([1, 1, 1, 1], dtype=np.uint8)
    node_ids = np.arange(n, dtype=np.int64)
    np.savez_compressed(path / "graph_data.npz", node_ids=node_ids, x=x, edge_index=edge_index, y=labels, train_mask=train_mask)


def _make_graph_root(tmp_path: Path, feat_dim: int = 22, y: np.ndarray | None = None) -> Path:
    root = tmp_path / "graphs"
    for split in ("train", "val", "test"):
        _write_graph(root / split / f"{split}_case0", feat_dim=feat_dim, y=y)
        _write_graph(root / split / f"{split}_case1", feat_dim=feat_dim, y=y)
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


def test_residual_alpha_scales_correction() -> None:
    model = NodeMLP(in_dim=22, hidden_dim=8, residual_head=True, residual_alpha=0.0, seg_prob_idx=(9, 13))
    x = torch.zeros((2, 22), dtype=torch.float32)
    x[:, 9:13] = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], dtype=torch.float32)
    out = model(x, None)
    expected = torch.log(torch.clamp(x[:, 9:13], min=1e-8))
    assert torch.allclose(out, expected, atol=1e-6)


def test_apply_class_mask_to_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 9.0, 3.0]], dtype=torch.float32)
    masked = apply_class_mask_to_logits(logits, [0, 1, 3])
    pred = int(torch.argmax(masked, dim=1).item())
    assert pred == 3


def test_train_supported_classes_detects_absent_class(tmp_path: Path) -> None:
    root = _make_graph_root(tmp_path, y=np.array([0, 1, 2, 2]))
    splits = load_graph_splits(root)
    supported = train_supported_classes(splits["train"])
    assert supported == [0, 1, 2]


def test_train_mlp_end_to_end(tmp_path: Path) -> None:
    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="mlp_test", cfg=TrainConfig(model="mlp", epochs=1, patience=1, seed=7, normalize_features=True, mask_unsupported_classes=True), graphs_root=str(root))
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "metrics_summary.json").exists()
    assert (run_dir / "run_config.json").exists()
    payload = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert "split_metrics" in payload and "train" in payload["split_metrics"]
    assert "predicted_class_counts" in payload["split_metrics"]["test"]


def test_train_graphsage_end_to_end_or_skip(tmp_path: Path) -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pytest.skip("torch_geometric not installed")

    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="sage_test", cfg=TrainConfig(model="graphsage", epochs=1, patience=1, seed=7), graphs_root=str(root))
    assert (run_dir / "best.pt").exists()


def test_json_safe_nan_to_null() -> None:
    payload = {"a": float("nan"), "b": [1.0, float("inf")]}
    out = json_safe(payload)
    assert out["a"] is None
    assert out["b"][1] is None


def test_feature_index_map_supports_legacy_and_new() -> None:
    assert "seg_probs_mean" in feature_index_map(14)
    assert "seg_probs_std" in feature_index_map(22)


def test_run_config_persists_new_knobs(tmp_path: Path) -> None:
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
        residual_head=True,
        residual_alpha=0.4,
        mask_unsupported_classes=True,
        grad_clip_norm=0.9,
        use_scheduler=True,
    )
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="cfg_persist", cfg=cfg, graphs_root=str(root))
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert payload["loss"] == "focal"
    assert payload["focal_gamma"] == pytest.approx(1.5)
    assert payload["residual_alpha"] == pytest.approx(0.4)
    assert payload["mask_unsupported_classes"] is True
    assert payload["grad_clip_norm"] == pytest.approx(0.9)


def test_selection_metric_is_recorded_and_used(tmp_path: Path) -> None:
    root = _make_graph_root(tmp_path)
    splits = load_graph_splits(root)
    cfg = TrainConfig(
        model="mlp",
        epochs=2,
        patience=2,
        seed=7,
        selection_metric="val_per_case_balanced_accuracy",
    )
    run_dir = run_training(splits, output_root=tmp_path / "runs", experiment_name="sel_metric", cfg=cfg, graphs_root=str(root))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert metrics["resolved_selection_metric"] == "val_per_case_balanced_accuracy"
    assert isinstance(metrics["best_selection_score"], float)
    assert run_cfg["resolved_selection_metric"] == "val_per_case_balanced_accuracy"


def test_residual_head_rejects_normalized_probs_without_raw_slice() -> None:
    model = NodeMLP(in_dim=22, hidden_dim=8, residual_head=True, residual_alpha=0.2, seg_prob_idx=(9, 13))
    x = torch.randn((2, 22), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="raw seg_probs_mean"):
        _ = model(x, None)
