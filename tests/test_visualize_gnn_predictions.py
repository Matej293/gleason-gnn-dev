from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.pipelines.gnn.models import GATNet, GCNNet, NodeMLP


def _load_viz_module():
    mod_path = Path(__file__).resolve().parents[1] / "src" / "cli" / "visualize_gnn_predictions.py"
    spec = importlib.util.spec_from_file_location("visualize_gnn_predictions", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed loading visualize_gnn_predictions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_graph_case(case_dir: Path) -> None:
    superpixels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int64,
    )
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    x = np.zeros((4, 22), dtype=np.float32)
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
    y = np.array([0, 1, 2, -1], dtype=np.int64)
    np.savez_compressed(
        case_dir / "graph_data.npz",
        node_ids=node_ids,
        x=x,
        edge_index=edge_index,
        y=y,
        superpixels=superpixels,
    )


def _write_run(run_dir: Path, *, with_norm: bool, residual_head: bool = True, feature_dropout: float = 0.2) -> None:
    model = NodeMLP(
        in_dim=22,
        hidden_dim=16,
        dropout=0.0,
        feature_dropout=feature_dropout,
        residual_head=residual_head,
        seg_prob_idx=(9, 13),
    )
    ckpt = {
        "model_state": model.state_dict(),
        "model": "mlp",
        "in_dim": 22,
        "hidden_dim": 16,
        "dropout": 0.0,
        "feature_dropout": feature_dropout,
        "residual_head": residual_head,
        "normalize_features": with_norm,
        "feature_index_map": {"seg_probs_mean": [9, 10, 11, 12]},
    }
    if with_norm:
        ckpt["norm_mean"] = np.zeros((22,), dtype=np.float32)
        ckpt["norm_std"] = np.ones((22,), dtype=np.float32)
    torch.save(ckpt, run_dir / "best.pt")

    run_cfg = {
        "model": "mlp",
        "hidden_dim": 16,
        "dropout": 0.0,
        "feature_dropout": feature_dropout,
        "residual_head": residual_head,
        "normalize_features": with_norm,
        "feature_dim": 22,
        "feature_index_map": {"seg_probs_mean": [9, 10, 11, 12]},
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg), encoding="utf-8")


def test_extract_superpixel_boundaries_toy_shape_and_non_empty() -> None:
    viz = _load_viz_module()
    sp = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int64,
    )
    boundaries = viz.extract_superpixel_boundaries(sp)
    assert boundaries.shape == sp.shape
    assert boundaries.dtype == np.bool_
    assert int(boundaries.sum()) > 0


def test_extract_node_centroids_returns_one_per_node() -> None:
    viz = _load_viz_module()
    sp = np.array(
        [
            [0, 0, 1],
            [0, 2, 2],
            [3, 3, 2],
        ],
        dtype=np.int64,
    )
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    centroids = viz.extract_node_centroids(node_ids, sp)
    assert set(centroids.keys()) == {0, 1, 2, 3}
    for x, y in centroids.values():
        assert 0.0 <= x < sp.shape[1]
        assert 0.0 <= y < sp.shape[0]


def test_unique_undirected_edges_deduplicates_bidirectional_pairs() -> None:
    viz = _load_viz_module()
    edge_index = np.array(
        [
            [0, 1, 1, 2, 2, 1, 3, 3],
            [1, 0, 2, 1, 1, 2, 3, 3],
        ],
        dtype=np.int64,
    )
    unique = viz.unique_undirected_edges(edge_index)
    assert unique == [(0, 1), (1, 2)]


def test_model_build_uses_feature_dropout_residual_head_and_seg_prob_idx() -> None:
    viz = _load_viz_module()
    model = viz._build_model_from_metadata(
        {
            "model": "mlp",
            "in_dim": 22,
            "hidden_dim": 8,
            "dropout": 0.1,
            "feature_dropout": 0.33,
            "residual_head": True,
            "feature_index_map": {"seg_probs_mean": [5, 6, 7, 8]},
        }
    )
    assert isinstance(model, NodeMLP)
    assert model.seg_prob_idx == (5, 9)
    assert model.residual_head is True
    assert model.feature_dropout.p == pytest.approx(0.33)


def test_model_build_supports_gcn_or_skip() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pytest.skip("torch_geometric not installed")
    viz = _load_viz_module()
    model = viz._build_model_from_metadata(
        {
            "model": "gcn",
            "in_dim": 22,
            "hidden_dim": 8,
            "dropout": 0.1,
            "feature_dropout": 0.25,
            "residual_head": False,
            "feature_index_map": {"seg_probs_mean": [9, 10, 11, 12]},
        }
    )
    assert isinstance(model, GCNNet)
    assert model.feature_dropout.p == pytest.approx(0.25)


def test_model_build_supports_gat_or_skip() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pytest.skip("torch_geometric not installed")
    viz = _load_viz_module()
    model = viz._build_model_from_metadata(
        {
            "model": "gat",
            "in_dim": 22,
            "hidden_dim": 8,
            "dropout": 0.1,
            "feature_dropout": 0.25,
            "residual_head": False,
            "feature_index_map": {"seg_probs_mean": [9, 10, 11, 12]},
        }
    )
    assert isinstance(model, GATNet)
    assert model.conv1.heads == 4
    assert model.conv2.heads == 4
    assert model.conv1.concat is False
    assert model.conv2.concat is False


def test_compare_metrics_handles_missing_g5_and_tolerance() -> None:
    viz = _load_viz_module()
    expected = {
        "macro_f1": 0.5,
        "balanced_accuracy": 0.6,
        "per_class_f1": {"benign": 0.4, "g3": 0.5, "g4": 0.6, "g5": None},
    }
    observed = {
        "macro_f1": 0.50001,
        "balanced_accuracy": 0.60001,
        "per_class_f1": {"benign": 0.40001, "g3": 0.50001, "g4": 0.60001, "g5": None},
    }
    ok, details = viz._compare_metrics(expected, observed, tol=1e-3)
    assert ok is True
    assert details["max_abs_delta"] <= 1e-3

    bad, _ = viz._compare_metrics(expected, observed, tol=1e-8)
    assert bad is False


def test_normalization_stats_required_when_enabled(tmp_path: Path) -> None:
    graphs_root = tmp_path / "graphs"
    run_dir = tmp_path / "run"
    case_dir = graphs_root / "test" / "case0"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_graph_case(case_dir)
    _write_run(run_dir, with_norm=False)

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    ckpt["normalize_features"] = True
    torch.save(ckpt, run_dir / "best.pt")

    cmd = [
        sys.executable,
        "-m", "src.cli.visualize_gnn_predictions",
        "--graphs-root",
        str(graphs_root),
        "--run-dir",
        str(run_dir),
        "--split",
        "test",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert proc.returncode != 0
    assert "normalize_features=true" in (proc.stderr + proc.stdout)


def test_visualize_script_writes_outputs_and_summary_with_parity(tmp_path: Path) -> None:
    graphs_root = tmp_path / "graphs"
    run_dir = tmp_path / "run"
    case_dir = graphs_root / "test" / "case0"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_graph_case(case_dir)
    _write_run(run_dir, with_norm=True)

    metrics_summary = {
        "split_metrics": {
            "test": {
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "per_class_f1": {"benign": 1.0, "g3": 1.0, "g4": 1.0, "g5": None},
            }
        }
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary), encoding="utf-8")

    out_root = tmp_path / "viz_out"
    cmd = [
        sys.executable,
        "-m", "src.cli.visualize_gnn_predictions",
        "--graphs-root",
        str(graphs_root),
        "--run-dir",
        str(run_dir),
        "--split",
        "test",
        "--output-dir",
        str(out_root),
        "--max-cases",
        "1",
        "--seed",
        "7",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    children = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(children) == 1
    out_dir = children[0]

    assert (out_dir / "cases" / "case0.png").exists()
    assert (out_dir / "cases" / "case0_superpixels.png").exists()
    assert (out_dir / "cases" / "case0_graph_overlay.png").exists()
    assert (out_dir / "confusion_seg_only.png").exists()
    assert (out_dir / "confusion_model.png").exists()
    assert (out_dir / "per_class_f1_comparison.png").exists()
    assert (out_dir / "per_case_macro_f1_delta.png").exists()

    summary = json.loads((out_dir / "viz_summary.json").read_text(encoding="utf-8"))
    assert summary["parity_check_enabled"] is True
    assert summary["parity_passed"] is True
    assert summary["parity_compared_metrics"]["per_class_f1"]["g5"]["expected"] is None
    assert summary["overlay_style"] == "clean"


def test_parity_mismatch_fails_fast(tmp_path: Path) -> None:
    graphs_root = tmp_path / "graphs"
    run_dir = tmp_path / "run"
    case_dir = graphs_root / "test" / "case0"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_graph_case(case_dir)
    _write_run(run_dir, with_norm=True)

    metrics_summary = {
        "split_metrics": {
            "test": {
                "macro_f1": 0.0,
                "balanced_accuracy": 0.0,
                "per_class_f1": {"benign": 0.0, "g3": 0.0, "g4": 0.0, "g5": None},
            }
        }
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary), encoding="utf-8")

    out_root = tmp_path / "viz_out"
    cmd = [
        sys.executable,
        "-m", "src.cli.visualize_gnn_predictions",
        "--graphs-root",
        str(graphs_root),
        "--run-dir",
        str(run_dir),
        "--split",
        "test",
        "--output-dir",
        str(out_root),
        "--max-cases",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert proc.returncode != 0
    assert "Parity check failed" in (proc.stderr + proc.stdout)


def test_visualize_can_disable_parity_check(tmp_path: Path) -> None:
    graphs_root = tmp_path / "graphs"
    run_dir = tmp_path / "run"
    case_dir = graphs_root / "test" / "case0"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_graph_case(case_dir)
    _write_run(run_dir, with_norm=True)

    out_root = tmp_path / "viz_out"
    cmd = [
        sys.executable,
        "-m", "src.cli.visualize_gnn_predictions",
        "--graphs-root",
        str(graphs_root),
        "--run-dir",
        str(run_dir),
        "--split",
        "test",
        "--output-dir",
        str(out_root),
        "--max-cases",
        "1",
        "--parity-check",
        "off",
        "--output-versioning",
        "overwrite",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
    summary = json.loads((out_root / "viz_summary.json").read_text(encoding="utf-8"))
    assert summary["parity_check_enabled"] is False
    assert summary["parity_passed"] is True
