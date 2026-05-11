from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gnn.models import NodeMLP


def _write_graph_case(case_dir: Path, offset: float = 0.0) -> None:
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
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
            [0.05, 0.05, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    x[:, 0] = offset
    edge_index = np.array([[0, 1, 2, 3, 1, 2], [1, 0, 3, 2, 2, 1]], dtype=np.int64)
    y = np.array([0, 1, 2, -1], dtype=np.int64)
    np.savez_compressed(case_dir / "graph_data.npz", node_ids=node_ids, x=x, edge_index=edge_index, y=y, superpixels=superpixels)


def _write_run(run_dir: Path, *, model_name: str = "mlp", write_ckpt: bool = True) -> None:
    model = NodeMLP(in_dim=22, hidden_dim=16, dropout=0.0, feature_dropout=0.0, residual_head=False, seg_prob_idx=(9, 13))
    if write_ckpt:
        ckpt = {
            "model_state": model.state_dict(),
            "model": model_name,
            "in_dim": 22,
            "hidden_dim": 16,
            "dropout": 0.0,
            "feature_dropout": 0.0,
            "residual_head": False,
            "normalize_features": False,
            "feature_index_map": {"seg_probs_mean": [9, 10, 11, 12]},
        }
        torch.save(ckpt, run_dir / "best.pt")
    run_cfg = {
        "model": model_name,
        "hidden_dim": 16,
        "dropout": 0.0,
        "feature_dropout": 0.0,
        "residual_head": False,
        "normalize_features": False,
        "feature_dim": 22,
        "feature_index_map": {"seg_probs_mean": [9, 10, 11, 12]},
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg), encoding="utf-8")


def _payload(num_cases: int = 3) -> dict:
    methods = ["seg_only", "mlp", "graphsage", "gcn", "gat"]
    results = {}
    arrays = {}
    for i, m in enumerate(methods):
        base = 0.45 + 0.05 * i
        results[m] = {
            "train": {"macro_f1": base, "balanced_accuracy": base, "per_class_f1": {"benign": base, "g3": base, "g4": base, "g5": base}},
            "val": {"macro_f1": base, "balanced_accuracy": base, "per_class_f1": {"benign": base, "g3": base, "g4": base, "g5": base}},
            "test": {"macro_f1": base, "balanced_accuracy": base, "per_class_f1": {"benign": base, "g3": base, "g4": base, "g5": base}},
        }
        arrays[m] = {
            "train": {"macro_f1": [base + 0.01 * j for j in range(num_cases)]},
            "val": {"macro_f1": [base + 0.01 * j for j in range(num_cases)]},
            "test": {"macro_f1": [base + 0.01 * j for j in range(num_cases)]},
        }
    return {
        "results": results,
        "per_case_arrays": arrays,
        "verdict": {"leaderboard_test_macro_f1": [{"method": "gat", "test_macro_f1": 0.65, "test_balanced_accuracy": 0.65}]},
    }


def _write_minimal_setup(tmp_path: Path, *, num_cases: int = 3) -> tuple[Path, Path, Path]:
    graphs_root = tmp_path / "graphs"
    for idx in range(num_cases):
        case_dir = graphs_root / "test" / f"case{idx}"
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_graph_case(case_dir, offset=float(idx))
    runs_root = tmp_path / "gnn_runs"
    for m in ["mlp", "graphsage", "gcn", "gat"]:
        run_dir = runs_root / f"20260101_00000{1 + ['mlp','graphsage','gcn','gat'].index(m)}_baseline_{m}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run(run_dir, model_name="mlp")
    cmp_dir = runs_root / "20260101_000010_baseline_comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    (cmp_dir / "baseline_comparison.json").write_text(json.dumps(_payload(num_cases=num_cases)), encoding="utf-8")
    return graphs_root, runs_root, cmp_dir


def test_payload_validation_error_when_missing_per_case_arrays(tmp_path: Path) -> None:
    graphs_root, runs_root, cmp_dir = _write_minimal_setup(tmp_path)
    bad = _payload()
    bad.pop("per_case_arrays")
    (cmp_dir / "baseline_comparison.json").write_text(json.dumps(bad), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/visualize_gnn_baseline_comparison.py",
        "--comparison-dir",
        str(cmp_dir),
        "--graphs-root",
        str(graphs_root),
        "--gnn-runs-root",
        str(runs_root),
        "--output-versioning",
        "overwrite",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert proc.returncode != 0
    assert "per_case_arrays" in (proc.stderr + proc.stdout)


def test_run_resolver_picks_nearest_timestamp(tmp_path: Path) -> None:
    graphs_root, runs_root, cmp_dir = _write_minimal_setup(tmp_path)
    extra = runs_root / "20240101_000000_baseline_mlp"
    extra.mkdir(parents=True, exist_ok=True)
    _write_run(extra, model_name="mlp")
    out_dir = tmp_path / "viz_out"
    cmd = [
        sys.executable,
        "scripts/visualize_gnn_baseline_comparison.py",
        "--comparison-dir",
        str(cmp_dir),
        "--graphs-root",
        str(graphs_root),
        "--gnn-runs-root",
        str(runs_root),
        "--output-dir",
        str(out_dir),
        "--output-versioning",
        "overwrite",
        "--max-cases",
        "2",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
    summary = json.loads((out_dir / "comparison_viz_summary.json").read_text(encoding="utf-8"))
    assert summary["resolved_run_dirs"]["mlp"].endswith("20260101_000001_baseline_mlp")


def test_generates_combined_figures_and_case_montages(tmp_path: Path) -> None:
    graphs_root, runs_root, cmp_dir = _write_minimal_setup(tmp_path, num_cases=3)
    out_dir = tmp_path / "viz_out"
    cmd = [
        sys.executable,
        "scripts/visualize_gnn_baseline_comparison.py",
        "--comparison-dir",
        str(cmp_dir),
        "--graphs-root",
        str(graphs_root),
        "--gnn-runs-root",
        str(runs_root),
        "--output-dir",
        str(out_dir),
        "--output-versioning",
        "overwrite",
        "--max-cases",
        "3",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
    assert (out_dir / "leaderboard_test_macro_f1.png").exists()
    assert (out_dir / "macro_f1_by_split.png").exists()
    assert (out_dir / "delta_vs_seg_only_test.png").exists()
    assert (out_dir / "per_class_f1_test.png").exists()
    assert (out_dir / "per_case_macro_f1_delta_overlay.png").exists()
    case_imgs = list((out_dir / "cases").glob("*_comparison.png"))
    assert len(case_imgs) >= 2


def test_missing_best_pt_for_one_model_fails(tmp_path: Path) -> None:
    graphs_root, runs_root, cmp_dir = _write_minimal_setup(tmp_path)
    bad_run = runs_root / "20260101_000004_baseline_gat"
    (bad_run / "best.pt").unlink()
    cmd = [
        sys.executable,
        "scripts/visualize_gnn_baseline_comparison.py",
        "--comparison-dir",
        str(cmp_dir),
        "--graphs-root",
        str(graphs_root),
        "--gnn-runs-root",
        str(runs_root),
        "--output-versioning",
        "overwrite",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert proc.returncode != 0
    assert "Missing checkpoint" in (proc.stderr + proc.stdout)


def test_case_count_mismatch_fails(tmp_path: Path) -> None:
    graphs_root, runs_root, cmp_dir = _write_minimal_setup(tmp_path, num_cases=2)
    payload = _payload(num_cases=2)
    payload["per_case_arrays"]["mlp"]["test"]["macro_f1"] = [0.1]
    (cmp_dir / "baseline_comparison.json").write_text(json.dumps(payload), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/visualize_gnn_baseline_comparison.py",
        "--comparison-dir",
        str(cmp_dir),
        "--graphs-root",
        str(graphs_root),
        "--gnn-runs-root",
        str(runs_root),
        "--output-versioning",
        "overwrite",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert proc.returncode != 0
    assert "Case count mismatch" in (proc.stderr + proc.stdout)
