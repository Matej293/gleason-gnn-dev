#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from src.gnn.baselines import seg_only_predict
from src.gnn.data import load_graph_splits
from src.gnn.metrics import CLASS_NAMES, CaseEval, aggregate_case_metrics, json_safe
from src.gnn.train import TrainConfig, run_training

SPLITS = ("train", "val", "test")
METHODS = ("mlp", "graphsage", "gcn", "gat")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate seg-only plus MLP/GraphSAGE/GCN/GAT baselines.")
    p.add_argument("--graphs-root", required=True, type=str)
    p.add_argument("--output-dir", type=str, default="outputs/gnn_runs")
    p.add_argument("--profile", choices=["fast", "balanced", "thesis"], default="thesis")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def eval_seg_only(splits: dict) -> tuple[dict, dict, dict]:
    out = {}
    arrays = {}
    per_case = {}
    for split in SPLITS:
        cases = []
        pred_counts = {name: 0 for name in CLASS_NAMES}
        for s in tqdm(splits[split], desc=f"seg-only {split}", leave=False):
            pred = seg_only_predict(s.x)
            m = s.eval_mask
            for i, name in enumerate(CLASS_NAMES):
                pred_counts[name] += int((pred[m] == i).sum().item())
            cases.append(CaseEval(s.image_id, s.y[m].cpu().numpy(), pred[m].cpu().numpy()))
        _, per_case_mean, _, rows = aggregate_case_metrics(cases)
        out[split] = per_case_mean
        out[split]["predicted_class_counts"] = pred_counts
        arrays[split] = {
            "macro_f1": [rows[c.image_id]["macro_f1"] for c in cases],
            "balanced_accuracy": [rows[c.image_id]["balanced_accuracy"] for c in cases],
        }
        per_case[split] = rows
    return out, arrays, per_case


def _load_summary(run_dir: Path) -> tuple[dict, dict, dict]:
    payload = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    arr = json.loads((run_dir / "per_case_arrays.json").read_text(encoding="utf-8"))
    per_case = json.loads((run_dir / "per_case_metrics.json").read_text(encoding="utf-8"))
    result = {}
    for split in SPLITS:
        result[split] = {
            "macro_f1": payload["split_metrics"][split]["macro_f1"],
            "balanced_accuracy": payload["split_metrics"][split]["balanced_accuracy"],
            "per_class_f1": payload["split_metrics"][split]["per_class_f1"],
            "predicted_class_counts": payload["split_metrics"][split].get("predicted_class_counts", {}),
        }
    return result, arr, per_case


def _verdict(results: dict) -> dict:
    ranked_methods = ("mlp", "graphsage", "gcn", "gat")
    leaderboard = sorted(
        (
            {
                "method": method,
                "test_macro_f1": float(results[method]["test"]["macro_f1"] or 0.0),
                "test_balanced_accuracy": float(results[method]["test"]["balanced_accuracy"] or 0.0),
            }
            for method in ranked_methods
        ),
        key=lambda r: r["test_macro_f1"],
        reverse=True,
    )
    best = leaderboard[0]
    seg = results["seg_only"]["test"]
    best_delta_vs_seg = best["test_macro_f1"] - float(seg["macro_f1"] or 0.0)
    return {
        "ranking_metric": "test_macro_f1",
        "leaderboard_test_macro_f1": leaderboard,
        "best_method": best["method"],
        "best_method_delta_vs_seg_only_test_macro_f1": best_delta_vs_seg,
        "notes": "Ranking is based on test macro-F1 only. Balanced accuracy and per-class F1 are diagnostic.",
    }


def _add_deltas_and_case_summary(results: dict, per_case_metrics: dict) -> dict:
    deltas = {}
    for method in METHODS:
        deltas[method] = {}
        for split in SPLITS:
            seg_split = results["seg_only"][split]
            method_split = results[method][split]
            per_class_delta = {
                cls: (method_split["per_class_f1"].get(cls) or 0.0) - (seg_split["per_class_f1"].get(cls) or 0.0)
                for cls in CLASS_NAMES
            }
            deltas[method][split] = {
                "macro_f1": (method_split["macro_f1"] or 0.0) - (seg_split["macro_f1"] or 0.0),
                "balanced_accuracy": (method_split["balanced_accuracy"] or 0.0) - (seg_split["balanced_accuracy"] or 0.0),
                "per_class_f1": per_class_delta,
            }

    per_case_win_loss = {}
    for method in METHODS:
        per_case_win_loss[method] = {}
        for split in SPLITS:
            seg_rows = per_case_metrics["seg_only"][split]
            m_rows = per_case_metrics[method][split]
            wins = losses = ties = 0
            dvals = []
            for image_id, seg_row in seg_rows.items():
                m_row = m_rows.get(image_id)
                if m_row is None:
                    continue
                delta = float((m_row.get("macro_f1") or 0.0) - (seg_row.get("macro_f1") or 0.0))
                dvals.append(delta)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
            per_case_win_loss[method][split] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mean_delta_macro_f1": float(sum(dvals) / max(len(dvals), 1)),
            }
    return {"split_deltas_vs_seg_only": deltas, "per_case_win_loss": per_case_win_loss}


def main() -> None:
    args = parse_args()
    print("[1/4] Loading graph splits...")
    splits = load_graph_splits(args.graphs_root)

    print("[2/4] Evaluating seg-only baseline...")
    seg_metrics, seg_arrays, seg_per_case = eval_seg_only(splits)
    results = {"seg_only": seg_metrics}
    per_case_arrays = {"seg_only": seg_arrays}
    per_case_metrics = {"seg_only": seg_per_case}

    common = dict(seed=args.seed, normalize_features=True, use_class_weights=True)
    print("[3/6] Training/evaluating MLP baseline...")
    mlp_dir = run_training(splits, output_root=args.output_dir, experiment_name="baseline_mlp", cfg=TrainConfig(model="mlp", **common), graphs_root=args.graphs_root)
    results["mlp"], per_case_arrays["mlp"], per_case_metrics["mlp"] = _load_summary(mlp_dir)

    print("[4/6] Training/evaluating GraphSAGE baseline...")
    sage_dir = run_training(splits, output_root=args.output_dir, experiment_name="baseline_graphsage", cfg=TrainConfig(model="graphsage", **common), graphs_root=args.graphs_root)
    results["graphsage"], per_case_arrays["graphsage"], per_case_metrics["graphsage"] = _load_summary(sage_dir)
    print("[5/6] Training/evaluating GCN baseline...")
    gcn_dir = run_training(splits, output_root=args.output_dir, experiment_name="baseline_gcn", cfg=TrainConfig(model="gcn", **common), graphs_root=args.graphs_root)
    results["gcn"], per_case_arrays["gcn"], per_case_metrics["gcn"] = _load_summary(gcn_dir)
    print("[6/6] Training/evaluating GAT baseline...")
    gat_dir = run_training(splits, output_root=args.output_dir, experiment_name="baseline_gat", cfg=TrainConfig(model="gat", **common), graphs_root=args.graphs_root)
    results["gat"], per_case_arrays["gat"], per_case_metrics["gat"] = _load_summary(gat_dir)

    extra = _add_deltas_and_case_summary(results, per_case_metrics)
    verdict = _verdict(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{ts}_baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "results": results,
        "per_case_arrays": per_case_arrays,
        "per_case_metrics": per_case_metrics,
        "split_deltas_vs_seg_only": extra["split_deltas_vs_seg_only"],
        "per_case_win_loss": extra["per_case_win_loss"],
        "verdict": verdict,
    }
    (out_dir / "baseline_comparison.json").write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")

    with (out_dir / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "macro_f1", "balanced_accuracy", "delta_macro_f1_vs_seg_only"])
        w.writeheader()
        for method in ("seg_only", *METHODS):
            for split in SPLITS:
                row = results[method][split]
                delta = 0.0 if method == "seg_only" else extra["split_deltas_vs_seg_only"][method][split]["macro_f1"]
                w.writerow({
                    "method": method,
                    "split": split,
                    "macro_f1": row.get("macro_f1"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "delta_macro_f1_vs_seg_only": delta,
                })

    print(f"Saved baseline comparison to: {out_dir}")


if __name__ == "__main__":
    main()
