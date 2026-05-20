#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select best GNN run directory for model+name prefix.")
    p.add_argument("--model", required=True, choices=["mlp", "graphsage", "gcn", "gat"])
    p.add_argument("--name-prefix", required=True, type=str)
    p.add_argument("--runs-root", default="outputs/gnn_runs", type=str)
    return p.parse_args()


def _score(metrics_path: Path) -> float:
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return float("-inf")
    score = data.get("best_selection_score")
    if score is None:
        score = (((data.get("split_metrics") or {}).get("val") or {}).get("macro_f1"))
    try:
        return float("-inf") if score is None else float(score)
    except Exception:
        return float("-inf")


def main() -> None:
    args = parse_args()
    cands = sorted(Path(args.runs_root).glob(f"*_{args.name_prefix}_{args.model}_seed*"))
    best = ""
    best_score = float("-inf")
    for d in cands:
        p = d / "metrics_summary.json"
        if not p.exists():
            continue
        s = _score(p)
        if s > best_score:
            best_score = s
            best = str(d)
    print(best)


if __name__ == "__main__":
    main()
