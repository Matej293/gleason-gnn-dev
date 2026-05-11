#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.gnn.data import load_graph_splits
from src.gnn.metrics import json_safe
from src.gnn.train import TrainConfig, run_training


PROFILES = {
    "fast": {"epochs": 40, "patience": 10, "hidden_dim": 32, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    "balanced": {"epochs": 120, "patience": 25, "hidden_dim": 64, "dropout": 0.25, "lr": 8e-4, "weight_decay": 1e-4},
    "thesis": {"epochs": 200, "patience": 40, "hidden_dim": 64, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train node classifier on superpixel graph artifacts.")
    p.add_argument("--graphs-root", required=True, type=str)
    p.add_argument("--model", choices=["mlp", "graphsage", "gcn", "gat"], default="graphsage")
    p.add_argument("--profile", choices=["fast", "balanced", "thesis"], default="thesis")
    p.add_argument("--selection-metric", type=str, default="val_per_case_macro_f1")
    p.add_argument("--normalize-features", action="store_true")
    p.add_argument("--loss", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--residual-head", action="store_true")
    p.add_argument("--residual-alpha", type=float, default=0.2)
    p.add_argument("--mask-unsupported-classes", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--feature-dropout", type=float, default=0.1)
    p.add_argument("--edge-dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--scheduler", action="store_true")
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--scheduler-patience", type=int, default=8)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="outputs/gnn_runs")
    p.add_argument("--name", type=str, default="gnn_node_classifier")
    return p.parse_args()


def _resolve_scheduler(args: argparse.Namespace) -> bool:
    if args.scheduler and args.no_scheduler:
        raise ValueError("Use only one of --scheduler or --no-scheduler")
    if args.scheduler:
        return True
    if args.no_scheduler:
        return False
    return True


def _build_cfg(args: argparse.Namespace, seed: int) -> TrainConfig:
    preset = PROFILES[args.profile]
    hidden_dim = int(args.hidden_dim) if args.hidden_dim is not None else int(preset["hidden_dim"])
    dropout = float(args.dropout) if args.dropout is not None else float(preset["dropout"])
    lr = float(args.lr) if args.lr is not None else float(preset["lr"])
    weight_decay = float(args.weight_decay) if args.weight_decay is not None else float(preset["weight_decay"])
    epochs = int(args.epochs) if args.epochs is not None else int(preset["epochs"])
    patience = int(args.patience) if args.patience is not None else int(preset["patience"])
    return TrainConfig(
        model=args.model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        feature_dropout=float(args.feature_dropout),
        edge_dropout=float(args.edge_dropout),
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        seed=seed,
        normalize_features=bool(args.normalize_features),
        use_class_weights=True,
        loss=args.loss,
        focal_gamma=float(args.focal_gamma),
        residual_head=bool(args.residual_head),
        residual_alpha=float(args.residual_alpha),
        mask_unsupported_classes=bool(args.mask_unsupported_classes),
        grad_clip_norm=float(args.grad_clip_norm),
        use_scheduler=_resolve_scheduler(args),
        scheduler_patience=int(args.scheduler_patience),
        scheduler_factor=float(args.scheduler_factor),
        scheduler_min_lr=float(args.scheduler_min_lr),
        amp=bool(args.amp),
        selection_metric=args.selection_metric,
    )


def main() -> None:
    args = parse_args()
    splits = load_graph_splits(args.graphs_root)
    run_dirs = []
    for seed_offset in range(int(args.seeds)):
        seed = int(args.seed) + seed_offset
        cfg = _build_cfg(args, seed=seed)
        exp_name = args.name if int(args.seeds) == 1 else f"{args.name}_seed{seed}"
        run_dir = run_training(splits=splits, output_root=args.output_dir, experiment_name=exp_name, cfg=cfg, graphs_root=args.graphs_root)
        run_dirs.append({"seed": seed, "run_dir": str(run_dir)})
        print(f"Saved run (seed={seed}) to: {run_dir}")

    if int(args.seeds) > 1:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir) / f"{ts}_{args.name}_multi_seed_summary.json"
        out.write_text(json.dumps(json_safe({"num_seeds": int(args.seeds), "runs": run_dirs}), indent=2) + "\n", encoding="utf-8")
        print(f"Saved multi-seed summary to: {out}")


if __name__ == "__main__":
    main()
