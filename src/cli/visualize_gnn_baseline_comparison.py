#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipelines.gnn.baselines import seg_only_predict
from src.pipelines.gnn.metrics import CaseEval, aggregate_case_metrics, json_safe
from src.pipelines.gnn.viz_runtime import (
    apply_norm_x as _apply_norm_x,
    build_model_from_metadata as _build_model_from_metadata,
    labels_to_map as _labels_to_map,
    load_case_npz as _load_case_npz,
    resolve_norm_stats as _resolve_norm_stats,
    seg_prob_idx_from_meta as _seg_prob_idx_from_meta,
)
from src.pipelines.graph.viz_helpers import (
    extract_node_centroids,
    extract_superpixel_boundaries,
    unique_undirected_edges,
)

CLASS_NAMES = ["benign", "g3", "g4", "g5"]
DEFAULT_MODELS = ["seg_only", "mlp", "graphsage", "gcn", "gat"]
COLORS = np.array(
    [
        [40, 40, 40],
        [74, 144, 226],
        [245, 158, 11],
        [220, 38, 38],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize baseline comparison outputs for GNN methods.")
    p.add_argument("--comparison-dir", required=True, type=str)
    p.add_argument("--graphs-root", required=True, type=str)
    p.add_argument("--gnn-runs-root", default="outputs/gnn_runs", type=str)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default=None, type=str)
    p.add_argument("--output-versioning", default="timestamp", choices=["timestamp", "overwrite", "require-empty"])
    p.add_argument("--max-cases", default=12, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--log-wandb", dest="log_wandb", action="store_true")
    p.add_argument("--no-log-wandb", dest="log_wandb", action="store_false")
    p.set_defaults(log_wandb=True)
    p.add_argument("--wandb-project", default="prostate-lesion-segmentation", type=str)
    p.add_argument("--wandb-entity", default=None, type=str)
    p.add_argument("--wandb-run-name", default=None, type=str)
    p.add_argument("--wandb-tags", nargs="+", default=None)
    p.add_argument("--wandb-log-max-case-images", default=24, type=int)
    return p.parse_args()


def _parse_ts(path: Path) -> datetime:
    prefix = path.name[:15]
    return datetime.strptime(prefix, "%Y%m%d_%H%M%S")


def _resolve_output_dir(comparison_dir: Path, output_dir: Path | None, split: str, mode: str) -> Path:
    root = output_dir if output_dir is not None else (comparison_dir / f"viz_{split}")
    if mode == "overwrite":
        root.mkdir(parents=True, exist_ok=True)
        return root
    if mode == "require-empty":
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Output directory must be empty for require-empty mode: {root}")
        root.mkdir(parents=True, exist_ok=True)
        return root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = root / ts
    target.mkdir(parents=True, exist_ok=False)
    return target


def _load_and_validate_payload(comparison_dir: Path) -> dict:
    payload_path = comparison_dir / "baseline_comparison.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing baseline comparison payload: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("results"), dict):
        raise ValueError("Invalid payload: expected 'results' dict.")
    if not isinstance(payload.get("per_case_arrays"), dict):
        raise ValueError("Invalid payload: expected 'per_case_arrays' dict.")
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict) or not isinstance(verdict.get("leaderboard_test_macro_f1"), list):
        raise ValueError("Invalid payload: expected 'verdict.leaderboard_test_macro_f1'.")
    return payload


def _resolve_run_dirs(comparison_dir: Path, gnn_runs_root: Path, models: list[str]) -> dict[str, Path]:
    comparison_ts = _parse_ts(comparison_dir)
    out: dict[str, Path] = {}
    for method in models:
        if method == "seg_only":
            continue
        candidates = sorted(gnn_runs_root.glob(f"*_baseline_{method}"))
        if not candidates:
            raise FileNotFoundError(f"Could not resolve run directory for model '{method}' under {gnn_runs_root}")
        parsed = []
        for p in candidates:
            try:
                ts = _parse_ts(p)
            except ValueError:
                continue
            parsed.append((abs((ts - comparison_ts).total_seconds()), p))
        if not parsed:
            raise RuntimeError(f"No timestamped run dirs found for model '{method}' under {gnn_runs_root}")
        out[method] = min(parsed, key=lambda t: t[0])[1]
    return out



def _colorize(label_map: np.ndarray) -> np.ndarray:
    return COLORS[np.clip(label_map, 0, 3)]


def render_boundary_overlay(ax: plt.Axes, base_rgb: np.ndarray, boundary_mask: np.ndarray) -> None:
    ax.imshow(base_rgb)
    yy, xx = np.nonzero(boundary_mask)
    if yy.size > 0:
        ax.scatter(xx, yy, s=0.6, c=[(1.0, 1.0, 1.0)], alpha=0.65, marker="s", linewidths=0)


def render_graph_overlay(ax: plt.Axes, base_rgb: np.ndarray, centroids: dict[int, tuple[float, float]], edges: list[tuple[int, int]]) -> None:
    ax.imshow(base_rgb)
    for a, b in edges:
        if a in centroids and b in centroids:
            x0, y0 = centroids[a]
            x1, y1 = centroids[b]
            ax.plot([x0, x1], [y0, y1], color=(1.0, 1.0, 1.0), alpha=0.8, linewidth=0.7)
    if centroids:
        order = sorted(centroids.keys())
        xs = [centroids[k][0] for k in order]
        ys = [centroids[k][1] for k in order]
        ax.scatter(xs, ys, s=13.0, c=[(0.0, 0.0, 0.0)], edgecolors=[(1.0, 1.0, 1.0)], linewidths=0.5, alpha=0.95)


def _safe(v: float | None) -> float:
    if v is None:
        return 0.0
    if isinstance(v, float) and np.isnan(v):
        return 0.0
    return float(v)


def _load_run_artifacts(run_dir: Path) -> dict:
    ckpt_path = run_dir / "best.pt"
    run_cfg_path = run_dir / "run_config.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_cfg_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    meta = dict(run_cfg)
    meta.update(ckpt)
    model = _build_model_from_metadata(meta)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    normalize_features = bool(ckpt.get("normalize_features", run_cfg.get("normalize_features", False)))
    norm_mean, norm_std = _resolve_norm_stats(ckpt, run_cfg)
    if normalize_features and (norm_mean is None or norm_std is None):
        raise RuntimeError(f"normalize_features=true but normalization stats are missing for run: {run_dir}")
    return {
        "model": model,
        "normalize_features": normalize_features,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


def _plot_leaderboard(payload: dict, out: Path) -> str:
    rows = payload["verdict"]["leaderboard_test_macro_f1"]
    methods = [r["method"] for r in rows]
    values = [_safe(r.get("test_macro_f1")) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ypos = np.arange(len(methods))
    ax.barh(ypos, values, color="#2563eb")
    ax.set_yticks(ypos, methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Leaderboard: Test Macro-F1")
    fig.tight_layout()
    p = out / "leaderboard_test_macro_f1.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def _plot_macro_by_split(results: dict, models: list[str], out: Path) -> str:
    splits = ["train", "val", "test"]
    x = np.arange(len(splits))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        vals = [_safe(results[m][s]["macro_f1"]) for s in splits]
        ax.bar(x - 0.4 + width * (i + 0.5), vals, width=width, label=m)
    ax.set_xticks(x, splits)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 by Split")
    ax.legend()
    fig.tight_layout()
    p = out / "macro_f1_by_split.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def _plot_delta_vs_seg(results: dict, models: list[str], out: Path) -> str:
    seg_m = _safe(results["seg_only"]["test"]["macro_f1"])
    seg_ba = _safe(results["seg_only"]["test"]["balanced_accuracy"])
    others = [m for m in models if m != "seg_only"]
    x = np.arange(len(others))
    macro = [_safe(results[m]["test"]["macro_f1"]) - seg_m for m in others]
    ba = [_safe(results[m]["test"]["balanced_accuracy"]) - seg_ba for m in others]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38
    ax.bar(x - w / 2, macro, width=w, label="delta macro_f1")
    ax.bar(x + w / 2, ba, width=w, label="delta balanced_accuracy")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, others)
    ax.set_title("Test Delta vs seg_only")
    ax.legend()
    fig.tight_layout()
    p = out / "delta_vs_seg_only_test.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def _plot_per_class_f1_test(results: dict, models: list[str], out: Path) -> str:
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        per_class = results[m]["test"]["per_class_f1"]
        vals = [_safe(per_class.get(c)) for c in CLASS_NAMES]
        ax.bar(x - 0.4 + width * (i + 0.5), vals, width=width, label=m)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-case Mean F1")
    ax.set_title("Per-Class F1 (Test)")
    ax.legend()
    fig.tight_layout()
    p = out / "per_class_f1_test.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def _plot_per_case_delta_overlay(case_ids: list[str], per_case_arrays: dict, models: list[str], split: str, out: Path) -> str:
    seg = np.asarray(per_case_arrays["seg_only"][split]["macro_f1"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(case_ids))
    for m in models:
        if m == "seg_only":
            continue
        arr = np.asarray(per_case_arrays[m][split]["macro_f1"], dtype=np.float64)
        ax.plot(x, arr - seg, label=f"{m} - seg_only", linewidth=1.4, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=90, fontsize=7)
    ax.set_ylabel("Macro-F1 Delta")
    ax.set_title("Per-Case Macro-F1 Delta Overlay")
    ax.legend()
    fig.tight_layout()
    p = out / "per_case_macro_f1_delta_overlay.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def _choose_cases(case_ids: list[str], per_case_arrays: dict, split: str, best_method: str, max_cases: int) -> list[str]:
    seg = np.asarray(per_case_arrays["seg_only"][split]["macro_f1"], dtype=np.float64)
    method_names = [m for m in per_case_arrays.keys() if m != "seg_only"]
    matrix = np.stack([np.asarray(per_case_arrays[m][split]["macro_f1"], dtype=np.float64) for m in method_names], axis=0)
    spread = np.max(matrix, axis=0) - np.min(matrix, axis=0)
    best = np.asarray(per_case_arrays[best_method][split]["macro_f1"], dtype=np.float64)
    delta = best - seg
    k = len(case_ids) if max_cases <= 0 else min(max_cases, len(case_ids))
    k_each = max(1, min(k, max(1, k // 3)))
    top_spread = np.argsort(-spread)[:k_each]
    top_vs = np.argsort(-delta)[:k_each]
    worst_vs = np.argsort(delta)[:k_each]
    order = [*top_spread.tolist(), *top_vs.tolist(), *worst_vs.tolist()]
    selected = []
    for idx in order:
        cid = case_ids[int(idx)]
        if cid not in selected:
            selected.append(cid)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for idx in np.argsort(-spread).tolist():
            cid = case_ids[int(idx)]
            if cid not in selected:
                selected.append(cid)
            if len(selected) >= k:
                break
    return selected


def _maybe_init_wandb(args: argparse.Namespace, output_dir: Path, summary: dict) -> Any | None:
    if not args.log_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[warn] WandB logging requested but unavailable ({exc}); continuing without WandB.")
        return None
    run_name = args.wandb_run_name or f"gnn_compare_viz_{output_dir.name}"
    tags = args.wandb_tags if args.wandb_tags is not None else ["gnn", "comparison-viz", args.split]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=tags,
        config=json_safe(summary.get("config", {})),
    )


def _wandb_log_outputs(
    run: Any,
    output_files: dict[str, str],
    per_case_manifest: dict[str, str],
    summary: dict,
    max_case_images: int,
) -> None:
    if run is None:
        return
    import wandb

    metrics = summary.get("selected_case_metrics_per_case_mean", {})
    leaderboard = summary.get("leaderboard_snapshot", [])
    best_row = leaderboard[0] if leaderboard else {}
    run.summary["best_method"] = best_row.get("method")
    run.summary["best_test_macro_f1"] = best_row.get("test_macro_f1")
    run.summary["best_test_balanced_accuracy"] = best_row.get("test_balanced_accuracy")
    run.summary["selected_case_count"] = len(per_case_manifest)
    run.summary["output_dir"] = summary.get("config", {}).get("output_dir")
    run.summary["source_comparison_dir"] = summary.get("source", {}).get("comparison_dir")
    run.summary["source_graphs_root"] = summary.get("source", {}).get("graphs_root")
    for method, vals in metrics.items():
        for metric_name, metric_val in vals.items():
            run.summary[f"selected/{method}/{metric_name}"] = metric_val

    log_payload = {}
    for key, file_path in output_files.items():
        p = Path(file_path)
        if p.exists():
            log_payload[f"plots/{key}"] = wandb.Image(str(p), caption=key)

    items = sorted(per_case_manifest.items())
    if max_case_images >= 0:
        items = items[:max_case_images]
    for image_id, file_path in items:
        p = Path(file_path)
        if p.exists():
            log_payload[f"cases/{image_id}"] = wandb.Image(str(p), caption=image_id)

    if log_payload:
        run.log(log_payload)
    summary_path = Path(summary["config"]["output_dir"]) / "comparison_viz_summary.json"
    if summary_path.exists():
        artifact = wandb.Artifact("comparison_viz_summary", type="report")
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact)
    run.finish()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    _ = rng  # deterministic selection hook retained for future random tie-breaks.

    comparison_dir = Path(args.comparison_dir)
    graphs_root = Path(args.graphs_root)
    gnn_runs_root = Path(args.gnn_runs_root)
    payload = _load_and_validate_payload(comparison_dir)
    results = payload["results"]
    per_case_arrays = payload["per_case_arrays"]

    models = args.models if args.models is not None else [m for m in DEFAULT_MODELS if m in results]
    if "seg_only" not in models:
        models = ["seg_only"] + models
    required = ["seg_only", "mlp", "graphsage", "gcn", "gat"]
    missing = [m for m in required if m not in models or m not in results or m not in per_case_arrays]
    if missing:
        raise RuntimeError(f"V1 requires seg_only, mlp, graphsage, gcn, gat in payload/models. Missing: {missing}")

    run_dirs = _resolve_run_dirs(comparison_dir, gnn_runs_root, models)
    run_artifacts = {m: _load_run_artifacts(run_dirs[m]) for m in models if m != "seg_only"}

    split_dir = graphs_root / args.split
    case_paths = sorted(split_dir.glob("*/graph_data.npz"))
    if not case_paths:
        raise RuntimeError(f"No graph_data.npz under {split_dir}")
    case_ids = [p.parent.name for p in case_paths]

    for method in models:
        arr = per_case_arrays[method][args.split]["macro_f1"]
        if len(arr) != len(case_ids):
            raise RuntimeError(
                f"Case count mismatch for method '{method}' split '{args.split}': "
                f"payload={len(arr)} graph_cases={len(case_ids)}"
            )

    output_dir = _resolve_output_dir(comparison_dir, Path(args.output_dir) if args.output_dir else None, args.split, args.output_versioning)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}
    output_files["leaderboard_test_macro_f1"] = _plot_leaderboard(payload, output_dir)
    output_files["macro_f1_by_split"] = _plot_macro_by_split(results, models, output_dir)
    output_files["delta_vs_seg_only_test"] = _plot_delta_vs_seg(results, models, output_dir)
    output_files["per_class_f1_test"] = _plot_per_class_f1_test(results, models, output_dir)
    output_files["per_case_macro_f1_delta_overlay"] = _plot_per_case_delta_overlay(case_ids, per_case_arrays, models, args.split, output_dir)

    leaderboard = payload["verdict"]["leaderboard_test_macro_f1"]
    best_method = leaderboard[0]["method"] if leaderboard else "mlp"
    selected_case_ids = _choose_cases(case_ids, per_case_arrays, args.split, best_method=best_method, max_cases=args.max_cases)
    selected_set = set(selected_case_ids)
    selected_paths = [p for p in case_paths if p.parent.name in selected_set]

    per_case_manifest = {}
    partial_artifacts: list[str] = []
    method_case_metrics: dict[str, list[CaseEval]] = {m: [] for m in models}

    for case_path in selected_paths:
        image_id = case_path.parent.name
        d = _load_case_npz(case_path)
        node_ids = d["node_ids"].astype(np.int64, copy=False)
        x_raw = torch.from_numpy(d["x"].astype(np.float32, copy=False))
        edge_index = torch.from_numpy(d["edge_index"].astype(np.int64, copy=False))
        y = d["y"].astype(np.int64, copy=False)
        valid = (y >= 0) & (y <= 3)

        preds: dict[str, np.ndarray] = {}
        preds["seg_only"] = seg_only_predict(x_raw).cpu().numpy().astype(np.int64)
        for m in models:
            if m == "seg_only":
                continue
            artifact = run_artifacts[m]
            x = x_raw
            if artifact["normalize_features"]:
                x = _apply_norm_x(x_raw, artifact["norm_mean"], artifact["norm_std"])
            with torch.no_grad():
                raw_seg = None
                if bool(getattr(artifact["model"], "residual_uses_raw_seg_probs", False)):
                    seg_start, seg_end = getattr(artifact["model"], "seg_prob_idx", (9, 13))
                    raw_seg = x_raw[:, int(seg_start) : int(seg_end)]
                logits = artifact["model"](x, edge_index, raw_seg_probs=raw_seg)
                preds[m] = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

        gt_map = _labels_to_map(d["superpixels"], node_ids, y)
        pred_maps = {m: _labels_to_map(d["superpixels"], node_ids, preds[m]) for m in models}
        boundaries = extract_superpixel_boundaries(d["superpixels"])
        centroids = extract_node_centroids(node_ids, d["superpixels"])
        edges = unique_undirected_edges(d["edge_index"].astype(np.int64, copy=False))

        ordered = ["seg_only", "mlp", "graphsage", "gcn", "gat"]
        fig, axes = plt.subplots(2, 7, figsize=(26, 8.4))
        top = axes[0]
        bot = axes[1]
        top[0].imshow(_colorize(gt_map))
        top[0].set_title("GT")
        for i, m in enumerate(ordered, start=1):
            top[i].imshow(_colorize(pred_maps[m]))
            top[i].set_title(m)
        best_map = pred_maps[best_method] if best_method in pred_maps else pred_maps["mlp"]
        seg_map = pred_maps["seg_only"]
        diff = (best_map != gt_map).astype(np.float32) - (seg_map != gt_map).astype(np.float32)
        im = top[6].imshow(diff, cmap="bwr", vmin=-1.0, vmax=1.0)
        top[6].set_title("error-highlights")
        render_boundary_overlay(bot[0], _colorize(seg_map), boundaries)
        bot[0].set_title("superpixels")
        for i, m in enumerate(ordered, start=1):
            render_graph_overlay(bot[i], _colorize(pred_maps[m]), centroids, edges)
            bot[i].set_title(f"{m} graph")
        render_graph_overlay(bot[6], _colorize(best_map), centroids, edges)
        bot[6].set_title("best graph")
        for ax in axes.ravel():
            ax.axis("off")
        fig.colorbar(im, ax=top[6], fraction=0.046, pad=0.04)
        fig.suptitle(image_id)
        fig.tight_layout()
        out_path = cases_dir / f"{image_id}_comparison.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        per_case_manifest[image_id] = str(out_path)

        for m in models:
            method_case_metrics[m].append(CaseEval(image_id=image_id, y_true=y[valid], y_pred=preds[m][valid]))

    if len(per_case_manifest) != len(selected_case_ids):
        partial_artifacts.append("some selected cases were not rendered")

    summary = {
        "source": {
            "comparison_dir": str(comparison_dir),
            "comparison_payload": str(comparison_dir / "baseline_comparison.json"),
            "graphs_root": str(graphs_root),
            "gnn_runs_root": str(gnn_runs_root),
        },
        "split": args.split,
        "selected_models": models,
        "resolved_run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "generated_files": output_files,
        "cases_dir": str(cases_dir),
        "case_montages": per_case_manifest,
        "leaderboard_snapshot": payload["verdict"]["leaderboard_test_macro_f1"],
        "selected_case_ids": selected_case_ids,
        "missing_or_partial_artifacts": partial_artifacts,
        "config": {
            "seed": args.seed,
            "max_cases": args.max_cases,
            "output_versioning": args.output_versioning,
            "output_dir": str(output_dir),
        },
        "selected_case_metrics_per_case_mean": {},
    }
    for m in models:
        _, per_case_mean, _, _ = aggregate_case_metrics(method_case_metrics[m])
        summary["selected_case_metrics_per_case_mean"][m] = per_case_mean
    (output_dir / "comparison_viz_summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    run = _maybe_init_wandb(args, output_dir, summary)
    _wandb_log_outputs(
        run=run,
        output_files=output_files,
        per_case_manifest=per_case_manifest,
        summary=summary,
        max_case_images=args.wandb_log_max_case_images,
    )
    print(f"Saved comparison visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
