#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

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
COLORS = np.array(
    [
        [40, 40, 40],      # benign
        [74, 144, 226],    # g3
        [245, 158, 11],    # g4
        [220, 38, 38],     # g5
    ],
    dtype=np.uint8,
)
OVERLAY_STYLES = {
    "clean": {
        "boundary_color": (1.0, 1.0, 1.0),
        "boundary_alpha": 0.65,
        "boundary_lw": 0.6,
        "edge_color": (1.0, 1.0, 1.0),
        "edge_alpha": 0.8,
        "edge_lw": 0.7,
        "node_color": (0.0, 0.0, 0.0),
        "node_edge_color": (1.0, 1.0, 1.0),
        "node_size": 13.0,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize GNN node predictions on superpixel graphs.")
    p.add_argument("--graphs-root", required=True, type=str, help="Path like outputs/graphs/<run_name>")
    p.add_argument("--run-dir", required=True, type=str, help="Path like outputs/gnn_runs/<run_name>")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default=None, type=str)
    p.add_argument("--max-cases", default=12, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--overlay-style", default="clean", choices=sorted(OVERLAY_STYLES.keys()))
    p.add_argument("--parity-check", default="on", choices=["on", "off"])
    p.add_argument("--parity-tol", default=1e-6, type=float)
    p.add_argument("--expected-metrics", default=None, type=str)
    p.add_argument("--output-versioning", default="timestamp", choices=["timestamp", "overwrite", "require-empty"])
    return p.parse_args()



def _metric_delta(expected: float | None, observed: float | None) -> float:
    if isinstance(expected, float) and np.isnan(expected):
        expected = None
    if isinstance(observed, float) and np.isnan(observed):
        observed = None
    if expected is None and observed is None:
        return 0.0
    if expected is None or observed is None:
        return float("inf")
    return abs(float(expected) - float(observed))


def _extract_split_metrics(payload: dict, split: str) -> dict:
    split_metrics = payload.get("split_metrics")
    if not isinstance(split_metrics, dict) or split not in split_metrics:
        raise ValueError(f"Expected metrics payload missing split '{split}'.")
    m = split_metrics[split]
    if not isinstance(m, dict):
        raise ValueError("Expected split metrics must be a dict.")
    return m


def _compare_metrics(expected: dict, observed: dict, tol: float) -> tuple[bool, dict]:
    details = {
        "macro_f1": {"expected": expected.get("macro_f1"), "observed": observed.get("macro_f1")},
        "balanced_accuracy": {"expected": expected.get("balanced_accuracy"), "observed": observed.get("balanced_accuracy")},
        "per_class_f1": {},
    }
    for cls in CLASS_NAMES:
        exp_v = None
        if isinstance(expected.get("per_class_f1"), dict):
            exp_v = expected["per_class_f1"].get(cls)
        obs_v = None
        if isinstance(observed.get("per_class_f1"), dict):
            obs_v = observed["per_class_f1"].get(cls)
        details["per_class_f1"][cls] = {"expected": exp_v, "observed": obs_v}

    max_delta = 0.0
    max_delta = max(max_delta, _metric_delta(details["macro_f1"]["expected"], details["macro_f1"]["observed"]))
    max_delta = max(max_delta, _metric_delta(details["balanced_accuracy"]["expected"], details["balanced_accuracy"]["observed"]))
    for cls_vals in details["per_class_f1"].values():
        max_delta = max(max_delta, _metric_delta(cls_vals["expected"], cls_vals["observed"]))
    details["max_abs_delta"] = max_delta
    details["tolerance"] = float(tol)
    return bool(max_delta <= tol), details


def _resolve_output_dir(run_dir: Path, split: str, output_dir: Path | None, mode: str) -> Path:
    root = output_dir if output_dir is not None else (run_dir / f"viz_{split}")
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



def _colorize(label_map: np.ndarray) -> np.ndarray:
    label_map = np.clip(label_map, 0, 3)
    return COLORS[label_map]


def render_boundary_overlay(ax: plt.Axes, base_rgb: np.ndarray, boundary_mask: np.ndarray, style: dict[str, float | tuple]) -> None:
    ax.imshow(base_rgb)
    yy, xx = np.nonzero(boundary_mask)
    if yy.size > 0:
        ax.scatter(
            xx,
            yy,
            s=style["boundary_lw"],
            c=[style["boundary_color"]],
            alpha=float(style["boundary_alpha"]),
            marker="s",
            linewidths=0,
        )


def render_graph_overlay(
    ax: plt.Axes,
    base_rgb: np.ndarray,
    centroids: dict[int, tuple[float, float]],
    edges: list[tuple[int, int]],
    style: dict[str, float | tuple],
) -> None:
    ax.imshow(base_rgb)
    for a, b in edges:
        if a not in centroids or b not in centroids:
            continue
        x0, y0 = centroids[a]
        x1, y1 = centroids[b]
        ax.plot([x0, x1], [y0, y1], color=style["edge_color"], alpha=float(style["edge_alpha"]), linewidth=float(style["edge_lw"]))
    if centroids:
        order = sorted(centroids.keys())
        xs = [centroids[k][0] for k in order]
        ys = [centroids[k][1] for k in order]
        ax.scatter(
            xs,
            ys,
            s=float(style["node_size"]),
            c=[style["node_color"]],
            edgecolors=[style["node_edge_color"]],
            linewidths=0.5,
            alpha=0.95,
        )


def _plot_confusion(cm: np.ndarray, title: str, out_path: Path, normalize: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    m = cm.astype(np.float64)
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        m = m / row_sums
    im = ax.imshow(m, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            txt = f"{m[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _safe_val(v: float) -> float:
    return 0.0 if np.isnan(v) else float(v)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    graphs_root = Path(args.graphs_root)
    run_dir = Path(args.run_dir)
    output_dir = _resolve_output_dir(run_dir, args.split, Path(args.output_dir) if args.output_dir else None, args.output_versioning)
    style = OVERLAY_STYLES[args.overlay_style]

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    run_cfg_path = run_dir / "run_config.json"
    run_cfg = {}
    if run_cfg_path.exists():
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))

    meta = dict(run_cfg)
    meta.update(ckpt)
    model = _build_model_from_metadata(meta)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normalize_features = bool(ckpt.get("normalize_features", run_cfg.get("normalize_features", False)))
    norm_mean, norm_std = _resolve_norm_stats(ckpt, run_cfg, require_1d=True)
    if normalize_features and (norm_mean is None or norm_std is None):
        raise RuntimeError("normalize_features=true but normalization stats are missing from checkpoint/run_config.")

    split_dir = graphs_root / args.split
    case_paths = sorted(split_dir.glob("*/graph_data.npz"))
    if not case_paths:
        raise RuntimeError(f"No graph_data.npz under {split_dir}")
    total_case_count = len(case_paths)
    parity_enabled = (args.parity_check == "on")
    if parity_enabled and args.max_cases > 0 and args.max_cases < total_case_count:
        raise RuntimeError(
            "Parity check requires full split evaluation. "
            f"Found {total_case_count} cases but max-cases={args.max_cases}. "
            "Use --max-cases -1 (or <=0) for full split, or disable parity-check."
        )

    if args.max_cases > 0 and len(case_paths) > args.max_cases:
        idx = np.arange(len(case_paths))
        rng.shuffle(idx)
        case_paths = [case_paths[i] for i in sorted(idx[: args.max_cases].tolist())]

    per_case_dir = output_dir / "cases"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    cases_best_dir = output_dir / "cases_best_delta"
    cases_worst_dir = output_dir / "cases_worst_delta"
    cases_best_dir.mkdir(parents=True, exist_ok=True)
    cases_worst_dir.mkdir(parents=True, exist_ok=True)

    model_cases: list[CaseEval] = []
    seg_cases: list[CaseEval] = []
    case_deltas: list[tuple[str, float]] = []
    case_output_manifest: dict[str, dict[str, str]] = {}

    for case_path in case_paths:
        payload = _load_case_npz(case_path)
        image_id = case_path.parent.name

        x_raw = torch.from_numpy(payload["x"].astype(np.float32, copy=False))
        x = x_raw
        if normalize_features:
            x = _apply_norm_x(x_raw, norm_mean, norm_std)
        edge_index = torch.from_numpy(payload["edge_index"].astype(np.int64, copy=False))
        y = payload["y"].astype(np.int64, copy=False)
        node_ids = payload["node_ids"].astype(np.int64, copy=False)
        superpixels = payload["superpixels"].astype(np.int64, copy=False)

        valid = (y >= 0) & (y <= 3)
        y_true = y[valid]

        with torch.no_grad():
            raw_seg = None
            if bool(getattr(model, "residual_uses_raw_seg_probs", False)):
                seg_start, seg_end = getattr(model, "seg_prob_idx", (9, 13))
                raw_seg = x_raw[:, int(seg_start) : int(seg_end)]
            logits = model(x, edge_index, raw_seg_probs=raw_seg)
            pred_model = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
        pred_seg = seg_only_predict(x_raw).cpu().numpy().astype(np.int64)

        model_cases.append(CaseEval(image_id=image_id, y_true=y_true, y_pred=pred_model[valid]))
        seg_cases.append(CaseEval(image_id=image_id, y_true=y_true, y_pred=pred_seg[valid]))

        m_model = aggregate_case_metrics([model_cases[-1]])[1]
        m_seg = aggregate_case_metrics([seg_cases[-1]])[1]
        case_deltas.append((image_id, _safe_val(m_model["macro_f1"]) - _safe_val(m_seg["macro_f1"])))

        gt_map = _labels_to_map(superpixels, node_ids, y)
        seg_map = _labels_to_map(superpixels, node_ids, pred_seg)
        model_map = _labels_to_map(superpixels, node_ids, pred_model)
        boundaries = extract_superpixel_boundaries(superpixels)
        centroids = extract_node_centroids(node_ids, superpixels)
        graph_edges = unique_undirected_edges(payload["edge_index"].astype(np.int64, copy=False))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()
        axes[0].imshow(_colorize(gt_map))
        axes[0].set_title("GT Node Labels")
        axes[1].imshow(_colorize(seg_map))
        axes[1].set_title("Seg-Only Baseline")
        axes[2].imshow(_colorize(model_map))
        axes[2].set_title(f"{str(ckpt['model']).upper()}")
        diff = (model_map != gt_map).astype(np.float32) - (seg_map != gt_map).astype(np.float32)
        im = axes[3].imshow(diff, cmap="bwr", vmin=-1.0, vmax=1.0)
        axes[3].set_title("Error Delta\n(blue=model better)")
        render_boundary_overlay(axes[4], _colorize(seg_map), boundaries, style)
        axes[4].set_title("Superpixel Map")
        render_graph_overlay(axes[5], _colorize(model_map), centroids, graph_edges, style)
        axes[5].set_title("Graph Overlay")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        fig.suptitle(image_id)
        fig.tight_layout()
        case_panel_path = per_case_dir / f"{image_id}.png"
        fig.savefig(case_panel_path, dpi=170)
        plt.close(fig)

        fig_sp, ax_sp = plt.subplots(1, 1, figsize=(5.6, 5.6))
        render_boundary_overlay(ax_sp, _colorize(seg_map), boundaries, style)
        ax_sp.set_title(f"{image_id} Superpixels")
        ax_sp.axis("off")
        fig_sp.tight_layout()
        case_superpixel_path = per_case_dir / f"{image_id}_superpixels.png"
        fig_sp.savefig(case_superpixel_path, dpi=170)
        plt.close(fig_sp)

        fig_go, ax_go = plt.subplots(1, 1, figsize=(5.6, 5.6))
        render_graph_overlay(ax_go, _colorize(model_map), centroids, graph_edges, style)
        ax_go.set_title(f"{image_id} Graph Overlay")
        ax_go.axis("off")
        fig_go.tight_layout()
        case_graph_path = per_case_dir / f"{image_id}_graph_overlay.png"
        fig_go.savefig(case_graph_path, dpi=170)
        plt.close(fig_go)
        case_output_manifest[image_id] = {
            "panel": str(case_panel_path),
            "superpixels": str(case_superpixel_path),
            "graph_overlay": str(case_graph_path),
        }

    model_micro, model_per_case, model_cm, _ = aggregate_case_metrics(model_cases)
    seg_micro, seg_per_case, seg_cm, _ = aggregate_case_metrics(seg_cases)

    observed_split_metrics = {
        "macro_f1": model_per_case["macro_f1"],
        "balanced_accuracy": model_per_case["balanced_accuracy"],
        "per_class_f1": model_per_case["per_class_f1"],
    }
    expected_metrics_path = Path(args.expected_metrics) if args.expected_metrics else (run_dir / "metrics_summary.json")
    parity_passed = True
    parity_details = None
    if parity_enabled:
        expected_payload = json.loads(expected_metrics_path.read_text(encoding="utf-8"))
        expected_split_metrics = _extract_split_metrics(expected_payload, args.split)
        parity_passed, parity_details = _compare_metrics(expected_split_metrics, observed_split_metrics, float(args.parity_tol))
        if not parity_passed:
            raise RuntimeError(
                "Parity check failed for visualization inference. "
                f"split={args.split}, tol={args.parity_tol}, details={json.dumps(json_safe(parity_details))}"
            )

    _plot_confusion(seg_cm, "Seg-Only Confusion Matrix", output_dir / "confusion_seg_only.png")
    _plot_confusion(model_cm, f"{str(ckpt['model']).upper()} Confusion Matrix", output_dir / "confusion_model.png")
    _plot_confusion(seg_cm, "Seg-Only Confusion Matrix (Norm)", output_dir / "confusion_seg_only_norm.png", normalize=True)
    _plot_confusion(model_cm, f"{str(ckpt['model']).upper()} Confusion Matrix (Norm)", output_dir / "confusion_model_norm.png", normalize=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xloc = np.arange(len(CLASS_NAMES))
    seg_vals = [_safe_val(seg_per_case["per_class_f1"][k]) for k in CLASS_NAMES]
    model_vals = [_safe_val(model_per_case["per_class_f1"][k]) for k in CLASS_NAMES]
    w = 0.35
    ax.bar(xloc - w / 2, seg_vals, width=w, label="seg_only")
    ax.bar(xloc + w / 2, model_vals, width=w, label=str(ckpt["model"]))
    ax.set_xticks(xloc, CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-Case Mean F1")
    ax.set_title("Per-Class F1 Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_f1_comparison.png", dpi=170)
    plt.close(fig)

    case_deltas_sorted = sorted(case_deltas, key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [v for _, v in case_deltas_sorted]
    labels = [k for k, _ in case_deltas_sorted]
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Per-Case Macro-F1 Delta (Model - Seg-Only)")
    ax.set_ylabel("Delta Macro F1")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "per_case_macro_f1_delta.png", dpi=170)
    plt.close(fig)

    worst = [k for k, _ in case_deltas_sorted[:3]]
    best = [k for k, _ in case_deltas_sorted[-3:]]
    best_manifest = {k: case_output_manifest[k] for k in best if k in case_output_manifest}
    worst_manifest = {k: case_output_manifest[k] for k in worst if k in case_output_manifest}
    (cases_best_dir / "manifest.json").write_text(json.dumps(json_safe(best_manifest), indent=2) + "\n", encoding="utf-8")
    (cases_worst_dir / "manifest.json").write_text(json.dumps(json_safe(worst_manifest), indent=2) + "\n", encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "graphs_root": str(graphs_root),
        "split": args.split,
        "checkpoint_path": str(ckpt_path),
        "run_config_path": str(run_cfg_path),
        "expected_metrics_path": str(expected_metrics_path),
        "parity_check_enabled": parity_enabled,
        "parity_tolerance": float(args.parity_tol),
        "parity_passed": parity_passed,
        "parity_compared_metrics": parity_details,
        "num_cases_visualized": len(case_paths),
        "metrics": {
            "seg_only": {
                "micro_over_nodes": seg_micro,
                "per_case_mean": seg_per_case,
            },
            "model": {
                "model_name": str(ckpt["model"]),
                "micro_over_nodes": model_micro,
                "per_case_mean": model_per_case,
            },
        },
        "output_files": {
            "confusion_seg_only": str(output_dir / "confusion_seg_only.png"),
            "confusion_model": str(output_dir / "confusion_model.png"),
            "per_class_f1_comparison": str(output_dir / "per_class_f1_comparison.png"),
            "per_case_macro_f1_delta": str(output_dir / "per_case_macro_f1_delta.png"),
            "cases_dir": str(per_case_dir),
            "case_superpixels_pattern": str(per_case_dir / "*_superpixels.png"),
            "case_graph_overlay_pattern": str(per_case_dir / "*_graph_overlay.png"),
            "cases_best_delta_dir": str(cases_best_dir),
            "cases_worst_delta_dir": str(cases_worst_dir),
        },
        "overlay_style": str(args.overlay_style),
        "clean_thesis_style": style,
        "case_outputs": case_output_manifest,
    }
    summary["best_delta_cases"] = best
    summary["worst_delta_cases"] = worst
    (output_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
