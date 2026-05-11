#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import load_config
from src.eval_utils import collate_consensus_batch, resolve_split_manifest_path, safe_read_json
from src.gleason_consensus_dataset import GleasonConsensusDataset
from src.graph_pipeline import (
    assign_majority_node_labels,
    build_edges,
    compute_node_features,
    generate_slic_superpixels,
)
from src.models import build_model
from src.utils import ensure_cuda_binary_compatibility, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build superpixel graph artifacts from model predictions. "
            "Uses a trained checkpoint from a specific run directory."
        )
    )
    p.add_argument(
        "--run",
        required=True,
        type=str,
        help="Run directory containing config.yaml and checkpoints/.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename in run/checkpoints or absolute path (default: best.pt).",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        required=True,
        help="Which split to export graphs for.",
    )
    p.add_argument("--output-dir", type=str, default="outputs/graphs")
    p.add_argument("--num-segments", type=int, default=300)
    p.add_argument("--compactness", type=float, default=10.0)
    p.add_argument("--superpixel-preset", choices=["low", "med", "high"], default=None)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--edge-policy", choices=["touch", "knn", "touch_plus_knn"], default="touch")
    p.add_argument("--edge-knn-k", type=int, default=2)
    p.add_argument("--edge-knn-max-distance", type=float, default=0.0, help="0 disables distance threshold.")
    p.add_argument("--min-majority-fraction", type=float, default=0.6)
    p.add_argument("--tiny-superpixel-max-pixels", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=None, help="Override loader batch size.")
    p.add_argument("--num-workers", type=int, default=None, help="Override loader worker count.")
    p.add_argument(
        "--min-supervised-ratio",
        type=float,
        default=0.01,
        help="Fail build if supervised node ratio falls below this threshold.",
    )
    return p.parse_args()


def _resolve_superpixel_params(args: argparse.Namespace) -> tuple[int, float]:
    if args.superpixel_preset is None:
        return int(args.num_segments), float(args.compactness)
    presets = {
        "low": (220, 6.0),
        "med": (300, 10.0),
        "high": (420, 16.0),
    }
    return presets[args.superpixel_preset]


def _resolve_resize_divisor(cfg: dict) -> int:
    model_name = str(cfg.get("model", "deconver")).strip().lower()
    if model_name == "deconver":
        deconver_strides = tuple(int(x) for x in cfg.get("deconver_strides", [1, 2, 2, 2]))
        return int(math.prod([s for s in deconver_strides if s > 1])) or 1
    return 8


def _resolve_checkpoint(run_dir: Path, checkpoint_arg: str) -> Path:
    ckpt_arg = Path(checkpoint_arg)
    if ckpt_arg.exists():
        return ckpt_arg.resolve()
    ckpt = run_dir / "checkpoints" / checkpoint_arg
    if ckpt.exists():
        return ckpt.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def _extract_logits(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        logits = out.get("out")
        if isinstance(logits, torch.Tensor):
            return logits
        raise TypeError("Model output dict must contain tensor under key 'out'.")
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unsupported model output type: {type(out)!r}")


def _select_indices(dataset: GleasonConsensusDataset, split_manifest: Path, split: str) -> list[int]:
    manifest = safe_read_json(split_manifest)
    if split == "all":
        ids = set(str(x.get("image_id", "")) for x in dataset.items)
    else:
        key = f"{split}_image_ids"
        ids = set(str(x) for x in manifest.get(key, []))
        if not ids:
            raise RuntimeError(f"No IDs found for split key: {key}")
    return [i for i, item in enumerate(dataset.items) if str(item.get("image_id", "")) in ids]


def main() -> None:
    args = parse_args()
    num_segments, compactness = _resolve_superpixel_params(args)
    run_dir = Path(args.run).resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config missing: {cfg_path}")
    cfg = load_config(str(cfg_path))

    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    ensure_cuda_binary_compatibility(device)

    model = build_model(cfg).to(device)
    load_checkpoint(str(ckpt_path), model=model, device=device)
    model.eval()

    dataset = GleasonConsensusDataset(
        data_root=cfg["data_root"],
        consensus_root=cfg["consensus_root"],
        image_subdirs=tuple(str(x) for x in cfg.get("image_subdirs", ["Train_imgs", "Test_imgs"])),
        transform=None,
        renormalize_probs=bool(cfg.get("renormalize_probs", True)),
        enforce_background_ignore=bool(cfg.get("enforce_background_ignore", True)),
        otsu_close_radius=int(cfg.get("otsu_close_radius", 3)),
        otsu_min_object_size=int(cfg.get("otsu_min_object_size", 4096)),
        otsu_min_hole_size=int(cfg.get("otsu_min_hole_size", 4096)),
        probs_eps=float(cfg.get("probs_eps", 1e-8)),
        load_qc_report=False,
        max_long_side=int(cfg.get("max_long_side", 0)) or None,
        resize_divisor=_resolve_resize_divisor(cfg),
    )

    split_manifest = resolve_split_manifest_path(cfg)
    indices = _select_indices(dataset, split_manifest, args.split)
    subset = Subset(dataset, indices)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("val_batch_size", 1))
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.get("num_workers", 0))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_consensus_batch,
    )

    run_out = Path(args.output_dir).resolve() / run_dir.name / args.split
    run_out.mkdir(parents=True, exist_ok=True)
    per_graph_stats: list[dict[str, float | int | str]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Build checkpoint graphs", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"]
            ignore_mask = batch["ignore_mask"]
            tissue_mask = batch["tissue_mask"]
            image_ids = [str(x) for x in batch["image_id"]]

            out = model(images)
            logits = _extract_logits(out)
            probs = torch.softmax(logits.float(), dim=1).detach().cpu()

            for i, image_id in enumerate(image_ids):
                image_rgb = (batch["image"][i].numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                sp = generate_slic_superpixels(
                    image_rgb=image_rgb,
                    tissue_mask=tissue_mask[i].numpy().astype(np.uint8),
                    num_segments=num_segments,
                    compactness=compactness,
                    sigma=args.sigma,
                )
                edge_index = build_edges(
                    sp,
                    policy=args.edge_policy,
                    knn_k=int(args.edge_knn_k),
                    knn_max_distance=None if float(args.edge_knn_max_distance) <= 0.0 else float(args.edge_knn_max_distance),
                )
                y, train_mask = assign_majority_node_labels(
                    superpixels=sp,
                    hard_mask=hard_mask[i].numpy().astype(np.int64),
                    ignore_mask=ignore_mask[i].numpy().astype(np.uint8),
                    min_majority_fraction=args.min_majority_fraction,
                )
                node_ids, x = compute_node_features(
                    image_rgb=image_rgb,
                    superpixels=sp,
                    seg_probs=probs[i].numpy().astype(np.float32),
                )
                feature_version = "v2" if x.shape[1] >= 22 else "v1"
                valid_sp = sp[sp >= 0]
                counts_sp = np.bincount(valid_sp) if valid_sp.size else np.zeros((0,), dtype=np.int64)
                num_nodes = int(node_ids.shape[0])
                edge_count_undirected = int(edge_index.shape[1] // 2)
                tiny_count = int((counts_sp <= int(args.tiny_superpixel_max_pixels)).sum()) if counts_sp.size else 0
                empty_or_degenerate = int((counts_sp <= 1).sum()) if counts_sp.size else 0
                per_graph_stats.append(
                    {
                        "image_id": image_id,
                        "num_nodes": num_nodes,
                        "num_edges_undirected": edge_count_undirected,
                        "tiny_superpixel_fraction": float(tiny_count / max(num_nodes, 1)),
                        "empty_or_degenerate_superpixel_fraction": float(empty_or_degenerate / max(num_nodes, 1)),
                    }
                )

                out_dir = run_out / image_id
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_dir / "graph_data.npz",
                    node_ids=node_ids,
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    train_mask=train_mask.astype(np.uint8),
                    superpixels=sp.astype(np.int32),
                )

    counts = {"benign": 0, "g3": 0, "g4": 0, "g5": 0}
    supervised_counts = {"benign": 0, "g3": 0, "g4": 0, "g5": 0}
    supervised = 0
    total_nodes = 0
    invalid = 0
    isolated = 0
    feature_dim = None
    for graph_path in sorted(run_out.glob("*/graph_data.npz")):
        d = np.load(graph_path)
        y = d["y"].astype(np.int64)
        tm = d["train_mask"].astype(np.bool_)
        ei = d["edge_index"]
        x = d["x"]
        feature_dim = int(x.shape[1])
        for i,k in enumerate(["benign","g3","g4","g5"]):
            counts[k] += int((y == i).sum())
            supervised_counts[k] += int(((y == i) & tm).sum())
        supervised += int(tm.sum())
        total_nodes += int(y.shape[0])
        invalid += int(((y < 0) | (y > 3)).sum())
        deg = np.zeros((x.shape[0],), dtype=np.int64)
        if ei.size:
            deg += np.bincount(ei[0], minlength=x.shape[0])
        isolated += int((deg == 0).sum())

    node_counts = np.array([int(r["num_nodes"]) for r in per_graph_stats], dtype=np.int64)
    edge_counts = np.array([int(r["num_edges_undirected"]) for r in per_graph_stats], dtype=np.int64)
    tiny_fracs = np.array([float(r["tiny_superpixel_fraction"]) for r in per_graph_stats], dtype=np.float32)
    deg_fracs = np.array([float(r["empty_or_degenerate_superpixel_fraction"]) for r in per_graph_stats], dtype=np.float32)

    def _dist(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    meta = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_images": int(len(indices)),
        "output_dir": str(run_out),
        "uses_model_predictions": True,
        "feature_version": "v2" if (feature_dim or 0) >= 22 else "v1",
        "feature_dim": feature_dim,
        "feature_index_map": {
            "legacy_prob_slice": [9, 13],
        },
        "build_args": {
            "num_segments": num_segments,
            "compactness": compactness,
            "sigma": args.sigma,
            "min_majority_fraction": args.min_majority_fraction,
            "superpixel_preset": args.superpixel_preset,
            "tiny_superpixel_max_pixels": int(args.tiny_superpixel_max_pixels),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "edge_policy": args.edge_policy,
            "edge_knn_k": int(args.edge_knn_k),
            "edge_knn_max_distance": float(args.edge_knn_max_distance),
        },
        "superpixel_params": {
            "num_segments": num_segments,
            "compactness": compactness,
            "sigma": args.sigma,
            "preset": args.superpixel_preset,
        },
        "graph_params": {
            "edge_policy": args.edge_policy,
            "edge_knn_k": int(args.edge_knn_k),
            "edge_knn_max_distance": float(args.edge_knn_max_distance),
        },
        "superpixel_quality": {
            "tiny_superpixel_fraction_distribution": _dist(tiny_fracs),
            "empty_or_degenerate_superpixel_fraction_distribution": _dist(deg_fracs),
            "per_graph": per_graph_stats,
        },
        "graph_size_distribution": {
            "node_count": _dist(node_counts),
            "edge_count_undirected": _dist(edge_counts),
        },
        "validation_report": {
            "class_counts": counts,
            "supervised_class_counts": supervised_counts,
            "supervised_nodes": supervised,
            "total_nodes": total_nodes,
            "supervised_node_ratio": float(supervised / max(total_nodes, 1)),
            "invalid_labels": invalid,
            "isolated_nodes": isolated,
        },
    }
    if float(supervised / max(total_nodes, 1)) < float(args.min_supervised_ratio):
        raise RuntimeError(
            f"Supervised node ratio too low: {float(supervised / max(total_nodes, 1)):.4f} < {float(args.min_supervised_ratio):.4f}"
        )
    with (run_out / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Saved graph artifacts for {len(indices)} images to: {run_out}")


if __name__ == "__main__":
    main()
