#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.common.config import consensus_dataset_kwargs_from_config, load_config
from src.common.cli_utils import (
    ensure_output_dir,
    require_existing_dir,
    require_existing_file,
    resolve_checkpoint_path,
    validate_fraction,
    validate_non_negative_int,
    validate_positive_int,
)
from src.common.config_validation import validate_deconver_config
from src.eval.eval_utils import collate_consensus_batch, resolve_split_manifest_path, safe_read_json
from src.data.gleason_consensus_dataset import GleasonConsensusDataset
from src.pipelines.graph import (
    assign_majority_node_labels,
    build_edges,
    compute_node_features,
    generate_slic_superpixels,
)
from src.models import build_model
from src.common.model_outputs import extract_logits as _extract_logits
from src.common.utils import ensure_cuda_binary_compatibility, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build superpixel graph artifacts from model predictions using a trained segmentation checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = p.add_argument_group("I/O")
    io_group.add_argument(
        "--run",
        required=True,
        type=str,
        help="Run directory containing config.yaml and checkpoints/.",
    )
    io_group.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename in run/checkpoints or absolute checkpoint path.",
    )
    io_group.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        required=True,
        help="Dataset split to export.",
    )
    io_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs/graphs",
        help="Root directory for exported graph artifacts.",
    )

    sp_group = p.add_argument_group("Superpixel / Graph")
    sp_group.add_argument("--num-segments", type=int, default=300)
    sp_group.add_argument("--compactness", type=float, default=10.0)
    sp_group.add_argument("--superpixel-preset", choices=["low", "med", "high"], default=None)
    sp_group.add_argument("--sigma", type=float, default=1.0)
    sp_group.add_argument("--edge-policy", choices=["touch", "knn", "touch_plus_knn"], default="touch")
    sp_group.add_argument("--edge-knn-k", type=int, default=2)
    sp_group.add_argument(
        "--edge-knn-max-distance",
        type=float,
        default=0.0,
        help="Maximum KNN edge distance; 0 disables distance threshold.",
    )
    sp_group.add_argument("--min-majority-fraction", type=float, default=0.6)
    sp_group.add_argument("--tiny-superpixel-max-pixels", type=int, default=8)

    loader_group = p.add_argument_group("Data Loader")
    loader_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override loader batch size.",
    )
    loader_group.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override loader worker count.",
    )

    qc_group = p.add_argument_group("Safety Checks")
    qc_group.add_argument(
        "--min-supervised-ratio",
        type=float,
        default=0.01,
        help="Fail build if supervised node ratio falls below this threshold.",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    args.num_segments = validate_positive_int(args.num_segments, field_name="num_segments")
    args.edge_knn_k = validate_positive_int(args.edge_knn_k, field_name="edge_knn_k")
    args.tiny_superpixel_max_pixels = validate_non_negative_int(
        args.tiny_superpixel_max_pixels,
        field_name="tiny_superpixel_max_pixels",
    )
    args.min_majority_fraction = validate_fraction(
        args.min_majority_fraction,
        field_name="min_majority_fraction",
    )
    args.min_supervised_ratio = validate_fraction(
        args.min_supervised_ratio,
        field_name="min_supervised_ratio",
    )
    if args.batch_size is not None:
        args.batch_size = validate_positive_int(args.batch_size, field_name="batch_size")
    if args.num_workers is not None:
        args.num_workers = validate_non_negative_int(args.num_workers, field_name="num_workers")
    args.compactness = float(args.compactness)
    args.sigma = float(args.sigma)
    args.edge_knn_max_distance = float(args.edge_knn_max_distance)
    if args.compactness < 0.0:
        raise ValueError(f"compactness must be >= 0, got {args.compactness}.")
    if args.sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {args.sigma}.")
    if args.edge_knn_max_distance < 0.0:
        raise ValueError(
            "edge_knn_max_distance must be >= 0; use 0 to disable threshold."
        )


def _resolve_superpixel_params(args: argparse.Namespace) -> tuple[int, float]:
    if args.superpixel_preset is None:
        return int(args.num_segments), float(args.compactness)
    presets = {
        "low": (220, 6.0),
        "med": (300, 10.0),
        "high": (420, 16.0),
    }
    return presets[args.superpixel_preset]



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
    _validate_args(args)

    num_segments, compactness = _resolve_superpixel_params(args)
    run_dir = require_existing_dir(args.run, label="Run directory")
    cfg_path = require_existing_file(run_dir / "config.yaml", label="Run config")
    output_root = ensure_output_dir(args.output_dir, label="Graph output root")

    cfg = load_config(cfg_path)
    validate_deconver_config(
        cfg,
        for_eval=(args.split != "all"),
        require_paths=True,
    )

    ckpt_path = resolve_checkpoint_path(run_dir, args.checkpoint, prefer_best=False)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    ensure_cuda_binary_compatibility(device)

    model = build_model(cfg).to(device)
    load_checkpoint(str(ckpt_path), model=model, device=device)
    model.eval()

    dataset_kwargs = consensus_dataset_kwargs_from_config(cfg)
    dataset = GleasonConsensusDataset(**dataset_kwargs)

    split_manifest = resolve_split_manifest_path(cfg)
    if args.split != "all":
        split_manifest = require_existing_file(split_manifest, label="Split manifest")

    logger_msg = (
        "Graph export setup | run=%s checkpoint=%s split=%s output_root=%s split_manifest=%s"
    )
    print(
        logger_msg
        % (
            run_dir,
            ckpt_path,
            args.split,
            output_root,
            split_manifest,
        )
    )

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

    run_out = output_root / run_dir.name / args.split
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
