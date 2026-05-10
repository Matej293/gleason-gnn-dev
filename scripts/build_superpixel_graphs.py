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
    build_touch_adjacency_edges,
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
        default="test",
        help="Which split to export graphs for (default: test).",
    )
    p.add_argument("--output-dir", type=str, default="outputs/graphs")
    p.add_argument("--num-segments", type=int, default=300)
    p.add_argument("--compactness", type=float, default=10.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--min-majority-fraction", type=float, default=0.6)
    return p.parse_args()


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
    loader = DataLoader(
        subset,
        batch_size=int(cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_consensus_batch,
    )

    run_out = Path(args.output_dir).resolve() / run_dir.name / args.split
    run_out.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Build checkpoint graphs", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"]
            ignore_mask = batch["ignore_mask"]
            tissue_mask = batch["tissue_mask"]
            image_ids = [str(x) for x in batch["image_id"]]

            out = model(images)
            logits = out[0] if isinstance(out, list) else out
            probs = torch.softmax(logits.float(), dim=1).detach().cpu()

            for i, image_id in enumerate(image_ids):
                image_rgb = (batch["image"][i].numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                sp = generate_slic_superpixels(
                    image_rgb=image_rgb,
                    tissue_mask=tissue_mask[i].numpy().astype(np.uint8),
                    num_segments=args.num_segments,
                    compactness=args.compactness,
                    sigma=args.sigma,
                )
                edge_index = build_touch_adjacency_edges(sp)
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

    meta = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_images": int(len(indices)),
        "output_dir": str(run_out),
        "uses_model_predictions": True,
    }
    with (run_out / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Saved graph artifacts for {len(indices)} images to: {run_out}")


if __name__ == "__main__":
    main()

