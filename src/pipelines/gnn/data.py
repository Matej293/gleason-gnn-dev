from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class GraphSample:
    image_id: str
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    supervision_mask: torch.Tensor
    eval_mask: torch.Tensor
    raw_seg_probs: torch.Tensor | None = None


REQUIRED_KEYS = ("node_ids", "x", "edge_index", "y", "train_mask")


def _validate_npz(payload: dict[str, np.ndarray], path: Path) -> None:
    for key in REQUIRED_KEYS:
        if key not in payload:
            raise ValueError(f"Missing key '{key}' in {path}")

    x = payload["x"]
    edge_index = payload["edge_index"]
    y = payload["y"]
    train_mask = payload["train_mask"]

    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [N,F] in {path}, got {x.shape}")
    n = x.shape[0]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E] in {path}, got {edge_index.shape}")
    if y.shape != (n,):
        raise ValueError(f"y must be [N] in {path}, got {y.shape}")
    if train_mask.shape != (n,):
        raise ValueError(f"train_mask must be [N] in {path}, got {train_mask.shape}")


def _load_graph_npz(path: Path, split: str) -> GraphSample:
    data = np.load(path)
    payload = {k: data[k] for k in data.files}
    _validate_npz(payload, path)

    x = payload["x"].astype(np.float32, copy=False)
    edge_index = payload["edge_index"].astype(np.int64, copy=False)
    y = payload["y"].astype(np.int64, copy=False)
    train_mask = payload["train_mask"].astype(np.bool_, copy=False)

    valid_class = (y >= 0) & (y <= 3)
    supervision = train_mask & valid_class if split == "train" else valid_class
    eval_mask = valid_class
    raw_seg_probs = None
    if x.shape[1] >= 13:
        raw_seg_probs = torch.from_numpy(x[:, 9:13].copy())

    return GraphSample(
        image_id=path.parent.name,
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(y),
        supervision_mask=torch.from_numpy(supervision),
        eval_mask=torch.from_numpy(eval_mask),
        raw_seg_probs=raw_seg_probs,
    )


def load_graph_split(graphs_root: str | Path, split: str) -> list[GraphSample]:
    split_dir = Path(graphs_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Graph split dir not found: {split_dir}")

    graph_paths = sorted(split_dir.glob("*/graph_data.npz"))
    if not graph_paths:
        raise RuntimeError(f"No graph_data.npz files found under: {split_dir}")

    return [_load_graph_npz(path, split=split) for path in graph_paths]


def load_graph_splits(graphs_root: str | Path) -> dict[str, list[GraphSample]]:
    return {
        "train": load_graph_split(graphs_root, "train"),
        "val": load_graph_split(graphs_root, "val"),
        "test": load_graph_split(graphs_root, "test"),
    }


def feature_index_map(feature_dim: int) -> dict[str, list[int] | int]:
    if feature_dim == 14:
        return {
            "mean_rgb": [0, 1, 2],
            "std_rgb": [3, 4, 5],
            "area": 6,
            "centroid_xy": [7, 8],
            "seg_probs_mean": [9, 10, 11, 12],
            "entropy": 13,
        }
    if feature_dim >= 22:
        return {
            "mean_rgb": [0, 1, 2],
            "std_rgb": [3, 4, 5],
            "area": 6,
            "centroid_xy": [7, 8],
            "seg_probs_mean": [9, 10, 11, 12],
            "entropy": 13,
            "seg_probs_std": [14, 15, 16, 17],
            "seg_top2_margin": 18,
            "neighbor_prob_contrast": 19,
            "compactness_proxy": 20,
            "boundary_touch_ratio": 21,
        }
    raise ValueError(f"Unsupported feature dimension: {feature_dim}")
