from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.gnn.models import GATNet, GCNNet, GraphSAGENet, NodeMLP


def seg_prob_idx_from_meta(meta: dict) -> tuple[int, int]:
    fmap = meta.get("feature_index_map")
    if isinstance(fmap, dict) and "seg_probs_mean" in fmap:
        seg = fmap["seg_probs_mean"]
        if isinstance(seg, (list, tuple)) and len(seg) >= 1:
            start = int(seg[0])
            return (start, start + 4)
    return (9, 13)


def build_model_from_metadata(meta: dict) -> torch.nn.Module:
    model_name = str(meta["model"])
    in_dim = int(meta["in_dim"] if "in_dim" in meta else meta["feature_dim"])
    hidden_dim = int(meta["hidden_dim"])
    dropout = float(meta["dropout"])
    feature_dropout = float(meta.get("feature_dropout", 0.0))
    residual_head = bool(meta.get("residual_head", False))
    residual_alpha = float(meta.get("residual_alpha", 0.2))
    kwargs = dict(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        feature_dropout=feature_dropout,
        residual_head=residual_head,
        residual_alpha=residual_alpha,
        seg_prob_idx=seg_prob_idx_from_meta(meta),
    )
    if model_name == "mlp":
        model = NodeMLP(**kwargs)
    elif model_name == "graphsage":
        model = GraphSAGENet(**kwargs)
    elif model_name == "gcn":
        model = GCNNet(**kwargs)
    elif model_name == "gat":
        model = GATNet(**kwargs)
    else:
        raise ValueError(f"Unsupported model type in checkpoint: {model_name}")
    uses_raw = bool(meta.get("residual_uses_raw_seg_probs", False))
    setattr(model, "residual_uses_raw_seg_probs", uses_raw)
    setattr(model, "allow_legacy_normalized_residual", not uses_raw)
    return model


def resolve_norm_stats(
    ckpt: dict,
    run_cfg: dict,
    *,
    require_1d: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mean = ckpt.get("norm_mean")
    std = ckpt.get("norm_std")
    if mean is None:
        mean = run_cfg.get("norm_mean")
    if std is None:
        std = run_cfg.get("norm_std")
    if mean is None or std is None:
        return None, None
    mean_np = np.asarray(mean, dtype=np.float32)
    std_np = np.asarray(std, dtype=np.float32)
    if require_1d and (mean_np.ndim != 1 or std_np.ndim != 1):
        raise ValueError("Normalization stats must be 1D vectors.")
    if mean_np.shape != std_np.shape:
        raise ValueError("Normalization mean/std shapes do not match.")
    std_np = std_np.copy()
    std_np[std_np < 1e-6] = 1.0
    return mean_np, std_np


def apply_norm_x(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return (x - torch.from_numpy(mean)) / torch.from_numpy(std)


def load_case_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    required = ("node_ids", "x", "edge_index", "y", "superpixels")
    for key in required:
        if key not in d.files:
            raise ValueError(f"Missing '{key}' in {path}")
    return {key: d[key] for key in d.files}


def labels_to_map(
    superpixels: np.ndarray,
    node_ids: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(superpixels, dtype=np.int64)
    lookup = {int(nid): int(lbl) for nid, lbl in zip(node_ids.tolist(), labels.tolist())}
    for nid in np.unique(superpixels).tolist():
        out[superpixels == nid] = lookup.get(int(nid), 0)
    return out


__all__ = [
    "apply_norm_x",
    "build_model_from_metadata",
    "labels_to_map",
    "load_case_npz",
    "resolve_norm_stats",
    "seg_prob_idx_from_meta",
]
