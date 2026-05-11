from __future__ import annotations

import torch

from src.gnn.data import feature_index_map


def seg_only_predict(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected x [N,F], got {tuple(x.shape)}")
    fmap = feature_index_map(int(x.shape[1]))
    idx = fmap["seg_probs_mean"]
    probs = x[:, idx[0] : idx[-1] + 1]
    return torch.argmax(probs, dim=1)
