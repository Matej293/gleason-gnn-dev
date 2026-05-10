from __future__ import annotations

import numpy as np
import torch


def to_pyg_data(
    features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
):
    """
    Convert numpy graph arrays to torch_geometric.data.Data.
    """
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required for PyG conversion. Install it to train GNN models."
        ) from exc

    return Data(
        x=torch.from_numpy(features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.from_numpy(labels).long(),
        train_mask=torch.from_numpy(train_mask.astype(np.bool_)),
    )

