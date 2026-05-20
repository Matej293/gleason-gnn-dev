from __future__ import annotations

import torch
from torch import nn


class NodeMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.2,
        feature_dropout: float = 0.0,
        residual_head: bool = False,
        residual_alpha: float = 0.2,
        seg_prob_idx: tuple[int, int] = (9, 13),
    ) -> None:
        super().__init__()
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.residual_head = bool(residual_head)
        self.residual_alpha = float(residual_alpha)
        self.seg_prob_idx = seg_prob_idx
        self.residual_uses_raw_seg_probs = True
        self.allow_legacy_normalized_residual = False

    def _base_logits(self, x: torch.Tensor, raw_seg_probs: torch.Tensor | None = None) -> torch.Tensor:
        if raw_seg_probs is None:
            if x.shape[1] < self.seg_prob_idx[1]:
                raise RuntimeError("Residual head enabled but seg_probs_mean slice is unavailable in input features.")
            probs = x[:, self.seg_prob_idx[0] : self.seg_prob_idx[1]]
            probs_sum = probs.sum(dim=1)
            looks_like_probs = bool(
                torch.all((probs >= -1e-6) & (probs <= 1.0 + 1e-6)).item()
                and torch.all(torch.isfinite(probs_sum)).item()
                and torch.all(torch.abs(probs_sum - 1.0) < 1e-2).item()
            )
            if (not looks_like_probs) and (not self.allow_legacy_normalized_residual):
                raise RuntimeError(
                    "Residual head requires raw seg_probs_mean when features are normalized. "
                    "Pass raw_seg_probs to model forward."
                )
        else:
            probs = raw_seg_probs
            if probs.ndim != 2 or probs.shape[1] != 4:
                raise RuntimeError(f"raw_seg_probs must have shape [N,4], got {tuple(probs.shape)}")
        return torch.log(torch.clamp(probs, min=1e-8))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        raw_seg_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index
        h = self.feature_dropout(x)
        logits = self.net(h)
        if self.residual_head:
            base = self._base_logits(x, raw_seg_probs=raw_seg_probs)
            logits = base + (self.residual_alpha * logits)
        return logits


class GraphSAGENet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.2,
        feature_dropout: float = 0.0,
        residual_head: bool = False,
        residual_alpha: float = 0.2,
        seg_prob_idx: tuple[int, int] = (9, 13),
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for GraphSAGE. Install PyG with a torch/cuda-matched wheel."
            ) from exc

        self.feature_dropout = nn.Dropout(feature_dropout)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.residual_head = bool(residual_head)
        self.residual_alpha = float(residual_alpha)
        self.seg_prob_idx = seg_prob_idx
        self.residual_uses_raw_seg_probs = True
        self.allow_legacy_normalized_residual = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        raw_seg_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_in = self.feature_dropout(x)
        h = self.conv1(x_in, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        logits = self.head(h)
        if self.residual_head:
            base = NodeMLP._base_logits(self, x, raw_seg_probs=raw_seg_probs)
            logits = base + (self.residual_alpha * logits)
        return logits


class GCNNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.2,
        feature_dropout: float = 0.0,
        residual_head: bool = False,
        residual_alpha: float = 0.2,
        seg_prob_idx: tuple[int, int] = (9, 13),
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for GCN. Install PyG with a torch/cuda-matched wheel."
            ) from exc

        self.feature_dropout = nn.Dropout(feature_dropout)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.residual_head = bool(residual_head)
        self.residual_alpha = float(residual_alpha)
        self.seg_prob_idx = seg_prob_idx
        self.residual_uses_raw_seg_probs = True
        self.allow_legacy_normalized_residual = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        raw_seg_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_in = self.feature_dropout(x)
        h = self.conv1(x_in, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        logits = self.head(h)
        if self.residual_head:
            base = NodeMLP._base_logits(self, x, raw_seg_probs=raw_seg_probs)
            logits = base + (self.residual_alpha * logits)
        return logits


class GATNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.2,
        feature_dropout: float = 0.0,
        residual_head: bool = False,
        residual_alpha: float = 0.2,
        seg_prob_idx: tuple[int, int] = (9, 13),
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for GAT. Install PyG with a torch/cuda-matched wheel."
            ) from exc

        self.feature_dropout = nn.Dropout(feature_dropout)
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.residual_head = bool(residual_head)
        self.residual_alpha = float(residual_alpha)
        self.seg_prob_idx = seg_prob_idx

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        raw_seg_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_in = self.feature_dropout(x)
        h = self.conv1(x_in, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        logits = self.head(h)
        if self.residual_head:
            base = NodeMLP._base_logits(self, x, raw_seg_probs=raw_seg_probs)
            logits = base + (self.residual_alpha * logits)
        return logits
