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
        self.seg_prob_idx = seg_prob_idx

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        h = self.feature_dropout(x)
        logits = self.net(h)
        if self.residual_head and x.shape[1] >= self.seg_prob_idx[1]:
            base = torch.log(torch.clamp(x[:, self.seg_prob_idx[0] : self.seg_prob_idx[1]], min=1e-8))
            logits = logits + base
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
        self.seg_prob_idx = seg_prob_idx

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_in = self.feature_dropout(x)
        h = self.conv1(x_in, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        logits = self.head(h)
        if self.residual_head and x.shape[1] >= self.seg_prob_idx[1]:
            base = torch.log(torch.clamp(x[:, self.seg_prob_idx[0] : self.seg_prob_idx[1]], min=1e-8))
            logits = logits + base
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
        self.seg_prob_idx = seg_prob_idx

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_in = self.feature_dropout(x)
        h = self.conv1(x_in, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        logits = self.head(h)
        if self.residual_head and x.shape[1] >= self.seg_prob_idx[1]:
            base = torch.log(torch.clamp(x[:, self.seg_prob_idx[0] : self.seg_prob_idx[1]], min=1e-8))
            logits = logits + base
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
        self.seg_prob_idx = seg_prob_idx

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_in = self.feature_dropout(x)
        h = self.conv1(x_in, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        logits = self.head(h)
        if self.residual_head and x.shape[1] >= self.seg_prob_idx[1]:
            base = torch.log(torch.clamp(x[:, self.seg_prob_idx[0] : self.seg_prob_idx[1]], min=1e-8))
            logits = logits + base
        return logits
