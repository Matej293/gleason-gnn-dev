from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainBatch:
    """Typed training batch container used by loop helpers."""

    images: torch.Tensor
    soft_probs: torch.Tensor
    hard_mask: torch.Tensor
    ignore_mask: torch.Tensor
    image_ids: list[str]
    sample_weights: torch.Tensor


@dataclass(frozen=True)
class LossOutputs:
    """Model loss outputs with scalar diagnostics."""

    loss: torch.Tensor
    soft_loss: float
    hard_dice_loss: float
    valid_fraction: float

    @property
    def stats(self) -> dict[str, float]:
        return {
            "soft_loss": self.soft_loss,
            "hard_dice_loss": self.hard_dice_loss,
            "valid_fraction": self.valid_fraction,
        }


@dataclass
class EpochStats:
    """Mutable epoch accumulators with deterministic averaging."""

    loss_sum: float = 0.0
    soft_sum: float = 0.0
    hard_sum: float = 0.0
    valid_fraction_sum: float = 0.0
    optimizer_steps: int = 0
    num_batches_seen: int = 0

    def add(self, *, loss: float, soft_loss: float, hard_dice_loss: float, valid_fraction: float) -> None:
        self.loss_sum += float(loss)
        self.soft_sum += float(soft_loss)
        self.hard_sum += float(hard_dice_loss)
        self.valid_fraction_sum += float(valid_fraction)
        self.num_batches_seen += 1

    def mark_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def averages(self, fallback_batches: int) -> dict[str, float]:
        denom = max(1, int(fallback_batches))
        return {
            "loss": self.loss_sum / denom,
            "soft_loss": self.soft_sum / denom,
            "hard_dice_loss": self.hard_sum / denom,
            "valid_fraction": self.valid_fraction_sum / denom,
        }


@dataclass(frozen=True)
class RunContext:
    """Stable run directory context shared across save/log operations."""

    run_dir: Path
    checkpoint_dir: Path
    split_manifest_copy_path: Path
    summary_path: Path
