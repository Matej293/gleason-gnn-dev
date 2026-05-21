from __future__ import annotations

import pytest
import torch

from src.common.utils import load_checkpoint, load_pretrained_checkpoint, save_checkpoint


class TinyHeadNet(torch.nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
        )
        self.head = torch.nn.Linear(8, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "ckpt.pt"

    save_checkpoint(model, opt, epoch=3, path=str(ckpt), best_val_dice=0.5, best_challenge_score=0.6)
    out = load_checkpoint(path=ckpt, model=model, optimizer=opt, device=torch.device("cpu"))

    assert out["epoch"] == 3
    assert float(out["best_val_dice"]) == 0.5
    assert float(out["best_challenge_score"]) == 0.6


def test_load_pretrained_checkpoint_skips_mismatched_head(tmp_path):
    torch.manual_seed(7)
    source_model = TinyHeadNet(out_channels=1)
    source_opt = torch.optim.AdamW(source_model.parameters(), lr=1e-3)

    ckpt = tmp_path / "source.pt"
    save_checkpoint(source_model, source_opt, epoch=1, path=str(ckpt))

    target_model = TinyHeadNet(out_channels=4)
    target_head_before = target_model.head.weight.detach().clone()

    info = load_pretrained_checkpoint(
        path=ckpt,
        model=target_model,
        device=torch.device("cpu"),
    )

    assert info["loaded_count"] > 0
    assert "head.weight" in info["skipped_shape_mismatch_keys"]
    assert "head.bias" in info["skipped_shape_mismatch_keys"]

    assert torch.equal(
        target_model.backbone[0].weight,
        source_model.backbone[0].weight,
    )
    assert torch.equal(
        target_model.backbone[2].weight,
        source_model.backbone[2].weight,
    )
    assert torch.equal(target_model.head.weight, target_head_before)


def test_load_checkpoint_remains_strict_for_shape_mismatch(tmp_path):
    source = torch.nn.Linear(4, 1)
    target = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(source.parameters(), lr=1e-3)

    ckpt = tmp_path / "strict.pt"
    save_checkpoint(source, opt, epoch=1, path=str(ckpt))

    with pytest.raises(RuntimeError):
        load_checkpoint(path=ckpt, model=target, device=torch.device("cpu"))
