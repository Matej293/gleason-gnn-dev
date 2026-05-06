from __future__ import annotations

import torch

from src.utils import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "ckpt.pt"

    save_checkpoint(model, opt, epoch=3, path=str(ckpt), best_val_dice=0.5, best_composite_score=0.6)
    out = load_checkpoint(path=ckpt, model=model, optimizer=opt, device=torch.device("cpu"))

    assert out["epoch"] == 3
    assert float(out["best_val_dice"]) == 0.5
    assert float(out["best_composite_score"]) == 0.6
