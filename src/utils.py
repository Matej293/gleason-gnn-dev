from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def create_run_dir(base_output_dir: str, experiment_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / f"{ts}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_metadata(run_dir: Path, cfg: dict) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": cfg.get("experiment_name", "unknown"),
    }
    out = run_dir / "metadata.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_config_copy(run_dir: Path, cfg: dict) -> None:
    out = run_dir / "config.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_latest_pointer(base_output_dir: str, run_dir: Path) -> None:
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)
    latest = base / "latest_run.txt"
    latest.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_val_dice: float | None = None,
    best_composite_score: float | None = None,
    last_hd95: float | None = None,
) -> None:
    payload: dict[str, object] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "best_composite_score": best_composite_score,
        "last_hd95": last_hd95,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict:
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(str(path), map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format: {path}")

    model_state = ckpt.get("model", ckpt)
    if not isinstance(model_state, dict):
        raise ValueError(f"Missing model state in checkpoint: {path}")
    model.load_state_dict(model_state)

    if optimizer is not None and isinstance(ckpt.get("optimizer"), dict):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt.get("scheduler"), dict):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and isinstance(ckpt.get("scaler"), dict):
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt


def rotate_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    epoch_ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    excess = len(epoch_ckpts) - keep_last_n
    if excess <= 0:
        return
    for ckpt in epoch_ckpts[:excess]:
        ckpt.unlink(missing_ok=True)


def ensure_cuda_binary_compatibility(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")


__all__ = [
    "create_run_dir",
    "ensure_cuda_binary_compatibility",
    "ensure_dir",
    "load_checkpoint",
    "rotate_checkpoints",
    "save_checkpoint",
    "save_config_copy",
    "save_latest_pointer",
    "save_metadata",
]
