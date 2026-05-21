from __future__ import annotations

import collections
import importlib
import json
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

logger = logging.getLogger(__name__)

_UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)")


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
    best_challenge_score: float | None = None,
    last_hd95: float | None = None,
) -> None:
    payload: dict[str, object] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "best_challenge_score": best_challenge_score,
        "last_hd95": last_hd95,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def _resolve_safe_global(symbol: str) -> object | None:
    resolvers: dict[str, Callable[[], object]] = {
        "list": lambda: list,
        "tuple": lambda: tuple,
        "set": lambda: set,
        "dict": lambda: dict,
        "str": lambda: str,
        "int": lambda: int,
        "float": lambda: float,
        "bool": lambda: bool,
        "collections.defaultdict": lambda: collections.defaultdict,
        "collections.OrderedDict": lambda: collections.OrderedDict,
        "monai.handlers.metric_logger.MetricLoggerKeys": lambda: getattr(
            importlib.import_module("monai.handlers.metric_logger"),
            "MetricLoggerKeys",
        ),
    }
    resolver = resolvers.get(symbol)
    if resolver is None:
        return None
    return resolver()


def _load_torch_checkpoint_weights_only(
    path: str | Path,
    *,
    map_location: torch.device | str,
) -> Any:
    path_str = str(path)

    safe_globals: list[object] = []
    safe_symbols: set[str] = set()

    while True:
        try:
            if safe_globals:
                with torch.serialization.safe_globals(safe_globals):
                    return torch.load(path_str, map_location=map_location, weights_only=True)
            return torch.load(path_str, map_location=map_location, weights_only=True)
        except TypeError:
            # Backward compatibility for torch versions without weights_only.
            return torch.load(path_str, map_location=map_location)
        except pickle.UnpicklingError as exc:
            match = _UNSUPPORTED_GLOBAL_RE.search(str(exc))
            if match is None:
                raise

            symbol = match.group(1)
            if symbol in safe_symbols:
                raise

            safe_obj = _resolve_safe_global(symbol)
            if safe_obj is None:
                raise

            safe_globals.append(safe_obj)
            safe_symbols.add(symbol)
            logger.info("Allowlisting safe global for checkpoint load: %s", symbol)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict:
    map_location: torch.device | str = device if device is not None else "cpu"
    ckpt = _load_torch_checkpoint_weights_only(path, map_location=map_location)
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


def _normalize_state_dict_prefix(
    state_dict: dict[str, Any],
    target_keys: set[str],
) -> dict[str, Any]:
    if not state_dict:
        return state_dict

    candidate_dicts = [
        state_dict,
        {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        },
        {
            (f"_orig_mod.{k}" if not k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        },
    ]

    best = state_dict
    best_match_count = -1
    for candidate in candidate_dicts:
        match_count = sum(1 for k in candidate if k in target_keys)
        if match_count > best_match_count:
            best = candidate
            best_match_count = match_count
    return best


def load_pretrained_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a pretrained checkpoint into ``model`` using only shape-compatible keys.

    This is intended for warm-starting when architecture is mostly shared but
    output heads differ (for example, out_channels changed).
    """
    map_location: torch.device | str = device if device is not None else "cpu"
    ckpt = _load_torch_checkpoint_weights_only(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format: {path}")

    model_state = ckpt.get("model", ckpt)
    if not isinstance(model_state, dict):
        raise ValueError(f"Missing model state in checkpoint: {path}")

    target_state = model.state_dict()
    normalized_state = _normalize_state_dict_prefix(
        model_state,
        target_keys=set(target_state.keys()),
    )

    compatible_state: dict[str, Any] = {}
    skipped_unexpected: list[str] = []
    skipped_shape_mismatch: list[str] = []

    for key, value in normalized_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped_unexpected.append(key)
            continue

        value_shape = getattr(value, "shape", None)
        if value_shape != target_value.shape:
            skipped_shape_mismatch.append(key)
            continue

        compatible_state[key] = value

    if not compatible_state:
        raise ValueError(
            "No compatible parameters found while loading pretrained checkpoint: "
            f"{path}"
        )

    load_result = model.load_state_dict(compatible_state, strict=False)

    return {
        "checkpoint": ckpt,
        "loaded_keys": sorted(compatible_state.keys()),
        "loaded_count": int(len(compatible_state)),
        "target_param_count": int(len(target_state)),
        "skipped_unexpected_keys": sorted(skipped_unexpected),
        "skipped_shape_mismatch_keys": sorted(skipped_shape_mismatch),
        "missing_keys_after_load": sorted(load_result.missing_keys),
        "unexpected_keys_after_load": sorted(load_result.unexpected_keys),
    }


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
    "load_pretrained_checkpoint",
    "rotate_checkpoints",
    "save_checkpoint",
    "save_config_copy",
    "save_latest_pointer",
    "save_metadata",
]
