from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.gnn.data import GraphSample, feature_index_map
from src.gnn.metrics import CLASS_NAMES, CaseEval, aggregate_case_metrics, json_safe
from src.gnn.models import GATNet, GCNNet, GraphSAGENet, NodeMLP


@dataclass
class TrainConfig:
    model: str = "graphsage"
    hidden_dim: int = 64
    dropout: float = 0.3
    feature_dropout: float = 0.1
    edge_dropout: float = 0.0
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 40
    seed: int = 42
    deterministic: bool = True
    use_class_weights: bool = True
    normalize_features: bool = True
    loss: str = "ce"
    focal_gamma: float = 2.0
    residual_head: bool = False
    residual_alpha: float = 0.2
    mask_unsupported_classes: bool = False
    grad_clip_norm: float = 1.0
    use_scheduler: bool = True
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    amp: bool = False
    selection_metric: str = "val_per_case_macro_f1"


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def train_supported_classes(samples: list[GraphSample], num_classes: int = 4) -> list[int]:
    present = np.zeros((num_classes,), dtype=np.bool_)
    for s in samples:
        y = s.y[s.supervision_mask].cpu().numpy()
        present[np.unique(y[(y >= 0) & (y < num_classes)])] = True
    return [i for i in range(num_classes) if present[i]]


def _ensure_seg_prob_slice(fmap: dict[str, list[int] | int], in_dim: int) -> tuple[int, int]:
    seg_slice = fmap.get("seg_probs_mean")
    if not isinstance(seg_slice, list) or len(seg_slice) != 4:
        raise ValueError("feature_index_map must contain seg_probs_mean with 4 indices.")
    start = int(seg_slice[0])
    expected = list(range(start, start + 4))
    if [int(x) for x in seg_slice] != expected:
        raise ValueError(f"seg_probs_mean must be contiguous. Got: {seg_slice}")
    end = start + 4
    if start < 0 or end > in_dim:
        raise ValueError(f"seg_probs_mean slice [{start}:{end}] invalid for feature dim {in_dim}")
    return start, end


def apply_class_mask_to_logits(logits: torch.Tensor, allowed_classes: list[int] | None) -> torch.Tensor:
    if not allowed_classes:
        return logits
    masked = logits.clone()
    class_mask = torch.zeros((masked.shape[1],), dtype=torch.bool, device=masked.device)
    class_mask[allowed_classes] = True
    masked[:, ~class_mask] = -1e9
    return masked


def _class_weights(samples: list[GraphSample], device: torch.device) -> torch.Tensor | None:
    counts = np.zeros((4,), dtype=np.int64)
    for s in samples:
        y = s.y[s.supervision_mask].cpu().numpy()
        counts += np.bincount(y, minlength=4)[:4]
    if counts.sum() == 0:
        return None
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _fit_normalization(samples: list[GraphSample]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([s.x[s.supervision_mask].cpu().numpy() for s in samples], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_norm(
    sample: GraphSample,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    seg_prob_idx: tuple[int, int],
) -> GraphSample:
    if mean is None or std is None:
        return sample
    x = (sample.x - torch.from_numpy(mean)) / torch.from_numpy(std)
    raw_seg_probs = sample.x[:, seg_prob_idx[0] : seg_prob_idx[1]].clone()
    return GraphSample(
        sample.image_id,
        x.float(),
        sample.edge_index,
        sample.y,
        sample.supervision_mask,
        sample.eval_mask,
        raw_seg_probs=raw_seg_probs,
    )


def _build_model(cfg: TrainConfig, in_dim: int, seg_prob_idx: tuple[int, int]) -> torch.nn.Module:
    kwargs = {
        "in_dim": in_dim,
        "hidden_dim": cfg.hidden_dim,
        "dropout": cfg.dropout,
        "feature_dropout": cfg.feature_dropout,
        "residual_head": cfg.residual_head,
        "residual_alpha": cfg.residual_alpha,
        "seg_prob_idx": seg_prob_idx,
    }
    if cfg.model == "mlp":
        return NodeMLP(**kwargs)
    if cfg.model == "graphsage":
        return GraphSAGENet(**kwargs)
    if cfg.model == "gcn":
        return GCNNet(**kwargs)
    if cfg.model == "gat":
        return GATNet(**kwargs)
    raise ValueError(f"Unsupported model: {cfg.model}")


def _forward(model: torch.nn.Module, sample: GraphSample, device: torch.device, edge_dropout: float) -> torch.Tensor:
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)
    raw_seg_probs = sample.raw_seg_probs.to(device) if sample.raw_seg_probs is not None else None
    if edge_dropout > 0.0 and edge_index.numel() > 0:
        keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= edge_dropout
        edge_index = edge_index[:, keep]
    return model(x, edge_index, raw_seg_probs=raw_seg_probs)


def _resolve_selection_value(eval_metrics: dict, selection_metric: str) -> float:
    key = str(selection_metric or "val_per_case_macro_f1").strip().lower()
    if key in {"val_per_case_macro_f1", "macro_f1", "val_macro_f1"}:
        v = eval_metrics.get("macro_f1")
    elif key in {"val_per_case_balanced_accuracy", "balanced_accuracy", "val_balanced_accuracy"}:
        v = eval_metrics.get("balanced_accuracy")
    elif key in {"val_micro_macro_f1", "micro_macro_f1"}:
        v = (eval_metrics.get("micro_over_nodes") or {}).get("macro_f1")
    elif key in {"val_micro_balanced_accuracy", "micro_balanced_accuracy"}:
        v = (eval_metrics.get("micro_over_nodes") or {}).get("balanced_accuracy")
    else:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")
    if v is None:
        return -1.0
    out = float(v)
    return out if np.isfinite(out) else -1.0


def _loss(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None, loss_name: str, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    if loss_name == "ce":
        return ce.mean()
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def _evaluate_split(
    model: torch.nn.Module,
    samples: list[GraphSample],
    device: torch.device,
    allowed_classes: list[int] | None,
) -> tuple[dict, dict[str, dict], np.ndarray, dict[str, list[float]]]:
    model.eval()
    cases = []
    num_total = 0
    num_supervised = 0
    pred_counts = {name: 0 for name in CLASS_NAMES}
    with torch.no_grad():
        for s in samples:
            logits = _forward(model, s, device, edge_dropout=0.0)
            logits = apply_class_mask_to_logits(logits, allowed_classes)
            pred = torch.argmax(logits, dim=1)
            m = s.eval_mask
            y_true = s.y[m].cpu().numpy()
            y_pred = pred[m].cpu().numpy()
            cases.append(CaseEval(image_id=s.image_id, y_true=y_true, y_pred=y_pred))
            num_total += int(m.sum().item())
            num_supervised += int(s.supervision_mask.sum().item())
            for i, name in enumerate(CLASS_NAMES):
                pred_counts[name] += int((y_pred == i).sum())

    micro, per_case_mean, cm, case_rows = aggregate_case_metrics(cases)
    arrays = {
        "macro_f1": [case_rows[c.image_id]["macro_f1"] for c in cases],
        "balanced_accuracy": [case_rows[c.image_id]["balanced_accuracy"] for c in cases],
    }
    split_metrics = {
        "macro_f1": per_case_mean["macro_f1"],
        "balanced_accuracy": per_case_mean["balanced_accuracy"],
        "per_class_f1": per_case_mean["per_class_f1"],
        "predicted_class_counts": pred_counts,
        "num_nodes_total": num_total,
        "num_nodes_supervised": num_supervised,
        "micro_over_nodes": micro,
    }
    return split_metrics, case_rows, cm, arrays


def _make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device="cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast_context(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)


def run_training(splits: dict[str, list[GraphSample]], output_root: str | Path, experiment_name: str, cfg: TrainConfig, graphs_root: str | None = None) -> Path:
    _set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = int(splits["train"][0].x.shape[1])
    fmap = feature_index_map(in_dim)
    seg_idx = _ensure_seg_prob_slice(fmap, in_dim)
    allowed_classes = train_supported_classes(splits["train"]) if cfg.mask_unsupported_classes else None

    norm_mean, norm_std = (None, None)
    if cfg.normalize_features:
        norm_mean, norm_std = _fit_normalization(splits["train"])
        splits = {k: [_apply_norm(s, norm_mean, norm_std, seg_idx) for s in v] for k, v in splits.items()}

    model = _build_model(cfg, in_dim=in_dim, seg_prob_idx=seg_idx).to(device)
    class_weights = _class_weights(splits["train"], device) if cfg.use_class_weights else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = _make_grad_scaler(device=device, enabled=cfg.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.scheduler_factor),
        patience=int(cfg.scheduler_patience),
        min_lr=float(cfg.scheduler_min_lr),
    ) if cfg.use_scheduler else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{ts}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    best_state = None
    best_tuple = (-1.0, -1.0, -1.0, -1)
    no_improve = 0

    log_path = run_dir / "train_log.jsonl"
    epoch_iter = tqdm(range(cfg.epochs), desc=f"train:{cfg.model}", leave=True)
    for epoch in epoch_iter:
        model.train()
        loss_acc = 0.0
        steps = 0
        for s in splits["train"]:
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device=device, enabled=cfg.amp):
                logits = _forward(model, s, device, edge_dropout=cfg.edge_dropout)
                y = s.y.to(device)
                mask = s.supervision_mask.to(device)
                if int(mask.sum().item()) == 0:
                    continue
                loss = _loss(logits[mask], y[mask], class_weights, cfg.loss, cfg.focal_gamma)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            loss_acc += float(loss.item())
            steps += 1

        eval_val, _, _, _ = _evaluate_split(model, splits["val"], device, allowed_classes=allowed_classes)
        selection_value = _resolve_selection_value(eval_val, cfg.selection_metric)
        if scheduler is not None:
            scheduler.step(selection_value)
        cur_tuple = (
            selection_value,
            float(eval_val["macro_f1"] if eval_val["macro_f1"] is not None else -1.0),
            float(eval_val["balanced_accuracy"] if eval_val["balanced_accuracy"] is not None else -1.0),
            -epoch,
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe({"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "train_loss": (loss_acc / max(steps, 1)), "val": eval_val})) + "\n")

        if cur_tuple > best_tuple:
            best_tuple = cur_tuple
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        epoch_iter.set_postfix({
            "loss": f"{(loss_acc / max(steps, 1)):.4f}",
            "val_f1": f"{(eval_val['macro_f1'] or 0.0):.4f}",
            "pat": no_improve,
        })
        if no_improve >= cfg.patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = -1
    model.load_state_dict(best_state)

    torch.save({
        "model_state": best_state,
        "model": cfg.model,
        "in_dim": in_dim,
        "hidden_dim": cfg.hidden_dim,
        "dropout": cfg.dropout,
        "feature_dropout": cfg.feature_dropout,
        "best_epoch": best_epoch,
        "residual_head": cfg.residual_head,
        "residual_alpha": cfg.residual_alpha,
        "residual_uses_raw_seg_probs": True,
        "normalize_features": cfg.normalize_features,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "feature_index_map": fmap,
        "active_classes": allowed_classes,
    }, run_dir / "best.pt")

    metrics_summary = {
        "split_metrics": {},
        "best_epoch": best_epoch,
        "selection_metric": cfg.selection_metric,
        "resolved_selection_metric": str(cfg.selection_metric),
        "best_selection_score": float(best_tuple[0]),
    }
    per_case_metrics: dict[str, dict] = {}
    per_case_arrays: dict[str, dict[str, list[float]]] = {}
    comparison_rows = []
    test_cm = None

    for split in ("train", "val", "test"):
        split_metrics, case_rows, cm, arrays = _evaluate_split(model, splits[split], device, allowed_classes=allowed_classes)
        metrics_summary["split_metrics"][split] = split_metrics
        per_case_metrics[split] = case_rows
        per_case_arrays[split] = arrays
        comparison_rows.append({"method": cfg.model, "split": split, "macro_f1": split_metrics["macro_f1"], "balanced_accuracy": split_metrics["balanced_accuracy"]})
        if split == "test":
            test_cm = cm

    (run_dir / "metrics_summary.json").write_text(json.dumps(json_safe(metrics_summary), indent=2) + "\n", encoding="utf-8")
    (run_dir / "per_case_metrics.json").write_text(json.dumps(json_safe(per_case_metrics), indent=2) + "\n", encoding="utf-8")
    (run_dir / "per_case_arrays.json").write_text(json.dumps(json_safe(per_case_arrays), indent=2) + "\n", encoding="utf-8")
    with (run_dir / "comparison_ready.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "macro_f1", "balanced_accuracy"])
        w.writeheader()
        for row in comparison_rows:
            w.writerow(json_safe(row))

    run_config = asdict(cfg)
    run_config.update({
        "graph_root": graphs_root,
        "feature_dim": in_dim,
        "feature_index_map": fmap,
        "best_epoch": best_epoch,
        "class_weight_policy": "inverse_frequency" if cfg.use_class_weights else "none",
        "active_classes": allowed_classes,
        "residual_uses_raw_seg_probs": True,
        "resolved_selection_metric": str(cfg.selection_metric),
        "best_selection_score": float(best_tuple[0]),
    })
    (run_dir / "run_config.json").write_text(json.dumps(json_safe(run_config), indent=2) + "\n", encoding="utf-8")

    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": str(device), "deterministic": cfg.deterministic, "seed": cfg.seed}
    (run_dir / "environment.json").write_text(json.dumps(json_safe(env), indent=2) + "\n", encoding="utf-8")

    if test_cm is None:
        test_cm = np.zeros((4, 4), dtype=np.int64)
    np.save(run_dir / "confusion_matrix.npy", test_cm)

    return run_dir
