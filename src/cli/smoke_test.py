#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.common.config import (
    consensus_dataset_kwargs_from_config,
    consensus_train_val_transforms_from_config,
    load_config,
    resolve_inference_mode,
    resolve_resized_sliding_window_overlap,
    resolve_resized_sliding_window_patch_size,
)
from src.common.config_validation import validate_deconver_config
from src.eval.eval_utils import collate_consensus_batch, compute_multiclass_metrics
from src.data.gleason_consensus_dataset import GleasonConsensusDataset
from src.common.model_outputs import extract_logits
from src.models import build_model


def _infer_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    inference_mode: str,
    resized_sliding_window_patch_size: tuple[int, int],
    resized_sliding_window_overlap: float,
) -> torch.Tensor:
    mode = str(inference_mode).strip().lower()
    if mode == "resized_full":
        if images.ndim != 4:
            raise ValueError(f"Expected images shape [B,C,H,W], got {tuple(images.shape)}")
        h, w = int(images.shape[-2]), int(images.shape[-1])
        multiple = 32
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        x = images
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode="replicate")
        logits = extract_logits(model(x))
        if pad_h > 0 or pad_w > 0:
            logits = logits[..., :h, :w]
        return logits
    if mode == "resized_sliding_window":
        from monai.inferers import sliding_window_inference

        def _predictor(window: torch.Tensor) -> torch.Tensor:
            return extract_logits(model(window))

        return sliding_window_inference(
            inputs=images,
            roi_size=resized_sliding_window_patch_size,
            sw_batch_size=1,
            predictor=_predictor,
            overlap=resized_sliding_window_overlap,
        )
    raise ValueError(
        "Unsupported inference_mode: "
        f"{inference_mode!r}. Expected 'resized_full' or 'resized_sliding_window'."
    )


def main() -> int:
    cfg_path = Path("configs/deconver.yaml")
    if not cfg_path.exists():
        print("FAIL: config missing", file=sys.stderr)
        return 1

    cfg = load_config(str(cfg_path))
    validate_deconver_config(cfg, for_eval=False, require_paths=False)

    data_root = Path(str(cfg.get("data_root", "./data")))
    consensus_root = Path(str(cfg.get("consensus_root", "./data/consensus")))
    if not data_root.exists() or not consensus_root.exists():
        print("SKIP: data paths not found")
        return 0

    _, val_transform = consensus_train_val_transforms_from_config(cfg)
    ds = GleasonConsensusDataset(
        **consensus_dataset_kwargs_from_config(cfg, transform=val_transform)
    )
    if len(ds) == 0:
        print("SKIP: no consensus samples discovered")
        return 0

    sample = ds[0]
    batch = collate_consensus_batch([sample])

    inference_mode = resolve_inference_mode(cfg)
    resized_sliding_window_patch_size = resolve_resized_sliding_window_patch_size(cfg)
    resized_sliding_window_overlap = resolve_resized_sliding_window_overlap(cfg)

    model = build_model(cfg).eval()

    with torch.inference_mode():
        logits = _infer_logits(
            model=model,
            images=batch["image"],
            inference_mode=inference_mode,
            resized_sliding_window_patch_size=resized_sliding_window_patch_size,
            resized_sliding_window_overlap=resized_sliding_window_overlap,
        )

    metrics = compute_multiclass_metrics(
        logits=logits.float(),
        hard_mask=batch["hard_mask"],
        ignore_mask=batch["ignore_mask"],
        include_background_in_dice=bool(cfg.get("include_background_in_dice", False)),
    )
    print(
        "OK: smoke test passed",
        {
            k: (None if torch.isnan(torch.tensor(v)) else round(v, 4))
            for k, v in metrics.items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
