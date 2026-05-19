#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.config import consensus_dataset_kwargs_from_config, load_config
from src.config_validation import validate_deconver_config
from src.eval_utils import collate_consensus_batch, compute_multiclass_metrics
from src.gleason_consensus_dataset import GleasonConsensusDataset
from src.models import build_model


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

    ds = GleasonConsensusDataset(**consensus_dataset_kwargs_from_config(cfg))
    if len(ds) == 0:
        print("SKIP: no consensus samples discovered")
        return 0

    sample = ds[0]
    batch = collate_consensus_batch([sample])

    model = build_model(cfg).eval()
    with torch.inference_mode():
        out = model(batch["image"])
        logits = out[0] if isinstance(out, list) else out

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
