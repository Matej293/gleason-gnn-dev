# ProstateLesionSegmentation (2D Deconver Only)

This repository is a simplified 2D-only training pipeline for Gleason consensus segmentation using the vendored `deconver` model.

## Train

```bash
PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml
```

Training logs are written to Weights & Biases (W&B) per epoch.  
Set `WANDB_API_KEY` for online logging, or set `wandb_mode: offline` in config for local-only logging.

## Evaluate

```bash
PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run outputs/runs/<run_name>
```

## Smoke Test

```bash
PYTHONPATH=. python scripts/smoke_test_2d.py
```

## Regenerate Consensus Data

```bash
PYTHONPATH=. python scripts/build_consensus_2d.py --dataset-root data --output-root data/consensus
```

This runs the vendored STAPLE consensus builder and writes outputs to:

```text
data/consensus/<image_id>/
  consensus_probs_compact.npz
  consensus_hard_mask.png
  ignore_mask.png
  qc_report.json
```

## Tests

```bash
PYTHONPATH=. pytest -q tests
```

## Kept configs

- `configs/deconver_2d.yaml`
- `configs/deconver_2d_local.yaml`

Both configs are set up for TITAN V / Volta compatibility:

- `use_amp: true`
- `amp_dtype: fp16`
- `use_compile: false`

## Expected data layout

```text
data/
  Train_imgs/
    <image_id>.jpg|png|jpeg
  Test_imgs/
    <image_id>.jpg|png|jpeg
  consensus/
    <image_id>/
      consensus_probs_compact.npz
      consensus_hard_mask.png
      ignore_mask.png
      qc_report.json (optional)
```
