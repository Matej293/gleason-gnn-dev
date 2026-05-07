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

Weighted-fusion example (new):

```bash
PYTHONPATH=. python scripts/build_consensus_2d.py \
  --dataset-root data \
  --output-root data/consensus \
  --consensus-fusion-mode weighted \
  --ignore-threshold-loose 0.30 \
  --ignore-threshold-strict 0.50 \
  --single-rater-ignore-policy confidence_mask \
  --disable-gpu \
  --workers 8
```

Make target:

```bash
make consensus-weighted
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

## Metrics Tracked

The training and evaluation pipeline now tracks the following metrics.

| Metric | Where tracked | Meaning |
|---|---|---|
| `train/loss` | Train | Total training loss (`lambda_soft * soft_loss + lambda_dice * hard_term`) |
| `train/soft_loss` | Train | Soft-label term (CE/KL or focal variant depending on `loss_variant`) |
| `train/hard_dice_loss` | Train | Hard supervision overlap loss term (Dice/Tversky path) |
| `train/valid_pixel_fraction` | Train | Fraction of non-ignored pixels used for supervision |
| `train/lr` | Train | Learning rate per epoch |
| `val/loss` | Validation | Total validation loss |
| `val/macro_dice` | Validation, Eval summary | Mean Dice across active classes (background inclusion depends on config) |
| `val/miou` | Validation, Eval summary | Mean IoU across active classes |
| `val/grade5_dice` | Validation, Eval summary | Dice for Gleason 5 class |
| `val/grade5_iou` | Validation, Eval summary | IoU for Gleason 5 class |
| `val/dice_benign` | Validation, Eval summary | Dice for benign class |
| `val/dice_g3` | Validation, Eval summary | Dice for Gleason 3 class |
| `val/dice_g4` | Validation, Eval summary | Dice for Gleason 4 class |
| `val/dice_g5` | Validation, Eval summary | Dice for Gleason 5 class |
| `val/iou_benign` | Validation, Eval summary | IoU for benign class |
| `val/iou_g3` | Validation, Eval summary | IoU for Gleason 3 class |
| `val/iou_g4` | Validation, Eval summary | IoU for Gleason 4 class |
| `val/iou_g5` | Validation, Eval summary | IoU for Gleason 5 class |
| `val/sensitivity` | Validation, Eval summary | Tumor-vs-benign recall |
| `val/precision` | Validation, Eval summary | Tumor-vs-benign precision |
| `val/iou_tumor_vs_benign` | Validation, Eval summary | Tumor-vs-benign IoU |
| `val/ignored_pixel_fraction` | Validation, Eval summary | Fraction of pixels ignored by `ignore_mask` |
| `val/tumor_pixels_ignored_fraction` | Validation, Eval summary | Fraction of tumor GT pixels that were ignored |
| `val/composite_score` | Validation | Checkpoint ranking score (`best_ckpt_w_macro_dice`, `best_ckpt_w_sensitivity`) |
| `mean_loo_dice_multiclass` | Eval summary (when enabled) | Mean leave-one-rater-out Dice from `qc_report.json` |
| `num_loo_entries` | Eval summary (when enabled) | Number of LOO entries used in that mean |

Notes:
- Validation metrics are logged during training and written in evaluation output JSON.
- LOO aggregate metrics are included when `eval_leave_one_rater_out: true`.

## New Consensus/Training Options (4-class upgrade)

### Consensus builder

- `consensus_fusion_mode`: `staple_unweighted` (baseline) or `weighted` (uses `weights_per_pathologist`)
- `ignore_threshold_loose`, `ignore_threshold_strict`: confidence threshold used to build `ignore_mask.png`
- `single_rater_ignore_policy`: `confidence_mask` (default) or `all_ignore`

`qc_report.json` now includes:

- `consensus_fusion.effective_fusion_mode`
- `consensus_fusion.used_weights_per_pathologist`
- `consensus_fusion.ignore_threshold_used` (when applicable)
- `final_thresholds_used.ignore_confidence_threshold`

### Training/evaluation config

- `class_loss_weights`: per-class weights `[benign, G3, G4, G5]`  
  (`class_weights` is still supported for backward compatibility)
- `loss_variant`: `soft_dice` (default), `focal_dice`, or `tversky_dice`
- `eval_leave_one_rater_out`: when `true`, logs/reports LOO-consensus diagnostics

## How To Run The New Changes

1. Build consensus with weighted fusion.
```bash
make consensus-weighted
```

Equivalent raw command:
```bash
PYTHONPATH=. python scripts/build_consensus_2d.py \
  --dataset-root data \
  --output-root data/consensus \
  --consensus-fusion-mode weighted \
  --ignore-threshold-loose 0.30 \
  --ignore-threshold-strict 0.50 \
  --single-rater-ignore-policy confidence_mask \
  --disable-gpu \
  --workers 8
```

2. Train with upgraded loss controls (edit config first).
```yaml
# configs/deconver_2d_local.yaml
loss_variant: focal_dice
class_loss_weights: [1.0, 1.2, 1.2, 2.5]
eval_leave_one_rater_out: true
```

```bash
PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml
```

3. Evaluate checkpoint (includes LOO summary when enabled in run config).
```bash
PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run outputs/runs/<run_name>
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
