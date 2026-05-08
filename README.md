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

## Fast Class Distribution Count (Train Split)

Use this to count class pixels/images from `consensus_hard_mask.png` for `train_image_ids`
from your split manifest (fast scan, no dataset transform overhead).

```bash
PYTHONPATH=. python scripts/count_class_distribution_fast.py --config configs/deconver_2d_local.yaml
```

Output includes:
- `Pixel counts`: total pixels per class across the train split
- `Pixel fractions`: normalized class frequency
- `Images containing class`: how many train images contain each class
- `Missing hard masks`: train IDs that did not have a mask file

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
  --target-ignore-tissue-frac 0.05 \
  --target-ignore-total-frac 0.12 \
  --ignore-threshold-min 0.05 \
  --ignore-threshold-max 0.35 \
  --auto-calibrate-ignore-threshold \
  --boundary-dilate-px 1 \
  --edge-smooth-open-px 0 \
  --edge-smooth-close-px 1 \
  --remove-small-islands-px 64 \
  --fill-small-holes-px 64 \
  --single-rater-ignore-policy confidence_mask \
  --disable-gpu \
  --workers 8
```

Make target:

```bash
make consensus-weighted
```

Background-ignore audit:

```bash
make audit-background-ignore
```

This writes:

```text
outputs/background_ignore_audit.json
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
- `target_ignore_tissue_frac`, `target_ignore_total_frac`: per-image ignore targets for auto-calibration
- `ignore_threshold_min`, `ignore_threshold_max`: calibration bounds
- `auto_calibrate_ignore_threshold`: lower threshold per image to reduce excessive ignore
- `boundary_dilate_px`, `edge_smooth_open_px`, `edge_smooth_close_px`, `remove_small_islands_px`, `fill_small_holes_px`: edge/refinement controls
- `single_rater_ignore_policy`: `confidence_mask` (default) or `all_ignore`

`qc_report.json` now includes:

- `consensus_fusion.effective_fusion_mode`
- `consensus_fusion.used_weights_per_pathologist`
- `consensus_fusion.ignore_threshold_used` (when applicable)
- `consensus_fusion.ignored_total_fraction`
- `consensus_fusion.ignored_tissue_fraction`
- `consensus_fusion.ignored_boundary_fraction`
- `consensus_fusion.boundary_length_before_refine` / `boundary_length_after_refine`
- `consensus_fusion.small_component_count_before_refine` / `small_component_count_after_refine`
- `consensus_fusion.excessive_tissue_ignore`
- `final_thresholds_used.ignore_confidence_threshold`

### Training/evaluation config

- `class_loss_weights`: per-class weights `[benign, G3, G4, G5]`  
  (`class_weights` is still supported for backward compatibility)
- `loss_variant`: `soft_dice` (default), `focal_dice`, or `tversky_dice`
- `eval_leave_one_rater_out`: when `true`, logs/reports LOO-consensus diagnostics
- `enforce_background_ignore`: defaults to `true`; forces non-tissue pixels to ignore during dataset loading

### Tissue/background handling in training

The model is trained on 4 tissue classes (`benign`, `G3`, `G4`, `G5`). Background is not a fifth class.

During dataset loading, tissue/background is estimated from RGB via Otsu + morphology. With
`enforce_background_ignore: true` (default), all detected non-tissue pixels are forced to `ignore=1`
before loss computation. This prevents accidental supervision on whitespace/background even if stored
`ignore_mask.png` does not fully cover it.

Important: this safety check validates consistency with loader-derived tissue/background, not pathology
ground-truth background labels.

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
  --target-ignore-tissue-frac 0.05 \
  --target-ignore-total-frac 0.12 \
  --ignore-threshold-min 0.05 \
  --ignore-threshold-max 0.35 \
  --auto-calibrate-ignore-threshold \
  --boundary-dilate-px 1 \
  --edge-smooth-open-px 0 \
  --edge-smooth-close-px 1 \
  --remove-small-islands-px 64 \
  --fill-small-holes-px 64 \
  --single-rater-ignore-policy confidence_mask \
  --disable-gpu \
  --workers 8
```

2. Audit background-ignore safety (recommended before training).

```bash
make audit-background-ignore
```

Read `outputs/background_ignore_audit.json`:

- `bg_not_ignored_*`: background leakage after `enforce_background_ignore` cleanup (should be near zero)
- `tissue_ignored_*`: how much tissue is ignored after cleanup

3. Train with upgraded loss controls (edit config first).
```yaml
# configs/deconver_2d_local.yaml
loss_variant: focal_dice
class_loss_weights: [1.0, 1.2, 1.2, 2.5]
eval_leave_one_rater_out: true
```

```bash
PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml
```

4. Evaluate checkpoint (includes LOO summary when enabled in run config).
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
