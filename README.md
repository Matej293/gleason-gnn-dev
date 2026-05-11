# ProstateLesionSegmentation (2D Gleason Consensus Segmentation + Graph Prep)

This repository provides a 2D pipeline for Gleason consensus segmentation and
region-graph preparation:
- segmentation models: `deconver` and `unet_lite`
- consensus-aware training/evaluation with tissue-based background ignore
- superpixel graph artifact export for downstream GNN experiments

## Train

`deconver`:
```bash
PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml
```

`unet_lite` (fast baseline):
```bash
PYTHONPATH=. python -m src.train_deconver_2d --config configs/unet_lite_2d_local.yaml
```

Training logs are written to Weights & Biases (W&B) per epoch.  
Set `WANDB_API_KEY` for online logging, or set `wandb_mode: offline` in config for local-only logging.

## Evaluate

```bash
PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run outputs/runs/<run_name>
```

Evaluation JSON includes:
- `aggregate_raw`: per-case mean metrics from raw argmax predictions
- `aggregate_post`: per-case mean metrics after postprocessing
- `aggregate`: alias of `aggregate_post` (backward compatibility)
- `per_case`: both `raw_*` and `post_*` metrics per image

## Smoke Test

```bash
PYTHONPATH=. python scripts/smoke_test_2d.py
```

## Build Superpixel Graph Artifacts

Create superpixel-based node/edge/feature artifacts for GNN training from
model predictions (checkpoint-driven, thesis default):

```bash
PYTHONPATH=. python scripts/build_superpixel_graphs.py \
  --run outputs/runs/<run_name> \
  --split test
```

Example with a real UNet-lite run:
```bash
PYTHONPATH=. python scripts/build_superpixel_graphs.py \
  --run outputs/runs/20260510_184849_unet_lite_2d_consensus_local \
  --split test
```

Outputs are saved to:

```text
outputs/graphs/<run_name>/<split>/<image_id>/graph_data.npz
```

## Train GNN Node Classifier

Install PyTorch Geometric (PyG) with a wheel that matches your Torch/CUDA build.
Example (adjust the `cu*` selector to your local torch wheel):

```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Train a GNN baseline on prepared graph splits (`mlp`, `graphsage`, `gcn`, `gat`):

```bash
PYTHONPATH=. python scripts/train_gnn_node_classifier.py \
  --graphs-root outputs/graphs/<run_name> \
  --model graphsage
```

Run baseline comparison (`seg_only`, `mlp`, `graphsage`, `gcn`, `gat`):

```bash
PYTHONPATH=. python scripts/eval_gnn_baselines.py \
  --graphs-root outputs/graphs/<run_name>
```

Compare + visualize baselines:

```bash
make gnn-eval GNN_GRAPHS_ROOT=outputs/graphs/<run_name>
make gnn-compare-viz \
  GNN_COMPARISON_DIR=outputs/gnn_runs/<timestamp>_baseline_comparison \
  GNN_GRAPHS_ROOT=outputs/graphs/<run_name>
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
| `val_raw/*` | Validation | Metrics on raw argmax predictions |
| `val_post/*` | Validation | Metrics after postprocessing |
| `val/composite_score` | Validation | Checkpoint ranking score from selected source (`raw` or `post`) |
| `mean_loo_dice_multiclass` | Eval summary (when enabled) | Mean leave-one-rater-out Dice from `qc_report.json` |
| `num_loo_entries` | Eval summary (when enabled) | Number of LOO entries used in that mean |

Notes:
- Validation metrics are logged as per-case means (not batch means) during training and evaluation.
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

### Loss Computation (Important)

The total training loss is:

```text
total_loss = lambda_soft * soft_term + lambda_dice * hard_overlap_term
```

Where:
- `soft_dice`:
  - `soft_term` = soft-label CE (`soft_label_loss: ce`) or KL (`soft_label_loss: kl`) vs STAPLE probabilities
  - `hard_overlap_term` = Dice loss
- `tversky_dice`:
  - same soft-label term as above
  - hard term uses Tversky loss (`alpha=0.3`, `beta=0.7`)
- `focal_dice`:
  - `soft_term` is replaced by hard-label focal CE (with class weights)
  - this variant does **not** use soft-label CE/KL

Additional details:
- `use_confidence_mask` + `confidence_threshold` excludes low-consensus pixels from both terms.
- `exclude_absent_classes_in_dice_loss` prevents absent classes from contributing to hard overlap loss.

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

4. Evaluate checkpoint (includes raw/post per-case and aggregate summaries).
```bash
PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run outputs/runs/<run_name>
```

5. Build superpixel graph artifacts for GNN-stage experiments.
```bash
PYTHONPATH=. python scripts/build_superpixel_graphs.py \
  --run outputs/runs/<run_name> \
  --split test
```

## Configs

- `configs/deconver_2d.yaml`
- `configs/deconver_2d_local.yaml`
- `configs/unet_lite_2d_local.yaml`

All provided local configs are set up for TITAN V / Volta compatibility:

- `use_amp: true`
- `amp_dtype: fp16`
- `use_compile`: model/config dependent (`deconver` default false, `unet_lite` fast config true)

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
