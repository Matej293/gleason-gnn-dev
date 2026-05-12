# ProstateLesionSegmentation

Segmentation and graph-preparation pipeline for the Gleason 2019 challenge dataset.

## Scope

This repository covers:

- Segmentation training (`deconver`, `unet_lite`, `pspnet_gleason`)
- Consensus label generation (STAPLE and weighted fusion)
- Checkpoint evaluation (raw and postprocessed metrics)
- Superpixel graph export
- GNN baselines (`mlp`, `graphsage`, `gcn`, `gat`) and visualization

## Repository structure

- `src/`: training, datasets, model code, consensus, graph pipeline, GNN modules
- `scripts/`: runnable CLI entry points
- `configs/`: training/evaluation configs
- `tests/`: `pytest` test suite
- `data/`: dataset and consensus outputs (local only)
- `outputs/`: training runs, graph artifacts, reports, visualizations

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

`data/` is not tracked in git.

Register/download Gleason 2019 here:
https://gleason2019.grand-challenge.org/Register/

Expected layout:

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
      qc_report.json
```

## Segmentation pipeline

1. Build consensus labels.

```bash
make consensus-weighted
# baseline alternative:
# make consensus
```

2. Optional sanity check for tissue/background ignore handling.

```bash
make audit-background-ignore
```

3. Train one segmentation model.

```bash
# deconver
make train CONFIG=configs/deconver_local.yaml

# unet_lite
make train CONFIG=configs/unet_lite_local.yaml

# pspnet_gleason
make train CONFIG=configs/pspnet_gleason_local.yaml
```

4. Evaluate the segmentation run.

```bash
make eval RUN=outputs/runs/<run_name>
```

5. Run segmentation sanity checks.

```bash
make smoke
make test
```

Segmentation outputs:

- Trained runs are stored under `outputs/runs/<run_name>/`
- Use that `<run_name>` as input to the GNN pipeline

## GNN pipeline

1. Export graph artifacts from a completed segmentation run.

```bash
make gnn-build-all RUN=outputs/runs/<run_name>
# split-specific alternative:
# make gnn-build RUN=outputs/runs/<run_name> SPLIT=train|val|test
```

2. Run GNN baseline evaluation.

```bash
make gnn-eval GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>
```

3. Train GNN models.

```bash
# single model
make gnn-train GNN_GRAPHS_ROOT=outputs/graphs/<graph_run> GNN_MODEL=graphsage

# full set: mlp, graphsage, gcn, gat
make gnn-train-all GNN_GRAPHS_ROOT=outputs/graphs/<graph_run> GNN_TRAIN_NAME=thesis
```

4. Visualize predictions and model comparisons.

```bash
make gnn-viz GNN_GRAPHS_ROOT=outputs/graphs/<graph_run> GNN_RUN_DIR=outputs/gnn_runs/<run_dir>
make gnn-viz-best GNN_GRAPHS_ROOT=outputs/graphs/<graph_run> GNN_TRAIN_NAME=thesis
make gnn-compare-viz GNN_COMPARISON_DIR=outputs/gnn_runs/<comparison_dir> GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>
```

Conventions:

- `<run_name>`: segmentation run folder under `outputs/runs/`
- `<graph_run>`: graph folder under `outputs/graphs/`, typically derived from `<run_name>`

## Core Make targets

```bash
make help
make train CONFIG=configs/deconver_local.yaml
make eval RUN=outputs/runs/<run_name>
make consensus
make consensus-weighted
make gnn-build RUN=outputs/runs/<run_name> SPLIT=test
make gnn-eval GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>
```

## CLI equivalents

```bash
PYTHONPATH=. python -m src.train_deconver --config configs/deconver_local.yaml
PYTHONPATH=. python scripts/evaluate_checkpoint.py --run outputs/runs/<run_name>
PYTHONPATH=. python scripts/build_superpixel_graphs.py --run outputs/runs/<run_name> --split test
PYTHONPATH=. python scripts/train_gnn_node_classifier.py --graphs-root outputs/graphs/<graph_run> --model graphsage
PYTHONPATH=. python scripts/eval_gnn_baselines.py --graphs-root outputs/graphs/<graph_run>
```

## Evaluation output

`evaluate_checkpoint.py` writes summary JSON with:

- `aggregate_raw`: per-case means from raw argmax predictions
- `aggregate_post`: per-case means after postprocessing
- `aggregate`: alias of `aggregate_post` (compatibility)
- `per_case`: per-image raw and post metrics

## Consensus and training notes

Weighted consensus entry point:

```bash
PYTHONPATH=. python scripts/build_consensus.py \
  --dataset-root data \
  --output-root data/consensus \
  --consensus-fusion-mode weighted
```

Useful config keys:

- `consensus_fusion_mode`: `staple_unweighted` or `weighted`
- `auto_calibrate_ignore_threshold`
- `single_rater_ignore_policy`: `confidence_mask` or `all_ignore`
- `enforce_background_ignore`

Training uses 4 tissue classes: `benign`, `G3`, `G4`, `G5` (background excluded).

Loss form:

```text
total_loss = lambda_soft * soft_term + lambda_dice * hard_overlap_term
```

Supported variants:

- `soft_dice`
- `tversky_dice` (`alpha=0.3`, `beta=0.7`)
- `focal_dice`

## Weights & Biases

- Set `WANDB_API_KEY` for online logging
- Use offline mode in config when needed
- Keep local `wandb/` directory untracked

## Testing

```bash
PYTHONPATH=. pytest -q tests
pytest -q tests/test_graph_pipeline.py
```
