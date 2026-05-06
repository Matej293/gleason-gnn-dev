# AGENTS.md - ProstateLesionSegmentation (Simplified)

## Scope

This repository is intentionally reduced to:

- 2D Gleason consensus training: `src/train_deconver_2d.py`
- 2D checkpoint evaluation: `scripts/evaluate_checkpoint_2d.py`
- Deconver model factory: `src/models/__init__.py`

No Docker, no 3D pipeline, and no SSL pretraining remain.

## Primary commands

- Train: `PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml`
- Evaluate: `PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run outputs/runs/<run_name>`
- Smoke test: `PYTHONPATH=. python scripts/smoke_test_2d.py`
- Tests: `PYTHONPATH=. pytest -q tests`

## Configs kept

- `configs/deconver_2d.yaml`
- `configs/deconver_2d_local.yaml`

For TITAN V / Volta:

- `use_amp: true`
- `amp_dtype: fp16`
- `use_compile: false`

## Dataset assumptions

- Images are discovered from `data/Train_imgs` and `data/Test_imgs` by image id.
- Consensus labels live in `data/consensus/<image_id>/` with:
  - `consensus_probs_compact.npz`
  - `consensus_hard_mask.png`
  - `ignore_mask.png`
  - optional `qc_report.json`

## Outputs

- Training artifacts: `outputs/runs/<timestamp>_<experiment_name>/`
- Evaluation summary: `outputs/runs/<run_name>/evaluation_2d_summary.json`
