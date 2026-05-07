# Repository Guidelines

## Project Structure & Module Organization
Core training logic lives in `src/train_deconver_2d.py`, with model construction in `src/models/__init__.py` and evaluation helpers in `src/eval_utils.py`. The 2D checkpoint evaluator is `scripts/evaluate_checkpoint_2d.py`; quick pipeline validation is `scripts/smoke_test_2d.py`.

Data assumptions are fixed:
- training/test images: `data/Train_imgs`, `data/Test_imgs`
- consensus labels per image: `data/consensus/<image_id>/`

Tests are under `tests/`. Configs are in `configs/` (`deconver_2d.yaml`, `deconver_2d_local.yaml`). Run artifacts are written to `outputs/runs/<timestamp>_<experiment_name>/`.

## Build, Test, and Development Commands
- `PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml`  
  Start local 2D Gleason consensus training.
- `PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run outputs/runs/<run_name>`  
  Evaluate a saved run and write `evaluation_2d_summary.json`.
- `PYTHONPATH=. python scripts/smoke_test_2d.py`  
  Run a fast end-to-end sanity check.
- `PYTHONPATH=. pytest -q tests`  
  Run the repository test suite.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8 defaults. Keep modules focused and small; prefer explicit function names (`load_consensus_targets`, `evaluate_checkpoint_2d`) over abbreviations.

Naming patterns:
- files/modules: `snake_case.py`
- functions/variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`

Keep config keys consistent with existing YAML files (`use_amp`, `amp_dtype`, `use_compile`).

## Testing Guidelines
Use `pytest` with tests named `test_*.py` and functions named `test_*`. Add or update tests when changing data loading, metric computation, checkpoint I/O, or visualization outputs.

Target practical coverage of changed code paths rather than broad, unfocused tests. For model/config changes, run both:
1. `PYTHONPATH=. pytest -q tests`
2. `PYTHONPATH=. python scripts/smoke_test_2d.py`

## Commit & Pull Request Guidelines
Recent commits use concise, lowercase, imperative-style summaries (for example: `improving masks and consensus`). Follow that style and keep subject lines specific.

For PRs:
- describe what changed and why
- list affected commands/configs
- include before/after metric snippets when behavior changes
- link related issue(s)
- note any data-path or runtime assumptions

## Configuration & Hardware Notes
For TITAN V / Volta GPUs, keep:
- `use_amp: true`
- `amp_dtype: fp16`
- `use_compile: false`
