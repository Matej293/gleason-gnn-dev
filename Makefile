.PHONY: help train eval smoke test consensus consensus-weighted viz-consensus-gt audit-background-ignore
.DEFAULT_GOAL := help

MAX_CASES ?= 64

help:
	@echo "Targets:"
	@echo "  make train"
	@echo "  make smoke"
	@echo "  make test"
	@echo "  make eval RUN=outputs/runs/<run_name>"
	@echo "  make consensus"
	@echo "  make consensus-weighted"
	@echo "  make viz-consensus-gt [MAX_CASES=64]"
	@echo "  make audit-background-ignore"

train:
	PYTHONPATH=. python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml

eval:
	@if [ -z "$(RUN)" ]; then echo "Usage: make eval RUN=outputs/runs/<run_name>"; exit 1; fi
	PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py --run $(RUN)

smoke:
	PYTHONPATH=. python scripts/smoke_test_2d.py

test:
	PYTHONPATH=. pytest -q tests

consensus:
	PYTHONPATH=. python scripts/build_consensus_2d.py --dataset-root data --output-root data/consensus

consensus-weighted:
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

viz-consensus-gt:
	PYTHONPATH=. python scripts/generate_consensus_gt_viz.py --max-cases $(MAX_CASES)

audit-background-ignore:
	PYTHONPATH=. python scripts/audit_background_ignore.py
