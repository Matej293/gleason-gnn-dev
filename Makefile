.PHONY: help train eval smoke test consensus
.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  make train"
	@echo "  make smoke"
	@echo "  make test"
	@echo "  make eval RUN=outputs/runs/<run_name>"
	@echo "  make consensus"

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
