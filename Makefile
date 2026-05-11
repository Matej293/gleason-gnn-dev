.PHONY: help train eval smoke test consensus consensus-weighted viz-consensus-gt audit-background-ignore gnn-build gnn-build-all gnn-eval gnn-train gnn-train-all gnn-viz gnn-viz-best gnn-compare-viz 
.DEFAULT_GOAL := help

MAX_CASES ?= 64
CONFIG ?= configs/deconver_local.yaml
RUN ?= outputs/runs/<run_name>
GNN_GRAPHS_ROOT ?= outputs/graphs/20260510_022358_deconver_consensus_local
GNN_PROFILE ?= thesis
GNN_SEED ?= 42
GNN_MODEL ?= graphsage
GNN_RUN_DIR ?= outputs/gnn_runs/<run_dir>
GNN_TRAIN_NAME ?= thesis_graphsage
GNN_LOSS ?= focal
GNN_FOCAL_GAMMA ?= 2.0
GNN_HIDDEN_DIM ?=
GNN_DROPOUT ?=
GNN_FEATURE_DROPOUT ?= 0.1
GNN_EDGE_DROPOUT ?= 0.0
GNN_LR ?=
GNN_WEIGHT_DECAY ?=
GNN_EPOCHS ?=
GNN_PATIENCE ?=
GNN_BUILD_BATCH_SIZE ?= 4
GNN_BUILD_NUM_WORKERS ?= 8
GNN_CHECKPOINT ?=
GNN_SUPERPIXEL_PRESET ?=
GNN_NUM_SEGMENTS ?= 300
GNN_COMPACTNESS ?= 10.0
GNN_SIGMA ?= 1.0
GNN_TINY_SUPERPIXEL_MAX_PIXELS ?= 8
GNN_COMPARISON_DIR ?= outputs/gnn_runs/<comparison_dir>
GNN_RUNS_ROOT ?= outputs/gnn_runs
GNN_VIZ_SPLIT ?= test
GNN_MAX_CASES ?= 12
GNN_MODELS ?= mlp graphsage gcn gat
GNN_TRAIN_SEEDS ?= 3
GNN_SELECTION_METRIC ?= val_per_case_macro_f1
GNN_BUILD_SPLITS ?= train val test
GNN_EDGE_POLICY ?= touch
GNN_EDGE_KNN_K ?= 2
GNN_EDGE_KNN_MAX_DISTANCE ?= 0
GNN_PARITY_CHECK ?= on

help:
	@echo "Targets:"
	@echo "  make train [CONFIG=configs/deconver_local.yaml]"
	@echo "  make smoke"
	@echo "  make test"
	@echo "  make eval RUN=outputs/runs/<run_name>"
	@echo "  make consensus"
	@echo "  make consensus-weighted"
	@echo "  make viz-consensus-gt [MAX_CASES=64]"
	@echo "  make audit-background-ignore"
	@echo "  make gnn-build RUN=outputs/runs/<run_name> SPLIT=<train|val|test|all>"
	@echo "  make gnn-build-all RUN=outputs/runs/<run_name> [GNN_BUILD_SPLITS='train val test']"
	@echo "  make gnn-eval [GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>] [GNN_PROFILE=thesis]"
	@echo "  make gnn-train [GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>] [GNN_MODEL=<mlp|graphsage|gcn|gat>]"
	@echo "  make gnn-train-all [GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>] [GNN_TRAIN_NAME=<name_prefix>]"
	@echo "  make gnn-viz [GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>] GNN_RUN_DIR=outputs/gnn_runs/<run_dir>"
	@echo "  make gnn-viz-best [GNN_GRAPHS_ROOT=outputs/graphs/<graph_run>] [GNN_TRAIN_NAME=<name_prefix>]"
	@echo "  make gnn-compare-viz GNN_COMPARISON_DIR=outputs/gnn_runs/<comparison_dir> [GNN_GRAPHS_ROOT=...]"
	@echo "  gnn-build defaults: GNN_BUILD_BATCH_SIZE=4 GNN_BUILD_NUM_WORKERS=8"
	@echo "  gnn edge defaults: GNN_EDGE_POLICY=touch GNN_EDGE_KNN_K=2 GNN_EDGE_KNN_MAX_DISTANCE=0"

train:
	PYTHONPATH=. python -m src.train_deconver --config $(CONFIG)

eval:
	@if [ -z "$(RUN)" ]; then echo "Usage: make eval RUN=outputs/runs/<run_name>"; exit 1; fi
	PYTHONPATH=. python scripts/evaluate_checkpoint.py --run $(RUN) --save-viz --log-wandb-viz --log-wandb-metrics

smoke:
	PYTHONPATH=. python scripts/smoke_test.py

test:
	PYTHONPATH=. pytest -q tests

consensus:
	PYTHONPATH=. python scripts/build_consensus.py --dataset-root data --output-root data/consensus

consensus-weighted:
	PYTHONPATH=. python scripts/build_consensus.py \
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

gnn-build:
	@if [ -z "$(RUN)" ]; then echo "Usage: make gnn-build RUN=outputs/runs/<run_name> SPLIT=<train|val|test|all>"; exit 1; fi
	@if [ -z "$(SPLIT)" ]; then echo "Usage: make gnn-build RUN=outputs/runs/<run_name> SPLIT=<train|val|test|all>"; exit 1; fi
	PYTHONPATH=. python scripts/build_superpixel_graphs.py --run $(RUN) --split $(SPLIT) --batch-size $(GNN_BUILD_BATCH_SIZE) --num-workers $(GNN_BUILD_NUM_WORKERS) --num-segments $(GNN_NUM_SEGMENTS) --compactness $(GNN_COMPACTNESS) --sigma $(GNN_SIGMA) --tiny-superpixel-max-pixels $(GNN_TINY_SUPERPIXEL_MAX_PIXELS) --edge-policy $(GNN_EDGE_POLICY) --edge-knn-k $(GNN_EDGE_KNN_K) --edge-knn-max-distance $(GNN_EDGE_KNN_MAX_DISTANCE) $(if $(GNN_SUPERPIXEL_PRESET),--superpixel-preset $(GNN_SUPERPIXEL_PRESET),) $(if $(GNN_CHECKPOINT),--checkpoint $(GNN_CHECKPOINT),)

gnn-build-all:
	@if [ -z "$(RUN)" ]; then echo "Usage: make gnn-build-all RUN=outputs/runs/<run_name> [GNN_BUILD_SPLITS='train val test']"; exit 1; fi
	@for SPLIT in $(GNN_BUILD_SPLITS); do \
		echo "Building graph split: $$SPLIT"; \
		PYTHONPATH=. python scripts/build_superpixel_graphs.py --run $(RUN) --split $$SPLIT --batch-size $(GNN_BUILD_BATCH_SIZE) --num-workers $(GNN_BUILD_NUM_WORKERS) --num-segments $(GNN_NUM_SEGMENTS) --compactness $(GNN_COMPACTNESS) --sigma $(GNN_SIGMA) --tiny-superpixel-max-pixels $(GNN_TINY_SUPERPIXEL_MAX_PIXELS) --edge-policy $(GNN_EDGE_POLICY) --edge-knn-k $(GNN_EDGE_KNN_K) --edge-knn-max-distance $(GNN_EDGE_KNN_MAX_DISTANCE) $(if $(GNN_SUPERPIXEL_PRESET),--superpixel-preset $(GNN_SUPERPIXEL_PRESET),) $(if $(GNN_CHECKPOINT),--checkpoint $(GNN_CHECKPOINT),) || exit 1; \
	done

gnn-eval:
	PYTHONPATH=. python scripts/eval_gnn_baselines.py --graphs-root $(GNN_GRAPHS_ROOT) --profile $(GNN_PROFILE) --seed $(GNN_SEED)

gnn-train:
	PYTHONPATH=. python scripts/train_gnn_node_classifier.py \
		--graphs-root $(GNN_GRAPHS_ROOT) \
		--model $(GNN_MODEL) \
		--profile $(GNN_PROFILE) \
		--normalize-features \
		--loss $(GNN_LOSS) \
		--focal-gamma $(GNN_FOCAL_GAMMA) \
		--feature-dropout $(GNN_FEATURE_DROPOUT) \
		--edge-dropout $(GNN_EDGE_DROPOUT) \
		--seed $(GNN_SEED) \
		--selection-metric $(GNN_SELECTION_METRIC) \
		$(if $(GNN_HIDDEN_DIM),--hidden-dim $(GNN_HIDDEN_DIM),) \
		$(if $(GNN_DROPOUT),--dropout $(GNN_DROPOUT),) \
		$(if $(GNN_LR),--lr $(GNN_LR),) \
		$(if $(GNN_WEIGHT_DECAY),--weight-decay $(GNN_WEIGHT_DECAY),) \
		$(if $(GNN_EPOCHS),--epochs $(GNN_EPOCHS),) \
		$(if $(GNN_PATIENCE),--patience $(GNN_PATIENCE),) \
		--name $(GNN_TRAIN_NAME)

gnn-train-all:
	@for MODEL in $(GNN_MODELS); do \
		echo "Training $$MODEL"; \
		PYTHONPATH=. python scripts/train_gnn_node_classifier.py \
			--graphs-root $(GNN_GRAPHS_ROOT) \
			--model $$MODEL \
			--profile $(GNN_PROFILE) \
			--normalize-features \
			--residual-head \
			--residual-alpha 0.2 \
			--mask-unsupported-classes \
			--loss $(GNN_LOSS) \
			--focal-gamma $(GNN_FOCAL_GAMMA) \
			--feature-dropout $(GNN_FEATURE_DROPOUT) \
			--edge-dropout $(GNN_EDGE_DROPOUT) \
			--grad-clip-norm 1.0 \
			--scheduler \
			--selection-metric $(GNN_SELECTION_METRIC) \
			--seed $(GNN_SEED) \
			--seeds $(GNN_TRAIN_SEEDS) \
			$(if $(GNN_HIDDEN_DIM),--hidden-dim $(GNN_HIDDEN_DIM),) \
			$(if $(GNN_DROPOUT),--dropout $(GNN_DROPOUT),) \
			$(if $(GNN_LR),--lr $(GNN_LR),) \
			$(if $(GNN_WEIGHT_DECAY),--weight-decay $(GNN_WEIGHT_DECAY),) \
			$(if $(GNN_EPOCHS),--epochs $(GNN_EPOCHS),) \
			$(if $(GNN_PATIENCE),--patience $(GNN_PATIENCE),) \
			--name "$(GNN_TRAIN_NAME)_$$MODEL" || exit 1; \
	done

gnn-viz:
	@if [ "$(GNN_RUN_DIR)" = "outputs/gnn_runs/<run_dir>" ]; then echo "Usage: make gnn-viz GNN_RUN_DIR=outputs/gnn_runs/<run_dir> [GNN_GRAPHS_ROOT=...]"; exit 1; fi
	PYTHONPATH=. python scripts/visualize_gnn_predictions.py --graphs-root $(GNN_GRAPHS_ROOT) --run-dir $(GNN_RUN_DIR) --split $(GNN_VIZ_SPLIT) --seed $(GNN_SEED)

gnn-viz-best:
	@for MODEL in $(GNN_MODELS); do \
		BEST_RUN="$$(PYTHONPATH=. python scripts/select_best_gnn_run.py --model "$$MODEL" --name-prefix "$(GNN_TRAIN_NAME)" --runs-root "$(GNN_RUNS_ROOT)")"; \
		if [ -z "$$BEST_RUN" ]; then \
			echo "No run found for $$MODEL with prefix '$(GNN_TRAIN_NAME)' - skipping."; \
			continue; \
		fi; \
		echo "Best $$MODEL run: $$BEST_RUN"; \
		PYTHONPATH=. python scripts/visualize_gnn_predictions.py --graphs-root $(GNN_GRAPHS_ROOT) --run-dir "$$BEST_RUN" --split $(GNN_VIZ_SPLIT) --seed $(GNN_SEED) --parity-check $(GNN_PARITY_CHECK) --max-cases -1 || exit 1; \
	done

gnn-compare-viz:
	@if [ "$(GNN_COMPARISON_DIR)" = "outputs/gnn_runs/<comparison_dir>" ]; then echo "Usage: make gnn-compare-viz GNN_COMPARISON_DIR=outputs/gnn_runs/<comparison_dir> [GNN_GRAPHS_ROOT=...]"; exit 1; fi
	PYTHONPATH=. python scripts/visualize_gnn_baseline_comparison.py \
		--comparison-dir $(GNN_COMPARISON_DIR) \
		--graphs-root $(GNN_GRAPHS_ROOT) \
		--gnn-runs-root $(GNN_RUNS_ROOT) \
		--split $(GNN_VIZ_SPLIT) \
		--max-cases $(GNN_MAX_CASES) \
		--seed $(GNN_SEED)
