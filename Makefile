# Makefile — convenience targets for reproducibility and development.
#
# `reproduce-fast` is the one a reviewer should run. It re-evaluates the
# committed PPO models on the committed held-out sets for all four regimes
# (raw, calibrated, regime, regime_raw) and regenerates the statistical
# summaries. Runs in ~1 minute on a laptop CPU.
#
# `reproduce-full` additionally retrains every (variant × regime) combination
# from scratch. ~4 hours on a consumer CPU. Useful for verifying the committed
# models, not something a reviewer should sit through.

.PHONY: help reproduce-fast reproduce-full multi-seed test lint clean
.DEFAULT_GOAL := help

DATASETS := raw calibrated regime regime_raw
VARIANTS := A B C

help:
	@echo "Reproducibility:"
	@echo "  make reproduce-fast   Regenerate all eval JSONs + statistical summaries (~1 min)"
	@echo "  make reproduce-full   Retrain all variants on all regimes, then reproduce-fast (~4 h)"
	@echo "  make multi-seed       Multi-seed robustness sweep (5 seeds x 3 variants x 2 regimes, ~5 h)"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run the pytest suite"
	@echo "  make lint             Run ruff + mypy on the extension dirs"
	@echo "  make clean            Remove pycache and pytest artefacts"

reproduce-fast:
	@for ds in $(DATASETS); do \
		echo "=== Regenerating $$ds regime ==="; \
		python -m agent.evaluate --dataset $$ds || exit 1; \
		python -m experiments.statistical_analysis --dataset $$ds || exit 1; \
	done
	@echo ""
	@echo "Done. See experiments/results/ for updated JSONs and .md summaries."

reproduce-full:
	@for ds in $(DATASETS); do \
		for r in $(VARIANTS); do \
			echo "=== Training variant $$r on $$ds regime ==="; \
			python -m agent.train --reward $$r --dataset $$ds || exit 1; \
		done; \
	done
	$(MAKE) reproduce-fast

multi-seed:
	python -m experiments.multi_seed --datasets raw regime --variants A B C --seeds 0 1 2 3 4

test:
	python -m pytest -x -v

lint:
	python -m ruff check agent api environment integration
	python -m mypy agent api environment integration

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
