# Makefile for CNN-Based Semiconductor Wafer Defect Detection

CONDA_ENV ?= base
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)

.PHONY: install bootstrap doctor train dashboard test test-cov defense demo federated active_learn compress progressive ood docs

# 1. Installation
install:
	$(CONDA_RUN) python -s -m pip install -e ".[dev]"

bootstrap:
	$(CONDA_RUN) python -s scripts/bootstrap_env.py

doctor:
	$(CONDA_RUN) python -s scripts/doctor.py

# 2. Main Training Pipeline (Refactored)
train:
	$(CONDA_RUN) python -s train.py --model all --epochs 5

# 3. Interactive Dashboard (Streamlit)
dashboard:
	$(CONDA_RUN) python -s -m streamlit run scripts/dashboard.py

# 4. Advanced ML Workflows
federated:
	$(CONDA_RUN) python -s -m pytest tests/unit/test_federated.py -v

active_learn:
	$(CONDA_RUN) python -s scripts/active_learn.py --model cnn --n-iterations 3

compress:
	$(CONDA_RUN) python -s scripts/compress_model.py --model cnn --method quantize

progressive:
	$(CONDA_RUN) python -s scripts/progressive_train.py --model cnn

# 5. Testing & Validation
test:
	$(CONDA_RUN) python -s -m pytest -q

test-cov:
	$(CONDA_RUN) python -s -m pytest --cov=src --cov-report=term-missing

ood:
	$(CONDA_RUN) python -s -m pytest tests/test_improvements.py::test_ood_detection -v

demo:
	$(CONDA_RUN) powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_defense_demo.ps1

defense:
	$(CONDA_RUN) python -s scripts/finalize_submission.py

# 6. Documentation
docs:
	@echo "Documentation is located in the docs/ and docs/guides/ directories."
	@echo "Key Guides:"
	@echo "  - docs/guides/FEDERATED_LEARNING.md"
	@echo "  - docs/guides/INFERENCE_SERVER_README.md"
	@echo "  - docs/guides/UNCERTAINTY_QUANTIFICATION.md"
	@echo "  - docs/guides/COMPREHENSIVE_FEATURE_GUIDE.md"
