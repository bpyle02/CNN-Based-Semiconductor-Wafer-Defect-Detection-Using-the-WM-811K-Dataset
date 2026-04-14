# Makefile for CNN-Based Semiconductor Wafer Defect Detection

CONDA_ENV ?= py313
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)

.PHONY: install bootstrap doctor train dashboard test test-cov federated active_learn compress progressive ood smoke docs

# 1. Installation
install:
	$(CONDA_RUN) python -s -m pip install -e ".[dev]"

bootstrap:
	$(CONDA_RUN) python -s scripts/bootstrap_env.py

doctor:
	$(CONDA_RUN) python -s scripts/doctor.py

# 2. Main Training Pipeline
train:
	$(CONDA_RUN) python -s train.py --model all --epochs 5

train-cnn:
	$(CONDA_RUN) python -s train.py --model cnn --epochs 5

train-gpu:
	$(CONDA_RUN) python -s train.py --model all --epochs 20 --device cuda --batch-size 128

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
	$(CONDA_RUN) python -s -m pytest tests/unit -k ood -v

# 6. Smoke-test end-to-end path (no training loop): load data, build model, run forward pass, save checkpoint
smoke:
	$(CONDA_RUN) python -s train.py --model cnn --epochs 1 --smoke-test

# 7. Documentation
docs:
	@echo "Documentation is in docs/ and docs/guides/."
	@echo "Key guides:"
	@echo "  - docs/guides/FEDERATED_LEARNING.md"
	@echo "  - docs/guides/INFERENCE_SERVER_README.md"
	@echo "  - docs/guides/UNCERTAINTY_QUANTIFICATION.md"
	@echo "  - docs/COLAB_SETUP.md"
