# Makefile for CNN-Based Semiconductor Wafer Defect Detection

CONDA_ENV ?= py313
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)

.PHONY: help install bootstrap doctor train train-cnn train-gpu dashboard test test-cov federated active_learn compress progressive ood smoke smoke-bash docs precompute serve-api colab-sync lock pre-commit-install lint format check-all clean clean-cache

help:
	@echo ""
	@echo "Common targets (run 'make <target>'):"
	@echo ""
	@echo "  install              Editable install + dev extras"
	@echo "  test                 Full pytest suite"
	@echo "  lint                 flake8 over src/ and train.py"
	@echo "  format               black src/ train.py scripts/"
	@echo "  check-all            lint + test + doctor (pre-push sanity gate)"
	@echo ""
	@echo "  smoke-bash           30s end-to-end shell smoke test (imports + toy train)"
	@echo "  precompute           Build data/LSWMD_cache.npz from LSWMD_new.pkl"
	@echo "  train-cnn            Custom CNN only, 5 epochs, GPU"
	@echo "  train-gpu            All three baselines, 20 epochs, GPU"
	@echo ""
	@echo "  serve-api            Launch FastAPI inference server on :8000"
	@echo "  colab-sync           Print the Colab notebook URL for main"
	@echo ""
	@echo "  lock                 Regenerate requirements-lock.txt"
	@echo "  pre-commit-install   Install and register pre-commit hooks"
	@echo "  clean                Remove __pycache__, .pytest_cache, *.egg-info"
	@echo "  clean-cache          Also remove data/LSWMD_cache.npz and wafer_runs/"
	@echo ""

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

# 6. Smoke-test: 1-epoch CPU training on synthetic-augmented data. Requires
# data/LSWMD_new.pkl on disk (for CI / dataset-less envs use `make smoke-bash`).
smoke:
	$(CONDA_RUN) python -s train.py --model cnn --epochs 1 --batch-size 32 --device cpu --seed 42 --synthetic

# 7. Lint / format / reproducibility / housekeeping
lint:
	$(CONDA_RUN) python -m flake8 src train.py --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	$(CONDA_RUN) python -m black src train.py scripts

check-all: lint test doctor

precompute:
	$(CONDA_RUN) python -s scripts/precompute_tensors.py

serve-api:
	$(CONDA_RUN) python -s -m src.inference.server --host 0.0.0.0 --port 8000

colab-sync:
	@echo "https://colab.research.google.com/github/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/blob/main/docs/colab_quickstart.ipynb"

lock:
	$(CONDA_RUN) python -s scripts/generate_lock.py > requirements-lock.txt
	@echo "Regenerated requirements-lock.txt"

pre-commit-install:
	$(CONDA_RUN) python -s -m pip install pre-commit
	$(CONDA_RUN) pre-commit install

smoke-bash:
	bash scripts/smoke_test.sh

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info build dist

clean-cache: clean
	@rm -f data/LSWMD_cache.npz
	@rm -rf wafer_runs/

# 8. Documentation
docs:
	@echo "Documentation is in docs/ and docs/guides/."
	@echo "Key guides:"
	@echo "  - docs/guides/FEDERATED_LEARNING.md"
	@echo "  - docs/guides/INFERENCE_SERVER_README.md"
	@echo "  - docs/guides/UNCERTAINTY_QUANTIFICATION.md"
	@echo "  - docs/COLAB_SETUP.md"
