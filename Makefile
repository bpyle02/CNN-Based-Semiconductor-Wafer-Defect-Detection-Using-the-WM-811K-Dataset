# Makefile for CNN-Based Semiconductor Wafer Defect Detection

.PHONY: install bootstrap doctor train dashboard test test-cov defense demo federated active_learn compress progressive ood docs

# 1. Installation
install:
	python -m pip install -e ".[dev]"

bootstrap:
	python scripts/bootstrap_env.py

doctor:
	python scripts/doctor.py

# 2. Main Training Pipeline (Refactored)
train:
	python train.py --model all --epochs 5

# 3. Interactive Dashboard (Streamlit)
dashboard:
	streamlit run scripts/dashboard.py

# 4. Advanced ML Workflows
federated:
	pytest tests/unit/test_federated.py -v

active_learn:
	python scripts/active_learn.py --model cnn --n-iterations 3

compress:
	python scripts/compress_model.py --model cnn --method quantize

progressive:
	python scripts/progressive_train.py --model cnn

# 5. Testing & Validation
test:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=term-missing

ood:
	pytest tests/test_improvements.py::test_ood_detection -v

demo:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_defense_demo.ps1

defense:
	python scripts/finalize_submission.py

# 6. Documentation
docs:
	@echo "Documentation is located in the docs/ and docs/guides/ directories."
	@echo "Key Guides:"
	@echo "  - docs/guides/FEDERATED_LEARNING.md"
	@echo "  - docs/guides/INFERENCE_SERVER_README.md"
	@echo "  - docs/guides/UNCERTAINTY_QUANTIFICATION.md"
	@echo "  - docs/guides/COMPREHENSIVE_FEATURE_GUIDE.md"
