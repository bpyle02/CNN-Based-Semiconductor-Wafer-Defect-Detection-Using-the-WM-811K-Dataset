# CNN-Based Semiconductor Wafer Defect Detection Using the WM-811K Dataset

This repository contains a modular wafer-defect detection codebase built around the WM-811K dataset, with baseline CNN classifiers and a broader set of research-oriented extensions for inference, uncertainty estimation, anomaly detection, federated learning, synthetic augmentation, and model management.

## Verified Status

- Repository validation date: `2026-04-05`
- The standardized local conda environment is `base`.
- The default local verification command is `conda run -n base pytest -q`.
- The inference API now supports Grad-CAM overlays in prediction responses.
- Editable packaging is now defined in `pyproject.toml`, with optional dependency groups for dev, server, tuning, docs, dashboards, and export tooling.

The authoritative defense-facing artifacts are:

- `DEFENSE_SUBMISSION.md`
- `docs/DEFENSE_PACKET.md`
- `docs/presentation.tex`
- `docs/FINAL_STATUS_REPORT.md`

## What This Repository Demonstrates

The codebase is organized as a research software artifact rather than a single-script class project. It covers:

- Wafer-map loading and preprocessing
- A custom CNN and pretrained transfer-learning baselines
- A FastAPI inference server with trusted checkpoint loading, single-image prediction, batch prediction, and Grad-CAM overlays
- Monte Carlo dropout uncertainty estimation
- OOD and anomaly detection utilities
- Federated averaging with improved robust aggregation options
- Synthetic augmentation and generator training utilities
- Model registry, version metadata, and experiment-artifact manifests
- Integration and unit tests for core functionality

## Quick Start

### 1. Install dependencies

Use the standardized local Conda environment, `base`. It currently provides Python `3.13.9` and the full repo toolchain.

```bash
conda run -n base python -m pip install -e ".[dev]"
```

If you prefer an explicit bootstrap helper instead of a direct editable install:

```bash
conda run -n base python scripts/bootstrap_env.py
```

### 2. Diagnose the active environment

```bash
make doctor
conda run -n base python scripts/doctor.py --json
```

This is useful when `python`, `pip`, and `pytest` may be resolving to different environments.
The `make` targets already execute through `conda run -n base`, so they are the shortest path for day-to-day work.
If the doctor notes that packages resolve outside the conda prefix, the environment is still mixed; reinstall those packages into `base` to clean it up.

### 3. Run the validated test suite

```bash
make test
# or
conda run -n base python -m pytest -q
```

### 4. Use config overlays when needed

```bash
conda run -n base python train.py --config configs/base.yaml --config configs/train.yaml
```

Available overlays include `configs/base.yaml`, `configs/train.yaml`, `configs/inference.yaml`, and `configs/federated.yaml`.

### 5. Run the committee smoke demo

This exercises the inference path, model loading, prediction, and Grad-CAM generation without needing an external web server:

```bash
./scripts/run_defense_demo.ps1
```

The wrapper uses the same Python environment that resolves `pytest`. If `fastapi` is installed in that environment, the demo writes a Grad-CAM overlay to `docs/generated/defense_demo_gradcam.png`.

### 6. Build the committee bundle

```bash
conda run -n base python scripts/finalize_submission.py
```

This creates `SUBMISSION_FINAL/`, `SUBMISSION_FINAL.zip`, and a machine-readable `MANIFEST.json` with git state, artifact hashes, validation status, and optional checkpoint metadata.

## Repository Layout

```text
.
├── docs/
│   ├── DEFENSE_PACKET.md
│   ├── FINAL_STATUS_REPORT.md
│   ├── presentation.tex
│   └── wafer_defect_detection_report.tex
├── scripts/
│   ├── defense_smoke_demo.py
│   └── ...
├── src/
│   ├── analysis/
│   ├── augmentation/
│   ├── data/
│   ├── detection/
│   ├── federated/
│   ├── inference/
│   ├── models/
│   ├── training/
│   ├── config.py
│   ├── exceptions.py
│   └── model_registry.py
└── tests/
    ├── integration/
    ├── unit/
    └── ...
```

## Defense-Ready Claims

Claims that are justified directly by the checked-in repository:

- The codebase is modular and testable.
- Core inference, uncertainty, and federated-learning utilities are implemented.
- The public test suite is runnable through the standardized conda workflow: `make test` or `conda run -n base python -m pytest -q`.
- The repo now includes an explainability-enabled inference path and a reproducible validation workflow.

Claims that should come from experiment logs or committee-approved report tables rather than the repo alone:

- Final task accuracy, macro F1, and ablation numbers
- Deployment-readiness or production-readiness certification
- Any benchmark not tied to versioned checkpoints and datasets in this repository

## Live Demo Options

Minimal defense demo:

```bash
conda run -n base pytest -q
conda run -n base powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_defense_demo.ps1
```

If you have a trained checkpoint and FastAPI dependencies available, you can also run the inference stack directly through the server utilities in `src/inference/server.py`.
To start the server with an explicit trusted checkpoint root:

```bash
conda run -n base python scripts/inference_server.py \
  --model checkpoints/best_cnn.pth \
  --model-type cnn \
  --trusted-checkpoint-dir checkpoints
```

### Production Utilities

To ensure your environment and models are ready for production, use the following tools:

- **Environment Diagnosis**: `conda run -n base python scripts/doctor.py` - Checks for dependency health, NumPy/PyTorch sanity, and dataset availability.
- **Model Validation Audit**: `conda run -n base python scripts/validate_model.py --model cnn --checkpoint checkpoints/best_cnn.pth` - Performs a deep audit of model calibration (ECE), uncertainty quantification, and temperature scaling.
- **Inference Server**: `conda run -n base python -m src.inference.server` - Real-time REST API with batch support, multi-model registry, and performance metrics.

## Known Limitations

- The repository does not currently ship a committee-approved trained checkpoint.
- The inference API supports a single active model at a time rather than a persistent multi-model serving registry, and it now only accepts checkpoints from trusted repo-local directories.
- The custom exception hierarchy is now used in core runtime modules, but broader adoption across every subsystem remains incomplete.
- Historical course-era documents remain in `docs/`; use the defense packet and updated slide deck as the authoritative submission-facing materials.

## Recommended Submission Set

For committee review, package:

1. `DEFENSE_SUBMISSION.md`
2. `docs/DEFENSE_PACKET.md`
3. `docs/presentation.tex` and `docs/presentation.pdf`
4. `docs/FINAL_STATUS_REPORT.md`
5. `SUBMISSION_FINAL/` or `SUBMISSION_FINAL.zip`

## Dataset Note

The WM-811K dataset is not versioned into this repository. Place the dataset at `data/LSWMD_new.pkl` if you intend to run full training workflows that require the original wafer maps.
