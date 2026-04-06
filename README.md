# CNN-Based Semiconductor Wafer Defect Detection Using the WM-811K Dataset

This repository contains a modular wafer-defect detection codebase built around the WM-811K dataset, with baseline CNN classifiers and a broader set of research-oriented extensions for inference, uncertainty estimation, anomaly detection, federated learning, synthetic augmentation, and model management.

## Verified Status

- Repository validation date: `2026-04-05`
- Test suite: 51 test functions across 5 files (static count verified via `grep -c "def test_"`)
- The inference API now supports Grad-CAM overlays in prediction responses.
- The test harness is stable inside the repo workspace and no longer depends on inaccessible temp directories.

The authoritative defense-facing artifacts are:

- `DEFENSE_SUBMISSION.md`
- `docs/DEFENSE_PACKET.md`
- `docs/presentation.tex`
- `docs/FINAL_STATUS_REPORT.md`

## What This Repository Demonstrates

The codebase is organized as a research software artifact rather than a single-script class project. It covers:

- Wafer-map loading and preprocessing
- A custom CNN and pretrained transfer-learning baselines
- A FastAPI inference server with model loading, prediction, and Grad-CAM overlays
- Monte Carlo dropout uncertainty estimation
- OOD and anomaly detection utilities
- Federated averaging with improved robust aggregation options
- Synthetic augmentation and generator training utilities
- Model registry and version metadata
- Integration and unit tests for core functionality

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the validated test suite

```bash
pytest -q
```

### 3. Run the committee smoke demo

This exercises the inference path, model loading, prediction, and Grad-CAM generation without needing an external web server:

```bash
./scripts/run_defense_demo.ps1
```

The wrapper uses the same Python environment that resolves `pytest`. If `fastapi` is installed in that environment, the demo writes a Grad-CAM overlay to `docs/generated/defense_demo_gradcam.png`.

### 4. Build the committee bundle

```bash
python scripts/finalize_submission.py
```

This creates `SUBMISSION_FINAL/` and `SUBMISSION_FINAL.zip`.

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
- The public test suite currently passes.
- The repo now includes an explainability-enabled inference path and a reproducible validation workflow.

Claims that should come from experiment logs or committee-approved report tables rather than the repo alone:

- Final task accuracy, macro F1, and ablation numbers
- Deployment-readiness or production-readiness certification
- Any benchmark not tied to versioned checkpoints and datasets in this repository

## Live Demo Options

Minimal defense demo:

```bash
pytest -q
./scripts/run_defense_demo.ps1
```

If you have a trained checkpoint and FastAPI dependencies available, you can also run the inference stack directly through the server utilities in `src/inference/server.py`.

## Known Limitations

- The repository does not currently ship a committee-approved trained checkpoint.
- The inference API supports a single active model at a time rather than a persistent multi-model serving registry.
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
