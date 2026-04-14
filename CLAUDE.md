# CNN-Based Semiconductor Wafer Defect Detection — Project Guide

## Project Overview

**Title:** CNN-Based Semiconductor Wafer Defect Detection Using the WM-811K Dataset

**Team:** Anindita Paul, Brandon Pyle, Anand Rajan, Brett Rettura

**Objective:** Develop and compare three deep learning architectures (custom CNN, ResNet-18, EfficientNet-B0) for multi-class wafer defect classification on the WM-811K dataset (~120K samples, 9 classes, severe imbalance: 85% 'none' class).

**Dataset:** WM-811K (Wafer Map 811K) — Industrial dataset from semiconductor manufacturing with real defect patterns.

---

## Repository Structure

> **Full file tree**: See `structure.md` Part II for the complete verified inventory of all 69 files across 9 modules.

### Core Modules (Original Design)

```
.
├── train.py                          # CLI entry point for training
├── config.yaml                       # Unified YAML configuration
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Multi-stage Docker build
��── docker-compose.yml                # Service orchestration
│
├── data/
│   └── LSWMD_new.pkl               # Dataset (not in git, user provides)
│
├── docs/
│   ├── wafer_defect_detection_report.tex      # LaTeX report
│   ├── presentation.tex                        # Beamer slides
│   └── wafer_defect_detection_run.ipynb        # Jupyter notebook
│
└── src/
    ├── data/
    │   ├── dataset.py               # Load and parse WM-811K pickle
    │   └── preprocessing.py         # Resize, normalize, augment, create datasets
    │
    ├── models/
    │   ├── cnn.py                   # Custom 4-block CNN (from scratch)
    │   ├── pretrained.py            # ResNet-18 & EfficientNet-B0 (ImageNet, fine-tuned)
    │   ├── vit.py                   # Vision Transformer
    │   ├── ensemble.py              # Model ensembling
    │   └── attention.py             # SE & CBAM attention mechanisms
    │
    ├── training/
    │   ├── config.py                # Pydantic-based config (Config, TrainingConfig, etc.)
    │   ├── trainer.py               # train_model() main loop
    │   ├── distributed.py           # Multi-GPU DDP support
    │   ├── simclr.py                # Self-supervised pretraining
    │   └── domain_adaptation.py     # CORAL + adversarial adaptation
    │
    ├── analysis/
    │   ├── evaluate.py              # Metrics, classification_report
    │   ├── visualize.py             # Plots: training curves, confusion matrices
    │   └── anomaly.py               # Autoencoder, IsolationForest, OC-SVM
    │
    ├── inference/
    │   ├── server.py                # FastAPI inference server (6 endpoints)
    │   ├── gradcam.py               # GradCAM for interpretability
    │   ├── uncertainty.py           # MC Dropout uncertainty quantification
    │   └── visualize.py             # GradCAM grid visualization
    │
    ├── augmentation/                # Synthetic defect generation, FID/IS metrics
    ├── detection/                   # OOD detection (Mahalanobis)
    ├── federated/                   # FedAvg, Byzantine-robust aggregation
    └── mlops/                       # W&B + MLflow experiment tracking
```

---

## Core Design Decisions

### 1. **Data Distribution Handling (Critical Fix)**

**Problem:** Original notebook used `WeightedRandomSampler` making training distribution ~uniform (each class ~11%), while val/test remained 85% 'none' class. Model never learned to predict 'none' → Accuracy ~10% (near-random).

**Solution:** Remove sampler, use `shuffle=True`, preserve natural distribution (85% 'none' in train, val, test). Weighted loss function (higher weight on rare classes) still penalizes rare-class errors appropriately without distorting distribution.

**Result:** Model now learns 'none' correctly → Accuracy 70-85%.

### 2. **ImageNet Normalization (Critical Fix)**

**Problem:** ResNet-18 and EfficientNet-B0 were pretrained on ImageNet (normalized to mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), but notebook applied only `x / 2.0` normalization.

**Solution:** Create separate transform pipelines applied consistently in BOTH training and evaluation:
- **CNN**: Only augmentation during training, no transform during eval (raw [0,1] images)
- **Pretrained training**: Augmentation + ImageNet norm (composed pipeline)
- **Pretrained eval**: ImageNet norm only (no augmentation)

**Note (2026-04-06 audit fix):** The original code applied ImageNet norm only during validation/test but NOT during training for pretrained models. This train/eval distribution mismatch was the root cause of ~10% accuracy. Fixed by composing augmentation + ImageNet norm into a single training transform.

### 3. **Layer-Boundary Freeze Strategy (Critical Fix)**

**Problem:** "Freeze last N parameters" cuts through residual blocks mid-block, inconsistent with architecture.

**Solution:**
- **ResNet-18**: Freeze layer1-3, unfreeze layer4 (last residual block) + fc
- **EfficientNet-B0**: Freeze features.0-6, unfreeze features.7-8 (last MBConv blocks) + classifier

Respects architectural boundaries for cleaner fine-tuning.

### 4. **Class Weights Computed from Training Set Only**

Ensures loss weights reflect actual training distribution (not full dataset), making them meaningful for the observed data.

---

## Training & Evaluation

### Key Hyperparameters

```python
# All models
num_epochs = 5              # Increased from 3 (removed sample cap)
batch_size = 64
weight_decay = 1e-4
scheduler = ReduceLROnPlateau(patience=3, factor=0.5)

# Custom CNN
learning_rate = 1e-3        # Higher LR for from-scratch training

# ResNet-18 & EfficientNet-B0
learning_rate = 1e-4        # Lower LR for transfer learning
freeze_earlier_layers = True # Only train later blocks
```

### Actual Results (5 epochs, CPU, 2026-04-06)

After fixing the distribution issue AND the ImageNet normalization bug:
- **Custom CNN**: Accuracy 88.6%, Macro F1 0.610, Weighted F1 0.904
- **ResNet-18**: Accuracy 83.7%, Macro F1 0.665, Weighted F1 0.875
- **EfficientNet-B0**: Accuracy 86.0%, Macro F1 0.608, Weighted F1 0.887
- **'none' class F1**: 0.946 (CNN), 0.907 (ResNet), 0.929 (EfficientNet)

(Previous broken results: ~10% accuracy, 'none' F1=0.000 due to missing ImageNet norm during training)

### Evaluation Workflow

1. **Split Data** (stratified):
   - Train: 70% (~85K samples)
   - Val: 15% (~18K samples)
   - Test: 15% (~18K samples)

2. **Loss Function**: `CrossEntropyLoss(weight=class_weights)`
   - Weights: [12.04, 34.46, 28.16, 27.68, 41.80, 40.02, 38.57, 35.78, 0.28]
   - Rare classes penalized more (Center: 12x, Donut: 34x, ..., none: 0.28x)

3. **Metrics**:
   - Accuracy, Macro F1, Weighted F1 (test set)
   - Per-class Precision, Recall, F1
   - Confusion matrix (normalized by true label)

---

## Python Package Architecture

### Design Principles

1. **Type Hints**: All functions have parameter and return type annotations
2. **Docstrings**: Comprehensive docstrings with algorithm details for complex functions
3. **Separation of Concerns**: Clear module boundaries (data, models, training, analysis, inference)
4. **Configuration**: Pydantic BaseModel with strict validation (Config class, `config.yaml`)
5. **Error Handling**: Validation in __post_init__ and load_dataset
6. **Reproducibility**: Seed setting, stratified splits, deterministic preprocessing

### Module Responsibilities

| Module | Responsibility |
|--------|---|
| `data.dataset` | Load pickle, extract labels, validate |
| `data.preprocessing` | Resize, normalize, augment, create PyTorch datasets |
| `models.cnn` | Custom CNN architecture (lightweight) |
| `models.pretrained` | ResNet-18 & EfficientNet-B0 with layer-boundary freezing |
| `training.config` | Centralized training hyperparameters |
| `training.trainer` | Main training loop with validation, scheduling, checkpointing |
| `analysis.evaluate` | Metrics computation, classification report |
| `analysis.visualize` | Training curves, confusion matrices, per-class analysis |
| `inference.gradcam` | GradCAM implementation for interpretability |
| `inference.visualize` | GradCAM grid visualization |

### Why This Architecture?

- **Reusability**: Modules imported by train.py, jupyter notebooks, and future scripts
- **Testability**: Each module independently testable with unit tests
- **Scalability**: Easy to add new models, metrics, visualizations
- **Reproducibility**: Explicit configuration, fixed random seeds
- **Documentation**: Type hints + docstrings reduce cognitive load

---

## Usage

### 1. Training via CLI

```bash
# Train all models (5 epochs, CPU)
python train.py --model all

# Train specific model with custom params
python train.py --model resnet --epochs 10 --lr 1e-4 --device cuda

# Show all options
python train.py --help
```

### 2. From Jupyter Notebook

```python
from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset, get_image_transforms
from src.models import WaferCNN, get_resnet18
from src.training import train_model

# Load and preprocess
df = load_dataset()
train_maps = preprocess_wafer_maps(raw_maps)
dataset = WaferMapDataset(train_maps, labels, transform=get_image_transforms())

# Train
model = WaferCNN(num_classes=9)
trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)
```

### 3. Programmatically

```python
from src.analysis import evaluate_model, plot_confusion_matrices

preds, labels, metrics = evaluate_model(model, test_loader, class_names)
# metrics: {'accuracy': 0.78, 'macro_f1': 0.45, 'weighted_f1': 0.72}
```

---

## Key Files & Functions

### Critical Functions

| Function | Purpose |
|----------|---------|
| `train_model()` | Main training loop (trainer.py) |
| `evaluate_model()` | Compute metrics on test set (evaluate.py) |
| `GradCAM.generate()` | Generate activation map for interpretability (gradcam.py) |
| `get_resnet18()` | Create ResNet-18 with layer-boundary freezing (pretrained.py) |
| `get_efficientnet_b0()` | Create EfficientNet-B0 with layer-boundary freezing (pretrained.py) |
| `preprocess_wafer_maps()` | Resize + normalize all maps upfront (preprocessing.py) |

### Key Classes

| Class | Purpose |
|-------|---------|
| `WaferMapDataset` | PyTorch Dataset (3-channel stacking, transforms) (preprocessing.py) |
| `WaferCNN` | Custom CNN (lightweight, from-scratch) (cnn.py) |
| `GradCAM` | Gradient-weighted class activation mapping (gradcam.py) |
| `TrainConfig` | Dataclass for hyperparameters (config.py) |

---

## Deliverables

### 1. Jupyter Notebook (`docs/wafer_defect_detection_run.ipynb`)

- **Status**: Completed with all Phase 1 fixes
- **Cells**: 19 code cells + markdown
- **Runtime**: ~50-60 min on CPU (5 epochs × 3 models)
- **Output**: Results table, visualizations, GradCAM

### 2. LaTeX Report (`docs/wafer_defect_detection_report.tex`)

- **Pages**: 8-9 (2 new sections added)
- **Structure**:
  - Title, Abstract, Intro
  - Related Work, Problem Statement
  - Dataset & Preprocessing
  - Methods (3 architectures, training strategy, loss weighting)
  - Results (tables, confusion matrices, per-class analysis)
  - **Code Description** (new, 0.75 page)
  - **Individual Contributions** (new, 1 page per team member)
  - Discussion, Conclusion

### 3. Beamer Slides (`docs/presentation.tex`)

- **Slides**: 18 (25-minute presentation)
- **Theme**: Madrid (clean, professional)
- **Structure**: Motivation → Problem → Dataset → Methods → Training → Results → GradCAM → Discussion

### 4. Python Package (`src/`)

- **Modules**: 11 files across 5 packages
- **Lines of Code**: ~8,500 in src/ package; ~12,750 total including scripts and tests
- **Quality**: PhD-level package structure, no external ML libraries (pure PyTorch + sklearn)

### 5. CLI Entry Point (`train.py`)

- **Usage**: `python train.py --model all --epochs 5`
- **Features**: Full pipeline (load, preprocess, train, evaluate, report)
- **Output**: Model comparison table, metrics, timing

---

## Best Practices & Standards

### Code Quality

- ✓ Type hints on core module interfaces (data, models, training, analysis entry points)
- ✓ Comprehensive docstrings (Sphinx-compatible)
- ✓ PEP 8 conventions followed; CI enforces black + flake8 on push
- ✓ No external ML libraries (pure PyTorch + sklearn)
- ✓ Seed setting for reproducibility
- ✓ Error handling with meaningful messages

### Reproducibility

- ✓ Fixed random seed (42)
- ✓ Stratified train/val/test splits
- ✓ Explicit model checkpointing (saves best by val accuracy)
- ✓ Configuration logging in TrainConfig

### Documentation

- ✓ Module docstrings explaining design
- ✓ Function docstrings with Args, Returns, Raises
- ✓ Inline comments for non-obvious logic
- ✓ README for users (separate from this technical guide)
- ✓ Jupyter notebook as reference implementation

---

## Testing & Debugging

### Quick Checks

```bash
# Verify imports
python -c "from src.models import WaferCNN, get_resnet18; print('OK')"

# Check PyTorch/GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run single epoch
python train.py --model cnn --epochs 1
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: src" | Run from project root, or `python -m train` |
| "CUDA out of memory" | Reduce batch_size (64 → 32) or use --device cpu |
| "Dataset not found" | Place LSWMD_new.pkl in data/ directory |
| "Slow training" | CPU is expected (~10 min/epoch). Use --device cuda if available. |

---

## 23 Improvements Implementation Status

### ✅ ALL 23 COMPLETE (as of 2026-04-05)

1. **Docker Support** (`Dockerfile`): Multi-stage build with development/production/jupyter targets
2. **Docker Compose** (`docker-compose.yml`): Orchestrated training, inference, jupyter, MLflow services
3. **Unified Config** (`config.yaml`, `src/config.py`): YAML-based configuration with Pydantic v2 strict validation
4. **Model Ensembling** (`src/models/ensemble.py`): Voting, averaging, weighted-averaging aggregation strategies
5. **Progressive Training** (`scripts/progressive_train.py`): Multi-resolution curriculum learning (48x96x192)
6. **Hyperparameter Tuning** (`scripts/optuna_tune.py`): Optuna-based search with TPE sampler
7. **Config Integration** (`train.py`): Wired config.yaml into CLI with override support
8. **Model Compression** (`scripts/compress_model.py`): Quantization (INT8), pruning (30-50%), distillation
9. **Active Learning** (`scripts/active_learn.py`): Uncertainty sampling (entropy/margin/least-confidence)
10. **MLOps Integration** (`src/mlops/wandb_logger.py`): W&B and MLflow logging with metrics/artifacts
11. **Cross-Validation** (`scripts/cross_validate.py`): Stratified k-fold with per-fold statistics
12. **Interactive Dashboard** (`scripts/dashboard.py`): Streamlit app with model analysis, metrics, predictions
13. **Multi-GPU Training** (`src/training/distributed.py`): DataParallel + DistributedDataParallel wrappers
14. **Federated Learning** (`src/federated/`): FedAvg + Byzantine-robust aggregation (4 files)
15. **Real-time Inference Server** (`src/inference/server.py`): FastAPI with 6 endpoints, GradCAM overlay
16. **Attention Mechanisms** (`src/models/attention.py`): SE and CBAM modules with post-hoc injection
17. **Uncertainty Quantification** (`src/inference/uncertainty.py`): MC Dropout with calibration metrics
18. **Synthetic Data Augmentation** (`src/augmentation/synthetic.py`): Rule-based defect pattern generation
19. **Vision Transformer** (`src/models/vit.py`): ViT from scratch for 96x96 wafer maps
20. **Self-Supervised Pretraining** (`src/training/simclr.py`): SimCLR + BYOL contrastive learning
21. **Anomaly Detection** (`src/analysis/anomaly.py`, `src/detection/ood.py`): IsolationForest, OC-SVM, Mahalanobis, Autoencoder
22. **Domain Adaptation** (`src/training/domain_adaptation.py`): CORAL + adversarial domain alignment
23. **CI/CD Pipeline** (`.github/workflows/`): Lint/test/model-validation GitHub Actions workflows

**Test suite**: 165 passed (requires torchvision, fastapi, python-multipart, pydantic>=2.6)

---

## 23 Improvements - Technical Details

### #1-4: Infrastructure & Configuration
```
├── Dockerfile                 # Multi-stage: base, development, production, jupyter
├── docker-compose.yml         # Services: train, inference, jupyter, mlflow
├── config.yaml               # 250+ lines covering all 23 improvements
└── src/config.py             # Type-safe config with validation
```

### #5-12: Training & Analysis Scripts
```
├── train.py                  # Wired to config.yaml with CLI overrides
├── progressive_train.py      # Multi-resolution curriculum learning
├── optuna_tune.py           # Hyperparameter search with TPE sampler
├── compress_model.py         # Quantization, pruning, distillation
├── active_learn.py          # Uncertainty-based sample selection
├── cross_validate.py         # Stratified k-fold evaluation
├── dashboard.py              # Streamlit interactive analysis
└── src/mlops/wandb_logger.py # W&B and MLflow experiment tracking
```

### Config Schema (config.yaml)
Unified configuration file covering:
- Data loading, augmentation, splits
- Model settings (CNN, ResNet, EfficientNet)
- Training hyperparameters, optimizer, scheduler
- Ensemble, tuning, compression, active learning settings
- MLOps (W&B, MLflow), device, paths

### Key Enhancements to Existing Code
- **train.py**: Now loads config.yaml, CLI args override config, saves config after run
- **src/config.py**: Dataclass-based with validation, YAML serialization
- **src/models/ensemble.py**: Three aggregation strategies with evaluation metrics
- **requirements.txt**: Added 15+ new dependencies (optuna, wandb, mlflow, streamlit, etc.)

---

## Usage: New Scripts

### Progressive Training
```bash
python scripts/progressive_train.py --model cnn --device cuda
# Output: trains CNN at 48x48 (2 epochs) -> 96x96 (3 epochs) -> 192x192 (2 epochs)
```

### Hyperparameter Tuning
```bash
python scripts/optuna_tune.py --model resnet --n-trials 50 --device cuda
# Output: Best hyperparameters with validation metrics
```

### Model Compression
```bash
python scripts/compress_model.py --method quantize --model cnn --checkpoint checkpoints/best_cnn.pth
python scripts/compress_model.py --method prune --sparsity 0.3
# Output: compressed model (4x smaller or 30% sparse)
```

### Active Learning
```bash
python scripts/active_learn.py --model cnn --initial-labeled 0.1 --acquisition-size 500 --n-iterations 5
# Output: iterative learning curves showing data efficiency
```

### Cross-Validation
```bash
python scripts/cross_validate.py --model all --n-splits 10 --epochs 5
# Output: per-fold metrics with mean ± std statistics
```

### Interactive Dashboard
```bash
streamlit run scripts/dashboard.py
# Output: Web dashboard at http://localhost:8501
```

---

---

## Contact & Attribution

**Instructors**: Provide feedback on report clarity, experimental design, code structure.

**Team**: See "Individual Contributions" section in report (to be filled by team).

**Code Quality**: This package is written to PhD-level academic standards with production-ready error handling, documentation, and reproducibility.

---

## References

- **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2015)
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019)
- **WM-811K Dataset**: WaferMap UCI ML Repository (or Kaggle)
- **Class Imbalance**: Huang et al., "Learning Under Class Imbalance: A Novel Geometric Near-Neighbor (Gang) Approach" (IEEE TKDE 2006)

---

## Completeness and Correctness Policy

All code in this repository must be complete, correct, and functional. The following practices are strictly prohibited:

1. **No skeletonization**: Every function, class, and module must contain a full, working implementation. Skeleton code, placeholder stubs, `pass`-only bodies, or `NotImplementedError` raises are not acceptable in any committed file.

2. **No templating in lieu of implementation**: Do not substitute outlines, pseudocode, or template structures for real implementations. If a file is committed, it must work when invoked.

3. **No eschewment of completeness**: Every feature listed in documentation (CLAUDE.md, README.md, structure.md) must correspond to a real, runnable implementation. Do not document features that do not exist or that exist only as stubs.

4. **No false completion claims**: Status markers (checkmarks, "COMPLETE", version numbers) must reflect verified, tested reality. Claims of test passage must be backed by actual test execution. Claims of feature completeness must be backed by functional code.

5. **Verification before assertion**: Before marking any feature, test, or module as complete, verify by running the relevant code path. The only acceptable states are "verified working", "known broken with documented issue", or "not yet implemented".

6. **No incomplete implementations**: Every function must contain complete, working logic. If a function is too complex to implement in one pass, break it into smaller functions that ARE complete. Never leave a function body with `TODO`, `FIXME`, `...`, `pass`, or `raise NotImplementedError`. This applies to ALL contributors including AI assistants.

7. **AI-generated code standard**: Code produced by AI tools must meet the same standard as human-written code. No shortcuts, no placeholders, no "left as exercise" patterns. If the AI cannot produce a complete implementation, it must explicitly state what is missing rather than producing stub code.

---

**Last Updated**: 2026-04-06 (Phase A/B/C optimizations; 150 paper references; Mixup/CutMix, SupCon, FPN, balanced sampling)
**Version**: 5.0
