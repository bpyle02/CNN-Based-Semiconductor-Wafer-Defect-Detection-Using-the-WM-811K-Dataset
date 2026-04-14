# Project Structure: CNN-Based Semiconductor Wafer Defect Detection

## Part I: Original Design (Initial Project Proposal)

The project was initially scoped into four responsibility areas with the following
deliverables and target folder layout.

### 1. Data Collection & Preprocessing

**Responsibilities:**
- Gather wafer map dataset (WM-811K, ~120K samples, 9 classes)
- Standardize formats and normalize data
- Handle missing labels, class imbalance, and noise
- Implement train/test split and augmentation (rotations, flips, etc.)

**Deliverables:**
- `data/` folder with dataset
- `dataset.py` -- load and parse WM-811K pickle
- `preprocessing.py` -- resize, normalize, augment, create PyTorch datasets
- Documentation of data pipeline

### 2. Model Design and Training

**Responsibilities:**
- Choose baseline models (CNN, ResNet, Vision Transformer, etc.)
- Build the model architecture
- Implement training loop, validation loop, and loss functions
- Hyperparameter tuning

**Deliverables:**
- `models/` folder with architecture implementations
- `train.py` -- CLI entry point
- `evaluate.py` -- metrics and classification reports
- Model architecture files
- Documentation of model design

### 3. Analysis, Visualization & Reporting

**Responsibilities:**
- Track experiments (learning curves, metrics)
- Visualize defect distributions, confusion matrices, and Grad-CAM / saliency maps
- Compare models and summarize results
- Write technical analysis for the final report

**Deliverables:**
- `analysis/` folder
- Plots, charts, and visualizations
- Final model comparison table

### 4. Pipeline and UI/Inference

**Responsibilities:**
- Build a clean project structure
- Write inference script to load the model and predict defect type for a new wafer map
- Ensure reproducibility by writing README and requirements.txt
- Environment setup instructions

**Deliverables:**
- `inference/` folder
- Inference pipeline
- Final integrated project folder
- Optional: web UI

### Original Target Folder Structure

```
project/
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── analysis/
│   └── inference/
├── notebooks/
├── README.md
└── requirements.txt
```

---

## Part II: Evolved Architecture (Current Implementation)

The codebase has grown beyond the original 4-area design into a comprehensive
deep learning platform with 23 advanced improvements spanning federated learning,
self-supervised pretraining, uncertainty quantification, model compression, and
production deployment infrastructure.

### Responsibility Area Mapping: Original to Current

| Original Area | Original Deliverables | Current Implementation |
|---------------|----------------------|----------------------|
| **1. Data** | `dataset.py`, `preprocessing.py` | `src/data/dataset.py`, `src/data/preprocessing.py` + synthetic augmentation pipeline (`src/augmentation/`) |
| **2. Models** | `model.py`, `train.py`, `evaluate.py` | 6 architecture files (`src/models/`), 7 training files (`src/training/`), CLI entry (`train.py`) |
| **3. Analysis** | `analysis/`, plots, comparison table | `src/analysis/` (3 modules) + anomaly detection + OOD detection (`src/detection/`) |
| **4. Pipeline** | `inference.py`, README, requirements | `src/inference/` (4 modules), FastAPI server, Docker, CI/CD, Makefile |

### Current Verified File Tree

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml                              # Lint, format, type check, unit tests
│       └── model_validation.yml                # Model creation, data pipeline, benchmark tests
│
├── data/
│   ├── README.md                               # Dataset download instructions
│   └── LSWMD_new.pkl                           # WM-811K dataset (~120K wafer maps)
│
├── docs/
│   ├── guides/                                 # 11 comprehensive feature guides
│   │   ├── COMPREHENSIVE_FEATURE_GUIDE.md
│   │   ├── FEDERATED_LEARNING.md
│   │   ├── FEDERATED_LEARNING_GUIDE.md
│   │   ├── INFERENCE_SERVER_ARCHITECTURE.md
│   │   ├── INFERENCE_SERVER_QUICKSTART.md
│   │   ├── INFERENCE_SERVER_README.md
│   │   ├── OOD_DETECTION_GUIDE.md
│   │   ├── SYNTHETIC_AUGMENTATION_GUIDE.md
│   │   ├── TEAM_README.md
│   │   ├── UNCERTAINTY_QUANTIFICATION.md
│   │   └── UNCERTAINTY_QUICKSTART.md
│   ├── wafer_defect_detection_report.tex       # IEEE-format technical report (LaTeX source)
│   ├── wafer_defect_detection_report.pdf       # Compiled report
│   ├── presentation.tex                        # Beamer slides (LaTeX source)
│   ├── presentation.pdf                        # Compiled presentation
│   ├── wafer_defect_detection_run.ipynb        # Reference Jupyter notebook
│   ├── old_wafer_defect_detection.ipynb        # Legacy notebook (baseline)
│   ├── DEFENSE_PACKET.md                       # Defense committee materials
│   ├── FINAL_STATUS_REPORT.md                  # Project completion status
│   ├── COMPREHENSIVE_REVIEW.md                 # Detailed feature review
│   ├── IMPROVEMENTS.md                         # Improvement descriptions
│   └── IMPROVEMENTS_STATUS.md                  # Implementation status tracking
│
├── scripts/
│   ├── train.py                                # Refactored training with BaseTrainer
│   ├── train_legacy.py                         # Legacy training script
│   ├── train_with_synthetic_augmentation.py    # Training with synthetic data
│   ├── progressive_train.py                    # Multi-resolution curriculum (48->96->192)
│   ├── distributed_train.py                    # Multi-GPU training (DDP)
│   ├── optuna_tune.py                          # Hyperparameter search (Optuna/TPE)
│   ├── compress_model.py                       # Quantization, pruning, distillation
│   ├── active_learn.py                         # Uncertainty-based sample selection
│   ├── synthetic_augment.py                    # Synthetic data augmentation
│   ├── cross_validate.py                       # Stratified k-fold evaluation
│   ├── inference_server.py                     # FastAPI server CLI wrapper
│   ├── uncertainty_example.py                  # MC Dropout demonstration
│   ├── dashboard.py                            # Streamlit interactive dashboard
│   ├── colab_runner.py                         # Google Colab setup and execution
│   ├── extract_and_update_report.py            # Extract metrics, update LaTeX report
│   ├── finalize_submission.py                  # Build final submission bundle
│   ├── defense_smoke_demo.py                   # Defense committee smoke test
│   ├── test_advanced_features.py               # Advanced feature integration tests
│   ├── run_defense_demo.ps1                    # PowerShell defense demo wrapper
│   └── wait_and_finalize.sh                    # Shell orchestration script
│
├── src/
│   ├── __init__.py                             # Package root (version 0.1.0)
│   ├── config.py                               # YAML configuration loader (DataConfig, TrainingConfig, ModelConfig)
│   ├── exceptions.py                           # Custom exception hierarchy (5 exception types)
│   ├── model_registry.py                       # Model versioning and metadata tracking
│   │
│   ├── data/
│   │   ├── __init__.py                         # Defensive imports (graceful torchvision fallback)
│   │   ├── dataset.py                          # load_dataset(), extract_failure_label() for WM-811K
│   │   └── preprocessing.py                    # WaferMapDataset, transforms, ImageNet normalization
│   │
│   ├── models/
│   │   ├── __init__.py                         # Model exports with optional dependency guards
│   │   ├── cnn.py                              # WaferCNN: 4-block custom CNN (96x96 input, 9 classes)
│   │   ├── pretrained.py                       # ResNet-18 & EfficientNet-B0 with layer-boundary freezing
│   │   ├── vit.py                              # Vision Transformer (patch embedding, encoder, ViT-Tiny/Small)
│   │   ├── ensemble.py                         # EnsembleModel (voting, averaging, weighted averaging)
│   │   ├── attention.py                        # SE blocks, CBAM blocks, post-hoc injection utilities
│   │   └── attention_examples.py               # Attention mechanism usage demonstrations
│   │
│   ├── training/
│   │   ├── __init__.py                         # Exports TrainConfig, train_model
│   │   ├── config.py                           # TrainConfig dataclass (epochs, LR, scheduler, seed)
│   │   ├── trainer.py                          # train_model(): forward/backward, validation, checkpointing
│   │   ├── base_trainer.py                     # BaseTrainer: YAML config, seed management, device setup
│   │   ├── distributed.py                      # Multi-GPU: setup/cleanup DDP, rank/world_size utilities
│   │   ├── simclr.py                           # SimCLR self-supervised pretraining (NT-Xent loss)
│   │   └── domain_adaptation.py                # CORAL alignment + adversarial domain adaptation
│   │
│   ├── analysis/
│   │   ├── __init__.py                         # Exports evaluate_model, plotting functions
│   │   ├── evaluate.py                         # Metrics: accuracy, macro/weighted F1, classification report
│   │   ├── visualize.py                        # Training curves, confusion matrices, model comparison
│   │   └── anomaly.py                          # Autoencoder, IsolationForest, OC-SVM, Mahalanobis
│   │
│   ├── inference/
│   │   ├── __init__.py                         # Defensive imports (FastAPI optional)
│   │   ├── server.py                           # FastAPI inference server (6 endpoints, CORS, health)
│   │   ├── gradcam.py                          # GradCAM: forward/backward hooks, activation maps
│   │   ├── uncertainty.py                      # MC Dropout: epistemic/aleatoric uncertainty, calibration
│   │   └── visualize.py                        # GradCAM grid overlay visualization
│   │
│   ├── augmentation/
│   │   ├── __init__.py                         # Exports SyntheticDataGenerator, FIDScorer
│   │   ├── synthetic.py                        # DefectSimulator: rule-based pattern generation (5 types)
│   │   ├── train_generator.py                  # GAN training with FID evaluation
│   │   └── evaluation.py                       # FIDScorer (Frechet Inception Distance), InceptionScorer
│   │
│   ├── detection/
│   │   ├── __init__.py                         # Exports OOD detectors
│   │   └── ood.py                              # MahalanobisDetector, OutOfDistributionDetector
│   │
│   ├── federated/
│   │   ├── __init__.py                         # Exports Byzantine-robust aggregators
│   │   ├── fed_avg.py                          # FedAvg server/client (learning rate scheduling)
│   │   ├── client.py                           # Federated client with checkpoint handling
│   │   ├── server.py                           # Federated server with Dirichlet non-IID partitioning
│   │   └── aggregation.py                      # Krum, MultiKrum, ByzantineRobustAggregator
│   │
│   └── mlops/
│       ├── __init__.py                         # Exports loggers
│       └── wandb_logger.py                     # WandBLogger + MLFlowLogger (graceful no-op fallback)
│
├── tests/
│   ├── conftest.py                             # Pytest fixtures (workspace, device, sample_data)
│   ├── test_improvements.py                    # Model registry, Byzantine aggregation, OOD tests
│   ├── test_uncertainty.py                     # Uncertainty quantification tests
│   ├── unit/
│   │   ├── test_federated.py                   # Federated learning unit tests
│   │   └── test_inference_server.py            # Inference server API tests
│   └── integration/
│       └── test_full_pipeline.py               # End-to-end training pipeline test
│
├── train.py                                    # Primary CLI entry point (--model, --epochs, --device)
├── setup.py                                    # Platform detection (Colab/Kaggle/Local) + dependency install
├── config.yaml                                 # Unified YAML configuration (all 23 features)
├── requirements.txt                            # 78 dependencies (core + optional + dev)
├── Dockerfile                                  # Multi-stage: base, development, production, jupyter
├── docker-compose.yml                          # 4 services: train, inference, jupyter, mlflow
├── Makefile                                    # Build targets: install, train, test, dashboard, defense
├── pytest.ini                                  # Test configuration and path exclusions
├── README.md                                   # User-facing project overview
├── DEFENSE_SUBMISSION.md                       # Defense committee submission guide
├── structure.md                                # This file
└── .gitignore                                  # Comprehensive exclusion rules
```

### Module Inventory Summary

| Module | Files | LOC (approx.) | Purpose |
|--------|-------|---------------|---------|
| `src/data` | 3 | ~415 | Dataset loading, preprocessing, transforms |
| `src/models` | 7 | ~1,800 | CNN, ResNet-18, EfficientNet-B0, ViT, attention, ensemble |
| `src/training` | 7 | ~1,300 | Training loops, config, distributed, SimCLR, domain adaptation |
| `src/analysis` | 4 | ~800 | Evaluation metrics, visualization, anomaly detection |
| `src/inference` | 5 | ~1,600 | FastAPI server, GradCAM, uncertainty quantification |
| `src/augmentation` | 4 | ~350 | Synthetic defect generation, GAN training, FID/IS metrics |
| `src/detection` | 2 | ~150 | Out-of-distribution detection (Mahalanobis, threshold) |
| `src/federated` | 5 | ~1,650 | FedAvg, Byzantine-robust aggregation, client/server |
| `src/mlops` | 2 | ~150 | Weights & Biases, MLflow experiment tracking |
| `src/` (root) | 4 | ~430 | Config, exceptions, model registry, package init |
| `scripts/` | 20 | ~3,500 | Training variants, tuning, compression, dashboard, CI |
| `tests/` | 6 | ~600 | Unit, integration, and feature tests |
| **Total** | **69** | **~12,750** | |

### 23 Improvements: Feature-to-File Mapping

| # | Feature | Primary Files |
|---|---------|--------------|
| 1 | Docker Support | `Dockerfile` |
| 2 | Docker Compose | `docker-compose.yml` |
| 3 | Unified Config | `config.yaml`, `src/config.py` |
| 4 | Model Ensembling | `src/models/ensemble.py` |
| 5 | Progressive Training | `scripts/progressive_train.py` |
| 6 | Hyperparameter Tuning | `scripts/optuna_tune.py` |
| 7 | Config Integration | `train.py` (wired to config.yaml) |
| 8 | Model Compression | `scripts/compress_model.py` |
| 9 | Active Learning | `scripts/active_learn.py` |
| 10 | MLOps Integration | `src/mlops/wandb_logger.py` |
| 11 | Cross-Validation | `scripts/cross_validate.py` |
| 12 | Interactive Dashboard | `scripts/dashboard.py` |
| 13 | Multi-GPU Training | `src/training/distributed.py` |
| 14 | Federated Learning | `src/federated/` (4 files) |
| 15 | Real-time Inference | `src/inference/server.py` |
| 16 | Attention Mechanisms | `src/models/attention.py` |
| 17 | Uncertainty Quantification | `src/inference/uncertainty.py` |
| 18 | Synthetic Data Augmentation | `src/augmentation/synthetic.py` |
| 19 | Vision Transformer | `src/models/vit.py` |
| 20 | Self-Supervised Pretraining | `src/training/simclr.py` |
| 21 | Anomaly Detection | `src/analysis/anomaly.py`, `src/detection/ood.py` |
| 22 | Domain Adaptation | `src/training/domain_adaptation.py` |
| 23 | CI/CD Pipeline | `.github/workflows/ci.yml`, `.github/workflows/model_validation.yml` |

### Architectural Principles

1. **Modular separation**: Each concern (data, models, training, analysis, inference, augmentation, detection, federated, mlops) has its own package with explicit exports
2. **Defensive imports**: Optional dependencies (torchvision, FastAPI, wandb, mlflow) are guarded with graceful fallbacks so core functionality works without them
3. **No circular dependencies**: All imports flow one-way (verified by static analysis)
4. **Configuration-driven**: `config.yaml` provides centralized settings; CLI arguments override for flexibility
5. **Custom exception hierarchy**: `src/exceptions.py` defines typed exceptions for each domain (DataLoadError, ModelError, TrainingError, InferenceError, FederatedError)
6. **Reproducibility**: Fixed seeds (42), stratified splits, deterministic preprocessing, model checkpointing
7. **Production readiness**: Docker multi-stage builds, FastAPI inference server, CI/CD workflows, health checks