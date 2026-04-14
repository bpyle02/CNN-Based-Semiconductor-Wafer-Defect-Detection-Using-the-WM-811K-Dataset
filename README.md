# CNN-Based Semiconductor Wafer Defect Detection — WM-811K

AI 570 - Team 4: Anindita Paul, Brandon Pyle, Anand Rajan, Brett Rettura

A PyTorch pipeline for multi-class wafer defect classification on the WM-811K dataset (~120K labeled wafer maps, 9 classes). Compares a custom CNN, ResNet-18, and EfficientNet-B0 with transfer learning, plus optional extensions (Grad-CAM, MC dropout uncertainty, OOD detection, federated learning, ensembling, SimCLR).

<p>
  <a href="https://colab.research.google.com/github/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/blob/main/docs/colab_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

---

## TL;DR — replicate in 5 commands

```bash
git clone https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset.git
cd CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset
conda env create -f environment.yml       # creates the "py313" env
conda run -n py313 pip install -e ".[dev]"
# Place LSWMD_new.pkl at data/LSWMD_new.pkl (see Dataset section below)
conda run -n py313 pytest -q              # ~188 tests
conda run -n py313 python train.py --model cnn --epochs 5 --device cpu    # ~30–45 min on CPU
```

GPU run (recommended):

```bash
conda run -n py313 python train.py --model all --epochs 20 --device cuda --batch-size 128
```

Or open the Colab badge above and run the notebook end-to-end on a free T4.

---

## 1. Prerequisites

- **Python 3.10–3.13** (3.13 tested; project caps at `<3.14` in `pyproject.toml`)
- **Conda** (Miniconda or Anaconda) — used for the canonical `py313` environment
- **Git**
- **Optional for GPU**: CUDA 12.x + matching PyTorch build

Verify:

```bash
python --version
conda --version
git --version
```

## 2. Environment setup

### Option A — Conda (recommended for Windows/macOS/Linux)

```bash
conda create -n py313 python=3.13 -y
conda activate py313
pip install -e ".[dev]"
```

Or with the checked-in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate py313
```

### Option B — venv / pip only

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -e ".[dev]"
```

### Verify the environment

```bash
python scripts/doctor.py
```

This prints a health check: Python version, PyTorch version, CUDA availability, dataset presence, and dependency status.

## 3. Dataset

The WM-811K dataset is **not** versioned in this repo (2.2 GB).

1. Obtain `LSWMD_new.pkl` (the preprocessed WM-811K pickle used by the team).
2. Place it here:

   ```
   data/LSWMD_new.pkl
   ```

3. Confirm:

   ```bash
   python -c "import pandas as pd; df = pd.read_pickle('data/LSWMD_new.pkl'); print(df.shape, df.columns.tolist())"
   ```

Any other storage path works too — pass `--data-path /path/to/LSWMD_new.pkl` to `train.py`.

### If you don't have the pickle

The raw WM-811K is on Kaggle: <https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map>. The `scripts/` directory has helpers to reprocess the original CSV into the same schema as `LSWMD_new.pkl`.

## 4. Quick validation

```bash
pytest -q                       # runs the test suite in ~1–2 min on CPU
python train.py --help          # CLI self-describes
```

## 5. Training

### CPU smoke run (~30–45 min per model, 5 epochs)

```bash
python train.py --model cnn --epochs 5 --device cpu --batch-size 64
```

### Full GPU run (~15–25 min total on a T4, ~8 min on A100)

```bash
python train.py --model all --epochs 20 --device cuda --batch-size 128
```

### With config overlays

```bash
python train.py --config configs/base.yaml --config configs/train.yaml
```

Overlays available: `configs/base.yaml`, `configs/train.yaml`, `configs/inference.yaml`, `configs/federated.yaml`. Later overlays override earlier ones.

### Outputs

Each training run saves:

- `checkpoints/<model>_best.pth` — best weights by validation macro-F1
- `checkpoints/<model>_best.pth.sha256` — integrity hash
- `results/<model>_metrics.json` — accuracy, macro F1, weighted F1, per-class F1, confusion matrix
- `results/<model>_training_curves.png` — loss / accuracy plots
- `results/<model>_confusion_matrix.png`

## 6. Running in Google Colab (free T4 GPU)

See [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md) or open the Colab badge at the top of this README.

Minimum steps inside a Colab notebook:

```python
!git clone https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset.git
%cd CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset
!pip install -q -e ".[dev]"

# Upload dataset to /content/data/LSWMD_new.pkl (or mount Google Drive)
from google.colab import drive; drive.mount('/content/drive')
!cp '/content/drive/MyDrive/datasets/LSWMD_new.pkl' data/

!python train.py --model all --epochs 20 --device cuda --batch-size 128
```

Use **Runtime → Change runtime type → T4 GPU** before running.

## 7. Expected results (5 epochs, same split, seed=42)

| Model           | Accuracy  | Macro F1  | Weighted F1 | CPU time (per epoch) | T4 time (per epoch) |
|-----------------|-----------|-----------|-------------|----------------------|---------------------|
| Custom CNN      | 0.78–0.80 | 0.42–0.48 | 0.70–0.74   | ~6 min               | ~25 s               |
| ResNet-18       | 0.84–0.86 | 0.50–0.56 | 0.78–0.82   | ~9 min               | ~35 s               |
| EfficientNet-B0 | 0.82–0.84 | 0.48–0.54 | 0.76–0.80   | ~8 min               | ~30 s               |

Accuracy is dominated by the 85% "none" class. **Macro F1 is the primary imbalance-aware metric.** Numbers are indicative — they depend on the specific split and hardware.

## 8. CLI reference

```
python train.py [OPTIONS]

--model {cnn,cnn_fpn,resnet,efficientnet,vit,swin,ride,all}
--epochs N
--batch-size N
--lr FLOAT
--device {cuda,cpu}
--seed N                         (default 42)
--data-path PATH                 (default data/LSWMD_new.pkl)
--config FILE                    (repeatable; overlays merged in order)
--synthetic                      (augment rare classes with synthetic samples)
--mixup                          (enable Mixup/CutMix)
--balanced-sampling              (class-balanced batches)
--uncertainty                    (MC Dropout after training)
--pretrained-checkpoint PATH     (load SimCLR/SupCon backbone)
--wandb / --mlflow               (experiment tracking)
--distributed                    (DataParallel multi-GPU)
```

## 9. Inference server

```bash
pip install -e ".[server]"
python -m src.inference.server --checkpoint checkpoints/cnn_best.pth --model cnn
# POST images to http://localhost:8000/predict, Grad-CAM overlays returned in the response
```

See `docs/guides/INFERENCE_SERVER_README.md` for the API shape.

## 10. Repository layout

```
.
├── train.py                    # CLI entry point
├── config.yaml                 # Canonical defaults
├── configs/                    # base.yaml, train.yaml, inference.yaml, federated.yaml
├── src/
│   ├── data/                   # dataset loading, preprocessing, transforms
│   ├── models/                 # CNN, ResNet, EfficientNet, ViT, Swin, FPN, RIDE, attention
│   ├── training/               # trainer, config, DDP, SimCLR, SupCon, losses, EMA
│   ├── analysis/               # metrics, visualization, anomaly/OOD, artifacts
│   ├── augmentation/           # synthetic wafer-map generation
│   ├── detection/              # defect pattern detection utilities
│   ├── federated/              # federated averaging client/server
│   ├── inference/              # FastAPI server, Grad-CAM, TTA, uncertainty
│   ├── mlops/                  # W&B, MLflow integration
│   ├── config.py               # Pydantic config schema
│   ├── exceptions.py           # Custom exception hierarchy
│   └── model_registry.py       # Checkpoint registry + hashing
├── tests/                      # unit + integration tests
├── scripts/                    # optuna_tune, compress_model, active_learn, ...
├── docs/
│   ├── COLAB_SETUP.md
│   ├── wafer_defect_detection_run.ipynb
│   ├── wafer_defect_detection_report.tex / .pdf
│   ├── presentation.tex
│   └── guides/                 # Feature-specific deep dives
├── references/                 # Reference papers (PDFs + text extracts)
├── Dockerfile                  # Multi-stage: base, dev, prod, jupyter
├── docker-compose.yml          # train / inference / jupyter / mlflow
└── Makefile                    # make install / train / test / smoke
```

## 11. Docker

```bash
docker-compose up train           # GPU training (needs nvidia-docker)
docker-compose up inference       # FastAPI inference server on :8000
docker-compose up jupyter         # Jupyter on :8888
```

## 12. Known limitations

- No trained checkpoint is versioned in the repo — run training first.
- The inference server serves one active model at a time.
- `distributed` mode currently uses `DataParallel`; full DDP needs `scripts/distributed_train.py`.
- Python 3.14 is not yet supported (tracking PyTorch 3.14 wheel availability).

## 13. Troubleshooting

**Imports fail after install.** Run `pip install -e ".[dev]"` from the repo root (not inside `src/`). Confirm with `python -c "import src.data"`.

**CUDA out of memory.** Lower `--batch-size` (try 64 or 32). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Dataset load error.** Confirm the file is at `data/LSWMD_new.pkl` and is a pandas pickle with `waferMap`, `failureType`, `trianTestLabel` columns.

**Tests fail on Windows with path errors.** Use a clean conda env (`py313`) rather than the base Anaconda install — the shared base env often has mixed 3.10/3.11 installs.

## 14. License & citation

Coursework for AI 570. Use of the WM-811K dataset is governed by Wu et al. (IEEE TSM 2015). If you reuse any code, please cite the upstream repository.

## 15. Contact

Open an issue on the upstream repo: <https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/issues>
