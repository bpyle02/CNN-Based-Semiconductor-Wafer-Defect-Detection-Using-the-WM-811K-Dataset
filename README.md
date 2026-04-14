# CNN-Based Semiconductor Wafer Defect Detection — WM-811K

AI 570 - Team 4: Anindita Paul, Brandon Pyle, Anand Rajan, Brett Rettura

A PyTorch pipeline for multi-class wafer defect classification on the WM-811K dataset (~120K labeled wafer maps, 9 classes). Compares a custom CNN, ResNet-18, and EfficientNet-B0 with transfer learning, plus optional extensions (Grad-CAM, MC dropout uncertainty, OOD detection, federated learning, ensembling, SimCLR).

<p>
  <a href="https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/actions/workflows/ci.yml"><img src="https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/actions/workflows/model_validation.yml"><img src="https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/actions/workflows/model_validation.yml/badge.svg?branch=main" alt="Model Validation"></a>
  <a href="https://colab.research.google.com/github/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/blob/main/docs/colab_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg" alt="Python 3.10-3.13">
  <img src="https://img.shields.io/badge/license-All%20Rights%20Reserved-lightgrey.svg" alt="All Rights Reserved">
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

## 6. Running in Google Colab (free T4 GPU) — step-by-step

The fastest path for most classmates. Expected wall-clock: **~20 minutes for all three models at 20 epochs on a T4**.

### Step 1 — put the dataset in Google Drive (do this once, before opening Colab)

1. Download `LSWMD_new.pkl` (2.2 GB) from wherever the team shared it, or from Kaggle: <https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map>.
2. In your browser, open Google Drive and create a folder: `My Drive → datasets`.
3. Drag `LSWMD_new.pkl` into that folder. Confirm the final path is `MyDrive/datasets/LSWMD_new.pkl`.

(If you put it somewhere else, edit the `DRIVE_DATASET_PATH` variable in cell 3 of the notebook.)

### Step 2 — open the notebook in Colab

Click the badge at the top of this README, or open this URL directly:

```
https://colab.research.google.com/github/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/blob/main/docs/colab_quickstart.ipynb
```

### Step 3 — switch the runtime to GPU

Menu bar → **Runtime** → **Change runtime type** → Hardware accelerator: **T4 GPU** → **Save**.

Colab will attach a new runtime. Wait until you see "Connected to Python 3 Google Compute Engine backend (GPU)" in the top-right.

### Step 4 — run the cells top-to-bottom

1. **Cell 1** — clones the repo and installs the package. ~1 min. If Colab prompts `RESTART RUNTIME`, click it, then re-run Cell 1.
2. **Cell 2** — verifies GPU. Expect `CUDA available: True` and `Device: Tesla T4`.
3. **Cell 3** — loads the dataset. It auto-detects this order:
   - already at `data/LSWMD_new.pkl` → use it
   - mount Google Drive and copy from `MyDrive/datasets/LSWMD_new.pkl`
   - if neither works, opens the browser upload dialog

   Approve Drive access when prompted. Copying 2.2 GB from Drive → runtime usually takes ~30–60 s.
4. **Cell 4** — sanity check: prints `shape: (811457, 7)` and the class distribution.
5. **Cell 5** — runs the test suite (~1 min on T4). Expect `188 passed`.
6. **Cell 6** — trains all three models. ~18–25 min on T4 with `EPOCHS=20, BATCH_SIZE=128`. If you hit CUDA OOM, edit the cell and drop `BATCH_SIZE` to 64 or 32.
7. **Cell 7** — prints the metrics table and displays confusion matrices.
8. **Cell 8** — copies `checkpoints/` and `results/` back to your Drive at `MyDrive/wafer_runs/<timestamp>/`. **Run this before closing the tab** — Colab disposes of the runtime on disconnect.

### Verifying GPU is actually being used

While Cell 6 is running, open a second Colab cell and run:

```python
!nvidia-smi
```

You should see one of the `python` / `torch` processes using 4–9 GB of GPU memory and the GPU utilization fluctuating between 60–100%. If utilization is stuck at 0%, the training is CPU-bound — stop and make sure Cell 2 confirmed CUDA availability.

### What if something fails

| Symptom | Fix |
|---|---|
| `No GPU detected` in Cell 2 | Runtime type wasn't set to GPU. Menu: Runtime → Change runtime type → T4 GPU. |
| `NOT found at /content/drive/MyDrive/datasets/LSWMD_new.pkl` | Either fix the Drive path, or edit `DRIVE_DATASET_PATH` in Cell 3 to match where you actually put the file. |
| `CUDA out of memory` in Cell 6 | Edit Cell 6 and set `BATCH_SIZE = 64` (or 32). Re-run the cell. |
| Session disconnects mid-training | Colab free tier times out after ~90 min of inactivity. Run Cell 8 sooner, or upgrade to Colab Pro. |
| `Dataset is not present` assertion | Run Cell 3 again after fixing the Drive path, or use Cell 3b to download from Kaggle. |

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
│   ├── colab_quickstart.ipynb        # Colab T4 end-to-end notebook
│   ├── COLAB_SETUP.md                # Colab instructions
│   ├── GPU_AND_RUN_GUIDE.md          # Hardware + wall-clock reference
│   ├── wafer_defect_detection_report.tex / .pdf
│   ├── presentation.tex
│   └── guides/                       # Feature deep-dive guides
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
