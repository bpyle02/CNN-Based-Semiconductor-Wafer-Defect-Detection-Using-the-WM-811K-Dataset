# Reproducing the Reported Metrics

This document tells a grader, a teammate, or future-you how to rerun the
pipeline from a clean clone and land within epsilon of the committed
baseline metrics. Keep it current whenever the baseline changes.

## What "reproducible" means here

There are two baselines in this repo:

1. **The committed `results/metrics.json` snapshot** — a 1-epoch Custom CNN
   run on a local machine. Weak by design (it's a smoke-test artifact, not
   a headline result) but deterministic given the pinned deps and seed.
   This is what `scripts/check_metrics.py` enforces on every PR.

2. **The Colab headline result** — 10-epoch Custom CNN on Colab T4:
   accuracy 0.9611, macro F1 0.7988, weighted F1 0.9632, ECE 0.0068.
   This is the number quoted in `CHANGELOG.md` and the README badge area.

Following the steps below **exactly** should land you within ±0.5% of the
committed baseline. The Colab headline is more sensitive to hardware and
driver stacks; expect ±1.5% on macro F1 between runs on different T4s.

## Required state

| item | value |
|---|---|
| Git SHA | `5cd09f2` (or any `main` commit that also ships a matching `results/metrics.baseline.json`) |
| Dataset | `data/LSWMD_new.pkl` |
| Dataset SHA-256 | `d15e6b4b0c99649a93b39e9d4f7bafd2b1f67a27c38272c1f50b072898b0faeb` |
| Python | 3.10–3.13 (3.13 used for baseline; see `requirements-lock.txt`) |
| PyTorch | `2.11.0` (see `requirements-lock.txt`) |
| Seed | `42` (passed via `--seed 42`) |
| Config | `config.yaml` + `configs/colab_fast.yaml` overlay for the Colab run |

Verify the dataset hash before training — a corrupted pkl silently
produces wildly different metrics:

```bash
python -c "
import hashlib
h = hashlib.sha256()
with open('data/LSWMD_new.pkl', 'rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        h.update(chunk)
print(h.hexdigest())
"
# expect: d15e6b4b0c99649a93b39e9d4f7bafd2b1f67a27c38272c1f50b072898b0faeb
```

## Reproduce the committed baseline (fast: ~30 min CPU, ~5 min GPU)

```bash
git clone https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset.git
cd CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset
git checkout main

# 1. Pin-exact dep install (not the looser requirements.txt)
conda env create -f environment.yml
conda activate py313
pip install -r requirements-lock.txt
pip install -e ".[dev]"

# 2. Drop the dataset in (see README §2 for sources)
cp /wherever/you/have/LSWMD_new.pkl data/

# 3. Run the 1-epoch CNN baseline
python train.py --model cnn --epochs 1 --batch-size 64 \
    --device cpu --seed 42

# 4. Verify the metrics landed within tolerance of the committed baseline
python scripts/check_metrics.py \
    --baseline results/metrics.baseline.json \
    --current results/metrics.json
# exit 0 on success; exit 1 if accuracy drops >2% or macro_f1 drops >3%
```

If `check_metrics.py` exits 1: **don't** hand-edit the baseline. Investigate
what drifted first. Common causes: dataset pkl mismatch, different torch
version, CPU vs GPU math differences, forgotten `--seed 42`.

## Reproduce the Colab headline result (slow: ~90 min on T4)

Open the Colab badge in the README or:
https://colab.research.google.com/github/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/blob/main/docs/colab_quickstart.ipynb

Run cells top-to-bottom. Key configuration the notebook applies on your
behalf:
- `--seed 42` (every subprocess)
- `--epochs 10`, `--batch-size 64`
- Custom CNN only (the headline number) plus ResNet-18 and EfficientNet-B0
- `configs/colab_fast.yaml` overlay when `USE_DRIVE_CKPT = True` (tighter
  early stopping + Drive-backed checkpoints)

After Cell 7 finishes, the metrics in `results/metrics.json` should land
within ±1.5% of the committed Colab headline (0.9611 / 0.7988).

## Updating the baseline on purpose

When a code change **intentionally** moves metrics (new model, new loss,
better hyperparameters), rerun the baseline and update both files in the
same PR:

```bash
python train.py --model cnn --epochs 1 --batch-size 64 --device cpu --seed 42
cp results/metrics.json results/metrics.baseline.json
git add results/metrics.json results/metrics.baseline.json CHANGELOG.md
git commit -m "Refresh baseline: <short reason>"
```

The PR template asks you to explicitly acknowledge metric regressions so
this step doesn't slip through unnoticed.

## Hardware notes

| env | wall-clock for 1 epoch CNN | notes |
|---|---|---|
| CPU (8-core i7, 32 GB) | ~27 min | baseline snapshot was produced here |
| Colab T4 | ~1–2 min | after precompute cache is built |
| Local RTX 4060 | ~45 sec | with `torch.compile` off |

CPU and GPU math don't produce bit-identical results; CPU is authoritative
for the committed baseline to stay deterministic in CI, which has no GPU.
