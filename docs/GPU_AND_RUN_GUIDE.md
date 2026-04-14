# Full Run and GPU Guide

Practical guidance for running the training pipeline at different scales and on different hardware.

## 1. Hardware recommendations

| Hardware              | Recommended use          | Wall-clock (5 ep, all 3 models) | Batch size | Notes |
|-----------------------|--------------------------|---------------------------------|------------|-------|
| CPU (laptop)          | Smoke tests, unit tests  | 2–4 hours                       | 32–64      | Fine for `pytest`, impractical for full training |
| Colab free T4 (16 GB) | Full training            | 15–25 min                       | 128        | **Default recommendation for classmates** |
| Colab L4 (24 GB)      | Full training + larger bs| 10–15 min                       | 256        | Colab Pro |
| Colab A100 (40 GB)    | Fast iteration           | 5–10 min                        | 256–512    | Colab Pro+ |
| Local RTX 3090/4090   | Full training            | 8–12 min                        | 256        | CUDA 12.x + PyTorch 2.x GPU build |
| 2× GPU (DDP)          | Scaled training          | ~half the single-GPU time       | 128/gpu    | Use `scripts/distributed_train.py` |

## 2. Expected per-epoch wall-clock (Custom CNN baseline)

| Device       | Time/epoch |
|--------------|------------|
| CPU (4 cores)| 5–8 min    |
| CPU (16 cores)| 2–4 min   |
| T4           | 20–30 s    |
| L4           | 15–20 s    |
| A100         | 6–10 s     |
| H100         | 3–5 s      |

ResNet-18 and EfficientNet-B0 run roughly 1.2–1.5× the CNN time.

## 3. Memory footprint

The full WM-811K pickle is ~2.2 GB on disk. The preprocessed tensors (96×96 grayscale, normalized) use:

- ~4 GB CPU RAM peak during preprocessing (pandas + numpy intermediates)
- ~3 GB VRAM for training at batch size 128 (CNN); ~5 GB for ResNet-18
- Reduce to batch size 64 or 32 if `CUDA out of memory` occurs

Set this env var in Colab to reduce fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 4. Speed-up tricks already implemented

- **Mixed precision (AMP)**: enabled by default when `--device cuda`
- **Pin memory + non-blocking transfers**: in DataLoader
- **Cosine LR schedule**: faster convergence than step decay
- **ImageNet normalization for pretrained models only**: reduces wasted compute for the custom CNN

## 5. Recommended full-run commands

### Free Colab T4 (dominant path for classmates)

```bash
python train.py --model all --epochs 20 --batch-size 128 --device cuda
```

Expected wall-clock: ~20 min.
Expected macro-F1: 0.50–0.58 (ResNet-18 usually best).

### Fast-iteration local GPU

```bash
python train.py --model cnn --epochs 30 --batch-size 256 --lr 2e-3 --device cuda
```

### Bare-minimum sanity check (CPU, no real training)

```bash
python train.py --model cnn --epochs 1 --batch-size 16 --device cpu
```

This will complete in ~2–3 minutes and is useful to confirm the data path, model, and trainer all work before committing a longer run.

## 6. Saving results off-runtime

On Colab, copy outputs to Drive so they survive a disconnect:

```python
import shutil, time, os
out = f'/content/drive/MyDrive/wafer_runs/{time.strftime("%Y%m%d_%H%M%S")}'
os.makedirs(out, exist_ok=True)
for d in ['checkpoints', 'results']:
    if os.path.isdir(d):
        shutil.copytree(d, os.path.join(out, d), dirs_exist_ok=True)
```

## 7. Multi-GPU (DDP)

```bash
# 2-GPU example
torchrun --nproc_per_node=2 scripts/distributed_train.py \
  --model all --epochs 20 --batch-size 128 --device cuda
```

DDP scales approximately linearly for batch size ≤ 256/gpu.

## 8. Hyperparameter sweeps (Optuna)

```bash
pip install -e '.[tuning]'
python scripts/optuna_tune.py --model cnn --n-trials 30 --timeout 3600
```

Writes results to `optuna_studies/` and logs per-trial metrics.

## 9. Inference after training

```bash
# REST API with Grad-CAM overlays
python -m src.inference.server --checkpoint checkpoints/resnet_best.pth --model resnet
# POST to http://localhost:8000/predict
```

## 10. Troubleshooting slow runs

| Symptom                        | Likely cause                        | Fix                                                                 |
|--------------------------------|-------------------------------------|---------------------------------------------------------------------|
| GPU at 0% utilization          | DataLoader is CPU-bound             | Increase `--num-workers` in config, or pre-cache tensors            |
| GPU at 100%, still slow        | Batch size too small                | Increase `--batch-size` (watch VRAM)                                |
| `CUDA out of memory`           | Batch too big                       | Halve `--batch-size`; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Epochs take longer each time   | Memory leak in metrics accumulation | Run `torch.cuda.empty_cache()` between epochs or restart kernel     |
| Colab times out at 90 min      | Free-tier session limit             | Checkpoint every epoch, resume with `--pretrained-checkpoint`       |
