# Colab Setup Guide

Run this in a Google Colab cell to train the models with GPU acceleration.

## Quick Start (Copy-Paste into Colab)

```python
# Cell 1: Clone repo and setup
!git clone https://github.com/YOUR_USERNAME/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset.git
%cd CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset

# Checkout feature branch (when ready)
# !git checkout feature/phd-complete-implementation

# Install dependencies
!python -m pip install -e ".[dev]"
```

```python
# Cell 2: Upload dataset
from google.colab import files
print("Upload LSWMD_new.pkl when prompted...")
uploaded = files.upload()
import shutil
shutil.move(list(uploaded.keys())[0], 'data/LSWMD_new.pkl')
print("Dataset moved to data/LSWMD_new.pkl")
```

Or use Google Drive (keep dataset in Drive):

```python
# Cell 2 (Alternative): Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Update data path to point to Drive
import os
os.makedirs('data', exist_ok=True)

# Create symlink (if dataset is in My Drive root)
# !ln -s /content/drive/MyDrive/LSWMD_new.pkl data/LSWMD_new.pkl
```

```python
# Cell 3: Verify GPU and imports
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
from src.models import WaferCNN, get_resnet18
print("Package imports OK")
```

```python
# Cell 4: Train models
!python train.py --model all --epochs 5 --device cuda --batch-size 64
```

---

## Full Walkthrough

### Step 1: Open Colab
Go to [Google Colab](https://colab.research.google.com) and create new notebook.

### Step 2: Enable GPU
- Click "Runtime" → "Change runtime type"
- Select GPU (T4 or L4)
- Click "Save"

### Step 3: Clone and Setup

**Cell 1:**
```python
!git clone https://github.com/YOUR_USERNAME/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset.git
%cd CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset
!git checkout feature/phd-complete-implementation
!python -m pip install -e ".[dev]"
```

### Step 4: Handle Dataset

**Option A: Upload from computer (Cell 2)**
```python
from google.colab import files
print("Select LSWMD_new.pkl file:")
uploaded = files.upload()
import shutil
filename = list(uploaded.keys())[0]
shutil.move(filename, 'data/LSWMD_new.pkl')
print(f"Dataset ready at data/LSWMD_new.pkl")
```

**Option B: Mount Google Drive (Cell 2)**
```python
from google.colab import drive
drive.mount('/content/drive')

# If dataset is at: My Drive/datasets/LSWMD_new.pkl
import os
os.symlink('/content/drive/MyDrive/datasets/LSWMD_new.pkl', 'data/LSWMD_new.pkl')
print("Dataset linked from Drive")
```

### Step 5: Verify Setup (Cell 3)
```python
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Available: {torch.cuda.is_available()}")

from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.training import train_model
from src.analysis import evaluate_model
print("All imports OK!")
```

### Step 6: Train (Cell 4)
```python
!python train.py --model all --epochs 5 --device cuda --batch-size 64
```

---

## Training Options

```bash
# All models (5 epochs)
python train.py --model all --epochs 5 --device cuda

# Single model
python train.py --model cnn --epochs 10 --device cuda
python train.py --model resnet --epochs 5 --device cuda

# Smaller batch size (if OOM)
python train.py --model all --epochs 5 --batch-size 32 --device cuda

# Custom learning rate
python train.py --model resnet --lr 5e-5 --epochs 5 --device cuda
```

---

## Save Results to Drive

After training, save checkpoints to Drive:

```python
!mkdir -p /content/drive/MyDrive/wafer_results
!cp -r checkpoints/ /content/drive/MyDrive/wafer_results/
!cp *.pkl /content/drive/MyDrive/wafer_results/ 2>/dev/null || echo "No pickle files"
print("Results saved to Drive")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: src` | Run `!pip install -e .` after setup |
| CUDA Out of Memory | Reduce batch size: `--batch-size 32` |
| Dataset not found | Verify at `data/LSWMD_new.pkl` or use Drive mount |
| Slow uploads | Use Drive mount instead of `files.upload()` |
| Connection timeout | Restart runtime and try again |

---

## Performance (Colab T4 GPU)

- **Custom CNN**: ~50s/epoch (5 epochs ≈ 4-5 min)
- **ResNet-18**: ~65s/epoch (5 epochs ≈ 5-6 min)
- **EfficientNet-B0**: ~60s/epoch (5 epochs ≈ 5-6 min)
- **Total (all 3 models)**: ~15-20 min

---

## Next Steps

1. Run training with all 3 models
2. Check results in notebook or Drive
3. Download checkpoints for local evaluation
4. (Coming) LaTeX report + Beamer slides in docs/
