#!/usr/bin/env python3
"""
Colab-specific runner: Setup, mount, and train in one script.
Run in Colab with: exec(open('colab_runner.py').read())
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def run_cmd(cmd):
    """Run shell command."""
    logger.info(f"\n> {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=False).returncode == 0

def setup_colab():
    """Full Colab setup and training pipeline."""

    logger.info("\n" + "="*70)
    logger.info("COLAB SETUP: Wafer Defect Detection")
    logger.info("="*70)

    if not is_colab():
        logger.warning("Warning: Not running in Colab. Some features may not work.")

    # Step 1: Clone repo
    logger.info("\n[1/6] Cloning repository...")
    repo_url = "https://github.com/parkianco/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset.git"
    run_cmd(f"git clone {repo_url}")
    os.chdir("CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset")

    # Step 2: Checkout branch
    logger.info("\n[2/6] Checking out feature branch...")
    run_cmd("git checkout feature/complete-implementation 2>/dev/null || echo 'Branch not available yet, using main'")

    # Step 3: Install dependencies
    logger.info("\n[3/6] Installing dependencies...")
    run_cmd("pip install --upgrade pip")
    run_cmd("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    run_cmd("pip install -q -e '.[dev]'")

    # Step 4: Setup dataset
    logger.info("\n[4/6] Dataset setup...")
    logger.info("\nOptions:")
    logger.info("  1. Upload from computer (files.upload())")
    logger.info("  2. Mount Google Drive")
    logger.info("  3. Skip (assume dataset exists)")

    choice = input("\nChoose option (1-3): ").strip()

    if choice == "1":
        from google.colab import files
        logger.info("\nSelect LSWMD_new.pkl from your computer...")
        uploaded = files.upload()
        if uploaded:
            import shutil
            filename = list(uploaded.keys())[0]
            os.makedirs("data", exist_ok=True)
            shutil.move(filename, "data/LSWMD_new.pkl")
            logger.info(f"Dataset moved to data/LSWMD_new.pkl")

    elif choice == "2":
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        logger.info("\nDataset in Drive? Specify path:")
        drive_path = input("Path (e.g., /content/drive/MyDrive/LSWMD_new.pkl): ").strip()
        if os.path.exists(drive_path):
            import shutil
            os.makedirs("data", exist_ok=True)
            shutil.copy(drive_path, "data/LSWMD_new.pkl")
            logger.info("Dataset copied from Drive")
        else:
            logger.warning(f"Path not found: {drive_path}")

    # Step 5: Verify setup
    logger.info("\n[5/6] Verifying setup...")
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        from src.models import WaferCNN, get_resnet18
        logger.info(f"✓ Package imports OK")
    except Exception as e:
        logger.warning(f"✗ Verification failed: {e}")
        return False

    # Step 6: Run training
    logger.info("\n[6/6] Training configuration...")
    logger.info("\nTraining options:")
    logger.info("  1. All models (5 epochs) - ~20 min")
    logger.info("  2. Custom CNN only (5 epochs) - ~4 min")
    logger.info("  3. ResNet-18 only (5 epochs) - ~5 min")
    logger.info("  4. EfficientNet-B0 only (5 epochs) - ~5 min")
    logger.info("  5. Custom configuration")

    train_choice = input("\nChoose option (1-5): ").strip()

    train_args = {
        "1": "--model all --epochs 5 --device cuda --batch-size 64",
        "2": "--model cnn --epochs 5 --device cuda --batch-size 64",
        "3": "--model resnet --epochs 5 --device cuda --batch-size 64",
        "4": "--model effnet --epochs 5 --device cuda --batch-size 64",
    }

    if train_choice in train_args:
        args = train_args[train_choice]
    elif train_choice == "5":
        model = input("Model (cnn/resnet/effnet/all): ").strip()
        epochs = input("Epochs (default 5): ").strip() or "5"
        batch_size = input("Batch size (default 64): ").strip() or "64"
        args = f"--model {model} --epochs {epochs} --device cuda --batch-size {batch_size}"
    else:
        logger.info("Invalid choice, using default (all models, 5 epochs)")
        args = "--model all --epochs 5 --device cuda --batch-size 64"

    # Start training
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    run_cmd(f"python train.py {args}")

    # Save to Drive (if available)
    logger.info("\n[Saving results...]")
    try:
        run_cmd("cp -r checkpoints /content/drive/MyDrive/wafer_results 2>/dev/null || echo 'Drive not mounted'")
        logger.info("Results saved to Drive/wafer_results")
    except:
        logger.info("Could not save to Drive (not mounted)")

    logger.info("\n" + "="*70)
    logger.info("Training complete!")
    logger.info("="*70)
    return True

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if is_colab():
        setup_colab()
    else:
        logger.info("This script is designed for Google Colab.")
        logger.info("For local setup, run: python -m pip install -e '.[dev]'")
