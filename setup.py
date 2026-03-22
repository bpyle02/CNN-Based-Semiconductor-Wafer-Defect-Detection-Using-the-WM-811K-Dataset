#!/usr/bin/env python3
"""
Setup script for installing dependencies and configuring environment.
Works on local machine, Colab, Kaggle, and other platforms.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command with error handling."""
    if description:
        print(f"\n{'='*70}")
        print(f"{description}")
        print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def is_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_kaggle():
    """Detect if running in Kaggle Notebooks."""
    return os.path.exists('/kaggle/input')

def setup_environment():
    """Setup environment based on platform."""
    platform = "Colab" if is_colab() else ("Kaggle" if is_kaggle() else "Local")
    print(f"\nDetected platform: {platform}")

    # Update pip
    run_command("pip install --upgrade pip", "Upgrading pip")

    # Install PyTorch (with CUDA support)
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    run_command(pytorch_cmd, "Installing PyTorch with CUDA support")

    # Install requirements
    run_command("pip install -r requirements.txt", "Installing project dependencies")

    # Verify installations
    verify_installations()

def verify_installations():
    """Verify critical packages are installed."""
    print(f"\n{'='*70}")
    print("Verifying installations")
    print(f"{'='*70}")

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
    }

    all_ok = True
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            all_ok = False

    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if gpu_available else "CPU"
        print(f"\n✓ GPU Available: {gpu_available}")
        print(f"  Device: {device}")
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        all_ok = False

    if all_ok:
        print(f"\n{'='*70}")
        print("Setup complete! Ready to train.")
        print(f"{'='*70}")
        return True
    else:
        print("\nSome packages failed to install. Please check errors above.")
        return False

if __name__ == "__main__":
    setup_environment()
