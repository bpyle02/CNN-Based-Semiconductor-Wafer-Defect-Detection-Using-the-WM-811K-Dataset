#!/usr/bin/env python3
"""
Bootstrap a local development environment for the wafer defect repository.

This script is the explicit replacement for the old bootstrap-style setup.py.
It installs the package in editable mode and can optionally install the CUDA
PyTorch wheels that the original helper hardcoded.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str], description: str) -> None:
    """Run a subprocess command and raise on failure."""
    logger.info("=" * 70)
    logger.info(description)
    logger.info("=" * 70)
    subprocess.run(args, check=True, cwd=REPO_ROOT)


def verify_installations() -> bool:
    """Verify the core runtime dependencies are importable."""
    packages = {
        "torch": "PyTorch",
        "torchvision": "Torchvision",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "matplotlib": "Matplotlib",
        "PIL": "Pillow",
    }

    all_ok = True
    for module_name, display_name in packages.items():
        try:
            __import__(module_name)
            logger.info("OK %s", display_name)
        except ImportError:
            logger.warning("MISSING %s", display_name)
            all_ok = False

    try:
        import torch

        gpu_available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if gpu_available else "CPU"
        logger.info("GPU Available: %s", gpu_available)
        logger.info("Device: %s", device)
    except Exception as exc:
        logger.warning("GPU check failed: %s", exc)
        all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the local development environment.")
    parser.add_argument(
        "--extras",
        default="dev",
        help='Editable-install extras to use, e.g. "dev" or "server,tuning".',
    )
    parser.add_argument(
        "--with-cuda-cu118",
        action="store_true",
        help="Install the CUDA 11.8 PyTorch wheels before the editable install.",
    )
    args = parser.parse_args()

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")

    if args.with_cuda_cu118:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
            ],
            "Installing CUDA 11.8 PyTorch wheels",
        )

    editable_target = f".[{args.extras}]" if args.extras else "."
    run_command(
        [sys.executable, "-m", "pip", "install", "-e", editable_target],
        f"Installing editable package {editable_target}",
    )

    if verify_installations():
        logger.info("Environment bootstrap complete.")
        return 0

    logger.warning("Environment bootstrap completed with missing dependencies.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    raise SystemExit(main())
