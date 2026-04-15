"""Print a GPU + environment snapshot (Cell 2 of kaggle_quickstart.ipynb)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

_DEFAULT_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_DOCKER_IMAGE",
    "PYTHONPATH",
)


def _try_import_torch():
    """Import torch lazily, returning None on failure.

    Factored out so tests can monkey-patch this to skip the potentially
    slow/hanging torch import entirely.
    """
    try:
        import torch  # noqa: WPS433 — optional runtime dep, imported lazily

        return torch
    except Exception as exc:  # pragma: no cover — defensive only
        print(f"torch:           <import failed: {exc!r}>")
        return None


def print_env_snapshot(env_vars: tuple[str, ...] = _DEFAULT_ENV_VARS) -> None:
    """Print Python/torch/CUDA versions, GPU specs, and selected env vars.

    Mirrors the inline body of Cell 2 so notebooks can call this in one line.
    Tolerates CPU-only or torch-less environments without raising.
    """
    # Emit the Python version line eagerly so even if a later step crashes
    # the caller can tell the helper started. ``platform.platform()`` has
    # been observed to hang or segfault on some Windows Python installs,
    # so guard it.
    print(f"Python:          {sys.version.split()[0]}", flush=True)
    try:
        print(f"Platform:        {platform.platform()}", flush=True)
    except Exception as exc:  # pragma: no cover — defensive only
        print(f"Platform:        <platform.platform() failed: {exc!r}>", flush=True)

    torch = _try_import_torch()

    if torch is not None:
        print(f"torch:           {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available:  {cuda_ok}")
        if cuda_ok:
            print(f"CUDA version:    {torch.version.cuda}")
            print(f"cuDNN version:   {torch.backends.cudnn.version()}")
            print(f"GPU count:       {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {props.name}  {props.total_memory / 1e9:.1f} GB")
        else:
            print(
                "NO GPU — open Settings on the right and set Accelerator = "
                "GPU T4 x2 (first T4 only used)."
            )

    for var in env_vars:
        print(f"env {var!r}: {os.environ.get(var, '<unset>')}")

    print()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        )
        print(out)
    except Exception:
        print("nvidia-smi not available (CPU runtime?).")
