"""Locate or download the WM-811K dataset (Cell 3 of kaggle_quickstart.ipynb)."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET_SLUG = "brandonpyle/wm-811k-wafer-map"
DEFAULT_TARGET = Path("data/LSWMD_new.pkl")
_ACCEPTED_NAMES = ("LSWMD_new.pkl", "LSWMD.pkl")
_MIN_SIZE_GB = 1.0


def _size_gb(path: str | os.PathLike[str]) -> float:
    return os.path.getsize(path) / 1e9


def _pick_pkl(paths: list[str]) -> str | None:
    """From a list of .pkl paths, return the first plausible full WM-811K dump.

    Accepts either LSWMD_new.pkl or LSWMD.pkl and requires size > 1 GB so we
    never link a tiny preview/sample file as the real dataset.
    """
    for path in paths:
        if os.path.basename(path) in _ACCEPTED_NAMES and _size_gb(path) > _MIN_SIZE_GB:
            return path
    return None


def locate_or_download_dataset(
    slug: str = DEFAULT_DATASET_SLUG,
    target_path: str | os.PathLike[str] = DEFAULT_TARGET,
    download_dir: str | os.PathLike[str] = "/kaggle/working/wm811k_dl",
    input_glob: str = "/kaggle/input/**/*.pkl",
) -> Path:
    """Ensure ``target_path`` exists as a link/copy of the WM-811K pkl.

    Resolution order:
    1. ``target_path`` already exists and is > 1 GB — reuse.
    2. Any ``/kaggle/input/**/*.pkl`` with an accepted filename — symlink.
    3. Kaggle CLI download of ``slug`` — installs the CLI if missing.

    Returns the resolved ``target_path`` as a :class:`pathlib.Path`.
    Raises :class:`RuntimeError` if no candidate is found.
    """
    target = Path(target_path)
    target.parent.mkdir(exist_ok=True, parents=True)

    # 1. Already linked from a prior run.
    if target.exists() and _size_gb(target) > _MIN_SIZE_GB:
        print(f"Dataset already at {target} ({_size_gb(target):.2f} GB)")
        return target

    src: str | None = None

    # 2. Kaggle may have auto-attached the dataset as a notebook input.
    input_candidates = glob.glob(input_glob, recursive=True)
    src = _pick_pkl(input_candidates)
    if src:
        print(f"Found attached at {src} ({_size_gb(src):.2f} GB)")

    # 3. Not attached. Fall back to the Kaggle CLI.
    if src is None:
        print(f"Dataset not attached; downloading {slug} via kaggle CLI...")
        dl_dir = Path(download_dir)
        dl_dir.mkdir(exist_ok=True, parents=True)
        kaggle_cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(dl_dir),
            "--unzip",
            "--quiet",
        ]
        try:
            subprocess.check_call(kaggle_cmd)
        except FileNotFoundError:
            # Kaggle CLI not on PATH — install, retry once.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
            subprocess.check_call(kaggle_cmd)
        dl_candidates = glob.glob(str(dl_dir / "**/*.pkl"), recursive=True)
        src = _pick_pkl(dl_candidates)
        if src:
            print(f"Downloaded to {src} ({_size_gb(src):.2f} GB)")

    if src is None:
        raise RuntimeError(
            f"Could not locate LSWMD_new.pkl or LSWMD.pkl. Verify Internet "
            f"is ON in notebook Settings, then re-run. If the Kaggle dataset "
            f"{slug} is private to you, log in under the same account in "
            f"Settings -> Account."
        )

    # Symlink so the pipeline's default path works without config changes.
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(src)
    print(f"Linked {target} -> {src}")
    print(f"Ready: {target} ({_size_gb(target):.2f} GB)")
    return target
