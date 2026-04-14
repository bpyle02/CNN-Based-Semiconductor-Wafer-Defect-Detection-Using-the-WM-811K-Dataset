#!/usr/bin/env python3
"""
Pre-resize WM-811K wafer maps to a uniform target size and cache to disk.

The raw dataset (LSWMD_new.pkl) stores 172,950 labeled wafer maps as
variable-shape numpy arrays (anywhere from ~26x26 to ~300x300). The default
WaferMapDataset lazily resizes each map to 96x96 using skimage every time
__getitem__ runs, which on Colab T4 is the bottleneck: the GPU stays idle
~80% of the time waiting for CPU-bound skimage calls.

This script resizes the entire labeled subset once using OpenCV (~10-50x
faster than skimage) with a process pool, then saves the result as a
single numpy .npz file. Training re-uses this cache and gets fed at
GPU-bound rates (~20-40 s/epoch on T4 instead of ~5-15 min/epoch).

Usage:
    python scripts/precompute_tensors.py
    python scripts/precompute_tensors.py --size 96 --input data/LSWMD_new.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Add repo root to sys.path for src imports when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_dataset
from src.data.dataset import KNOWN_CLASSES


def _resize_one(args: tuple[np.ndarray, int]) -> np.ndarray:
    """Resize one wafer map using OpenCV (releases the GIL)."""
    import cv2  # imported inside worker so Pool pickling works on Windows

    wm, size = args
    arr = np.asarray(wm, dtype=np.float32)
    resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    # Raw WM-811K values are {0, 1, 2}. Normalize to [0, 1] by dividing by 2.
    resized = (resized / 2.0).astype(np.float16)
    return resized


def build_cache(
    input_path: Path,
    output_path: Path,
    size: int = 96,
    workers: int = 0,
) -> Path:
    """Build the pre-resized tensor cache. Callable from train.py as a fallback.

    Args:
        input_path: raw LSWMD_new.pkl
        output_path: destination .npz
        size: target HxW (square)
        workers: process count; 0 means min(cpu_count(), 4)

    Returns:
        The output_path on success.

    Raises:
        FileNotFoundError: if input_path doesn't exist.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw dataset from {input_path} ...")
    df = load_dataset(input_path)

    mask = df["failureClass"].isin(KNOWN_CLASSES)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Labeled samples: {len(df):,} (of {mask.size:,} total)")

    wafer_maps = df["waferMap"].tolist()
    labels_str = df["failureClass"].to_numpy()

    n_workers = workers or min(cpu_count(), 4)
    logger.info(
        f"Resizing {len(wafer_maps):,} maps to {size}x{size} "
        f"using {n_workers} worker process(es) ..."
    )

    t0 = time.time()
    payload = [(wm, size) for wm in wafer_maps]

    if n_workers <= 1:
        resized_list = [_resize_one(p) for p in payload]
    else:
        with Pool(n_workers) as pool:
            resized_list = list(pool.imap(_resize_one, payload, chunksize=200))

    maps = np.stack(resized_list)
    elapsed = time.time() - t0
    logger.info(
        f"Resize complete in {elapsed:.1f}s "
        f"({len(wafer_maps)/elapsed:.0f} maps/s). "
        f"Cache shape: {maps.shape}, dtype: {maps.dtype}, "
        f"size on disk: {maps.nbytes/1e9:.2f} GB"
    )

    logger.info(f"Saving to {output_path} ...")
    np.savez(
        output_path,
        maps=maps,
        labels=labels_str,
        classes=np.array(KNOWN_CLASSES),
        target_size=np.array([size, size], dtype=np.int32),
    )
    logger.info("Done.")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/LSWMD_new.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/LSWMD_cache.npz"))
    parser.add_argument("--size", type=int, default=96, help="Target resolution (H=W)")
    parser.add_argument("--workers", type=int, default=0, help="Process count (0 = cpu_count)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        build_cache(args.input, args.output, size=args.size, workers=args.workers)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
