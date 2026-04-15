#!/usr/bin/env python3
"""
Pre-resize WM-811K wafer maps to a uniform target size and cache to disk.

The raw dataset (LSWMD_new.pkl) stores 172,950 labeled wafer maps as
variable-shape numpy arrays (anywhere from ~26x26 to ~300x300). The default
WaferMapDataset lazily resizes each map to 96x96 using skimage every time
__getitem__ runs, which on Colab T4 is the bottleneck: the GPU stays idle
~80% of the time waiting for CPU-bound skimage calls.

This script resizes the entire labeled subset once using OpenCV (~10-50x
faster than skimage) and saves the result as a single numpy .npz file.
Training re-uses this cache and gets fed at GPU-bound rates (~20-40 s/epoch
on T4 instead of ~5-15 min/epoch).

Memory discipline (important for Colab T4's 13.6 GB limit):
- Default path is SINGLE-PROCESS. Multiprocessing.Pool fork copies the
  parent RSS once per worker, which spikes memory to 6-10 GB easily
  and triggers the Colab OOM-killer (exit code -9). Use --workers > 1
  only on machines with >32 GB RAM.
- The output array is pre-allocated ONCE and filled in place, so we
  never hold 2x the output in memory (no `np.stack`).
- The pandas DataFrame is released as soon as wafer arrays and labels
  are extracted, with an explicit `gc.collect()` to free the backing.
- Peak RSS stays around 4.5 GB: well under Colab's 13.6 GB limit even
  with another process (the notebook kernel) holding 2-3 GB.

Usage:
    python scripts/precompute_tensors.py
    python scripts/precompute_tensors.py --size 96 --input data/LSWMD_new.pkl
    python scripts/precompute_tensors.py --workers 2        # opt-in, needs >16 GB
"""

from __future__ import annotations

import argparse
import gc
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
    """Resize one wafer map using OpenCV. Returns float16 in [0, 1]."""
    import cv2  # imported inside worker so Pool pickling works on Windows

    wm, size = args
    arr = np.asarray(wm, dtype=np.float32)
    resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    # Raw WM-811K values are {0, 1, 2}. Normalize to [0, 1] by dividing by 2.
    return (resized / 2.0).astype(np.float16)


def _resize_inplace(wafer_maps: list, out: np.ndarray, size: int) -> None:
    """Fill pre-allocated `out[i]` with the resized `wafer_maps[i]` for all i.

    Frees each source wafer from the list as soon as it's consumed so the
    overall footprint of wafer_maps + out shrinks monotonically (source
    memory drops by ~size**2 bytes per iteration; output pages are already
    allocated so the total RSS curve is flat or downward-sloping).
    """
    import cv2  # one-time import, not per-call

    n = len(wafer_maps)
    t0 = time.time()
    progress_every = max(n // 10, 1)

    for i in range(n):
        wm = wafer_maps[i]
        arr = np.asarray(wm, dtype=np.float32)
        resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = (resized / 2.0).astype(np.float16)
        wafer_maps[i] = None  # drop source ref so GC can reclaim variable-size array

        if (i + 1) % progress_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            logger.info(
                "  %d/%d (%.0f maps/s, ETA %.0fs)", i + 1, n, rate, eta
            )


def build_cache(
    input_path: Path,
    output_path: Path,
    size: int = 96,
    workers: int = 1,
) -> Path:
    """Build the pre-resized tensor cache.

    Args:
        input_path: raw LSWMD_new.pkl
        output_path: destination .npz
        size: target HxW (square)
        workers: process count. **Default 1 (single-process, memory-safe).**
            Set > 1 only on hosts with >=16 GB RAM — multiprocessing.Pool
            fork copies the parent RSS and will OOM Colab T4.

    Returns:
        The output_path on success.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset from %s ...", input_path)
    df = load_dataset(input_path)

    mask = df["failureClass"].isin(KNOWN_CLASSES)
    df = df[mask].reset_index(drop=True)
    n = len(df)
    logger.info("Labeled samples: %d", n)

    wafer_maps = df["waferMap"].tolist()
    # Copy the labels OUT of the DataFrame so we can drop df entirely.
    labels_str = df["failureClass"].to_numpy().copy()

    # Critical for Colab memory: drop the DataFrame (and its variable-shape
    # numpy object column) so subsequent allocations have real RAM to use.
    del df, mask
    gc.collect()

    logger.info(
        "Pre-allocating output array: (%d, %d, %d) float16 = %.2f GB",
        n, size, size, n * size * size * 2 / 1e9
    )
    maps = np.empty((n, size, size), dtype=np.float16)

    t0 = time.time()
    if workers <= 1:
        logger.info(
            "Resizing %d maps to %dx%d (single-process, memory-frugal) ...",
            n, size, size
        )
        _resize_inplace(wafer_maps, maps, size)
    else:
        logger.info(
            "Resizing %d maps to %dx%d using %d worker process(es). "
            "Note: pool forking doubles RSS; only use on hosts with "
            ">=16 GB RAM.",
            n, size, size, workers
        )
        with Pool(workers) as pool:
            for i, resized in enumerate(
                pool.imap(_resize_one,
                          ((wm, size) for wm in wafer_maps),
                          chunksize=200)
            ):
                maps[i] = resized

    elapsed = time.time() - t0
    logger.info(
        "Resize complete in %.1fs (%.0f maps/s). Shape: %s, dtype: %s, "
        "uncompressed size: %.2f GB",
        elapsed, n / max(elapsed, 1e-6),
        maps.shape, maps.dtype, maps.nbytes / 1e9
    )

    # Release the source list now that everything is in `maps`.
    del wafer_maps
    gc.collect()

    logger.info("Saving to %s ...", output_path)
    np.savez(
        output_path,
        maps=maps,
        labels=labels_str,
        classes=np.array(KNOWN_CLASSES),
        target_size=np.array([size, size], dtype=np.int32),
    )
    logger.info("Done. Cache file: %.2f GB", output_path.stat().st_size / 1e9)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/LSWMD_new.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/LSWMD_cache.npz"))
    parser.add_argument("--size", type=int, default=96, help="Target resolution (H=W)")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Process count. Default 1 (memory-safe). Only use > 1 on >=16 GB RAM hosts.",
    )
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
