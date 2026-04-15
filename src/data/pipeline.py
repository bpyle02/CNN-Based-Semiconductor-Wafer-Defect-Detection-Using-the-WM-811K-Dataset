"""Data pipeline: cache-aware load + stratified split for WM-811K.

This module owns the canonical ``load_and_preprocess_data`` implementation.
Scripts and CLIs should import from here rather than duplicating the logic.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import load_dataset
from src.data.dataset import KNOWN_CLASSES

logger = logging.getLogger(__name__)

SEED = 42


def load_and_preprocess_data(
    dataset_path,
    train_size: float = 0.70,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
    synthetic: bool = False,
    target_size=(96, 96),
) -> Dict[str, Any]:
    """Load, split, and optionally balance the WM-811K dataset.

    Uses the pre-resized NPZ cache produced by ``scripts/precompute_tensors.py``
    when present (memory-mapped to keep RSS bounded on Colab), otherwise falls
    back to loading the raw pickle.

    Returns a dict with keys: ``train_maps``, ``y_train``, ``val_maps``,
    ``y_val``, ``test_maps``, ``y_test``, ``loss_weights``, ``class_names``.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    logger.info(f"\n{'='*70}")
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info(f"{'='*70}")

    # Fast path: pre-resized cache from scripts/precompute_tensors.py.
    # We intentionally do NOT auto-build the cache from inside train.py
    # anymore. The cache-build spawns a multiprocessing Pool, and on
    # Colab T4 (13.6 GB) the combined footprint of (raw df) + (pool fork
    # copies) + (imminent training setup) exceeds RAM and the kernel
    # gets SIGKILLed (exit -9). Build the cache separately ahead of time:
    #     python scripts/precompute_tensors.py
    # or run the notebook's Cell 5b. The Colab quickstart Cell 6 also
    # runs a preflight that builds the cache as its own subprocess if
    # missing — that process exits cleanly and releases RAM before
    # train.py starts.
    cache_path = Path(dataset_path).parent / "LSWMD_cache.npz"
    maps_npy_path = cache_path.with_suffix(".maps.npy")
    if cache_path.exists():
        logger.info(f"Using pre-resized cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        cached_labels_str = cache["labels"]

        # Two cache layouts supported:
        #   1. Current: sidecar .npz (this file) + .maps.npy (memmap). RAM-lean.
        #   2. Legacy:  single .npz whose "maps" key holds the full array.
        #      Still readable but loads the whole 3.2 GB array into RAM.
        if maps_npy_path.exists():
            logger.info(f"Memory-mapping {maps_npy_path} (mmap_mode='r')")
            cached_maps = np.load(maps_npy_path, mmap_mode="r")
        elif "maps" in cache.files:
            logger.info("Legacy cache layout: loading full maps array into RAM")
            cached_maps = cache["maps"]
        else:
            raise RuntimeError(
                f"{cache_path} has no 'maps' key and {maps_npy_path} is missing. "
                "Regenerate the cache with scripts/precompute_tensors.py."
            )

        le = LabelEncoder().fit(np.array(KNOWN_CLASSES))
        labels = le.transform(cached_labels_str)
        # Hand wafer_maps as an object array so the existing downstream code
        # path (index + object-array assignment) works unchanged. The per-item
        # shape is already target_size, so WaferMapDataset takes its fast path.
        # With the memmap layout, each wafer_maps[i] is a memmap slice — the
        # backing data only pages into RAM when the DataLoader actually reads
        # it per-batch, which keeps training-time RSS bounded.
        wafer_maps = np.empty(len(cached_maps), dtype=object)
        for i in range(len(cached_maps)):
            wafer_maps[i] = cached_maps[i]
    else:
        logger.info("Loading dataset...")
        df = load_dataset(dataset_path)

        labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
        df_clean = df[labeled_mask].reset_index(drop=True)

        le = LabelEncoder()
        df_clean["label_encoded"] = le.fit_transform(df_clean["failureClass"])

        wafer_maps = df_clean["waferMap"].values
        labels = df_clean["label_encoded"].values

    logger.info(f"Total samples: {len(labels):,}")
    logger.info(f"Class distribution:")
    for i, cls in enumerate(KNOWN_CLASSES):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        logger.info(f"  {cls:12s}: {count:6,} ({pct:5.1f}%)")

    # Split: train (70%), val (15%), test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)), labels, test_size=test_size, stratify=labels, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp, random_state=seed
    )

    # WM-811K wafer maps have heterogeneous shapes; keep as object arrays so the
    # WaferMapDataset's lazy per-item resize handles them. Collapsing into a dense
    # np.array() fails with "inhomogeneous shape" for the raw dataset.
    train_maps = np.empty(len(X_train), dtype=object)
    train_maps[:] = [wafer_maps[i] for i in X_train]
    val_maps = np.empty(len(X_val), dtype=object)
    val_maps[:] = [wafer_maps[i] for i in X_val]
    test_maps = np.empty(len(X_test), dtype=object)
    test_maps[:] = [wafer_maps[i] for i in X_test]

    # Optional: balance training set with synthetic augmentation
    if synthetic:
        from src.augmentation.synthetic import balance_dataset_with_synthetic

        logger.info("Applying synthetic augmentation to balance training set...")
        train_maps, y_train = balance_dataset_with_synthetic(
            train_maps, y_train, target_per_class=None, size=target_size[0]
        )
        logger.info(f"  Training set after augmentation: {len(y_train):,} samples")

    # Compute class weights from training set
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [
            total_train / (len(KNOWN_CLASSES) * class_counts_train[c])
            for c in range(len(KNOWN_CLASSES))
        ],
        dtype=torch.float32,
    )
    logger.info(f"\nClass weights (from training set):")
    logger.info(f"  {[f'{w:.2f}' for w in loss_weights.tolist()]}")

    return {
        "train_maps": train_maps,
        "y_train": y_train,
        "val_maps": val_maps,
        "y_val": y_val,
        "test_maps": test_maps,
        "y_test": y_test,
        "loss_weights": loss_weights,
        "class_names": KNOWN_CLASSES,
    }


__all__ = ["load_and_preprocess_data", "SEED"]
