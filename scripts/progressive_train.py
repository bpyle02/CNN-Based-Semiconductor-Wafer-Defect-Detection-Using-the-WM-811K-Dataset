#!/usr/bin/env python3
"""
Progressive training script: multi-resolution training with increasing image sizes.

Implements curriculum learning by training models at progressively larger image sizes.
Typically: 48x48 (2 epochs) -> 96x96 (3 epochs) -> 192x192 (2 epochs).

This approach:
- Speeds up early training with small images
- Improves generalization by forcing models to learn at multiple scales
- Reduces computational cost early on

Usage:
    python progressive_train.py --model cnn --device cuda
    python progressive_train.py --model all --config custom_config.yaml
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.analysis import count_params, count_trainable, evaluate_model
from src.config import Config, load_config
from src.data import (
    KNOWN_CLASSES,
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    load_dataset,
    seed_worker,
)
from src.models import WaferCNN, get_efficientnet_b0, get_resnet18
from src.training import train_model
from src.training.base_trainer import BaseTrainer

SEED = 42


def preprocess_maps_to_size(raw_maps: list, target_size: int) -> np.ndarray:
    """
    Resize wafer maps to target size.

    Args:
        raw_maps: List of raw wafer map arrays
        target_size: Target image dimension (for square images)

    Returns:
        Array of preprocessed wafer maps with shape (N, 3, target_size, target_size)
    """
    processed = []
    for wmap in raw_maps:
        if isinstance(wmap, str):
            # Skip invalid maps
            processed.append(np.zeros((target_size, target_size), dtype=np.float32))
        else:
            try:
                # Normalize to [0, 1]
                normalized = (wmap.astype(np.float32) + 1) / 2.0
                # Resize using OpenCV
                resized = cv2.resize(
                    normalized, (target_size, target_size), interpolation=cv2.INTER_LINEAR
                )
                # Stack to 3 channels
                stacked = np.stack([resized] * 3, axis=0)
                processed.append(stacked)
            except Exception:
                processed.append(np.zeros((3, target_size, target_size), dtype=np.float32))
    return np.array(processed, dtype=np.float32)


def load_and_split_data(
    dataset_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Load dataset and return raw arrays (before resizing)."""
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    # Load raw dataset
    df = load_dataset(dataset_path)

    # Filter to known classes
    labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)
    logger.info(f"Filtered to {len(df_clean):,} labeled wafers")

    # Encode labels
    le = LabelEncoder()
    df_clean["label_encoded"] = le.fit_transform(df_clean["failureClass"])
    logger.info(f"Classes: {le.classes_.tolist()}")

    # Extract raw data
    wafer_maps_raw = df_clean["waferMap"].values
    labels = df_clean["label_encoded"].values

    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.arange(len(labels)), labels, test_size=0.30, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    logger.info(f"Split sizes: Train={len(y_train):,}, Val={len(y_val):,}, Test={len(y_test):,}")

    # Return raw maps and labels
    train_maps_raw = wafer_maps_raw[X_train]
    val_maps_raw = wafer_maps_raw[X_val]
    test_maps_raw = wafer_maps_raw[X_test]

    return train_maps_raw, val_maps_raw, test_maps_raw, y_train, y_val, y_test, le  # type: ignore


def create_dataloaders(
    train_maps: np.ndarray,
    val_maps: np.ndarray,
    test_maps: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    image_size: int,
    model_type: str = "cnn",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for current image size."""
    # Preprocess to target size
    logger.info(f"Preprocessing to {image_size}x{image_size}...")
    train_maps = preprocess_maps_to_size(train_maps, image_size)
    val_maps = preprocess_maps_to_size(val_maps, image_size)
    test_maps = preprocess_maps_to_size(test_maps, image_size)

    # Create datasets
    if model_type == "cnn":
        train_transform = get_image_transforms()
        val_transform = None
    else:  # pretrained models
        train_transform = torch.nn.Sequential(get_image_transforms(), get_imagenet_normalize())
        val_transform = get_imagenet_normalize()

    train_dataset = WaferMapDataset(train_maps, y_train, transform=train_transform)
    val_dataset = WaferMapDataset(val_maps, y_val, transform=val_transform)
    test_dataset = WaferMapDataset(test_maps, y_test, transform=val_transform)

    # Create loaders
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader


def main() -> int:
    """
    Main progressive training entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Load config
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(
        description="Progressive training with multi-resolution curriculum learning"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "effnet", "all"],
        default=config.training.default_model,
        help=f"Model to train (default: {config.training.default_model})",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=config.device)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    # Reload config if custom path
    if args.config:
        config = load_config(args.config)

    # Setup
    device = torch.device(config.device)
    logger.info(f"Device: {device}")

    # Load data (raw, before resizing)
    if args.data_path is None:
        args.data_path = Path(config.data.dataset_path)
        if not args.data_path.is_absolute():
            args.data_path = Path(__file__).parent / args.data_path

    train_maps_raw, val_maps_raw, test_maps_raw, y_train, y_val, y_test, le = load_and_split_data(
        args.data_path,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        seed=config.seed,
    )
    class_names = le.classes_.tolist()

    # Compute loss weights
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [
            total_train / (len(KNOWN_CLASSES) * class_counts_train[c])
            for c in range(len(KNOWN_CLASSES))
        ],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Progressive training stages
    if not config.progressive_training.enabled:
        logger.info(
            "Progressive training disabled in config. Set progressive_training.enabled = true"
        )
        return 1

    stages = config.progressive_training.stages
    logger.info(f"\n{'='*70}")
    logger.info(f"PROGRESSIVE TRAINING: {len(stages)} stages")
    logger.info(f"{'='*70}")

    models_to_train = (
        ["cnn", "resnet", "effnet"]
        if config.training.default_model == "all"
        else [config.training.default_model]
    )
    results = {}

    for model_type in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"PROGRESSIVE TRAINING: {model_type.upper()}")
        logger.info(f"{'='*70}")

        # Create model (once, then train across stages)
        if model_type == "cnn":
            model = WaferCNN(num_classes=len(class_names)).to(device)
            model_name = "Custom CNN"
        elif model_type == "resnet":
            model = get_resnet18(num_classes=len(class_names)).to(device)
            model_name = "ResNet-18"
        else:  # effnet
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)
            model_name = "EfficientNet-B0"

        logger.info(f"{model_name} Parameters:")
        logger.info(f"  Total: {count_params(model):,}")
        logger.info(f"  Trainable: {count_trainable(model):,}")

        stage_histories = []
        stage_start_time = time.time()

        # Progressive training loop
        for stage_idx, stage in enumerate(stages):
            image_size = stage["image_size"]
            epochs = stage["epochs"]
            lr_factor = stage.get("learning_rate_factor", 1.0)

            logger.info(
                f"\n  Stage {stage_idx + 1}/{len(stages)}: {image_size}x{image_size} ({epochs} epochs, LR factor={lr_factor})"
            )

            # Create dataloaders for this stage
            train_loader, val_loader, test_loader = create_dataloaders(
                train_maps_raw,
                val_maps_raw,
                test_maps_raw,
                y_train,
                y_val,
                y_test,
                batch_size=config.training.batch_size,
                image_size=image_size,
                model_type=model_type,
            )

            # Adjust learning rate for this stage
            if model_type == "cnn":
                base_lr = config.training.learning_rate.get("cnn", 1e-3)
            else:
                base_lr = config.training.learning_rate.get("resnet", 1e-4)

            stage_lr = base_lr * lr_factor
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=stage_lr,
                weight_decay=config.training.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

            # Train for this stage
            model, history = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler=scheduler,
                epochs=epochs,
                model_name=f"{model_name} (Stage {stage_idx + 1})",
                device=config.device,
            )
            stage_histories.append(history)

        # Evaluate on final test set
        _, test_loader_final, _ = create_dataloaders(
            train_maps_raw,
            val_maps_raw,
            test_maps_raw,
            y_train,
            y_val,
            y_test,
            batch_size=config.training.batch_size,
            image_size=stages[-1]["image_size"],  # Final size
            model_type=model_type,
        )

        preds, labels, metrics = evaluate_model(
            model, test_loader_final, class_names, model_name, config.device
        )

        total_time = time.time() - stage_start_time
        results[model_type] = {
            "model": model,
            "histories": stage_histories,
            "predictions": preds,
            "labels": labels,
            "metrics": metrics,
            "model_name": model_name,
            "total_time": total_time,
        }

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("PROGRESSIVE TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    for mtype, res in results.items():
        metrics = res["metrics"]
        logger.info(f"{res['model_name']}:")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"  Time:        {res['total_time']:.1f}s")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.exit(main())
