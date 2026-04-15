#!/usr/bin/env python3
"""
Distributed multi-GPU training launcher.

Supports DataParallel (single-machine, multiple GPUs) and
DistributedDataParallel (multi-machine training).

Usage:
    # Single machine, multiple GPUs (DataParallel)
    python distributed_train.py --model cnn --device-ids 0,1,2,3

    # Multi-machine distributed (DistributedDataParallel)
    python -m torch.distributed.launch --nproc_per_node=4 distributed_train.py
"""

import argparse
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import count_params, count_trainable, evaluate_model
from src.config import load_config
from src.data import (
    KNOWN_CLASSES,
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    load_dataset,
    preprocess_wafer_maps,
    seed_worker,
)
from src.models import WaferCNN, get_efficientnet_b0, get_resnet18
from src.training import train_model
from src.training.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    setup_distributed,
    wrap_model_dataparallel,
    wrap_model_distributed,
)

SEED = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess(
    dataset_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
    rank: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, DataLoader], LabelEncoder]:
    """Load dataset, preprocess, and create loaders."""
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("LOADING AND PREPROCESSING DATA")
        logger.info("=" * 70)

    # Load raw dataset
    df = load_dataset(dataset_path)

    # Filter to known classes
    labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)
    if rank == 0:
        logger.info(f"Filtered to {len(df_clean):,} labeled wafers")

    # Encode labels
    le = LabelEncoder()
    df_clean["label_encoded"] = le.fit_transform(df_clean["failureClass"])

    # Extract data
    wafer_maps = df_clean["waferMap"].values
    labels = df_clean["label_encoded"].values

    # Stratified train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.arange(len(labels)), labels, test_size=0.30, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    # Preprocess
    train_maps_raw = [wafer_maps[i] for i in X_train]
    val_maps_raw = [wafer_maps[i] for i in X_val]
    test_maps_raw = [wafer_maps[i] for i in X_test]

    train_maps = preprocess_wafer_maps(train_maps_raw)
    val_maps = preprocess_wafer_maps(val_maps_raw)
    test_maps = preprocess_wafer_maps(test_maps_raw)

    if rank == 0:
        logger.info(
            f"Split sizes: Train={len(y_train):,}, Val={len(y_val):,}, Test={len(y_test):,}"
        )

    # Compute loss weights
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [
            total_train / (len(KNOWN_CLASSES) * class_counts_train[c])
            for c in range(len(KNOWN_CLASSES))
        ],
        dtype=torch.float32,
    )

    # Create datasets
    train_aug = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    train_dataset_cnn = WaferMapDataset(train_maps, y_train, transform=train_aug)
    val_dataset_cnn = WaferMapDataset(val_maps, y_val, transform=None)
    test_dataset_cnn = WaferMapDataset(test_maps, y_test, transform=None)

    train_dataset_pre = WaferMapDataset(
        train_maps, y_train, transform=torch.nn.Sequential(train_aug, imagenet_norm)
    )
    val_dataset_pre = WaferMapDataset(val_maps, y_val, transform=imagenet_norm)
    test_dataset_pre = WaferMapDataset(test_maps, y_test, transform=imagenet_norm)

    # Create loaders with DistributedSampler if distributed
    batch_size = 64
    if is_distributed():
        train_sampler_cnn = DistributedSampler(
            train_dataset_cnn,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=seed,
        )
        train_sampler_pre = DistributedSampler(
            train_dataset_pre,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=seed,
        )
        g_cnn = torch.Generator().manual_seed(42)
        train_loader_cnn = DataLoader(
            train_dataset_cnn,
            batch_size=batch_size,
            sampler=train_sampler_cnn,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g_cnn,
        )
        g_pre = torch.Generator().manual_seed(42)
        train_loader_pre = DataLoader(
            train_dataset_pre,
            batch_size=batch_size,
            sampler=train_sampler_pre,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g_pre,
        )
    else:
        g_cnn = torch.Generator().manual_seed(42)
        train_loader_cnn = DataLoader(
            train_dataset_cnn,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g_cnn,
        )
        g_pre = torch.Generator().manual_seed(42)
        train_loader_pre = DataLoader(
            train_dataset_pre,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g_pre,
        )

    g_val_cnn = torch.Generator().manual_seed(42)
    val_loader_cnn = DataLoader(
        val_dataset_cnn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g_val_cnn,
    )
    test_loader_cnn = DataLoader(
        test_dataset_cnn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g_val_cnn,
    )
    g_val_pre = torch.Generator().manual_seed(42)
    val_loader_pre = DataLoader(
        val_dataset_pre,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g_val_pre,
    )
    test_loader_pre = DataLoader(
        test_dataset_pre,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g_val_pre,
    )

    loaders = {
        "cnn": (train_loader_cnn, val_loader_cnn, test_loader_cnn),
        "resnet": (train_loader_pre, val_loader_pre, test_loader_pre),
        "effnet": (train_loader_pre, val_loader_pre, test_loader_pre),
    }

    data = {
        "class_names": le.classes_.tolist(),
        "loss_weights": loss_weights,
    }

    return data, loaders, le


def main():
    """Main distributed training entry point."""
    # Check if running with torch.distributed.launch
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(description="Distributed multi-GPU training")
    parser.add_argument("--model", choices=["cnn", "resnet", "effnet", "all"], default="cnn")
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size)
    parser.add_argument(
        "--device-ids", type=str, default=None, help="GPU IDs (comma-separated, for DataParallel)"
    )
    parser.add_argument("--distributed", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--data-path", type=Path, default=None)

    args = parser.parse_args()

    # Setup distributed training
    is_dist = world_size > 1 or args.distributed
    if is_dist and rank == 0:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")

    if is_dist:
        setup_distributed(rank, world_size, backend="nccl" if torch.cuda.is_available() else "gloo")

    # Setup device
    if is_dist:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Device: {device}")

    set_seed(config.seed)

    # Load data
    if args.data_path is None:
        args.data_path = Path(config.data.dataset_path)
        if not args.data_path.is_absolute():
            args.data_path = Path(__file__).parent / args.data_path

    data, loaders, le = load_and_preprocess(args.data_path, seed=config.seed, rank=rank)
    class_names = data["class_names"]
    loss_weights = data["loss_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    models_to_train = ["cnn", "resnet", "effnet"] if args.model == "all" else [args.model]
    results = {}

    for model_type in models_to_train:
        if rank == 0:
            logger.info("\n" + "=" * 70)
            logger.info(f"DISTRIBUTED TRAINING: {model_type.upper()}")
            logger.info("=" * 70)

        # Create model
        if model_type == "cnn":
            model = WaferCNN(num_classes=len(class_names)).to(device)
            lr = 1e-3
            model_name = "Custom CNN"
        elif model_type == "resnet":
            model = get_resnet18(num_classes=len(class_names)).to(device)
            lr = 1e-4
            model_name = "ResNet-18"
        else:  # effnet
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)
            lr = 1e-4
            model_name = "EfficientNet-B0"

        # Wrap model for distributed training
        if is_dist:
            model = wrap_model_distributed(model, rank, world_size)
        else:
            if args.device_ids:
                device_ids = [int(x) for x in args.device_ids.split(",")]
                model = wrap_model_dataparallel(model, device_ids=device_ids)

        if rank == 0:
            total_params = count_params(model)
            trainable_params = count_trainable(model)
            logger.info(
                f"{model_name} Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )

        # Training setup
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Train
        train_loader, val_loader, test_loader = loaders[model_type]
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            model_name=model_name,
            device=str(device),
        )

        # Evaluate (only on rank 0)
        if rank == 0:
            preds, labels, metrics = evaluate_model(
                model, test_loader, class_names, model_name, str(device)
            )
            results[model_type] = {
                "metrics": metrics,
                "history": history,
            }

    # Summary (only on rank 0)
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        for mtype, res in results.items():
            metrics = res["metrics"]
            logger.info(f"Metrics:")
            logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1:    {metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    # Cleanup
    if is_dist:
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import os

    sys.exit(main())
