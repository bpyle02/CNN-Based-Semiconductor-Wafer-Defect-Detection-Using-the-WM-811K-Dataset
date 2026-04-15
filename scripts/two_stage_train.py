#!/usr/bin/env python3
"""Two-stage training for wafer defect detection.

Stage 1: Balanced pretraining on synthetic-augmented data (all classes equal count).
Stage 2: Fine-tuning on the real (imbalanced) distribution with frozen early layers.

This addresses the 150:1 class imbalance in WM-811K by first building a balanced
feature representation, then adapting to the real distribution.

Usage:
    python scripts/two_stage_train.py --model cnn --pretrain-epochs 30 --finetune-epochs 20 --device cpu
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.evaluate import evaluate_model
from src.augmentation.synthetic import balance_dataset_with_synthetic
from src.data.dataset import KNOWN_CLASSES, load_dataset
from src.data.preprocessing import (
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    seed_worker,
)
from src.models.cnn import WaferCNN
from src.models.pretrained import get_efficientnet_b0, get_resnet18
from src.training.losses import build_classification_loss
from src.training.trainer import train_model

logger = logging.getLogger(__name__)

SEED = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(
    dataset_path: Path,
    seed: int = SEED,
) -> dict:
    """Load WM-811K, filter to known classes, stratified split 70/15/15.

    Returns dict with train/val/test maps and labels, plus raw arrays for
    synthetic augmentation.
    """
    logger.info("Loading dataset from %s ...", dataset_path)
    df = load_dataset(dataset_path)

    labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean["label_encoded"] = le.fit_transform(df_clean["failureClass"])

    wafer_maps = df_clean["waferMap"].values
    labels = df_clean["label_encoded"].values

    logger.info("Total labeled samples: %d", len(labels))
    for i, cls in enumerate(KNOWN_CLASSES):
        count = int((labels == i).sum())
        pct = 100.0 * count / len(labels)
        logger.info("  %-12s: %6d (%5.1f%%)", cls, count, pct)

    # 70% train, 15% val, 15% test  (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)),
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=seed,
    )
    val_ratio = 0.15 / (1.0 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=seed,
    )

    train_maps = np.array([wafer_maps[i] for i in X_train])
    val_maps = np.array([wafer_maps[i] for i in X_val])
    test_maps = np.array([wafer_maps[i] for i in X_test])

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    return {
        "train_maps": train_maps,
        "y_train": y_train,
        "val_maps": val_maps,
        "y_val": y_val,
        "test_maps": test_maps,
        "y_test": y_test,
    }


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from label array."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor(
        [total / (num_classes * counts[c]) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights


def build_model(model_name: str, num_classes: int, device: str) -> nn.Module:
    """Construct the requested model architecture."""
    if model_name == "cnn":
        return WaferCNN(num_classes=num_classes, input_channels=3).to(device)
    if model_name == "resnet":
        return get_resnet18(num_classes=num_classes, pretrained=True, freeze_until="layer3").to(
            device
        )
    if model_name == "efficientnet":
        return get_efficientnet_b0(
            num_classes=num_classes, pretrained=True, freeze_until="features.6"
        ).to(device)
    raise ValueError(f"Unsupported model: {model_name}")


def make_transforms(model_name: str, augment: bool):
    """Return (train_transform, val_transform) for the given model.

    CNN uses raw augmentation; pretrained models compose augmentation
    with ImageNet normalization.
    """
    import torchvision.transforms as tv

    if model_name == "cnn":
        train_tf = get_image_transforms(augment=augment)
        val_tf = None
    else:
        imagenet_norm = get_imagenet_normalize()
        if augment:
            aug = get_image_transforms(augment=True)
            train_tf = tv.Compose([*aug.transforms, imagenet_norm])
        else:
            train_tf = tv.Compose([imagenet_norm])
        val_tf = tv.Compose([imagenet_norm])
    return train_tf, val_tf


def make_loaders(
    train_maps: np.ndarray,
    y_train: np.ndarray,
    val_maps: np.ndarray,
    y_val: np.ndarray,
    train_tf,
    val_tf,
    batch_size: int,
    seed: int,
) -> tuple:
    """Build DataLoaders for train and validation sets."""
    train_ds = WaferMapDataset(train_maps, y_train, transform=train_tf)
    val_ds = WaferMapDataset(val_maps, y_val, transform=val_tf)

    g = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "worker_init_fn": seed_worker,
        "generator": g,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def freeze_early_layers(model: nn.Module, model_name: str) -> None:
    """Freeze early feature-extraction layers, leaving later layers + classifier trainable.

    For CNN: freeze all feature blocks except the last two conv blocks.
    For ResNet-18: freeze conv1, bn1, layer1-layer3; unfreeze layer4 + fc.
    For EfficientNet-B0: freeze features.0-6; unfreeze features.7-8 + classifier.
    """
    if model_name == "cnn":
        # WaferCNN has .features (Sequential of conv blocks) and .classifier
        # Each conv block = [Conv, BN, ReLU, Conv, BN, ReLU, MaxPool] = 7 layers
        # With 4 blocks (default), freeze first 2 blocks (indices 0-13), unfreeze last 2 (14-27)
        block_size = 7  # layers per block when use_batch_norm=True
        total_layers = len(model.features)
        num_blocks = total_layers // block_size
        freeze_blocks = max(0, num_blocks - 2)
        freeze_up_to = freeze_blocks * block_size

        for i, layer in enumerate(model.features):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

        # classifier always trainable (already requires_grad=True by default)
        logger.info(
            "CNN: froze %d/%d feature layers (%d/%d blocks)",
            freeze_up_to,
            total_layers,
            freeze_blocks,
            num_blocks,
        )

    elif model_name == "resnet":
        # Freeze conv1, bn1, layer1, layer2, layer3  -- unfreeze layer4 + fc
        frozen_prefixes = ("conv1", "bn1", "layer1", "layer2", "layer3")
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_prefixes):
                param.requires_grad = False
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        logger.info("ResNet: %d/%d parameter groups trainable", trainable, total)

    elif model_name == "efficientnet":
        # Freeze features.0 through features.6 -- unfreeze features.7, features.8, classifier
        for name, param in model.named_parameters():
            if name.startswith("features."):
                parts = name.split(".")
                if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) <= 6:
                    param.requires_grad = False
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        logger.info("EfficientNet: %d/%d parameter groups trainable", trainable, total)

    else:
        logger.warning("Unknown model '%s'; skipping layer freezing", model_name)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters (used before Stage 1 pretraining)."""
    for param in model.parameters():
        param.requires_grad = True


def evaluate_and_collect(
    model: nn.Module,
    test_maps: np.ndarray,
    y_test: np.ndarray,
    val_tf,
    batch_size: int,
    seed: int,
    device: str,
    stage_name: str,
) -> dict:
    """Evaluate model on test set, return metrics dict."""
    test_ds = WaferMapDataset(test_maps, y_test, transform=val_tf)
    g = torch.Generator().manual_seed(seed)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    preds, labels_arr, metrics = evaluate_model(
        model, test_loader, KNOWN_CLASSES, stage_name, device
    )

    # Per-class F1
    prec, rec, f1_per, support = precision_recall_fscore_support(
        labels_arr, preds, average=None, zero_division=0
    )
    per_class = {}
    for i, cls in enumerate(KNOWN_CLASSES):
        per_class[cls] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_per[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "ece": float(metrics.get("ece", 0.0)),
        "per_class": per_class,
    }


def run_two_stage(args: argparse.Namespace) -> dict:
    """Execute the full two-stage training pipeline."""
    set_seed(args.seed)
    device = args.device
    num_classes = len(KNOWN_CLASSES)

    # ------------------------------------------------------------------ data
    data = load_and_split_data(args.data_path, seed=args.seed)
    train_maps = data["train_maps"]
    y_train = data["y_train"]
    val_maps = data["val_maps"]
    y_val = data["y_val"]
    test_maps = data["test_maps"]
    y_test = data["y_test"]

    train_tf, val_tf = make_transforms(args.model, augment=True)

    # ============================================================ STAGE 1
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: BALANCED PRETRAINING (synthetic augmentation)")
    logger.info("=" * 70)

    # Create balanced training set via synthetic oversampling
    balanced_maps, balanced_labels = balance_dataset_with_synthetic(
        train_maps, y_train, target_per_class=None, size=96
    )
    balanced_counts = Counter(balanced_labels)
    logger.info("Balanced training set: %d samples", len(balanced_labels))
    for cls_id in sorted(balanced_counts):
        logger.info("  class %d (%s): %d", cls_id, KNOWN_CLASSES[cls_id], balanced_counts[cls_id])

    # Class weights for balanced set (should be ~uniform, but compute anyway)
    balanced_weights = compute_class_weights(balanced_labels, num_classes).to(device)

    # Build model -- all layers unfrozen for pretraining
    model = build_model(args.model, num_classes, device)
    unfreeze_all(model)

    # Focal Loss for Stage 1 (gamma=2, balanced weights)
    stage1_criterion = build_classification_loss(
        "FocalLoss",
        class_weights=balanced_weights,
        focal_gamma=2.0,
        label_smoothing=0.0,
    )

    # Higher LR for pretraining
    pretrain_lr = {"cnn": 1e-3, "resnet": 5e-4, "efficientnet": 5e-4}.get(args.model, 1e-3)
    stage1_optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=pretrain_lr,
        weight_decay=1e-4,
    )

    # ReduceLROnPlateau for Stage 1
    stage1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        stage1_optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    balanced_train_loader, val_loader = make_loaders(
        balanced_maps,
        balanced_labels,
        val_maps,
        y_val,
        train_tf,
        val_tf,
        args.batch_size,
        args.seed,
    )

    logger.info(
        "Stage 1 config: lr=%.1e, epochs=%d, loss=FocalLoss(gamma=2)",
        pretrain_lr,
        args.pretrain_epochs,
    )

    t0 = time.time()
    model, stage1_history = train_model(
        model,
        balanced_train_loader,
        val_loader,
        stage1_criterion,
        stage1_optimizer,
        scheduler=stage1_scheduler,
        epochs=args.pretrain_epochs,
        model_name=f"Stage1-{args.model}",
        device=device,
        gradient_clip=1.0,
        monitored_metric="val_macro_f1",
    )
    stage1_time = time.time() - t0
    logger.info("Stage 1 complete in %.1fs", stage1_time)

    # Save Stage 1 checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    stage1_path = ckpt_dir / f"two_stage_{args.model}_stage1.pth"
    torch.save(model.state_dict(), stage1_path)
    logger.info("Stage 1 checkpoint saved: %s", stage1_path)

    # Evaluate Stage 1
    stage1_metrics = evaluate_and_collect(
        model,
        test_maps,
        y_test,
        val_tf,
        args.batch_size,
        args.seed,
        device,
        stage_name=f"Stage1-{args.model}",
    )
    stage1_metrics["time_sec"] = stage1_time

    # ============================================================ STAGE 2
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: FINE-TUNING ON REAL (IMBALANCED) DATA")
    logger.info("=" * 70)

    # Freeze early layers for fine-tuning
    freeze_early_layers(model, args.model)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %d / %d (%.1f%%)",
        trainable_count,
        total_count,
        100.0 * trainable_count / total_count,
    )

    # Class weights from real (imbalanced) training distribution
    real_weights = compute_class_weights(y_train, num_classes).to(device)

    # Standard CrossEntropy with class weights for Stage 2
    stage2_criterion = build_classification_loss(
        "CrossEntropyLoss",
        class_weights=real_weights,
        label_smoothing=0.05,
    )

    # Lower LR for fine-tuning
    finetune_lr = {"cnn": 1e-4, "resnet": 1e-5, "efficientnet": 1e-5}.get(args.model, 1e-4)
    stage2_optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=finetune_lr,
        weight_decay=1e-4,
    )

    # Cosine annealing LR schedule
    stage2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        stage2_optimizer,
        T_max=args.finetune_epochs,
        eta_min=1e-7,
    )

    real_train_loader, val_loader2 = make_loaders(
        train_maps,
        y_train,
        val_maps,
        y_val,
        train_tf,
        val_tf,
        args.batch_size,
        args.seed,
    )

    logger.info(
        "Stage 2 config: lr=%.1e, epochs=%d, loss=CrossEntropy+weights, scheduler=CosineAnnealing",
        finetune_lr,
        args.finetune_epochs,
    )

    t0 = time.time()
    model, stage2_history = train_model(
        model,
        real_train_loader,
        val_loader2,
        stage2_criterion,
        stage2_optimizer,
        scheduler=stage2_scheduler,
        epochs=args.finetune_epochs,
        model_name=f"Stage2-{args.model}",
        device=device,
        gradient_clip=1.0,
        monitored_metric="val_macro_f1",
    )
    stage2_time = time.time() - t0
    logger.info("Stage 2 complete in %.1fs", stage2_time)

    # Save final checkpoint
    stage2_path = ckpt_dir / f"two_stage_{args.model}_final.pth"
    torch.save(model.state_dict(), stage2_path)
    logger.info("Final checkpoint saved: %s", stage2_path)

    # Evaluate Stage 2
    stage2_metrics = evaluate_and_collect(
        model,
        test_maps,
        y_test,
        val_tf,
        args.batch_size,
        args.seed,
        device,
        stage_name=f"Stage2-{args.model}",
    )
    stage2_metrics["time_sec"] = stage2_time

    # ============================================================ RESULTS
    logger.info("\n" + "=" * 70)
    logger.info("TWO-STAGE TRAINING RESULTS COMPARISON")
    logger.info("=" * 70)

    header = f"{'Stage':<30} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'ECE':<10} {'Time (s)':<10}"
    logger.info(header)
    logger.info("-" * len(header))

    for label, m in [
        ("Stage 1 (balanced pretrain)", stage1_metrics),
        ("Stage 2 (real fine-tuned)", stage2_metrics),
    ]:
        logger.info(
            "%-30s %-12.4f %-12.4f %-12.4f %-10.4f %-10.1f",
            label,
            m["accuracy"],
            m["macro_f1"],
            m["weighted_f1"],
            m["ece"],
            m["time_sec"],
        )

    # Per-class F1 comparison
    logger.info("\nPer-class F1 comparison:")
    logger.info("%-12s  %12s  %12s", "Class", "Stage 1 F1", "Stage 2 F1")
    logger.info("-" * 40)
    for cls in KNOWN_CLASSES:
        s1_f1 = stage1_metrics["per_class"][cls]["f1"]
        s2_f1 = stage2_metrics["per_class"][cls]["f1"]
        delta = s2_f1 - s1_f1
        logger.info("%-12s  %12.4f  %12.4f  (%+.4f)", cls, s1_f1, s2_f1, delta)

    # Save results JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)
    results = {
        "model": args.model,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "device": device,
        "seed": args.seed,
        "stage1_balanced_pretrain": stage1_metrics,
        "stage2_real_finetune": stage2_metrics,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "checkpoints": {
            "stage1": str(stage1_path),
            "final": str(stage2_path),
        },
        "total_time_sec": stage1_metrics["time_sec"] + stage2_metrics["time_sec"],
    }

    metrics_path = results_dir / "two_stage_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", metrics_path)

    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Two-stage training: balanced pretrain + real fine-tune"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "efficientnet"],
        default="cnn",
        help="Model architecture (default: cnn)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=30,
        help="Epochs for Stage 1 balanced pretraining (default: 30)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=20,
        help="Epochs for Stage 2 real fine-tuning (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/LSWMD_new.pkl"),
        help="Path to WM-811K pickle file",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    run_two_stage(args)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.exit(main())
