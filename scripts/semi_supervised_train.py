#!/usr/bin/env python3
"""
FixMatch semi-supervised training script for wafer defect detection.

Leverages ALL wafer maps from WM-811K: labeled wafers (~172K) go through
standard supervised training while the remaining unlabeled wafers (~640K)
provide consistency regularization via pseudo-labels.

Usage:
    python scripts/semi_supervised_train.py --model cnn --epochs 50
    python scripts/semi_supervised_train.py --model resnet --threshold 0.95 --lambda-u 1.0
    python scripts/semi_supervised_train.py --model cnn --device cuda --epochs 100

References:
    [111] Sohn et al. (2020). "FixMatch". arXiv:2001.07685
    [117] Zhu et al. (2005). "Semi-Supervised Learning Literature Survey"
    [120] Lee (2013). "Pseudo-Label". arXiv:1908.02983
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import KNOWN_CLASSES, load_dataset
from src.data.preprocessing import (
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    seed_worker,
)
from src.models import WaferCNN, get_efficientnet_b0, get_resnet18
from src.training.losses import build_classification_loss
from src.training.semi_supervised import (
    FixMatchTrainer,
    UnlabeledWaferDataset,
    extract_unlabeled_maps,
    get_strong_transform,
    get_weak_transform,
)

logger = logging.getLogger(__name__)

SEED = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FixMatch semi-supervised training for wafer defect detection",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "resnet", "efficientnet"],
        help="Model architecture to train (default: cnn)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for labeled data (default: 64)",
    )
    parser.add_argument(
        "--unlabeled-batch-size",
        type=int,
        default=128,
        help="Batch size for unlabeled data (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3 for CNN, 1e-4 for pretrained)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Pseudo-label confidence threshold (default: 0.95)",
    )
    parser.add_argument(
        "--lambda-u",
        type=float,
        default=1.0,
        help="Weight for unsupervised loss (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to LSWMD_new.pkl (default: data/LSWMD_new.pkl)",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default: 0)",
    )
    return parser.parse_args()


def build_model(model_name: str, num_classes: int, device: str) -> tuple[nn.Module, str]:
    """Construct a model by name.

    Returns:
        (model, display_name) tuple.
    """
    if model_name == "cnn":
        model = WaferCNN(num_classes=num_classes)
        return model.to(device), "Custom CNN"

    if model_name == "resnet":
        model = get_resnet18(num_classes=num_classes)
        return model.to(device), "ResNet-18"

    if model_name == "efficientnet":
        model = get_efficientnet_b0(num_classes=num_classes)
        return model.to(device), "EfficientNet-B0"

    raise ValueError(f"Unknown model: {model_name}")


def get_transforms_for_model(
    model_name: str,
) -> tuple:
    """Return (train_transform, eval_transform) for the given model.

    Pretrained models need ImageNet normalization composed into the
    transform pipeline.
    """
    train_aug = get_image_transforms(augment=True, domain_augment=False)
    imagenet_norm = get_imagenet_normalize()

    if model_name in ("resnet", "efficientnet"):
        import torchvision.transforms as T

        # Compose augmentation + ImageNet normalization for pretrained models
        train_transform = T.Compose([train_aug, imagenet_norm])
        eval_transform = imagenet_norm
    else:
        train_transform = train_aug
        eval_transform = None

    return train_transform, eval_transform


def main() -> int:
    """Run FixMatch semi-supervised training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    set_seed(SEED)

    logger.info("=" * 70)
    logger.info("FIXMATCH SEMI-SUPERVISED TRAINING")
    logger.info("=" * 70)
    logger.info("Model:        %s", args.model)
    logger.info("Epochs:       %d", args.epochs)
    logger.info(
        "Batch size:   %d (labeled), %d (unlabeled)", args.batch_size, args.unlabeled_batch_size
    )
    logger.info("Threshold:    %.2f", args.threshold)
    logger.info("Lambda_u:     %.2f", args.lambda_u)
    logger.info("Device:       %s", args.device)

    # ------------------------------------------------------------------
    # 1. Load full dataset (ALL wafers, labeled + unlabeled)
    # ------------------------------------------------------------------
    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = REPO_ROOT / "data" / "LSWMD_new.pkl"
    else:
        dataset_path = Path(dataset_path)

    logger.info("\nLoading dataset from %s ...", dataset_path)
    df = load_dataset(dataset_path)
    logger.info("Total wafers in dataset: %d", len(df))

    # ------------------------------------------------------------------
    # 2. Separate labeled and unlabeled wafers
    # ------------------------------------------------------------------
    labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
    df_labeled = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    df_labeled["label_encoded"] = le.transform(df_labeled["failureClass"])

    labeled_maps = df_labeled["waferMap"].values
    labels = df_labeled["label_encoded"].values

    logger.info("\nLabeled wafers:   %d", len(labels))
    for i, cls in enumerate(KNOWN_CLASSES):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        logger.info("  %12s: %6d (%5.1f%%)", cls, count, pct)

    # Extract unlabeled maps
    unlabeled_maps = extract_unlabeled_maps(df, KNOWN_CLASSES)
    logger.info("Unlabeled wafers: %d", len(unlabeled_maps))

    # ------------------------------------------------------------------
    # 3. Split labeled data into train / val / test
    # ------------------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)),
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.15 / 0.85,
        stratify=y_temp,
        random_state=SEED,
    )

    train_maps = [labeled_maps[i] for i in X_train]
    val_maps = [labeled_maps[i] for i in X_val]
    test_maps = [labeled_maps[i] for i in X_test]

    logger.info("\nSplit sizes: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))

    # ------------------------------------------------------------------
    # 4. Create datasets and data loaders
    # ------------------------------------------------------------------
    train_transform, eval_transform = get_transforms_for_model(args.model)

    train_dataset = WaferMapDataset(train_maps, y_train, transform=train_transform)
    val_dataset = WaferMapDataset(val_maps, y_val, transform=eval_transform)
    test_dataset = WaferMapDataset(test_maps, y_test, transform=eval_transform)

    weak_t = get_weak_transform()
    strong_t = get_strong_transform()
    unlabeled_dataset = UnlabeledWaferDataset(
        unlabeled_maps,
        weak_transform=weak_t,
        strong_transform=strong_t,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    labeled_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.unlabeled_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(
        "DataLoaders: labeled=%d batches, unlabeled=%d batches, val=%d batches, test=%d batches",
        len(labeled_loader),
        len(unlabeled_loader),
        len(val_loader),
        len(test_loader),
    )

    # ------------------------------------------------------------------
    # 5. Build model, optimizer, loss, scheduler
    # ------------------------------------------------------------------
    num_classes = len(KNOWN_CLASSES)
    model, display_name = build_model(args.model, num_classes, args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("\n%s: %d params (%d trainable)", display_name, total_params, trainable_params)

    # Learning rate
    lr = args.lr
    if lr is None:
        lr = 1e-3 if args.model == "cnn" else 1e-4

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Class-weighted loss
    class_counts = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [total_train / (num_classes * class_counts[c]) for c in range(num_classes)],
        dtype=torch.float32,
    ).to(args.device)

    criterion = build_classification_loss(
        "CrossEntropyLoss",
        class_weights=loss_weights,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # ------------------------------------------------------------------
    # 6. Train with FixMatch
    # ------------------------------------------------------------------
    trainer = FixMatchTrainer(
        model=model,
        num_classes=num_classes,
        confidence_threshold=args.threshold,
        lambda_u=args.lambda_u,
        device=args.device,
    )

    trained_model, history = trainer.train(
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        scheduler=scheduler,
        gradient_clip=args.gradient_clip,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    trained_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(args.device)
            logits = trained_model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels_batch.tolist())

    test_acc = accuracy_score(all_targets, all_preds)
    test_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    logger.info("Test Accuracy:    %.4f", test_acc)
    logger.info("Test Macro F1:    %.4f", test_macro_f1)
    logger.info("Test Weighted F1: %.4f", test_weighted_f1)
    logger.info("\nClassification Report:")
    logger.info(
        "\n%s",
        classification_report(
            all_targets,
            all_preds,
            target_names=KNOWN_CLASSES,
            zero_division=0,
        ),
    )

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"fixmatch_{args.model}_best.pth"
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "model_name": args.model,
            "num_classes": num_classes,
            "class_names": KNOWN_CLASSES,
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
            "confidence_threshold": args.threshold,
            "lambda_u": args.lambda_u,
            "epochs": args.epochs,
        },
        checkpoint_path,
    )
    logger.info("Model checkpoint saved to %s", checkpoint_path)

    results_path = save_dir / f"fixmatch_{args.model}_results.json"
    results = {
        "model": args.model,
        "method": "fixmatch",
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "confidence_threshold": args.threshold,
        "lambda_u": args.lambda_u,
        "epochs_ran": history["epochs_ran"],
        "best_epoch": history["best_epoch"],
        "best_val_metric": history["best_metric"],
        "total_time_seconds": history["total_time"],
        "labeled_samples": len(y_train),
        "unlabeled_samples": len(unlabeled_maps),
        "history": {
            "train_loss": history["train_loss"],
            "val_macro_f1": history["val_macro_f1"],
            "pseudo_label_ratio": history["pseudo_label_ratio"],
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
