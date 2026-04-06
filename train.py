#!/usr/bin/env python3
"""
CLI entry point for training wafer defect models.

Usage:
    python train.py --model cnn --epochs 5 --device cuda
    python train.py --model all --epochs 5 --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_recall_fscore_support,
)
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.model_registry import save_checkpoint_with_hash

from src.config import load_config
from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset, get_image_transforms, get_imagenet_normalize, seed_worker
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.analysis import evaluate_model


KNOWN_CLASSES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]
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


def load_and_preprocess_data(dataset_path, test_size=0.15, val_size=0.15, seed=SEED):
    """Load and split data."""
    logger.info(f"\n{'='*70}")
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info(f"{'='*70}")

    logger.info("Loading dataset...")
    df = load_dataset(dataset_path)

    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    logger.info(f"Total samples: {len(labels):,}")
    logger.info(f"Class distribution:")
    for i, cls in enumerate(KNOWN_CLASSES):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        logger.info(f"  {cls:12s}: {count:6,} ({pct:5.1f}%)")

    # Split: train (70%), val (15%), test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)), labels, test_size=test_size,
        stratify=labels, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size),
        stratify=y_temp, random_state=seed
    )

    train_maps_raw = [wafer_maps[i] for i in X_train]
    val_maps_raw = [wafer_maps[i] for i in X_val]
    test_maps_raw = [wafer_maps[i] for i in X_test]

    logger.info(f"\nPreprocessing {len(train_maps_raw):,} training maps...")
    t0 = time.time()
    train_maps = np.array(preprocess_wafer_maps(train_maps_raw))
    logger.info(f"Preprocessing {len(val_maps_raw):,} validation maps...")
    val_maps = np.array(preprocess_wafer_maps(val_maps_raw))
    logger.info(f"Preprocessing {len(test_maps_raw):,} test maps...")
    test_maps = np.array(preprocess_wafer_maps(test_maps_raw))
    logger.info(f"Preprocessing complete in {time.time() - t0:.1f}s")

    # Compute class weights from training set
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [total_train / (len(KNOWN_CLASSES) * class_counts_train[c]) for c in range(len(KNOWN_CLASSES))],
        dtype=torch.float32
    )
    logger.info(f"\nClass weights (from training set):")
    logger.info(f"  {[f'{w:.2f}' for w in loss_weights.tolist()]}")

    return {
        'train_maps': train_maps, 'y_train': y_train,
        'val_maps': val_maps, 'y_val': y_val,
        'test_maps': test_maps, 'y_test': y_test,
        'loss_weights': loss_weights,
        'class_names': KNOWN_CLASSES,
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    """Train model with validation. Returns (model, epoch_history)."""
    best_acc = 0.0
    best_model = None
    history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_acc += (preds == labels).float().mean().item()

        val_acc /= len(val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}

        history.append({'epoch': epoch + 1, 'train_loss': float(train_loss), 'val_acc': float(val_acc)})
        logger.info(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

    # Restore best model
    if best_model:
        model.load_state_dict(best_model)

    return model, history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train wafer defect detection models')
    parser.add_argument('--model', choices=['cnn', 'resnet', 'effnet', 'all'], default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-path', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'),
                        help='Path to config.yaml (default: config.yaml)')
    args = parser.parse_args()

    # Load config.yaml defaults if available
    config = None
    if args.config and args.config.exists():
        config = load_config(str(args.config))
        logger.info(f"Loaded defaults from {args.config}")
    else:
        logger.info("No config.yaml found, using hardcoded defaults")

    # Resolve parameters: CLI > config.yaml > hardcoded defaults
    hw_default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_choice = args.model if args.model is not None else (
        config.training.default_model if config else 'cnn')
    # Map config's "efficientnet" to CLI's "effnet" if needed
    if model_choice == 'efficientnet':
        model_choice = 'effnet'

    epochs = args.epochs if args.epochs is not None else (
        config.training.epochs if config else 5)
    batch_size = args.batch_size if args.batch_size is not None else (
        config.training.batch_size if config else 64)
    device = args.device if args.device is not None else (
        config.device if config else hw_default_device)
    seed = args.seed if args.seed is not None else (
        config.seed if config else SEED)
    data_path = args.data_path if args.data_path is not None else (
        Path(config.data.dataset_path) if config else Path('data/LSWMD_new.pkl'))

    # Store resolved values back into args for downstream use
    args.model = model_choice
    args.epochs = epochs
    args.batch_size = batch_size
    args.device = device
    args.seed = seed
    args.data_path = data_path

    set_seed(args.seed)
    device = args.device
    logger.info(f"Device: {device}")
    logger.info(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"model={args.model}, seed={args.seed}")

    # Load data
    data = load_and_preprocess_data(args.data_path, seed=args.seed)
    train_maps = data['train_maps']
    y_train = data['y_train']
    val_maps = data['val_maps']
    y_val = data['y_val']
    test_maps = data['test_maps']
    y_test = data['y_test']
    loss_weights = data['loss_weights'].to(device)
    class_names = data['class_names']

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Create transforms
    try:
        import torchvision.transforms as tv_transforms
    except ImportError:
        tv_transforms = None

    train_aug = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    # Compose augmentation + ImageNet norm for pretrained training
    if tv_transforms is not None:
        pretrained_train_transform = tv_transforms.Compose([
            *train_aug.transforms,
            imagenet_norm,
        ])
        pretrained_val_transform = tv_transforms.Compose([imagenet_norm])
    else:
        pretrained_train_transform = imagenet_norm
        pretrained_val_transform = imagenet_norm

    # Output directories
    ckpt_dir = Path('checkpoints')
    results_dir = Path('results')
    ckpt_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    models_to_train = ['cnn', 'resnet', 'effnet'] if args.model == 'all' else [args.model]
    results = {}

    # Resolve learning rate: CLI --lr > config.yaml per-model > hardcoded default
    def _resolve_lr(model_key: str, hardcoded: float) -> float:
        if args.lr is not None:
            return args.lr
        if config and isinstance(config.training.learning_rate, dict):
            return config.training.learning_rate.get(model_key, hardcoded)
        if config and isinstance(config.training.learning_rate, (int, float)):
            return float(config.training.learning_rate)
        return hardcoded

    for model_name in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info(f"{'='*70}")

        # Create model with correct normalization per architecture
        if model_name == 'cnn':
            model = WaferCNN(num_classes=len(class_names)).to(device)
            display_name = "Custom CNN"
            lr = _resolve_lr('cnn', 1e-3)
            transforms_train = train_aug         # augmentation only
            transforms_val = None                # raw [0,1] images
        elif model_name == 'resnet':
            model = get_resnet18(num_classes=len(class_names)).to(device)
            display_name = "ResNet-18"
            lr = _resolve_lr('resnet', 1e-4)
            transforms_train = pretrained_train_transform  # augmentation + ImageNet norm
            transforms_val = pretrained_val_transform      # ImageNet norm only
        else:
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)
            display_name = "EfficientNet-B0"
            lr = _resolve_lr('efficientnet', 1e-4)
            transforms_train = pretrained_train_transform  # augmentation + ImageNet norm
            transforms_val = pretrained_val_transform      # ImageNet norm only

        # Create loaders
        train_dataset = WaferMapDataset(train_maps, y_train, transform=transforms_train)
        val_dataset = WaferMapDataset(val_maps, y_val, transform=transforms_val)
        test_dataset = WaferMapDataset(test_maps, y_test, transform=transforms_val)

        g = torch.Generator().manual_seed(42)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

        logger.info(f"Model: {display_name}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Batch size: {args.batch_size}")

        # Train
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        t0 = time.time()
        model, epoch_history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)
        train_time = time.time() - t0

        # Save checkpoint with integrity hash
        ckpt_path = ckpt_dir / f'best_{model_name}.pth'
        file_hash = save_checkpoint_with_hash(model.state_dict(), ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path} (SHA-256: {file_hash[:16]}...)")

        # Evaluate
        logger.info(f"\nEvaluating on test set...")
        preds, labels_arr, metrics = evaluate_model(model, test_loader, class_names, display_name, device)

        # Per-class metrics
        prec, rec, f1_per, support = precision_recall_fscore_support(
            labels_arr, preds, average=None, zero_division=0
        )
        per_class = {}
        for i, cls in enumerate(class_names):
            per_class[cls] = {
                'precision': float(prec[i]),
                'recall': float(rec[i]),
                'f1': float(f1_per[i]),
                'support': int(support[i]),
            }

        results[model_name] = {
            'display_name': display_name,
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1']),
            'time_sec': float(train_time),
            'per_class': per_class,
            'epoch_history': epoch_history,
        }

        logger.info(f"  Accuracy    : {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1    : {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
        logger.info(f"  Time        : {train_time:.1f}s")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"\n{'Model':<18} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Time (s)':<10}")
    logger.info("-" * 70)
    for model_name in models_to_train:
        r = results[model_name]
        logger.info(f"{model_name:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f} {r['time_sec']:<10.1f}")

    # Save metrics JSON
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")

    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(main())
