#!/usr/bin/env python3
"""
K-Fold cross-validation for comprehensive model evaluation.

Provides robust performance estimates by training on multiple fold splits.
Supports stratified k-fold to preserve class distributions.

Usage:
    python cross_validate.py --model cnn --n-splits 5 --device cuda
    python cross_validate.py --model all --n-splits 10 --test-folds [0,1,2]
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.training import train_model
from src.analysis import evaluate_model
from src.config import load_config


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


def load_and_preprocess(dataset_path: Path, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load dataset and preprocess."""
    print("Loading data...")
    df = load_dataset(dataset_path)

    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    # Preprocess all maps
    print(f"Preprocessing {len(wafer_maps):,} maps...")
    maps_raw = [wafer_maps[i] for i in range(len(wafer_maps))]
    maps_processed = preprocess_wafer_maps(maps_raw)

    print(f"Loaded: {len(maps_processed):,} samples, {len(le.classes_)} classes")
    return maps_processed, labels, le.classes_.tolist()


def compute_fold_loss_weights(y_fold: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute loss weights for a fold."""
    class_counts = Counter(y_fold)
    total = len(y_fold)
    weights = torch.tensor(
        [total / (num_classes * class_counts[c]) for c in range(num_classes)],
        dtype=torch.float32
    )
    return weights


def cross_validate(
    model_type: str,
    maps: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    n_splits: int = 5,
    batch_size: int = 64,
    epochs: int = 5,
    device: str = 'cuda',
    seed: int = SEED,
) -> Dict[str, Any]:
    """Perform k-fold cross-validation."""
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION: {model_type.upper()} ({n_splits}-Fold)")
    print(f"{'='*70}")

    # Create fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = {
        'accuracy': [],
        'macro_f1': [],
        'weighted_f1': [],
    }

    for fold_idx, (train_idxs, val_idxs) in enumerate(skf.split(maps, labels)):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}")

        # Split data
        maps_train, maps_val = maps[train_idxs], maps[val_idxs]
        labels_train, labels_val = labels[train_idxs], labels[val_idxs]

        print(f"    Train: {len(labels_train):,}, Val: {len(labels_val):,}")

        # Create datasets
        if model_type == 'cnn':
            train_transform = get_image_transforms()
            val_transform = None
        else:  # pretrained
            train_transform = torch.nn.Sequential(get_image_transforms(), get_imagenet_normalize())
            val_transform = get_imagenet_normalize()

        train_dataset = WaferMapDataset(maps_train, labels_train, transform=train_transform)
        val_dataset = WaferMapDataset(maps_val, labels_val, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Create model
        if model_type == 'cnn':
            model = WaferCNN(num_classes=len(class_names)).to(device)
        elif model_type == 'resnet':
            model = get_resnet18(num_classes=len(class_names)).to(device)
        else:  # effnet
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)

        # Compute loss weights for this fold
        loss_weights = compute_fold_loss_weights(labels_train, len(class_names))
        criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))

        # Setup optimizer
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4 if model_type != 'cnn' else 1e-3,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Train
        model, _ = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler=scheduler, epochs=epochs,
            model_name=f"{model_type.upper()} (Fold {fold_idx + 1})",
            device=device
        )

        # Evaluate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        accuracy = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)

        fold_results['accuracy'].append(accuracy)
        fold_results['macro_f1'].append(macro_f1)
        fold_results['weighted_f1'].append(weighted_f1)

        print(f"    Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

    # Compute statistics
    print(f"\n  {'='*70}")
    print(f"  FOLD STATISTICS (Cross-Validation)")
    print(f"  {'='*70}")
    for metric in fold_results.keys():
        values = fold_results[metric]
        print(f"  {metric.upper()}:")
        print(f"    Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print(f"    Min:  {np.min(values):.4f}")
        print(f"    Max:  {np.max(values):.4f}")

    return fold_results


def main():
    """Main cross-validation entry point."""
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(
        description='K-fold cross-validation for model evaluation'
    )
    parser.add_argument(
        '--model',
        choices=['cnn', 'resnet', 'effnet', 'all'],
        default='cnn',
        help='Model to validate'
    )
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per fold')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.device)
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--data-path', type=Path, default=None)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    # Load data
    if args.data_path is None:
        args.data_path = Path(config.data.dataset_path)
        if not args.data_path.is_absolute():
            args.data_path = Path(__file__).parent / args.data_path

    maps, labels, class_names = load_and_preprocess(args.data_path, seed=args.seed)

    # Cross-validate
    models_to_validate = ['cnn', 'resnet', 'effnet'] if args.model == 'all' else [args.model]
    results = {}

    for model_type in models_to_validate:
        fold_results = cross_validate(
            model_type, maps, labels, class_names,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
        )
        results[model_type] = fold_results

    # Summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    for model_type, fold_results in results.items():
        print(f"\n{model_type.upper()}:")
        for metric in fold_results.keys():
            values = fold_results[metric]
            print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
