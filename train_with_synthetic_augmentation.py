#!/usr/bin/env python3
"""
Training pipeline with synthetic data augmentation.

Integrates synthetic augmentation into the standard training workflow to
improve model performance on imbalanced datasets.

Example usage:
    # Use rule-based augmentation (fast, CPU)
    python train_with_synthetic_augmentation.py \\
        --model resnet --epochs 5 --augmentation rule-based

    # Use GAN-based augmentation (slower, better quality)
    python train_with_synthetic_augmentation.py \\
        --model cnn --epochs 10 --augmentation gan --gan-epochs 20

    # No augmentation (baseline)
    python train_with_synthetic_augmentation.py \\
        --model efficientnet --epochs 5 --augmentation none
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from src.data.dataset import load_dataset, KNOWN_CLASSES
from src.data.preprocessing import (
    preprocess_wafer_maps, WaferMapDataset, get_image_transforms, get_imagenet_normalize
)
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.training.trainer import train_model
from src.training.config import TrainConfig
from src.analysis.evaluate import evaluate_model
from src.analysis.visualize import plot_confusion_matrices, plot_training_curves
from src.augmentation.synthetic import (
    SyntheticDataAugmenter, balance_dataset_with_synthetic
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train with synthetic data augmentation"
    )
    parser.add_argument(
        '--model',
        choices=['cnn', 'resnet', 'efficientnet', 'all'],
        default='cnn',
        help='Model architecture (default: cnn)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--augmentation',
        choices=['none', 'rule-based', 'gan'],
        default='rule-based',
        help='Augmentation strategy (default: rule-based)'
    )
    parser.add_argument(
        '--gan-epochs',
        type=int,
        default=5,
        help='GAN training epochs (default: 5)'
    )
    parser.add_argument(
        '--target-samples-per-class',
        type=int,
        default=None,
        help='Target samples per class. If None, uses max class count.'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device (default: cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results_augmented',
        help='Output directory for results (default: ./results_augmented)'
    )

    return parser.parse_args()


def setup_seeds(seed: int) -> None:
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_and_preprocess_data(
    sample_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess wafer data."""
    print("\n=== Loading Dataset ===")
    df = load_dataset()

    # Filter to known classes
    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_filtered = df[labeled_mask].reset_index(drop=True)

    # Sample if needed
    if sample_size is not None and sample_size < len(df_filtered):
        print(f"Sampling {sample_size} examples for testing...")
        df_filtered = df_filtered.sample(n=sample_size, random_state=42)

    wafer_maps = np.array(df_filtered['waferMap'].tolist())
    labels = np.array([KNOWN_CLASSES.index(c) for c in df_filtered['failureClass']])

    print(f"Loaded {len(wafer_maps)} samples")
    print(f"Wafer shape: {wafer_maps[0].shape}")

    # Preprocess
    print("Preprocessing wafer maps...")
    preprocessed_maps = preprocess_wafer_maps(wafer_maps, target_size=(96, 96))
    preprocessed_maps = np.array(preprocessed_maps)

    return preprocessed_maps, labels


def augment_data_if_needed(
    maps: np.ndarray,
    labels: np.ndarray,
    augmentation_type: str,
    gan_epochs: int,
    target_samples: Optional[int],
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentation if requested."""
    if augmentation_type == 'none':
        return maps, labels

    print(f"\n=== Applying {augmentation_type} Augmentation ===")

    # Determine target
    if target_samples is None:
        from collections import Counter
        counts = Counter(labels)
        target_samples = max(counts.values())

    if augmentation_type == 'gan':
        # Train GAN
        augmenter = SyntheticDataAugmenter(
            generator_type='gan',
            image_size=96,
            device=device
        )
        print("Training GAN on wafer maps...")
        augmenter.train_generator(
            maps,
            epochs=gan_epochs,
            batch_size=32,
            verbose=True
        )
    else:  # rule-based
        augmenter = SyntheticDataAugmenter(
            generator_type='rule-based',
            image_size=96,
            device=device
        )

    # Augment
    augmented_maps, augmented_labels = augmenter.augment_dataset(
        maps, labels, target_samples, KNOWN_CLASSES
    )

    return augmented_maps, augmented_labels


def create_dataloaders(
    maps: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    model_type: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    # Stratified split
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(maps))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
    )

    train_maps, train_labels = maps[train_idx], labels[train_idx]
    val_maps, val_labels = maps[val_idx], labels[val_idx]
    test_maps, test_labels = maps[test_idx], labels[test_idx]

    # Transforms
    aug_transform = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    # Determine if using pretrained model
    use_pretrained = model_type in ['resnet', 'efficientnet']

    # Create datasets
    train_dataset = WaferMapDataset(
        list(train_maps),
        train_labels,
        transform=aug_transform
    )
    val_dataset = WaferMapDataset(
        list(val_maps),
        val_labels,
        transform=None
    )
    test_dataset = WaferMapDataset(
        list(test_maps),
        test_labels,
        transform=None
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_maps)} samples")
    print(f"  Val:   {len(val_maps)} samples")
    print(f"  Test:  {len(test_maps)} samples")

    return train_loader, val_loader, test_loader


def create_model(model_type: str, device: torch.device) -> nn.Module:
    """Create model."""
    if model_type == 'cnn':
        model = WaferCNN(num_classes=9)
    elif model_type == 'resnet':
        model = get_resnet18(num_classes=9, freeze_earlier_layers=True)
    elif model_type == 'efficientnet':
        model = get_efficientnet_b0(num_classes=9, freeze_earlier_layers=True)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return model.to(device)


def main() -> None:
    """Main execution."""
    args = parse_args()
    setup_seeds(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # Load data
    maps, labels = load_and_preprocess_data()

    # Show original distribution
    from collections import Counter
    counts = Counter(labels)
    print("\n--- Original Class Distribution ---")
    for class_idx in sorted(counts.keys()):
        print(f"  {KNOWN_CLASSES[class_idx]:12s}: {counts[class_idx]:5d}")

    # Augment if needed
    if args.augmentation != 'none':
        maps, labels = augment_data_if_needed(
            maps, labels,
            args.augmentation,
            args.gan_epochs,
            args.target_samples_per_class,
            device
        )

    # Show augmented distribution
    counts = Counter(labels)
    print("\n--- Final Class Distribution ---")
    for class_idx in sorted(counts.keys()):
        print(f"  {KNOWN_CLASSES[class_idx]:12s}: {counts[class_idx]:5d}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        maps, labels, args.batch_size, args.model
    )

    # Train models
    models_to_train = ['cnn', 'resnet', 'efficientnet'] if args.model == 'all' else [args.model]

    results = {}
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")

        model = create_model(model_type, device)

        # Setup training
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [12.04, 34.46, 28.16, 27.68, 41.80, 40.02, 38.57, 35.78, 0.28],
                dtype=torch.float32,
                device=device
            )
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr if model_type == 'cnn' else 1e-4,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        # Train
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=args.epochs, scheduler=scheduler, device=device,
            checkpoint_path=output_dir / f'{model_type}_best.pth'
        )

        # Evaluate
        print(f"\nEvaluating on test set...")
        test_preds, test_labels, metrics = evaluate_model(
            trained_model, test_loader, KNOWN_CLASSES, device
        )

        results[model_type] = {
            'model': trained_model,
            'history': history,
            'metrics': metrics,
            'predictions': test_preds,
            'true_labels': test_labels,
        }

        print(f"\n{model_type.upper()} Results:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    # Visualization
    if len(results) > 0:
        print("\n=== Visualizing Results ===")

        for model_type, res in results.items():
            # Training curves
            plot_training_curves(
                res['history'],
                save_path=output_dir / f'{model_type}_training_curves.png'
            )

            # Confusion matrix
            plot_confusion_matrices(
                res['predictions'],
                res['true_labels'],
                KNOWN_CLASSES,
                save_path=output_dir / f'{model_type}_confusion_matrix.png'
            )

    print(f"\n✓ Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
