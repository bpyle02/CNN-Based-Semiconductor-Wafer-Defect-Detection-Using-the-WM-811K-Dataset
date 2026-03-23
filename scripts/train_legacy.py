#!/usr/bin/env python3
"""
CLI entry point for training wafer defect detection models.

Orchestrates the full training pipeline:
  1. Load and preprocess dataset
  2. Create train/val/test splits
  3. Instantiate models
  4. Train with learning rate scheduling
  5. Evaluate and report metrics
  6. Generate visualizations

Usage:
    python train.py --model cnn --epochs 5 --batch-size 64
    python train.py --model resnet --lr 1e-4
    python train.py --model effnet --device cuda
"""

import argparse
import sys
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.training import train_model, TrainConfig
from src.analysis import evaluate_model, count_params, count_trainable
from src.config import Config, load_config


KNOWN_CLASSES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]
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
    batch_size: int = 64,
) -> Tuple[Dict[str, Any], Dict[str, DataLoader], LabelEncoder]:
    """
    Load dataset, preprocess, and create train/val/test splits.

    Returns:
        Tuple of (data_dict, loaders_dict, label_encoder)
    """
    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)

    # Load raw dataset
    df = load_dataset(dataset_path)

    # Filter to known classes
    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)
    print(f"\nFiltered to {len(df_clean):,} labeled wafers (removed {len(df) - len(df_clean):,})")

    # Encode labels
    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])
    print(f"Classes: {le.classes_.tolist()}")

    # Extract data
    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    # Stratified train/val/test split
    temp_size = test_size + val_size
    if not 0 < temp_size < 1:
        raise ValueError("test_size + val_size must be between 0 and 1")

    X_train, X_temp, y_train, y_temp = train_test_split(
        np.arange(len(labels)), labels, test_size=temp_size,
        stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / temp_size,
        stratify=y_temp, random_state=seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(y_train):,}")
    print(f"  Val:   {len(y_val):,}")
    print(f"  Test:  {len(y_test):,}")

    # Preprocess (resize + normalize)
    print(f"\nPreprocessing maps to 96x96...")
    train_maps_raw = [wafer_maps[i] for i in X_train]
    val_maps_raw = [wafer_maps[i] for i in X_val]
    test_maps_raw = [wafer_maps[i] for i in X_test]

    train_maps = preprocess_wafer_maps(train_maps_raw)
    val_maps = preprocess_wafer_maps(val_maps_raw)
    test_maps = preprocess_wafer_maps(test_maps_raw)

    # Compute loss weights from training set
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [total_train / (len(KNOWN_CLASSES) * class_counts_train[c]) for c in range(len(KNOWN_CLASSES))],
        dtype=torch.float32
    )
    print(f"\nClass weights (loss): {[f'{w:.2f}' for w in loss_weights.tolist()]}")

    # Create datasets
    train_aug = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    train_transform_pre = transforms.Compose([train_aug, imagenet_norm])
    val_transform_pre = imagenet_norm

    # Datasets
    train_dataset_cnn = WaferMapDataset(train_maps, y_train, transform=train_aug)
    val_dataset_cnn = WaferMapDataset(val_maps, y_val, transform=None)
    test_dataset_cnn = WaferMapDataset(test_maps, y_test, transform=None)

    train_dataset_pre = WaferMapDataset(train_maps, y_train, transform=train_transform_pre)
    val_dataset_pre = WaferMapDataset(val_maps, y_val, transform=val_transform_pre)
    test_dataset_pre = WaferMapDataset(test_maps, y_test, transform=val_transform_pre)

    # DataLoaders
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False, num_workers=0)

    train_loader_pre = DataLoader(train_dataset_pre, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_pre = DataLoader(val_dataset_pre, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader_pre = DataLoader(test_dataset_pre, batch_size=batch_size, shuffle=False, num_workers=0)

    loaders = {
        'cnn': (train_loader_cnn, val_loader_cnn, test_loader_cnn),
        'resnet': (train_loader_pre, val_loader_pre, test_loader_pre),
        'effnet': (train_loader_pre, val_loader_pre, test_loader_pre),
    }

    data = {
        'class_names': le.classes_.tolist(),
        'loss_weights': loss_weights,
        'test_datasets': {
            'cnn': test_dataset_cnn,
            'resnet': test_dataset_pre,
            'effnet': test_dataset_pre,
        }
    }

    return data, loaders, le


def main():
    """Main CLI entry point."""
    # Load config
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(
        description='Train CNN models for semiconductor wafer defect detection'
    )
    parser.add_argument(
        '--model',
        choices=['cnn', 'resnet', 'effnet', 'all'],
        default=config.training.default_model,
        help=f'Model to train (default: {config.training.default_model})'
    )
    parser.add_argument('--epochs', type=int, default=config.training.epochs,
                        help=f'Training epochs (default: {config.training.epochs})')
    parser.add_argument('--batch-size', type=int, default=config.training.batch_size,
                        help=f'Batch size (default: {config.training.batch_size})')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (auto per model)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.device,
                        help=f'Device (default: {config.device})')
    parser.add_argument('--seed', type=int, default=config.seed,
                        help=f'Random seed (default: {config.seed})')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to dataset pickle')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--save-config', type=str, default=None,
                        help='Save final config to file after applying CLI overrides')

    args = parser.parse_args()

    # Reload config if custom path provided
    if args.config:
        config = load_config(args.config)

    # Override config with CLI args
    config.training.default_model = args.model
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.device = args.device
    config.seed = args.seed

    # Setup
    set_seed(config.seed)
    device = torch.device(config.device)
    print(f"\nDevice: {config.device}")
    print(f"Config loaded from: config.yaml (default_model={config.training.default_model}, "
          f"epochs={config.training.epochs}, batch_size={config.training.batch_size})")

    # Load data
    if args.data_path is None:
        args.data_path = Path(config.data.dataset_path)
        if not args.data_path.is_absolute():
            args.data_path = Path(__file__).parent / args.data_path

    data, loaders, le = load_and_preprocess(args.data_path,
                                            test_size=config.data.test_size,
                                            val_size=config.data.val_size,
                                            seed=config.seed,
                                            batch_size=config.training.batch_size)
    class_names = data['class_names']
    loss_weights = data['loss_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    models_to_train = ['cnn', 'resnet', 'effnet'] if config.training.default_model == 'all' else [config.training.default_model]
    results = {}

    for model_type in models_to_train:
        print("\n" + "="*70)
        print(f"TRAINING {model_type.upper()}")
        print("="*70)

        # Create model
        if model_type == 'cnn':
            model = WaferCNN(num_classes=len(class_names)).to(device)
            lr = args.lr or config.training.learning_rate.get('cnn', 1e-3)
            model_name = "Custom CNN"
        elif model_type == 'resnet':
            model = get_resnet18(num_classes=len(class_names)).to(device)
            lr = args.lr or config.training.learning_rate.get('resnet', 1e-4)
            model_name = "ResNet-18"
        else:  # effnet
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)
            lr = args.lr or config.training.learning_rate.get('efficientnet', 1e-4)
            model_name = "EfficientNet-B0"

        # Training setup
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=config.training.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Train
        train_loader, val_loader, test_loader = loaders[model_type]
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler=scheduler, epochs=config.training.epochs, model_name=model_name,
            device=config.device
        )

        # Evaluate
        preds, labels, metrics = evaluate_model(
            model, test_loader, class_names, model_name, str(device)
        )

        results[model_type] = {
            'model': model,
            'history': history,
            'predictions': preds,
            'labels': labels,
            'metrics': metrics,
            'model_name': model_name,
        }

        # Parameter counts
        total, trainable = count_params(model), count_trainable(model)
        print(f"{model_name} Parameters:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,}")

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    for mtype, res in results.items():
        metrics = res['metrics']
        print(f"{res['model_name']}:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  Time:        {res['history']['total_time']:.1f}s")

    # Save config if requested
    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"\nConfig saved to: {args.save_config}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
