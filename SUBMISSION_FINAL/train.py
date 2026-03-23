#!/usr/bin/env python3
"""
CLI entry point for training wafer defect models.

Usage:
    python train.py --model cnn --epochs 5 --device cuda
    python train.py --model all --epochs 5 --device cpu
"""

import argparse
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset, get_image_transforms, get_imagenet_normalize
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
    print(f"\n{'='*70}")
    print("LOADING AND PREPROCESSING DATA")
    print(f"{'='*70}")

    print("Loading dataset...")
    df = load_dataset(dataset_path)

    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    print(f"Total samples: {len(labels):,}")
    print(f"Class distribution:")
    for i, cls in enumerate(KNOWN_CLASSES):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        print(f"  {cls:12s}: {count:6,} ({pct:5.1f}%)")

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

    print(f"\nPreprocessing {len(train_maps_raw):,} training maps...")
    t0 = time.time()
    train_maps = np.array(preprocess_wafer_maps(train_maps_raw))
    print(f"Preprocessing {len(val_maps_raw):,} validation maps...")
    val_maps = np.array(preprocess_wafer_maps(val_maps_raw))
    print(f"Preprocessing {len(test_maps_raw):,} test maps...")
    test_maps = np.array(preprocess_wafer_maps(test_maps_raw))
    print(f"Preprocessing complete in {time.time() - t0:.1f}s")

    # Compute class weights from training set
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [total_train / (len(KNOWN_CLASSES) * class_counts_train[c]) for c in range(len(KNOWN_CLASSES))],
        dtype=torch.float32
    )
    print(f"\nClass weights (from training set):")
    print(f"  {[f'{w:.2f}' for w in loss_weights.tolist()]}")

    return {
        'train_maps': train_maps, 'y_train': y_train,
        'val_maps': val_maps, 'y_val': y_val,
        'test_maps': test_maps, 'y_test': y_test,
        'loss_weights': loss_weights,
        'class_names': KNOWN_CLASSES,
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    """Train model with validation."""
    best_acc = 0.0
    best_model = None

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

        print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

    # Restore best model
    if best_model:
        model.load_state_dict(best_model)

    return model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train wafer defect detection models')
    parser.add_argument('--model', choices=['cnn', 'resnet', 'effnet', 'all'], default='cnn')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--data-path', type=Path, default=Path('data/LSWMD_new.pkl'))
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

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

    # Create datasets
    train_transform = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    models_to_train = ['cnn', 'resnet', 'effnet'] if args.model == 'all' else [args.model]
    results = {}

    for model_name in models_to_train:
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*70}")

        # Create model
        if model_name == 'cnn':
            model = WaferCNN(num_classes=len(class_names)).to(device)
            display_name = "Custom CNN"
            lr = args.lr or 1e-3
            transforms_train = train_transform
            transforms_val = None
        elif model_name == 'resnet':
            model = get_resnet18(num_classes=len(class_names)).to(device)
            display_name = "ResNet-18"
            lr = args.lr or 1e-4
            transforms_train = train_transform
            transforms_val = imagenet_norm
        else:
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)
            display_name = "EfficientNet-B0"
            lr = args.lr or 1e-4
            transforms_train = train_transform
            transforms_val = imagenet_norm

        # Create loaders
        train_dataset = WaferMapDataset(train_maps, y_train, transform=transforms_train)
        val_dataset = WaferMapDataset(val_maps, y_val, transform=transforms_val)
        test_dataset = WaferMapDataset(test_maps, y_test, transform=transforms_val)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print(f"Model: {display_name}")
        print(f"Learning rate: {lr}")
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Batch size: {args.batch_size}")

        # Train
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        t0 = time.time()
        model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)
        train_time = time.time() - t0

        # Evaluate
        print(f"\nEvaluating on test set...")
        preds, labels, metrics = evaluate_model(model, test_loader, class_names, display_name, device)

        results[model_name] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'time': train_time,
            'metrics': metrics
        }

        print(f"  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  Macro F1    : {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
        print(f"  Time        : {train_time:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<18} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Time (s)':<10}")
    print("-" * 70)
    for model_name in models_to_train:
        r = results[model_name]
        print(f"{model_name:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f} {r['time']:<10.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
