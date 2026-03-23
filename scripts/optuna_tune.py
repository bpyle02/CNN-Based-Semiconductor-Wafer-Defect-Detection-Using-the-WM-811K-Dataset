#!/usr/bin/env python3
"""
Hyperparameter tuning with Optuna.

Searches over learning rate, batch size, dropout rate, and weight decay
to maximize macro F1 score on validation set.

Usage:
    python optuna_tune.py --model cnn --n-trials 50
    python optuna_tune.py --model all --n-trials 100 --n-jobs 4
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.training import train_model
from src.analysis import evaluate_model
from src.config import Config, load_config


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


def load_and_preprocess(
    dataset_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
) -> Tuple[Dict[str, Any], Dict[str, DataLoader], LabelEncoder]:
    """Load dataset, preprocess, and create loaders."""
    print("Loading data...")
    df = load_dataset(dataset_path)

    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)
    print(f"Loaded {len(df_clean):,} samples")

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.arange(len(labels)), labels, test_size=0.30,
        stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        stratify=y_temp, random_state=seed
    )

    # Preprocess
    train_maps_raw = [wafer_maps[i] for i in X_train]
    val_maps_raw = [wafer_maps[i] for i in X_val]

    train_maps = preprocess_wafer_maps(train_maps_raw)
    val_maps = preprocess_wafer_maps(val_maps_raw)

    # Compute loss weights
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [total_train / (len(KNOWN_CLASSES) * class_counts_train[c]) for c in range(len(KNOWN_CLASSES))],
        dtype=torch.float32
    )

    # Create datasets
    train_aug = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    train_dataset_cnn = WaferMapDataset(train_maps, y_train, transform=train_aug)
    val_dataset_cnn = WaferMapDataset(val_maps, y_val, transform=None)

    train_dataset_pre = WaferMapDataset(train_maps, y_train, transform=torch.nn.Sequential(train_aug, imagenet_norm))
    val_dataset_pre = WaferMapDataset(val_maps, y_val, transform=imagenet_norm)

    data = {
        'class_names': le.classes_.tolist(),
        'loss_weights': loss_weights,
    }

    return data, (train_dataset_cnn, val_dataset_cnn, train_dataset_pre, val_dataset_pre), le


class HyperparameterTuner:
    """Optuna-based hyperparameter tuner."""

    def __init__(
        self,
        model_type: str,
        train_dataset,
        val_dataset,
        class_names: list,
        loss_weights: torch.Tensor,
        device: str = 'cuda',
        seed: int = SEED,
    ):
        self.model_type = model_type
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.loss_weights = loss_weights
        self.device = torch.device(device)
        self.seed = seed
        self.trial_count = 0

    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna."""
        self.trial_count += 1

        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        print(f"\n  Trial {self.trial_count}: LR={lr:.2e}, BS={batch_size}, Dropout={dropout_rate:.2f}, WD={weight_decay:.2e}")

        try:
            # Create model
            if self.model_type == 'cnn':
                model = WaferCNN(num_classes=len(self.class_names), dropout_rate=dropout_rate)
            elif self.model_type == 'resnet':
                model = get_resnet18(num_classes=len(self.class_names))
            else:  # effnet
                model = get_efficientnet_b0(num_classes=len(self.class_names))

            model = model.to(self.device)

            # Create loaders
            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            # Setup training
            criterion = nn.CrossEntropyLoss(weight=self.loss_weights.to(self.device))
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

            # Train for limited epochs
            model.train()
            best_val_f1 = 0.0
            patience_counter = 0
            max_patience = 3

            for epoch in range(3):  # Short training for tuning
                # Train
                train_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validate
                model.eval()
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        outputs = model(inputs)
                        val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                        val_targets.extend(targets.numpy())

                val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
                print(f"    Epoch {epoch + 1}: Val Macro F1={val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break

                model.train()

            print(f"    Best Val F1: {best_val_f1:.4f}")
            return best_val_f1

        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0


def main():
    """Main tuning entry point."""
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument(
        '--model',
        choices=['cnn', 'resnet', 'effnet', 'all'],
        default='cnn',
        help='Model to tune'
    )
    parser.add_argument('--n-trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.device)
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--data-path', type=Path, default=None)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    print(f"\nDevice: {args.device}")
    print(f"Tuning with Optuna: {args.n_trials} trials")

    # Load data
    if args.data_path is None:
        args.data_path = Path(config.data.dataset_path)
        if not args.data_path.is_absolute():
            args.data_path = Path(__file__).parent / args.data_path

    data, datasets, le = load_and_preprocess(args.data_path)
    class_names = data['class_names']
    loss_weights = data['loss_weights']
    train_dataset_cnn, val_dataset_cnn, train_dataset_pre, val_dataset_pre = datasets

    models_to_tune = ['cnn', 'resnet', 'effnet'] if args.model == 'all' else [args.model]
    results = {}

    for model_type in models_to_tune:
        print(f"\n{'='*70}")
        print(f"TUNING {model_type.upper()}")
        print(f"{'='*70}")

        # Select datasets
        if model_type == 'cnn':
            train_ds, val_ds = train_dataset_cnn, val_dataset_cnn
        else:
            train_ds, val_ds = train_dataset_pre, val_dataset_pre

        # Create tuner
        tuner = HyperparameterTuner(
            model_type, train_ds, val_ds, class_names, loss_weights,
            device=args.device, seed=args.seed
        )

        # Run optimization
        sampler = TPESampler(seed=args.seed)
        study = optuna.create_study(
            sampler=sampler,
            direction='maximize'
        )

        study.optimize(
            tuner.objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )

        # Summary
        print(f"\nBest trial:")
        print(f"  Macro F1: {study.best_value:.4f}")
        print(f"  Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        results[model_type] = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
        }

    # Save results
    print(f"\n{'='*70}")
    print("TUNING COMPLETE")
    print(f"{'='*70}")
    for mtype, res in results.items():
        print(f"{mtype}: Best Macro F1 = {res['best_value']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
