#!/usr/bin/env python3
"""
Active learning pipeline: iteratively select informative unlabeled samples.

Reduces labeled data requirements via uncertainty sampling.

Strategies:
- Entropy: -sum(p * log(p)) across predictions
- Margin: difference between top-2 class probabilities
- Least Confidence: 1 - max(p)

Workflow:
1. Start with small labeled pool
2. Train model on labeled data
3. Compute uncertainty on unlabeled data
4. Select most uncertain samples for labeling
5. Repeat

Usage:
    python active_learn.py --model cnn --initial-labeled 0.1 --acquisition-size 100
    python active_learn.py --model all --n-iterations 5 --strategy entropy
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.analysis import evaluate_model
from src.config import load_config
from src.training.base_trainer import BaseTrainer


KNOWN_CLASSES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]

def load_data_splits(
    dataset_path: Path,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load and split data into labeled pool and unlabeled pool."""
    print("Loading data...")
    df = load_dataset(dataset_path)

    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    # Split into training pool (for active learning) and test set
    X_pool, X_test, y_pool, y_test = train_test_split(
        np.arange(len(labels)), labels, test_size=0.20,
        stratify=labels, random_state=seed
    )

    pool_maps = [wafer_maps[i] for i in X_pool]
    test_maps = [wafer_maps[i] for i in X_test]

    # Preprocess
    pool_maps = np.array(preprocess_wafer_maps(pool_maps))
    test_maps = np.array(preprocess_wafer_maps(test_maps))

    print(f"Pool: {len(y_pool):,}, Test: {len(y_test):,}")

    return pool_maps, y_pool, test_maps, y_test, le.classes_.tolist()


def compute_uncertainty(
    model: nn.Module,
    dataset,
    batch_size: int = 64,
    device: str = 'cuda',
    method: str = 'entropy',
) -> np.ndarray:
    """Compute uncertainty scores for all samples."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    uncertainties = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            if method == 'entropy':
                # Entropy: -sum(p * log(p))
                unc = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            elif method == 'margin':
                # Margin: 1 - (p1 - p2)
                sorted_probs = torch.sort(probs, descending=True)[0]
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                unc = 1 - margin
            else:  # least confidence
                # 1 - max(p)
                unc = 1 - torch.max(probs, dim=1)[0]

            uncertainties.extend(unc.cpu().numpy())

    return np.array(uncertainties)


def select_uncertain_samples(
    uncertainties: np.ndarray,
    unlabeled_indices: np.ndarray,
    acquisition_size: int,
) -> List[int]:
    """Select most uncertain samples."""
    # Get top-k most uncertain unlabeled samples
    top_k_unlabeled = np.argsort(-uncertainties[unlabeled_indices])[:acquisition_size]
    selected = unlabeled_indices[top_k_unlabeled]
    return selected.tolist()


def main() -> int:
    """
    Main active learning entry point.

    Returns:
        Exit code (0 for success)
    """
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(description='Active learning pipeline')
    parser.add_argument(
        '--model',
        choices=['cnn', 'resnet', 'effnet', 'all'],
        default='cnn',
        help='Model to use'
    )
    parser.add_argument('--initial-labeled', type=float, default=0.1,
                        help='Fraction of pool to start with (0-1)')
    parser.add_argument('--acquisition-size', type=int, default=500,
                        help='Samples to label per iteration')
    parser.add_argument('--n-iterations', type=int, default=3,
                        help='Number of active learning iterations')
    parser.add_argument('--strategy', choices=['entropy', 'margin', 'least_confidence'],
                        default='entropy', help='Uncertainty strategy')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.device)
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--data-path', type=Path, default=None)

    args = parser.parse_args()

    # Setup
    if args.seed != config.seed:
        trainer.seed = args.seed
        trainer.set_seed()
    device = args.device
    print(f"Device: {device}")
    print(f"Active Learning: {args.strategy}, {args.n_iterations} iterations")

    # Load data
    if args.data_path is None:
        args.data_path = Path(config.data.dataset_path)
        if not args.data_path.is_absolute():
            args.data_path = Path(__file__).parent / args.data_path

    pool_maps, y_pool, test_maps, y_test, class_names = load_data_splits(args.data_path, seed=args.seed)

    # Compute loss weights (from full pool)
    class_counts = Counter(y_pool)
    total = len(y_pool)
    loss_weights = torch.tensor(
        [total / (len(KNOWN_CLASSES) * class_counts[c]) for c in range(len(KNOWN_CLASSES))],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Initialize labeled and unlabeled sets
    n_initial = int(len(y_pool) * args.initial_labeled)
    labeled_indices = np.random.choice(len(y_pool), n_initial, replace=False)
    unlabeled_indices = np.setdiff1d(np.arange(len(y_pool)), labeled_indices)

    print(f"Initial labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")

    # Active learning loop
    models_to_train = ['cnn', 'resnet', 'effnet'] if args.model == 'all' else [args.model]

    for model_type in models_to_train:
        print(f"\n{'='*70}")
        print(f"ACTIVE LEARNING: {model_type.upper()}")
        print(f"{'='*70}")

        # Create model
        if model_type == 'cnn':
            model = WaferCNN(num_classes=len(class_names)).to(device)
            model_name = "Custom CNN"
        elif model_type == 'resnet':
            model = get_resnet18(num_classes=len(class_names)).to(device)
            model_name = "ResNet-18"
        else:
            model = get_efficientnet_b0(num_classes=len(class_names)).to(device)
            model_name = "EfficientNet-B0"

        current_labeled = labeled_indices.copy()
        current_unlabeled = unlabeled_indices.copy()

        iteration_results = []

        for iteration in range(args.n_iterations):
            print(f"\n  Iteration {iteration + 1}/{args.n_iterations}")
            print(f"    Labeled: {len(current_labeled)}, Unlabeled: {len(current_unlabeled)}")

            # Create training dataset
            train_aug = get_image_transforms()
            imagenet_norm = get_imagenet_normalize()
            if model_type == 'cnn':
                train_transform = train_aug
            else:
                train_transform = transforms.Compose([train_aug, imagenet_norm])

            train_dataset = WaferMapDataset(
                pool_maps[current_labeled], y_pool[current_labeled], transform=train_transform
            )
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

            # Train model
            model.train()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4 if model_type != 'cnn' else 1e-3,
                weight_decay=1e-4
            )

            for epoch in range(2):  # Quick training
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            # Compute uncertainties on unlabeled data
            unlabeled_dataset = WaferMapDataset(
                pool_maps[current_unlabeled], y_pool[current_unlabeled], transform=None if model_type == 'cnn' else imagenet_norm
            )
            uncertainties = compute_uncertainty(
                model, unlabeled_dataset, device=device, method=args.strategy
            )

            # Select samples
            selected_idxs = select_uncertain_samples(
                uncertainties, np.arange(len(current_unlabeled)), args.acquisition_size
            )
            selected_pool_indices = current_unlabeled[selected_idxs]

            # Add to labeled set
            current_labeled = np.append(current_labeled, selected_pool_indices)
            current_unlabeled = np.setdiff1d(np.arange(len(y_pool)), current_labeled)

            # Evaluate on test set
            test_transform = None if model_type == 'cnn' else imagenet_norm
            test_dataset = WaferMapDataset(test_maps, y_test, transform=test_transform)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            model.eval()
            preds, labels, metrics = evaluate_model(
                model, test_loader, class_names, f"{model_name} (Iter {iteration + 1})", device
            )

            print(f"    Test Macro F1: {metrics['macro_f1']:.4f}")
            iteration_results.append({
                'iteration': iteration + 1,
                'labeled': len(current_labeled),
                'f1': metrics['macro_f1'],
                'accuracy': metrics['accuracy'],
            })

        # Summary
        print(f"\n  {model_name} Summary:")
        for res in iteration_results:
            print(f"    Iter {res['iteration']}: Labeled={res['labeled']:,}, Accuracy={res['accuracy']:.4f}, F1={res['f1']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
