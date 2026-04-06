#!/usr/bin/env python3
"""
Active learning pipeline: iteratively select informative unlabeled samples.

Reduces labeled data requirements via uncertainty sampling, with optional
priority-weighted selection adapted from Prioritized Experience Replay
(Schaul et al., 2016).

Strategies:
- Entropy: -sum(p * log(p)) across predictions
- Margin: difference between top-2 class probabilities
- Least Confidence: 1 - max(p)

Priority modes (--prioritized):
- Uniform: standard top-k selection (default without --prioritized)
- Prioritized: samples drawn proportional to uncertainty^alpha with
  importance-sampling correction via beta annealing

Workflow:
1. Start with small labeled pool
2. Train model on labeled data
3. Compute uncertainty on unlabeled data
4. Select most uncertain samples for labeling (uniform or prioritized)
5. Repeat

Usage:
    python active_learn.py --model cnn --initial-labeled 0.1 --acquisition-size 100
    python active_learn.py --model all --n-iterations 5 --strategy entropy
    python active_learn.py --model cnn --prioritized --alpha 0.6 --beta-start 0.4
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset, seed_worker, KNOWN_CLASSES
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.analysis import evaluate_model
from src.config import load_config
from src.training.base_trainer import BaseTrainer


SEED = 42


# ============================================================================
# Prioritized Sample Selector
# ============================================================================

class PrioritizedSampleSelector:
    """Selects samples for labeling using priority-weighted sampling.

    Adapted from Prioritized Experience Replay (Schaul et al., 2016).
    Instead of uniform random selection from the unlabeled pool,
    samples are selected proportional to their 'informativeness' priority.

    Priority metrics (computed from existing acquisition functions):
    - entropy: prediction entropy (higher = more uncertain = more informative)
    - margin: 1 - (p1 - p2) (higher = more informative)
    - least_confidence: 1 - max(p) (higher = more informative)

    Uses the same alpha/beta annealing from prioritized replay:
    - alpha controls how much prioritization (0 = uniform, 1 = full priority)
    - beta controls importance sampling correction (anneals to 1.0)
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
    ) -> None:
        """
        Initialize the prioritized sample selector.

        Args:
            alpha: Priority exponent. 0 = uniform sampling, 1 = full prioritization.
            beta_start: Initial importance-sampling correction exponent.
            beta_end: Final beta value (annealed to over training).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= beta_start <= 1.0:
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")
        if not 0.0 <= beta_end <= 1.0:
            raise ValueError(f"beta_end must be in [0, 1], got {beta_end}")

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self._last_priorities: Optional[np.ndarray] = None
        self._last_weights: Optional[np.ndarray] = None

    def compute_priorities(
        self,
        model: nn.Module,
        unlabeled_dataset,
        device: str = "cuda",
        metric: str = "entropy",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Compute priority scores for all unlabeled samples.

        Args:
            model: Trained model to compute predictions with.
            unlabeled_dataset: Dataset of unlabeled samples.
            device: Torch device string.
            metric: Acquisition function ('entropy', 'margin', 'least_confidence').
            batch_size: Batch size for inference.

        Returns:
            Array of priority scores (higher = more informative), one per sample.
        """
        model.eval()
        g = torch.Generator().manual_seed(42)
        loader = DataLoader(
            unlabeled_dataset, batch_size=batch_size, shuffle=False,
            worker_init_fn=seed_worker, generator=g,
        )

        raw_scores: List[np.ndarray] = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                if metric == "entropy":
                    scores = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                elif metric == "margin":
                    sorted_probs = torch.sort(probs, descending=True)[0]
                    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                    scores = 1.0 - margin
                else:  # least_confidence
                    scores = 1.0 - torch.max(probs, dim=1)[0]

                raw_scores.append(scores.cpu().numpy())

        priorities = np.concatenate(raw_scores)
        # Shift to ensure all priorities are strictly positive (required for
        # proportional sampling). Small epsilon prevents zero-probability entries.
        priorities = priorities - priorities.min() + 1e-6
        self._last_priorities = priorities
        return priorities

    def select_batch(
        self,
        priorities: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select batch_size samples with probability proportional to priority^alpha.

        Mirrors the proportional priority sampling from PrioritizedReplayBuffer:
        probability_i = priority_i^alpha / sum(priority_j^alpha)

        Importance-sampling weights correct for the non-uniform distribution:
        w_i = (N * P(i))^(-beta) / max(w)

        Args:
            priorities: Array of priority scores (one per unlabeled sample).
            batch_size: Number of samples to select.

        Returns:
            Tuple of (selected_indices, importance_weights).
        """
        n = len(priorities)
        if batch_size > n:
            batch_size = n

        # Proportional priority sampling: P(i) = p_i^alpha / sum(p_j^alpha)
        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()

        # Sample without replacement
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # Importance-sampling weights: w_i = (N * P(i))^(-beta) / max_w
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        self._last_weights = weights
        return indices, weights

    def update_beta(self, step: int, total_steps: int) -> None:
        """Anneal beta from beta_start to beta_end over training.

        Args:
            step: Current step (0-indexed).
            total_steps: Total number of steps to anneal over.
        """
        if total_steps <= 1:
            self.beta = self.beta_end
        else:
            fraction = min(step / (total_steps - 1), 1.0)
            self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

    def get_stats(self) -> Dict[str, float]:
        """Return statistics about the most recent selection."""
        stats: Dict[str, float] = {"alpha": self.alpha, "beta": self.beta}
        if self._last_priorities is not None:
            stats["priority_mean"] = float(self._last_priorities.mean())
            stats["priority_max"] = float(self._last_priorities.max())
            stats["priority_min"] = float(self._last_priorities.min())
            stats["priority_std"] = float(self._last_priorities.std())
        if self._last_weights is not None:
            stats["weight_mean"] = float(self._last_weights.mean())
            stats["weight_min"] = float(self._last_weights.min())
        return stats

def load_data_splits(
    dataset_path: Path,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load and split data into labeled pool and unlabeled pool."""
    logger.info("Loading data...")
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

    logger.info(f"Pool: {len(y_pool):,}, Test: {len(y_test):,}")

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
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

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
    parser.add_argument('--prioritized', action='store_true',
                        help='Use priority-weighted sampling instead of top-k selection')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Priority exponent (0=uniform, 1=full priority). Only with --prioritized')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Initial importance-sampling beta. Only with --prioritized')
    parser.add_argument('--beta-end', type=float, default=1.0,
                        help='Final beta value (annealed over iterations). Only with --prioritized')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.device)
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--data-path', type=Path, default=None)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    logger.info(f"Device: {device}")
    mode_str = "prioritized" if args.prioritized else "top-k"
    logger.info(f"Active Learning: {args.strategy} ({mode_str}), {args.n_iterations} iterations")

    # Initialize prioritized selector if requested
    selector: Optional[PrioritizedSampleSelector] = None
    if args.prioritized:
        selector = PrioritizedSampleSelector(
            alpha=args.alpha,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        )
        logger.info(f"Prioritized sampling: alpha={args.alpha}, "
                     f"beta={args.beta_start}->{args.beta_end}")

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

    logger.info(f"Initial labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")

    # Active learning loop
    models_to_train = ['cnn', 'resnet', 'effnet'] if args.model == 'all' else [args.model]

    for model_type in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"ACTIVE LEARNING: {model_type.upper()}")
        logger.info(f"{'='*70}")

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
            logger.info(f"\n  Iteration {iteration + 1}/{args.n_iterations}")
            logger.info(f"    Labeled: {len(current_labeled)}, Unlabeled: {len(current_unlabeled)}")

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
            g_train = torch.Generator().manual_seed(42)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g_train)

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

            # Select samples: prioritized or standard top-k
            if selector is not None:
                priorities = selector.compute_priorities(
                    model, unlabeled_dataset, device=device,
                    metric=args.strategy, batch_size=64,
                )
                selected_idxs, importance_weights = selector.select_batch(
                    priorities, args.acquisition_size,
                )
                # Anneal beta toward 1.0 over all iterations
                selector.update_beta(iteration + 1, args.n_iterations)

                # Log priority statistics
                stats = selector.get_stats()
                logger.info(f"    Priority stats: mean={stats['priority_mean']:.4f}, "
                            f"max={stats['priority_max']:.4f}, min={stats['priority_min']:.4f}, "
                            f"std={stats['priority_std']:.4f}")
                logger.info(f"    IS weights: mean={stats['weight_mean']:.4f}, "
                            f"min={stats['weight_min']:.4f}, beta={stats['beta']:.4f}")
            else:
                uncertainties = compute_uncertainty(
                    model, unlabeled_dataset, device=device, method=args.strategy
                )
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
            g_test = torch.Generator().manual_seed(42)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g_test)

            model.eval()
            preds, labels, metrics = evaluate_model(
                model, test_loader, class_names, f"{model_name} (Iter {iteration + 1})", device
            )

            logger.info(f"    Test Macro F1: {metrics['macro_f1']:.4f}")
            iteration_results.append({
                'iteration': iteration + 1,
                'labeled': len(current_labeled),
                'f1': metrics['macro_f1'],
                'accuracy': metrics['accuracy'],
            })

        # Summary
        logger.info(f"\n  {model_name} Summary:")
        for res in iteration_results:
            logger.info(f"    Iter {res['iteration']}: Labeled={res['labeled']:,}, Accuracy={res['accuracy']:.4f}, F1={res['f1']:.4f}")

    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(main())
