"""
Federated learning server script for coordinating FedAvg training.

Usage:
    python -m src.federated.server --model cnn --num-rounds 10 --clients-per-round 5
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder

from src.federated.fed_avg import FedAveragingServer, FedAveragingClient, FedAvgConfig
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.model_registry import save_checkpoint_with_hash
from src.training.config import TrainConfig
from src.data.dataset import KNOWN_CLASSES
from src.data.preprocessing import seed_worker


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_client_loaders(
    train_dataset: torch.utils.data.Dataset,
    num_clients: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    non_iid_alpha: Optional[float] = None,
) -> List[DataLoader]:
    """Partition training data across clients with optional non-IID distribution.

    Non-IID partitioning (when alpha < inf):
        Samples are distributed using Dirichlet distribution to create
        realistic non-IID data (some clients specialize in certain classes).

    Args:
        train_dataset: PyTorch Dataset to partition
        num_clients: Number of clients
        batch_size: Batch size for each client loader
        shuffle: Whether to shuffle data within each client
        num_workers: Number of data loading workers
        non_iid_alpha: Dirichlet concentration parameter. Lower alpha → more non-IID.
                      If None, uses uniform (IID) partitioning.

    Returns:
        List of DataLoaders, one per client
    """
    num_samples = len(train_dataset)
    indices = np.arange(num_samples)

    if non_iid_alpha is None:
        # IID partitioning: each client gets equal random subset
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID partitioning using Dirichlet distribution
        if hasattr(train_dataset, 'targets'):
            labels = np.array(train_dataset.targets)
        elif hasattr(train_dataset, 'tensors'):
            # TensorDataset case
            labels = train_dataset.tensors[1].numpy()
        else:
            # Fallback: use uniform partitioning
            np.random.shuffle(indices)
            client_indices = np.array_split(indices, num_clients)
            client_loaders = [
                DataLoader(
                    Subset(train_dataset, subset),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(42),
                )
                for subset in client_indices
            ]
            return client_loaders

        # Dirichlet-based non-IID partitioning
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(num_clients)]

        # For each class, distribute samples using Dirichlet
        for class_idx in range(num_classes):
            class_samples = indices[labels[indices] == class_idx]
            np.random.shuffle(class_samples)

            # Draw proportions from Dirichlet
            proportions = np.random.dirichlet(
                np.repeat(non_iid_alpha, num_clients)
            )
            proportions = (np.cumsum(proportions) * len(class_samples)).astype(int)
            proportions[-1] = len(class_samples)

            # Assign to clients
            start = 0
            for client_id, end in enumerate(proportions):
                client_indices[client_id].extend(class_samples[start:end])
                start = end

    client_loaders = [
        DataLoader(
            Subset(train_dataset, subset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
        for subset in client_indices
    ]

    if len(client_loaders[0].dataset) > 0:
        logger.info(
            f"Created {num_clients} clients with {num_samples} total samples. "
            f"Client 0 has {len(client_loaders[0].dataset)} samples."
        )

    return client_loaders


def federated_train(
    model_name: str = "cnn",
    num_rounds: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    num_clients: int = 10,
    batch_size: int = 32,
    non_iid_alpha: Optional[float] = None,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    log_file: Optional[str] = None,
) -> Tuple[nn.Module, dict]:
    """Execute federated learning training.

    Args:
        model_name: 'cnn', 'resnet', or 'efficientnet'
        num_rounds: Number of federation rounds
        clients_per_round: Number of clients sampled per round
        local_epochs: Local SGD epochs per client per round
        learning_rate: Global learning rate
        num_clients: Total number of clients
        batch_size: Batch size for each client
        non_iid_alpha: Dirichlet alpha for non-IID partitioning (None = IID)
        device: 'cuda' or 'cpu'
        seed: Random seed
        checkpoint_dir: Directory to save checkpoints
        log_file: Optional file to save metrics

    Returns:
        Tuple of (trained_model, results_dict)
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger.info(f"Loading data and creating {num_clients} clients...")

    # Load or create training data (dummy data for testing)
    try:
        from src.data import (
            load_dataset,
            preprocess_wafer_maps,
            WaferMapDataset,
            get_image_transforms,
            get_imagenet_normalize,
        )
        import torchvision.transforms as transforms

        logger.info("Loading WM-811K dataset...")
        df = load_dataset()
        labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
        df = df[labeled_mask].reset_index(drop=True)
        label_encoder = LabelEncoder()
        raw_maps = df['waferMap'].values
        labels = label_encoder.fit_transform(df['failureClass'])

        # Preprocess
        processed_maps = preprocess_wafer_maps(raw_maps)
        train_transform = get_image_transforms()
        if model_name != "cnn":
            train_transform = transforms.Compose([train_transform, get_imagenet_normalize()])
        train_dataset = WaferMapDataset(processed_maps, labels, transform=train_transform)

        logger.info(f"Loaded dataset with {len(train_dataset)} labeled samples")
    except Exception as e:
        logger.warning(f"Could not load real dataset: {e}")
        logger.info("Creating synthetic dummy dataset for demonstration...")

        # Create dummy dataset (100 samples, 9 classes)
        X = torch.randn(100, 3, 96, 96)
        y = torch.randint(0, 9, (100,))
        train_dataset = torch.utils.data.TensorDataset(X, y)

    # Create client data loaders
    client_loaders = create_client_loaders(
        train_dataset=train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        shuffle=True,
        non_iid_alpha=non_iid_alpha,
    )

    # Create model
    logger.info(f"Creating {model_name} model...")
    if model_name == "cnn":
        global_model = WaferCNN(num_classes=9)
    elif model_name == "resnet":
        global_model = get_resnet18(num_classes=9, pretrained=False)
    elif model_name == "efficientnet":
        global_model = get_efficientnet_b0(num_classes=9, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create FedAvg config
    fed_config = FedAvgConfig(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        verbose=True,
    )

    # Create clients and server
    logger.info("Initializing federated clients...")
    clients = []
    for client_id, loader in enumerate(client_loaders):
        client_model = WaferCNN(num_classes=9) if model_name == "cnn" else (
            get_resnet18(num_classes=9, pretrained=False) if model_name == "resnet"
            else get_efficientnet_b0(num_classes=9, pretrained=False)
        )
        client = FedAveragingClient(
            client_id=client_id,
            train_loader=loader,
            model=client_model,
            criterion=criterion,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        clients.append(client)

    logger.info(f"Creating federated server with {len(clients)} clients...")
    server = FedAveragingServer(
        model=global_model,
        clients=clients,
        config=fed_config,
        test_loader=None,
    )

    # Train
    logger.info("Starting federated training...")
    results = server.train()

    # Save results with integrity hash
    model_path = checkpoint_path / f"fed_{model_name}_final.pth"
    file_hash = save_checkpoint_with_hash(results['global_model'].state_dict(), model_path)
    logger.info(f"Model saved to {model_path} (SHA-256: {file_hash[:16]}...)")

    # Save metrics
    metrics = {
        'model': model_name,
        'num_rounds': num_rounds,
        'clients_per_round': clients_per_round,
        'local_epochs': local_epochs,
        'num_clients': num_clients,
        'round_losses': results['round_loss'],
        'round_accuracies': results['round_acc'],
        'test_accuracies': results['test_acc'],
        'total_time': results['total_time'],
        'communication_rounds': results['communication_rounds'],
    }

    if log_file:
        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {log_file}")

    return results['global_model'], metrics


def main() -> None:
    """CLI entry point for federated training."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Training (FedAvg)"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "efficientnet"],
        default="cnn",
        help="Model architecture",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federation rounds",
    )
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=5,
        help="Number of clients sampled per round",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local SGD epochs per client",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Total number of clients",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per client",
    )
    parser.add_argument(
        "--non-iid-alpha",
        type=float,
        default=None,
        help="Dirichlet alpha for non-IID partitioning (lower = more non-IID)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="JSON file to save metrics",
    )

    args = parser.parse_args()

    model, metrics = federated_train(
        model_name=args.model,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        non_iid_alpha=args.non_iid_alpha,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
    )

    logger.info("Federated training complete!")
    logger.info(f"Final average accuracy: {metrics['round_accuracies'][-1]:.4f}")
    logger.info(f"Total time: {metrics['total_time']:.1f}s")


if __name__ == "__main__":
    main()
