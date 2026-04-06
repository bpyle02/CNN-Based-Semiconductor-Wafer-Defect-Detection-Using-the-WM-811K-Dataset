"""
Federated learning client script for local training and model synchronization.

A federated client:
    1. Receives global model from server
    2. Trains on local data for E epochs
    3. Uploads model update to server
    4. Repeats until training complete

Usage:
    # Server and client are typically co-located, but this script
    # demonstrates a client-server interaction pattern.
    python -m src.federated.client --client-id 0 --model-type cnn
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.federated.fed_avg import FedAveragingClient
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.data.preprocessing import seed_worker
from src.model_registry import save_checkpoint_with_hash, verify_checkpoint


logger = logging.getLogger(__name__)


class ClientManager:
    """Manages a federated client: training, sync, checkpoint operations.

    Attributes:
        client_id: Unique client identifier
        client: FedAveragingClient instance
        checkpoint_dir: Directory for saving/loading model checkpoints
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Initialize client manager.

        Args:
            client_id: Unique client identifier
            train_loader: DataLoader for local training data
            model: PyTorch model
            criterion: Loss function
            local_epochs: Number of local SGD epochs per round
            learning_rate: Learning rate for local optimizer
            device: Compute device
            checkpoint_dir: Directory for checkpoints
        """
        self.client_id = client_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.client = FedAveragingClient(
            client_id=client_id,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
        )

        self.round_num = 0
        self.metadata = {
            'client_id': client_id,
            'total_samples': self.client.num_samples,
            'rounds_completed': 0,
        }

    def receive_global_model(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Synchronize with global model from server.

        Args:
            global_weights: Model state_dict from server
        """
        self.client.set_weights(global_weights)
        logger.info(
            f"Client {self.client_id}: Received global model for round {self.round_num}"
        )

    def train_local(self, learning_rate: float) -> Dict[str, Any]:
        """Perform local SGD training.

        Args:
            learning_rate: Current learning rate (from server schedule)

        Returns:
            Dictionary with training metrics
        """
        logger.info(
            f"Client {self.client_id}: Starting local training "
            f"({self.client.local_epochs} epochs, lr={learning_rate:.4f})"
        )

        loss, acc = self.client.train_local(learning_rate)

        metrics = {
            'client_id': self.client_id,
            'round': self.round_num,
            'local_loss': loss,
            'local_acc': acc,
            'num_samples': self.client.num_samples,
        }

        logger.info(
            f"Client {self.client_id}: Loss={loss:.4f}, Acc={acc:.4f} "
            f"({self.client.num_samples} samples)"
        )

        self.round_num += 1
        self.metadata['rounds_completed'] = self.round_num

        return metrics

    def upload_model(self) -> Dict[str, torch.Tensor]:
        """Return model update for server aggregation.

        Returns:
            Model state_dict
        """
        weights = self.client.get_weights()
        logger.info(f"Client {self.client_id}: Uploading model update")
        return weights

    def save_checkpoint(self) -> str:
        """Save local model checkpoint with integrity hash.

        Returns:
            Path to saved checkpoint
        """
        path = self.checkpoint_dir / f"client_{self.client_id}_round_{self.round_num}.pth"
        checkpoint = {
            'model': self.client.model.state_dict(),
            'round': self.round_num,
            'loss_history': self.client.local_loss_history,
            'acc_history': self.client.local_acc_history,
            'metadata': self.metadata,
        }
        file_hash = save_checkpoint_with_hash(checkpoint, path)
        logger.info(
            f"Client {self.client_id}: Checkpoint saved to {path} "
            f"(SHA-256: {file_hash[:16]}...)"
        )
        return str(path)

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint with integrity verification.

        Args:
            path: Path to checkpoint file
        """
        checkpoint_path = Path(path)
        if not verify_checkpoint(checkpoint_path):
            logger.warning(
                f"Client {self.client_id}: Checkpoint integrity verification "
                f"FAILED for {path}. File may be corrupted or tampered with."
            )

        checkpoint = torch.load(path)
        self.client.model.load_state_dict(checkpoint['model'])
        self.round_num = checkpoint['round']
        self.metadata = checkpoint['metadata']
        logger.info(f"Client {self.client_id}: Checkpoint loaded from {path}")

    def get_metadata(self) -> Dict[str, Any]:
        """Return client metadata for server coordination.

        Returns:
            Dictionary with client info
        """
        return {
            **self.metadata,
            'current_round': self.round_num,
            'loss_history': self.client.local_loss_history,
            'acc_history': self.client.local_acc_history,
        }


def simulate_client_training(
    client_id: int = 0,
    model_type: str = "cnn",
    num_rounds: int = 10,
    local_epochs: int = 2,
    learning_rate: float = 0.01,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Simulate a federated client training loop (for testing/demonstration).

    This simulates receiving global models, training locally, and uploading updates.

    Args:
        client_id: Client identifier
        model_type: 'cnn', 'resnet', or 'efficientnet'
        num_rounds: Number of federation rounds to simulate
        local_epochs: Local SGD epochs per round
        learning_rate: Learning rate
        device: Compute device

    Returns:
        Dictionary with training history
    """
    logger.info(f"Simulating federated client {client_id}...")

    # Create dummy dataset (10 samples for speed)
    X = torch.randn(10, 3, 96, 96)
    y = torch.randint(0, 9, (10,))
    dataset = torch.utils.data.TensorDataset(X, y)
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=5, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # Create model
    if model_type == "cnn":
        model = WaferCNN(num_classes=9)
    elif model_type == "resnet":
        model = get_resnet18(num_classes=9, pretrained=False)
    elif model_type == "efficientnet":
        model = get_efficientnet_b0(num_classes=9, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.CrossEntropyLoss()

    # Create client manager
    manager = ClientManager(
        client_id=client_id,
        train_loader=loader,
        model=model,
        criterion=criterion,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    # Simulate rounds
    history = {
        'client_id': client_id,
        'rounds': [],
    }

    for round_num in range(num_rounds):
        # Simulate receiving global model (would come from server)
        global_weights = model.state_dict()
        manager.receive_global_model(global_weights)

        # Local training
        metrics = manager.train_local(learning_rate)
        history['rounds'].append(metrics)

        # Upload model (in real system, would send to server)
        weights = manager.upload_model()

        logger.info(f"Round {round_num + 1}/{num_rounds} complete")

    manager.save_checkpoint()

    logger.info(f"Client {client_id} simulation complete!")
    logger.info(f"Final loss: {history['rounds'][-1]['local_loss']:.4f}")
    logger.info(f"Final accuracy: {history['rounds'][-1]['local_acc']:.4f}")

    return history


def main() -> None:
    """CLI entry point for client simulation."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Client (Simulation)"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client identifier",
    )
    parser.add_argument(
        "--model-type",
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
        "--local-epochs",
        type=int,
        default=2,
        help="Local SGD epochs per round",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON file to save history (optional)",
    )

    args = parser.parse_args()

    history = simulate_client_training(
        client_id=args.client_id,
        model_type=args.model_type,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        device=args.device,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"History saved to {args.output}")


if __name__ == "__main__":
    main()
