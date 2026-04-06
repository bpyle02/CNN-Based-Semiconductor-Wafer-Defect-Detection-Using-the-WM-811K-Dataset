"""
Federated Averaging (FedAvg) algorithm implementation.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (ICML 2017)

FedAvg protocol:
    1. Server initializes global model
    2. Each round: Sample subset of clients
    3. Each client trains on local data (E epochs, B batch size)
    4. Clients send model updates to server
    5. Server computes weighted average of client models
    6. Server sends updated model back to all clients

This implementation supports:
    - Configurable number of rounds and client sampling
    - Local SGD on each client
    - Weighted averaging by client data size
    - Learning rate scheduling
    - Model checkpointing
    - Evaluation on validation/test sets
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model_registry import save_checkpoint_with_hash, verify_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class FedAvgConfig:
    """Configuration for Federated Averaging training.

    Attributes:
        num_rounds: Number of federation rounds (default: 10)
        clients_per_round: Number of clients sampled per round (default: 10)
        local_epochs: Local SGD epochs on each client (default: 1)
        learning_rate: Global learning rate (default: 0.01)
        learning_rate_schedule: 'constant' or 'exponential' (default: 'constant')
        lr_decay_rate: Decay factor for exponential schedule (default: 0.99)
        device: Compute device ('cuda' or 'cpu')
        seed: Random seed for reproducibility (default: 42)
        verbose: Enable logging (default: True)
    """

    num_rounds: int = 10
    clients_per_round: int = 10
    local_epochs: int = 1
    learning_rate: float = 0.01
    learning_rate_schedule: str = "constant"
    lr_decay_rate: float = 0.99
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True
    aggregation_method: str = "weighted_avg"
    trim_ratio: float = 0.2
    byzantine_tolerance: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")
        if self.clients_per_round < 1:
            raise ValueError(f"clients_per_round must be >= 1, got {self.clients_per_round}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {self.local_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.learning_rate_schedule not in ("constant", "exponential"):
            raise ValueError(
                f"learning_rate_schedule must be 'constant' or 'exponential', "
                f"got {self.learning_rate_schedule}"
            )
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")
        if self.aggregation_method not in ("weighted_avg", "median", "trimmed_mean", "krum"):
            raise ValueError(
                "aggregation_method must be one of "
                "'weighted_avg', 'median', 'trimmed_mean', or 'krum'"
            )
        if not 0.0 <= self.trim_ratio < 0.5:
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {self.trim_ratio}")
        if self.byzantine_tolerance < 0:
            raise ValueError(
                f"byzantine_tolerance must be >= 0, got {self.byzantine_tolerance}"
            )


class FedAveragingClient:
    """Federated learning client performing local SGD.

    Each client maintains a copy of the global model and trains on its local data
    for E epochs. After training, the client sends its model parameters to the server.

    Attributes:
        client_id: Unique identifier for this client
        train_loader: DataLoader for local training data
        val_loader: Optional DataLoader for local validation
        model: PyTorch model (will be synced with global model each round)
        criterion: Loss function
        local_epochs: Number of local SGD epochs per round
        learning_rate: Initial learning rate
        device: Compute device
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
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initialize a federated client.

        Args:
            client_id: Unique client identifier
            train_loader: DataLoader for local training data
            model: PyTorch model (will be updated in place)
            criterion: Loss function
            local_epochs: Number of local SGD epochs
            learning_rate: Learning rate for local optimizer
            device: Compute device
            val_loader: Optional validation loader for local evaluation
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Track metrics
        self.local_loss_history: List[float] = []
        self.local_acc_history: List[float] = []
        self.num_samples = sum(len(batch[0]) for batch in train_loader)

    def train_local(self, current_lr: float) -> Tuple[float, float]:
        """Perform local SGD training for E epochs.

        Training procedure (per McMahan et al.):
            1. Load current global model from server (weights synced)
            2. For each local epoch:
               a. For each batch in local dataset:
                  - Forward pass
                  - Compute loss
                  - Backward pass
                  - SGD step
            3. Return updated model weights and training loss/accuracy

        Args:
            current_lr: Current learning rate (may be scheduled by server)

        Returns:
            Tuple of (avg_loss, accuracy) over local training
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                epoch_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += labels.size(0)

            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

            self.local_loss_history.append(epoch_loss)
            self.local_acc_history.append(epoch_acc)

        avg_loss = total_loss / self.local_epochs
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return copy of model weights (for uploading to server).

        Returns:
            Dictionary mapping parameter names to tensors
        """
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Update model with weights from server (global model).

        Args:
            weights: Dictionary mapping parameter names to tensors
        """
        self.model.load_state_dict(weights)

    def validate_local(self) -> Optional[float]:
        """Evaluate model on local validation set (optional).

        Returns:
            Validation accuracy if val_loader provided, else None
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        return val_acc


class ByzantineRobustAggregator:
    """Robust aggregation resistant to model poisoning attacks."""

    def __init__(
        self,
        method: str = 'median',
        trim_ratio: float = 0.2,
        byzantine_tolerance: int = 0,
    ) -> None:
        """
        method: 'median' (robust to 50% malicious),
                'trimmed_mean' (robust to 30% malicious),
                'krum' (robust to n-f clients, where f < n/3)
        """
        self.method = method
        self.trim_ratio = trim_ratio
        self.byzantine_tolerance = byzantine_tolerance

    def aggregate(self, client_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate client weights robustly."""
        if not client_weights:
            raise ValueError("client_weights cannot be empty")

        if self.method == 'median':
            return self._median_aggregation(client_weights)
        elif self.method == 'trimmed_mean':
            return self._trimmed_mean_aggregation(
                client_weights,
                trim_ratio=self.trim_ratio,
            )
        elif self.method == 'krum':
            return self._krum_aggregation(
                client_weights,
                byzantine_tolerance=self.byzantine_tolerance,
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

    def _median_aggregation(self, weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Median aggregation - robust to 50% poisoned clients."""
        aggregated = {}
        for key in weights_list[0].keys():
            values = torch.stack([w[key] for w in weights_list])
            aggregated[key] = torch.median(values, dim=0)[0]
        return aggregated

    def _trimmed_mean_aggregation(self, weights_list: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation - removes outlier updates."""
        aggregated = {}
        trim_count = int(len(weights_list) * trim_ratio)
        if trim_count * 2 >= len(weights_list):
            raise ValueError(
                "trim_ratio removes all client updates; reduce trim_ratio or "
                "increase clients_per_round"
            )

        for key in weights_list[0].keys():
            values = torch.stack([w[key] for w in weights_list])
            sorted_vals = torch.sort(values, dim=0)[0]
            if trim_count > 0:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            aggregated[key] = torch.mean(trimmed, dim=0)
        return aggregated

    @staticmethod
    def _flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten a state dict into a single vector for distance-based scoring."""
        return torch.cat([tensor.detach().float().reshape(-1) for tensor in weights.values()])

    def _krum_aggregation(
        self,
        weights_list: List[Dict[str, torch.Tensor]],
        byzantine_tolerance: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Krum aggregation using nearest-neighbor distance scoring."""
        num_clients = len(weights_list)
        min_required_clients = 2 * byzantine_tolerance + 3
        if num_clients < min_required_clients:
            raise ValueError(
                f"Krum requires at least {min_required_clients} clients for "
                f"byzantine_tolerance={byzantine_tolerance}, got {num_clients}"
            )

        flattened_weights = [self._flatten_weights(weights) for weights in weights_list]
        neighbor_count = num_clients - byzantine_tolerance - 2
        scores = torch.empty(num_clients, dtype=torch.float64)

        for i in range(num_clients):
            pairwise_distances = []
            for j in range(num_clients):
                if i == j:
                    continue
                distance = torch.norm(flattened_weights[i] - flattened_weights[j], p=2).pow(2)
                pairwise_distances.append(distance)

            nearest_distances = torch.sort(torch.stack(pairwise_distances))[0][:neighbor_count]
            scores[i] = nearest_distances.sum().item()

        best_idx = int(torch.argmin(scores).item())
        return copy.deepcopy(weights_list[best_idx])


class FedAveragingServer:
    """Federated learning server coordinating global model aggregation.

    Server responsibilities:
        1. Initialize and maintain global model
        2. Sample subset of clients each round
        3. Send global model to clients
        4. Receive model updates from clients
        5. Aggregate updates via weighted averaging
        6. Update global model
        7. Optional: evaluate on test set

    FedAvg Aggregation (weighted by client sample size):
        w_t+1 = Sum(n_k / n_total * w_k_t)
        where n_k = number of samples on client k
              n_total = total samples across all clients
              w_k_t = weights trained on client k at round t

    Attributes:
        model: Global PyTorch model
        clients: List of FedAveragingClient objects
        config: FedAvgConfig
        test_loader: Optional test set for server-side evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        clients: List[FedAveragingClient],
        config: FedAvgConfig,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initialize federated server.

        Args:
            model: Global PyTorch model (will be copied to clients)
            clients: List of FedAveragingClient objects
            config: FedAvgConfig with hyperparameters
            test_loader: Optional test loader for global evaluation
        """
        if config.clients_per_round > len(clients):
            raise ValueError(
                f"clients_per_round ({config.clients_per_round}) cannot exceed "
                f"number of clients ({len(clients)})"
            )

        self.global_model = model
        self.clients = clients
        self.config = config
        self.test_loader = test_loader
        self.device = config.device

        # Ensure all clients have same initial weights
        global_weights = self.global_model.state_dict()
        for client in self.clients:
            client.set_weights(copy.deepcopy(global_weights))

        # Set random seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Track metrics
        self.round_loss_history: List[float] = []
        self.round_acc_history: List[float] = []
        self.test_acc_history: List[float] = []
        self.communication_rounds = 0

        if config.verbose:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)

    def select_clients(self, num_clients: int) -> List[int]:
        """Sample subset of clients for current round.

        Args:
            num_clients: Number of clients to sample

        Returns:
            List of client indices
        """
        return list(np.random.choice(len(self.clients), num_clients, replace=False))

    def get_learning_rate(self, round_num: int) -> float:
        """Compute learning rate for current round (with optional scheduling).

        Args:
            round_num: Current federation round (0-indexed)

        Returns:
            Learning rate for this round
        """
        if self.config.learning_rate_schedule == "constant":
            return self.config.learning_rate
        elif self.config.learning_rate_schedule == "exponential":
            return self.config.learning_rate * (self.config.lr_decay_rate ** round_num)
        else:
            return self.config.learning_rate

    def aggregate_weights(
        self, client_indices: List[int], client_weights: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute average of client model weights."""
        
        if getattr(self.config, 'aggregation_method', 'weighted_avg') != 'weighted_avg':
            # Use Byzantine Robust Aggregator
            aggregator = ByzantineRobustAggregator(
                method=self.config.aggregation_method,
                trim_ratio=self.config.trim_ratio,
                byzantine_tolerance=self.config.byzantine_tolerance,
            )
            return aggregator.aggregate(client_weights)

        # Default weighted average
        selected_clients = [self.clients[idx] for idx in client_indices]
        total_samples = sum(client.num_samples for client in selected_clients)

        if total_samples == 0:
            raise ValueError("Selected clients have zero total samples")

        avg_weights = copy.deepcopy(client_weights[0])
        weight_scale = selected_clients[0].num_samples / total_samples

        for key in avg_weights:
            avg_weights[key] = avg_weights[key] * weight_scale

        for i in range(1, len(client_indices)):
            weight_scale = selected_clients[i].num_samples / total_samples
            for key in avg_weights:
                avg_weights[key] += client_weights[i][key] * weight_scale

        return avg_weights

    def evaluate_global(self) -> Optional[float]:
        """Evaluate global model on test set (optional).

        Returns:
            Test accuracy if test_loader provided, else None
        """
        if self.test_loader is None:
            return None

        self.global_model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = outputs.max(1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total
        self.test_acc_history.append(test_acc)
        return test_acc

    def train_round(self, round_num: int) -> Tuple[float, float, Optional[float]]:
        """Execute one federation round.

        Protocol:
            1. Sample clients_per_round clients
            2. Send global model to sampled clients
            3. Each client trains locally for local_epochs
            4. Server receives model updates
            5. Server aggregates via weighted averaging
            6. Optional: evaluate on test set

        Args:
            round_num: Current round number (0-indexed)

        Returns:
            Tuple of (avg_loss, avg_acc, test_acc) for this round
        """
        start_time = time.time()

        # Sample clients
        client_indices = self.select_clients(self.config.clients_per_round)
        sampled_clients = [self.clients[idx] for idx in client_indices]

        # Send global model to sampled clients
        global_weights = self.global_model.state_dict()
        for client in sampled_clients:
            client.set_weights(copy.deepcopy(global_weights))

        # Clients train locally
        current_lr = self.get_learning_rate(round_num)
        local_losses = []
        local_accs = []
        client_weights_list = []

        for client in sampled_clients:
            loss, acc = client.train_local(current_lr)
            local_losses.append(loss)
            local_accs.append(acc)
            client_weights_list.append(client.get_weights())

        # Server aggregates
        avg_weights = self.aggregate_weights(client_indices, client_weights_list)
        self.global_model.load_state_dict(avg_weights)

        # Metrics
        round_loss = np.mean(local_losses)
        round_acc = np.mean(local_accs)
        self.round_loss_history.append(round_loss)
        self.round_acc_history.append(round_acc)

        # Optional: evaluate on test set
        test_acc = self.evaluate_global()

        elapsed = time.time() - start_time
        self.communication_rounds += 1

        # Logging
        if self.config.verbose:
            msg = (
                f"Round {round_num + 1}/{self.config.num_rounds} | "
                f"Loss: {round_loss:.4f}, Acc: {round_acc:.4f}"
            )
            if test_acc is not None:
                msg += f", Test Acc: {test_acc:.4f}"
            msg += f" | Time: {elapsed:.1f}s"
            logger.info(msg)

        return round_loss, round_acc, test_acc

    def train(self) -> Dict[str, Any]:
        """Execute full federated training for all rounds.

        Returns:
            Dictionary containing:
                - 'global_model': Final trained global model
                - 'round_loss': List of average losses per round
                - 'round_acc': List of average accuracies per round
                - 'test_acc': List of test accuracies per round (if test_loader provided)
                - 'total_time': Total training time in seconds
                - 'communication_rounds': Total federation rounds executed
        """
        if self.config.verbose:
            logger.info(
                f"Starting FedAvg training: {self.config.num_rounds} rounds, "
                f"{self.config.clients_per_round} clients/round, "
                f"{self.config.local_epochs} local epochs"
            )

        start_time = time.time()

        for round_num in range(self.config.num_rounds):
            self.train_round(round_num)

        elapsed = time.time() - start_time

        if self.config.verbose:
            final_test_acc = (
                self.test_acc_history[-1] if self.test_acc_history else None
            )
            msg = (
                f"FedAvg training complete in {elapsed:.1f}s. "
                f"Final avg acc: {self.round_acc_history[-1]:.4f}"
            )
            if final_test_acc is not None:
                msg += f", Test acc: {final_test_acc:.4f}"
            logger.info(msg)

        return {
            'global_model': self.global_model,
            'round_loss': self.round_loss_history,
            'round_acc': self.round_acc_history,
            'test_acc': self.test_acc_history,
            'total_time': elapsed,
            'communication_rounds': self.communication_rounds,
        }

    def get_global_model(self) -> nn.Module:
        """Return the trained global model.

        Returns:
            PyTorch model
        """
        return self.global_model

    def save_checkpoint(self, path: str) -> None:
        """Save global model checkpoint with integrity hash.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'global_model': self.global_model.state_dict(),
            'round_loss': self.round_loss_history,
            'round_acc': self.round_acc_history,
            'test_acc': self.test_acc_history,
            'config': self.config.__dict__,
        }
        file_hash = save_checkpoint_with_hash(checkpoint, Path(path))
        if self.config.verbose:
            logger.info(f"Checkpoint saved to {path} (SHA-256: {file_hash[:16]}...)")

    def load_checkpoint(self, path: str) -> None:
        """Load global model from checkpoint with integrity verification.

        Args:
            path: Path to checkpoint file
        """
        checkpoint_path = Path(path)
        if not verify_checkpoint(checkpoint_path):
            logger.warning(
                f"Checkpoint integrity verification FAILED for {path}. "
                "File may be corrupted or tampered with."
            )

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.global_model.load_state_dict(checkpoint['global_model'])
        self.round_loss_history = checkpoint['round_loss']
        self.round_acc_history = checkpoint['round_acc']
        self.test_acc_history = checkpoint['test_acc']
        if self.config.verbose:
            logger.info(f"Checkpoint loaded from {path}")


def create_federated_setup(
    model_class: type,
    client_loaders: List[DataLoader],
    test_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    config: Optional[FedAvgConfig] = None,
    device: str = "cpu",
) -> Tuple[FedAveragingServer, List[FedAveragingClient]]:
    """Convenience function to create federated learning setup.

    Args:
        model_class: Model class to instantiate (must accept num_classes kwarg)
        client_loaders: List of DataLoaders for each client
        test_loader: Optional test loader for server evaluation
        criterion: Loss function (default: CrossEntropyLoss)
        config: FedAvgConfig (default: FedAvgConfig())
        device: Compute device

    Returns:
        Tuple of (server, clients)

    Example:
        >>> loaders = [train_loader_client_1, train_loader_client_2, ...]
        >>> server, clients = create_federated_setup(
        ...     model_class=WaferCNN,
        ...     client_loaders=loaders,
        ...     test_loader=test_loader,
        ...     config=FedAvgConfig(num_rounds=10, clients_per_round=5)
        ... )
        >>> history = server.train()
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if config is None:
        config = FedAvgConfig(device=device)

    # Create global model
    global_model = model_class(num_classes=9)
    global_model.to(device)

    # Create clients
    clients = []
    for client_id, loader in enumerate(client_loaders):
        client_model = model_class(num_classes=9)
        client = FedAveragingClient(
            client_id=client_id,
            train_loader=loader,
            model=client_model,
            criterion=criterion,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            device=device,
        )
        clients.append(client)

    # Create server
    server = FedAveragingServer(
        model=global_model,
        clients=clients,
        config=config,
        test_loader=test_loader,
    )

    return server, clients


if __name__ == "__main__":
    logger.info("FedAvg module loaded. Use FedAveragingServer and FedAveragingClient.")
