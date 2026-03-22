#!/usr/bin/env python
"""
Comprehensive test suite for federated learning implementation (improvement #14).

Tests:
    1. FedAvgConfig validation
    2. FedAveragingClient local training
    3. FedAveragingServer aggregation
    4. End-to-end federated training loop
    5. Non-IID data partitioning
    6. Learning rate scheduling
    7. Checkpoint save/load
"""

import copy
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.fed_avg import (
    FedAvgConfig,
    FedAveragingClient,
    FedAveragingServer,
    create_federated_setup,
)
from src.models import WaferCNN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fed_avg_config():
    """Test configuration validation."""
    logger.info("Test 1: FedAvgConfig validation")

    # Valid config
    config = FedAvgConfig(num_rounds=5, clients_per_round=3)
    assert config.num_rounds == 5
    assert config.clients_per_round == 3
    logger.info("  PASS: Valid config created")

    # Invalid num_rounds
    try:
        FedAvgConfig(num_rounds=0)
        assert False, "Should raise ValueError"
    except ValueError:
        logger.info("  PASS: Rejects num_rounds < 1")

    # Invalid learning_rate
    try:
        FedAvgConfig(learning_rate=-0.01)
        assert False, "Should raise ValueError"
    except ValueError:
        logger.info("  PASS: Rejects negative learning_rate")


def test_client_local_training():
    """Test FedAveragingClient local training."""
    logger.info("Test 2: Client local training")

    # Create dummy data
    X = torch.randn(20, 3, 96, 96)
    y = torch.randint(0, 9, (20,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    # Create client
    model = WaferCNN(num_classes=9)
    criterion = nn.CrossEntropyLoss()
    client = FedAveragingClient(
        client_id=0,
        train_loader=loader,
        model=model,
        criterion=criterion,
        local_epochs=2,
        learning_rate=0.01,
    )

    # Train
    loss, acc = client.train_local(0.01)
    assert loss >= 0, "Loss should be non-negative"
    assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"
    assert len(client.local_loss_history) == 2, "Should have 2 epochs"
    logger.info(f"  PASS: Local training complete (loss={loss:.4f}, acc={acc:.4f})")


def test_server_aggregation():
    """Test FedAveragingServer weight aggregation."""
    logger.info("Test 3: Server aggregation")

    # Create dummy data
    X = torch.randn(20, 3, 96, 96)
    y = torch.randint(0, 9, (20,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    # Create clients
    model = WaferCNN(num_classes=9)
    criterion = nn.CrossEntropyLoss()

    clients = []
    for i in range(3):
        client = FedAveragingClient(
            client_id=i,
            train_loader=loader,
            model=WaferCNN(num_classes=9),
            criterion=criterion,
            local_epochs=1,
        )
        clients.append(client)

    # Create server
    config = FedAvgConfig(num_rounds=1, clients_per_round=3)
    server = FedAveragingServer(
        model=model,
        clients=clients,
        config=config,
    )

    # Test aggregation
    client_indices = [0, 1, 2]
    client_weights = [client.get_weights() for client in clients]
    avg_weights = server.aggregate_weights(client_indices, client_weights)

    # Verify aggregated weights have correct shape
    for key in avg_weights:
        assert key in model.state_dict(), f"Missing key {key}"
        assert avg_weights[key].shape == model.state_dict()[key].shape, \
            f"Shape mismatch for {key}"

    logger.info("  PASS: Aggregation produces valid weights")


def test_federated_training_loop():
    """Test end-to-end federated training."""
    logger.info("Test 4: End-to-end federated training")

    # Create dummy data
    X = torch.randn(30, 3, 96, 96)
    y = torch.randint(0, 9, (30,))
    dataset = TensorDataset(X, y)

    # Create client loaders
    loaders = []
    for i in range(3):
        subset = list(range(i * 10, (i + 1) * 10))
        from torch.utils.data import Subset
        loader = DataLoader(Subset(dataset, subset), batch_size=5)
        loaders.append(loader)

    # Create setup
    config = FedAvgConfig(
        num_rounds=2,
        clients_per_round=2,
        local_epochs=1,
        verbose=False,
    )

    model = WaferCNN(num_classes=9)
    criterion = nn.CrossEntropyLoss()

    clients = []
    for i, loader in enumerate(loaders):
        client = FedAveragingClient(
            client_id=i,
            train_loader=loader,
            model=WaferCNN(num_classes=9),
            criterion=criterion,
            local_epochs=config.local_epochs,
        )
        clients.append(client)

    server = FedAveragingServer(
        model=model,
        clients=clients,
        config=config,
    )

    # Train
    results = server.train()

    assert len(results['round_loss']) == 2, "Should have 2 rounds"
    assert len(results['round_acc']) == 2, "Should have 2 rounds"
    assert results['communication_rounds'] == 2, "Should have 2 communication rounds"
    assert results['total_time'] > 0, "Training should take time"

    logger.info(
        f"  PASS: 2 rounds complete "
        f"(final acc={results['round_acc'][-1]:.4f}, time={results['total_time']:.1f}s)"
    )


def test_learning_rate_scheduling():
    """Test learning rate scheduling."""
    logger.info("Test 5: Learning rate scheduling")

    # Constant schedule
    config_const = FedAvgConfig(
        num_rounds=1,
        clients_per_round=1,
        learning_rate=0.01,
        learning_rate_schedule="constant",
    )
    X = torch.randn(10, 3, 96, 96)
    y = torch.randint(0, 9, (10,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    model = WaferCNN(num_classes=9)
    criterion = nn.CrossEntropyLoss()
    clients = [
        FedAveragingClient(0, loader, WaferCNN(num_classes=9), criterion)
    ]

    server_const = FedAveragingServer(model, clients, config_const)
    lr_0 = server_const.get_learning_rate(0)
    lr_5 = server_const.get_learning_rate(5)
    assert lr_0 == lr_5 == 0.01, "Constant schedule should be constant"
    logger.info("  PASS: Constant schedule works")

    # Exponential schedule
    config_exp = FedAvgConfig(
        num_rounds=1,
        clients_per_round=1,
        learning_rate=1.0,
        learning_rate_schedule="exponential",
        lr_decay_rate=0.5,
    )
    server_exp = FedAveragingServer(model, clients, config_exp)
    lr_0_exp = server_exp.get_learning_rate(0)
    lr_1_exp = server_exp.get_learning_rate(1)
    lr_2_exp = server_exp.get_learning_rate(2)
    assert lr_0_exp == 1.0, "Round 0 should be 1.0"
    assert abs(lr_1_exp - 0.5) < 1e-6, "Round 1 should be 0.5"
    assert abs(lr_2_exp - 0.25) < 1e-6, "Round 2 should be 0.25"
    logger.info("  PASS: Exponential schedule works")


def test_non_iid_partitioning():
    """Test non-IID data partitioning."""
    logger.info("Test 6: Non-IID data partitioning")

    from src.federated.server import create_client_loaders

    # Create dataset with class labels
    X = torch.randn(100, 3, 96, 96)
    y = torch.cat([torch.full((10,), i) for i in range(9)])
    y = torch.cat([y, torch.randint(0, 9, (10,))])  # Total 100 samples

    dataset = TensorDataset(X, y)

    # IID partitioning
    loaders_iid = create_client_loaders(
        dataset,
        num_clients=5,
        batch_size=10,
        non_iid_alpha=None,
    )
    assert len(loaders_iid) == 5, "Should create 5 clients"
    logger.info(f"  PASS: IID partitioning creates 5 clients")

    # Non-IID partitioning
    loaders_noniid = create_client_loaders(
        dataset,
        num_clients=5,
        batch_size=10,
        non_iid_alpha=0.5,  # More non-IID
    )
    assert len(loaders_noniid) == 5, "Should create 5 clients"
    logger.info(f"  PASS: Non-IID (alpha=0.5) partitioning creates 5 clients")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    logger.info("Test 7: Checkpoint save/load")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and train
        X = torch.randn(20, 3, 96, 96)
        y = torch.randint(0, 9, (20,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=5)

        model = WaferCNN(num_classes=9)
        criterion = nn.CrossEntropyLoss()
        clients = [
            FedAveragingClient(0, loader, WaferCNN(num_classes=9), criterion, local_epochs=1)
        ]

        config = FedAvgConfig(num_rounds=1, clients_per_round=1, verbose=False)
        server = FedAveragingServer(model, clients, config)
        server.train()

        # Save checkpoint
        ckpt_path = Path(tmpdir) / "test_ckpt.pth"
        server.save_checkpoint(str(ckpt_path))
        assert ckpt_path.exists(), "Checkpoint should be saved"
        logger.info(f"  PASS: Checkpoint saved to {ckpt_path}")

        # Load checkpoint
        model2 = WaferCNN(num_classes=9)
        clients2 = [
            FedAveragingClient(0, loader, WaferCNN(num_classes=9), criterion)
        ]
        server2 = FedAveragingServer(model2, clients2, config)
        server2.load_checkpoint(str(ckpt_path))

        # Verify weights match
        w1 = server.global_model.state_dict()
        w2 = server2.global_model.state_dict()
        for key in w1:
            assert torch.allclose(w1[key], w2[key], atol=1e-6), \
                f"Weights mismatch for {key}"
        logger.info("  PASS: Loaded checkpoint matches original")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("Federated Learning (FedAvg) Test Suite")
    logger.info("=" * 70)

    try:
        test_fed_avg_config()
        test_client_local_training()
        test_server_aggregation()
        test_federated_training_loop()
        test_learning_rate_scheduling()
        test_non_iid_partitioning()
        test_checkpoint_save_load()

        logger.info("=" * 70)
        logger.info("All tests PASSED")
        logger.info("=" * 70)
        return 0

    except Exception as e:
        logger.error(f"Test FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
