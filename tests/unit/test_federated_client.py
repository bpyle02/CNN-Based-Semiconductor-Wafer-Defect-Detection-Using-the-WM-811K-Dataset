"""Tests for src/federated/client.py (ClientManager wrapper)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import ClientManager


def _make_loader(n: int = 8, batch: int = 4) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 4)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch)


def _make_manager(tmp_path: Path, **kwargs) -> ClientManager:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    return ClientManager(
        client_id=kwargs.pop("client_id", 0),
        train_loader=kwargs.pop("train_loader", _make_loader()),
        model=model,
        criterion=nn.CrossEntropyLoss(),
        local_epochs=kwargs.pop("local_epochs", 1),
        learning_rate=kwargs.pop("learning_rate", 0.01),
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )


def test_manager_init_creates_checkpoint_dir(tmp_path: Path):
    mgr = _make_manager(tmp_path)
    assert mgr.checkpoint_dir.exists()
    assert mgr.client_id == 0
    assert mgr.round_num == 0
    assert mgr.metadata["total_samples"] == 8


def test_get_and_set_weights_roundtrip(tmp_path: Path):
    mgr = _make_manager(tmp_path)
    weights = mgr.upload_model()  # calls get_weights under the hood
    # Perturb model, then reload original weights.
    with torch.no_grad():
        for p in mgr.client.model.parameters():
            p.add_(1.0)
    mgr.receive_global_model(weights)
    # After receiving, params should match original weights exactly.
    for name, p in mgr.client.model.state_dict().items():
        assert torch.equal(p, weights[name])


def test_train_local_returns_metrics_and_increments_round(tmp_path: Path):
    mgr = _make_manager(tmp_path, local_epochs=1)
    metrics = mgr.train_local(learning_rate=0.01)
    assert metrics["client_id"] == 0
    assert metrics["round"] == 0
    assert "local_loss" in metrics and "local_acc" in metrics
    assert metrics["num_samples"] == 8
    assert mgr.round_num == 1
    assert mgr.metadata["rounds_completed"] == 1


def test_fit_roundtrip_preserves_weights_when_not_trained(tmp_path: Path):
    """get_weights after set_weights (no training) returns the same tensors."""
    mgr = _make_manager(tmp_path)
    original = mgr.upload_model()
    # Create a new state dict with same keys but different values.
    modified = {k: v + 0.5 for k, v in original.items()}
    mgr.receive_global_model(modified)
    roundtripped = mgr.upload_model()
    for k in modified:
        assert torch.allclose(roundtripped[k], modified[k])


def test_save_and_load_checkpoint(tmp_path: Path):
    mgr = _make_manager(tmp_path)
    mgr.train_local(0.01)
    original_round = mgr.round_num
    path = mgr.save_checkpoint()
    assert Path(path).exists()

    # Mutate state then reload.
    mgr.round_num = 999
    mgr.load_checkpoint(path)
    assert mgr.round_num == original_round


def test_get_metadata_contains_histories(tmp_path: Path):
    mgr = _make_manager(tmp_path)
    mgr.train_local(0.01)
    meta = mgr.get_metadata()
    assert "loss_history" in meta and "acc_history" in meta
    assert len(meta["loss_history"]) == 1
    assert meta["current_round"] == 1
