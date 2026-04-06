"""Unit tests for the supervised trainer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import FocalLoss
from src.training.trainer import train_model


def _make_loaders() -> tuple[DataLoader, DataLoader]:
    features = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.5, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, 0.5, 0.0],
            [2.0, 0.0, 0.0, 0.5],
            [0.0, 2.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(features, targets), batch_size=3, shuffle=False)
    return train_loader, val_loader


def _make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_train_model_records_step_scheduler_progress():
    train_loader, val_loader = _make_loaders()
    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        epochs=3,
        device="cpu",
        model_name="tiny-step",
    )

    assert history["epochs_ran"] == 3
    assert history["learning_rate"] == pytest.approx([0.1, 0.05, 0.025])


def test_train_model_early_stops_after_repeated_non_improvement():
    train_loader, val_loader = _make_loaders()
    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0)

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=6,
        device="cpu",
        model_name="tiny-es",
        early_stopping_enabled=True,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        monitored_metric="val_loss",
    )

    assert history["stopped_early"] is True
    assert history["epochs_ran"] <= 3


def test_train_model_accepts_mixed_precision_flag_on_cpu():
    train_loader, val_loader = _make_loaders()
    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=2,
        device="cpu",
        model_name="tiny-amp",
        mixed_precision=True,
    )

    assert history["mixed_precision"] is False
    assert history["epochs_ran"] == 2


def test_focal_loss_supports_label_smoothing():
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    loss = criterion(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
