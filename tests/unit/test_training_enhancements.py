"""Regression tests for the enhanced training loop and loss builders."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import FocalLoss, build_classification_loss
from src.training.trainer import train_model


def make_loader(num_samples: int = 12) -> DataLoader:
    features = torch.randn(num_samples, 3, 8, 8)
    labels = torch.randint(0, 3, (num_samples,))
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


def test_build_classification_loss_supports_label_smoothing():
    criterion = build_classification_loss(
        "CrossEntropyLoss",
        label_smoothing=0.1,
        reduction="mean",
    )

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1


def test_build_classification_loss_supports_focal_loss():
    weights = torch.tensor([1.0, 2.0, 3.0])
    criterion = build_classification_loss(
        "FocalLoss",
        class_weights=weights,
        focal_gamma=1.5,
        label_smoothing=0.05,
    )

    assert isinstance(criterion, FocalLoss)
    assert criterion.gamma == 1.5
    assert criterion.label_smoothing == 0.05
    # FocalLoss with gamma > 0 applies sqrt to weights to avoid double-compensation
    assert torch.allclose(criterion.weight, weights.sqrt())


def test_train_model_tracks_macro_f1_and_steps_standard_scheduler():
    model = make_model()
    train_loader = make_loader()
    val_loader = make_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        epochs=3,
        model_name="unit-test",
        device="cpu",
        monitored_metric="val_macro_f1",
    )

    assert len(history["val_macro_f1"]) == 3
    assert len(history["learning_rate"]) == 3
    assert history["learning_rate"][0] == 0.1
    assert history["learning_rate"][1] == 0.05
    assert history["learning_rate"][2] == 0.025
    assert history["best_metric_name"] == "val_macro_f1"


def test_train_model_supports_early_stopping():
    model = make_model()
    train_loader = make_loader()
    val_loader = make_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=6,
        model_name="unit-test",
        device="cpu",
        early_stopping_enabled=True,
        early_stopping_patience=1,
        monitored_metric="val_loss",
    )

    assert history["stopped_early"] is True
    assert history["epochs_ran"] < 6
