"""Tests for training entrypoint helper construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import train as train_cli
from src.models.cnn import WaferCNN


def test_build_optimizer_uses_configured_sgd_momentum() -> None:
    model = WaferCNN(num_classes=2)

    optimizer = train_cli.build_optimizer(
        model,
        optimizer_name="sgd",
        learning_rate=0.01,
        weight_decay=0.001,
        momentum=0.85,
        nesterov=False,
    )

    assert optimizer.defaults["momentum"] == pytest.approx(0.85)
    assert optimizer.defaults["nesterov"] is False


def test_build_model_uses_cnn_dropout_and_batch_norm_flags() -> None:
    model_cfg = SimpleNamespace(
        dropout_rate=0.4,
        input_channels=3,
        use_batch_norm=False,
    )

    model, display_name = train_cli.build_model("cnn", model_cfg, num_classes=3, device="cpu")

    assert display_name == "Custom CNN"
    assert isinstance(model, WaferCNN)
    assert model.classifier[0].p == pytest.approx(0.4)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())


def test_build_model_uses_pretrained_head_dropout_and_freeze_boundary() -> None:
    torchvision = pytest.importorskip("torchvision")
    assert torchvision is not None

    model_cfg = SimpleNamespace(
        dropout_rate=0.2,
        pretrained=False,
        freeze_until="layer2",
    )

    model, display_name = train_cli.build_model("resnet", model_cfg, num_classes=3, device="cpu")

    assert display_name == "ResNet-18"
    assert model.fc[0].p == pytest.approx(0.2)
    assert all(not param.requires_grad for name, param in model.named_parameters() if name.startswith("layer2"))
    assert all(param.requires_grad for name, param in model.named_parameters() if name.startswith("layer3"))
