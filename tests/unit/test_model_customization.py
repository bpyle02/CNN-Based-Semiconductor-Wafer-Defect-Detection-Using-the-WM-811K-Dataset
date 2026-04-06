"""Focused tests for configurable model heads and freeze strategies."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.cnn import WaferCNN
from src.models.pretrained import get_efficientnet_b0, get_resnet18


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.randn(2, 3, 96, 96)


class TestWaferCNNCustomization:
    def test_custom_head_and_channels(self, batch: torch.Tensor) -> None:
        model = WaferCNN(
            num_classes=4,
            feature_channels=(16, 32),
            dropout_rate=0.15,
            head_hidden_dim=64,
            head_dropout=0.25,
        )

        out = model(batch)

        assert out.shape == (2, 4)
        assert isinstance(model.classifier[0], nn.Dropout)
        assert model.classifier[0].p == pytest.approx(0.15)
        assert isinstance(model.classifier[1], nn.Linear)
        assert model.classifier[1].out_features == 64
        assert isinstance(model.classifier[3], nn.Dropout)
        assert model.classifier[3].p == pytest.approx(0.25)
        assert isinstance(model.classifier[-1], nn.Linear)
        assert model.classifier[-1].out_features == 4

    def test_linear_head_without_hidden_layer(self, batch: torch.Tensor) -> None:
        model = WaferCNN(num_classes=5, head_hidden_dim=None)
        out = model(batch)

        assert out.shape == (2, 5)
        assert len(model.classifier) == 2
        assert isinstance(model.classifier[1], nn.Linear)
        assert model.classifier[1].out_features == 5


class TestTransferLearningCustomization:
    def test_resnet_freeze_boundary_and_head_width(self, batch: torch.Tensor) -> None:
        model = get_resnet18(
            num_classes=7,
            pretrained=False,
            freeze_until="layer2",
            head_dropout=0.2,
            head_hidden_dim=32,
        )

        out = model(batch)

        assert out.shape == (2, 7)
        assert isinstance(model.fc, nn.Sequential)
        assert isinstance(model.fc[0], nn.Dropout)
        assert model.fc[0].p == pytest.approx(0.2)
        assert isinstance(model.fc[1], nn.Linear)
        assert model.fc[1].out_features == 32
        assert isinstance(model.fc[-1], nn.Linear)
        assert model.fc[-1].out_features == 7

        for name, param in model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                assert not param.requires_grad, f"{name} should be frozen"
            if name.startswith(("layer3", "layer4", "fc")):
                assert param.requires_grad, f"{name} should be trainable"

    def test_resnet_explicit_frozen_prefixes(self) -> None:
        model = get_resnet18(
            num_classes=7,
            pretrained=False,
            frozen_prefixes=("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
        )

        assert all(
            not param.requires_grad
            for name, param in model.named_parameters()
            if not name.startswith("fc")
        )
        assert all(param.requires_grad for name, param in model.named_parameters() if name.startswith("fc"))

    def test_efficientnet_no_freeze_with_custom_head(self, batch: torch.Tensor) -> None:
        model = get_efficientnet_b0(
            num_classes=6,
            pretrained=False,
            freeze_until=None,
            head_dropout=0.1,
            head_hidden_dim=48,
        )

        out = model(batch)

        assert out.shape == (2, 6)
        assert isinstance(model.classifier, nn.Sequential)
        assert model.classifier[0].p == pytest.approx(0.1)
        assert model.classifier[1].out_features == 48
        assert model.classifier[-1].out_features == 6
        assert all(param.requires_grad for param in model.parameters())
