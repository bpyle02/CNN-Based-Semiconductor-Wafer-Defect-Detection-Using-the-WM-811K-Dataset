"""
Custom CNN architecture for wafer defect classification.

Implements a lightweight convolutional neural network optimized for CPU inference
on 96x96 wafer map images.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WaferCNN(nn.Module):
    """
    Custom CNN for semiconductor wafer defect classification.

    Architecture:
        - Input: 3x96x96 (replicated grayscale channels)
        - Configurable convolutional feature stages
        - Global average pooling
        - Configurable classifier head depth and dropout
        - Output: logits for ``num_classes``

    The defaults preserve the existing architecture, but the feature widths and
    classifier head can now be adjusted from training code without changing the
    model definition itself.
    """

    def __init__(
        self,
        num_classes: int = 9,
        input_channels: int = 3,
        feature_channels: Sequence[int] | None = (32, 64, 128, 256),
        dropout_rate: float = 0.5,
        head_hidden_dim: int | None = 128,
        head_dropout: float | None = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Initialize the custom CNN.

        Args:
            num_classes: Number of output classes (default 9 for wafer defects)
            input_channels: Number of input channels. Default keeps the current
                replicated grayscale convention.
            feature_channels: Output width for each convolutional stage.
            dropout_rate: Dropout probability before the classifier head.
            head_hidden_dim: Optional hidden width for the classifier head. Set
                to ``None`` for a single linear layer.
            head_dropout: Dropout probability between hidden and output layers
                when ``head_hidden_dim`` is enabled.
            use_batch_norm: Whether to apply BatchNorm after each convolution.
        """
        super().__init__()

        if feature_channels is None:
            feature_channels = (32, 64, 128, 256)
        feature_channels = tuple(feature_channels)
        if not feature_channels:
            raise ValueError("feature_channels must contain at least one stage")
        if any(channel <= 0 for channel in feature_channels):
            raise ValueError("feature_channels values must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the interval [0, 1)")
        if head_hidden_dim is not None and head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive when provided")
        if head_dropout is None:
            head_dropout = 0.3
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in the interval [0, 1)")

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.dropout_rate = dropout_rate
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.use_batch_norm = use_batch_norm

        # Feature extraction (conv blocks)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in feature_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        classifier_layers: list[nn.Module] = [nn.Dropout(dropout_rate)]
        if head_hidden_dim is None:
            classifier_layers.append(nn.Linear(feature_channels[-1], num_classes))
        else:
            classifier_layers.extend(
                [
                    nn.Linear(feature_channels[-1], head_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden_dim, num_classes),
                ]
            )
        self.classifier = nn.Sequential(*classifier_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output logits (B, num_classes)
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = WaferCNN(num_classes=9)
    total, trainable = count_parameters(model)
    logger.info("WaferCNN:")
    logger.info(f"  Total parameters: {total:,}")
    logger.info(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    x = torch.randn(1, 3, 96, 96)
    y = model(x)
    logger.info(f"  Input shape: {x.shape}")
    logger.info(f"  Output shape: {y.shape}")
