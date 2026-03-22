"""
Custom CNN architecture for wafer defect classification.

Implements a lightweight convolutional neural network optimized for CPU inference
on 96x96 wafer map images.
"""

import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    """
    Custom CNN for semiconductor wafer defect classification.

    Architecture:
        - Input: 3x96x96 (replicated grayscale channels)
        - Conv blocks with ReLU, BatchNorm, MaxPool
        - Dropout for regularization
        - Fully connected layers with Dropout
        - Output: 9-class logits (8 defect types + 'none')

    Design rationale:
        - Lightweight (suitable for CPU training)
        - BatchNorm for training stability
        - Dropout (0.3-0.5) to prevent overfitting
        - No pre-training needed (task-specific architecture)
    """

    def __init__(self, num_classes: int = 9) -> None:
        """
        Initialize the custom CNN.

        Args:
            num_classes: Number of output classes (default 9 for wafer defects)
        """
        super().__init__()

        # Feature extraction (conv blocks)
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 96 -> 48

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48 -> 24

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24 -> 12

            # Block 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 12 -> 6
        )

        # Global average pooling: (B, 256, 6, 6) -> (B, 256)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (B, 3, 96, 96)

        Returns:
            Output logits (B, num_classes)
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> tuple:
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
    print(f"WaferCNN:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    x = torch.randn(1, 3, 96, 96)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
