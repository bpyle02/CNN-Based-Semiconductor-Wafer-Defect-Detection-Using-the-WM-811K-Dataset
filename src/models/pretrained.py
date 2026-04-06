"""
Pretrained model architectures for transfer learning on wafer defect classification.

Implements ResNet-18 and EfficientNet-B0 with proper layer-boundary freezing strategy.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def get_resnet18(num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet with fine-tuning on final residual block.

    Freezing strategy:
        - Freezes: conv1, bn1, layer1, layer2, layer3
        - Unfreezes: layer4 (last residual block), fc (classification head)
        - Replaces fc with new head with Dropout(0.5)

    This approach:
        - Preserves low-level ImageNet features (edges, textures)
        - Allows task-specific adaptation in final layers
        - Respects architectural block boundaries (not "last N params")

    Args:
        num_classes: Number of output classes (default 9)

    Returns:
        ResNet-18 model with unfrozen layer4 and custom fc
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze early layers at block boundaries
    for name, param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_efficientnet_b0(num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet with fine-tuning on final feature blocks.

    Freezing strategy:
        - Freezes: features.0-6 (early MBConv blocks)
        - Unfreezes: features.7, features.8 (last MBConv blocks), classifier
        - Replaces classifier with new head with Dropout(0.5)

    EfficientNet uses compound scaling with 8 feature blocks. Final two blocks
    (features.7-8) are most task-specific; earlier blocks capture generic features.

    Args:
        num_classes: Number of output classes (default 9)

    Returns:
        EfficientNet-B0 model with unfrozen features.7-8 and custom classifier
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze early feature blocks at architectural boundaries
    for name, param in model.named_parameters():
        if not (
            name.startswith('features.7') or
            name.startswith('features.8') or
            name.startswith('classifier')
        ):
            param.requires_grad = False

    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    return model


def count_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_frozen_params(model: nn.Module) -> Tuple[str, ...]:
    """
    Get list of frozen parameter names.

    Args:
        model: PyTorch model

    Returns:
        Tuple of frozen parameter names
    """
    return tuple(name for name, param in model.named_parameters() if not param.requires_grad)


if __name__ == "__main__":
    logger.info("ResNet-18:")
    resnet = get_resnet18(num_classes=9)
    total, trainable = count_params(resnet)
    frozen = len(get_frozen_params(resnet))
    logger.info(f"  Total parameters: {total:,}")
    logger.info(f"  Trainable parameters: {trainable:,}")
    logger.info(f"  Frozen parameters: {frozen:,}")
    logger.info(f"  Frozen layers: layer1, layer2, layer3")

    logger.info("\nEfficientNet-B0:")
    effnet = get_efficientnet_b0(num_classes=9)
    total, trainable = count_params(effnet)
    frozen = len(get_frozen_params(effnet))
    logger.info(f"  Total parameters: {total:,}")
    logger.info(f"  Trainable parameters: {trainable:,}")
    logger.info(f"  Frozen parameters: {frozen:,}")
    logger.info(f"  Frozen layers: features.0-6")

    # Test forward pass
    x = torch.randn(1, 3, 96, 96)
    y_resnet = resnet(x)
    y_effnet = effnet(x)
    logger.info(f"\nForward pass check:")
    logger.info(f"  ResNet-18 output shape: {y_resnet.shape}")
    logger.info(f"  EfficientNet-B0 output shape: {y_effnet.shape}")
