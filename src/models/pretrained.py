"""
Pretrained model architectures for transfer learning on wafer defect classification.

Implements ResNet-18 and EfficientNet-B0 with configurable freezing and
classifier-head construction utilities.

References:
    [1] He et al. (2016). "Deep Residual Learning for Image Recognition". arXiv:1512.03385
    [2] Tan & Le (2019). "EfficientNet: Rethinking Model Scaling". arXiv:1905.11946
    [56] (2020). "DL Approaches for Wafer Map Defect Pattern Recognition"
    [60] (2020). "Wafer Defect Pattern Recognition Using Transfer Learning"
    [74] Kornblith et al. (2019). "Do Better ImageNet Models Transfer Better?". arXiv:1805.08974
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

RESNET18_BACKBONE_PREFIXES: Tuple[str, ...] = (
    "conv1",
    "bn1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)
EFFICIENTNET_B0_BACKBONE_PREFIXES: Tuple[str, ...] = tuple(
    f"features.{index}" for index in range(9)
)


def build_classifier_head(
    in_features: int,
    num_classes: int,
    dropout_rate: float = 0.5,
    hidden_dim: Optional[int] = None,
    hidden_dropout: Optional[float] = None,
) -> nn.Sequential:
    """
    Build a small classifier head with optional hidden layer.

    Args:
        in_features: Input feature dimension from the backbone.
        num_classes: Number of output classes.
        dropout_rate: Dropout before the head.
        hidden_dim: Optional hidden width. When omitted, a single linear layer is used.
        hidden_dropout: Dropout after the hidden layer. Defaults to ``dropout_rate``.
    """
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the interval [0, 1)")
    if hidden_dim is not None and hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive when provided")

    if hidden_dropout is None:
        hidden_dropout = dropout_rate
    if not 0.0 <= hidden_dropout < 1.0:
        raise ValueError("hidden_dropout must be in the interval [0, 1)")

    layers: list[nn.Module] = [nn.Dropout(dropout_rate)]
    if hidden_dim is None:
        layers.append(nn.Linear(in_features, num_classes))
    else:
        layers.extend(
            [
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        )
    return nn.Sequential(*layers)


def set_frozen_prefixes(model: nn.Module, frozen_prefixes: Sequence[str]) -> None:
    """Freeze parameters whose names start with any of the provided prefixes."""
    prefixes = tuple(prefix for prefix in frozen_prefixes if prefix)
    if not prefixes:
        return

    for name, param in model.named_parameters():
        param.requires_grad = not any(name.startswith(prefix) for prefix in prefixes)


def resolve_frozen_prefixes(
    freeze_until: Optional[str],
    backbone_prefixes: Sequence[str],
) -> Tuple[str, ...]:
    """
    Resolve a ``freeze_until`` boundary into a tuple of frozen prefixes.

    ``freeze_until="layer3"`` means freeze everything up to and including
    ``layer3`` and leave later blocks plus the classifier trainable.
    """
    if freeze_until is None:
        return tuple()

    normalized = str(freeze_until).strip().lower()
    if normalized in {"", "none"}:
        return tuple()

    ordered = tuple(backbone_prefixes)
    if normalized not in ordered:
        allowed = ", ".join(ordered)
        raise ValueError(
            f"Unknown freeze boundary '{freeze_until}'. Expected one of: none, {allowed}"
        )

    boundary_index = ordered.index(normalized)
    return ordered[: boundary_index + 1]


# Ref [1]: He et al. (arXiv:1512.03385) — ResNet-18 with ImageNet pretraining
def get_resnet18(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_until: Optional[str] = "layer3",
    frozen_prefixes: Optional[Sequence[str]] = None,
    head_dropout: float = 0.5,
    head_hidden_dim: Optional[int] = None,
) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet with configurable fine-tuning.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights when available.
        freeze_until: Freeze backbone blocks up to this boundary. Set to
            ``None`` or ``"none"`` to leave the backbone trainable.
        frozen_prefixes: Explicit frozen parameter prefixes. When provided,
            this overrides ``freeze_until``.
        head_dropout: Dropout probability applied before the classifier head.
        head_hidden_dim: Optional hidden width for a two-layer classifier head.

    Returns:
        A ResNet-18 model with a configurable classifier head and freeze strategy.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = build_classifier_head(
        in_features=in_features,
        num_classes=num_classes,
        dropout_rate=head_dropout,
        hidden_dim=head_hidden_dim,
    )

    if frozen_prefixes is not None:
        set_frozen_prefixes(model, frozen_prefixes)
    else:
        resolved_frozen = resolve_frozen_prefixes(freeze_until, RESNET18_BACKBONE_PREFIXES)
        set_frozen_prefixes(model, resolved_frozen)

    return model


# Ref [2]: Tan & Le (arXiv:1905.11946) — EfficientNet-B0 compound scaling
def get_efficientnet_b0(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_until: Optional[str] = "features.6",
    frozen_prefixes: Optional[Sequence[str]] = None,
    head_dropout: float = 0.5,
    head_hidden_dim: Optional[int] = None,
) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet with configurable fine-tuning.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights when available.
        freeze_until: Freeze backbone blocks up to this boundary. Set to
            ``None`` or ``"none"`` to leave the backbone trainable.
        frozen_prefixes: Explicit frozen parameter prefixes. When provided,
            this overrides ``freeze_until``.
        head_dropout: Dropout probability applied before the classifier head.
        head_hidden_dim: Optional hidden width for a two-layer classifier head.

    Returns:
        An EfficientNet-B0 model with a configurable classifier head and freeze strategy.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = build_classifier_head(
        in_features=in_features,
        num_classes=num_classes,
        dropout_rate=head_dropout,
        hidden_dim=head_hidden_dim,
    )

    if frozen_prefixes is not None:
        set_frozen_prefixes(model, frozen_prefixes)
    else:
        resolved_frozen = resolve_frozen_prefixes(
            freeze_until,
            EFFICIENTNET_B0_BACKBONE_PREFIXES,
        )
        set_frozen_prefixes(model, resolved_frozen)

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
