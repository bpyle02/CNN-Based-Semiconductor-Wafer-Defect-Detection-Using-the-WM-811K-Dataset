"""Custom loss functions and builders for training on imbalanced datasets."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.

    The loss down-weights easy examples by scaling the standard cross-entropy
    term with ``(1 - pt) ** gamma``.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        moderate_weights: bool = True,
    ) -> None:
        super().__init__()
        # When both class weights and focal modulation (gamma > 0) are active,
        # rare classes get double-compensated: higher weight from class_weights
        # AND stronger (1-pt)^gamma scaling (since rare classes have lower pt).
        # Applying sqrt to the weights reduces this overlap (Lin et al., 2017
        # recommend lowering alpha as gamma increases).
        if moderate_weights and weight is not None and gamma > 0:
            weight = weight.sqrt()
        self.register_buffer("weight", weight.clone().detach() if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_classification_loss(
    loss_name: str = "CrossEntropyLoss",
    *,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    reduction: str = "mean",
    moderate_weights: bool = True,
) -> nn.Module:
    """Create a classification loss function from normalized config values."""
    if loss_name == "FocalLoss":
        return FocalLoss(
            weight=class_weights,
            gamma=focal_gamma,
            reduction=reduction,
            label_smoothing=label_smoothing,
            moderate_weights=moderate_weights,
        )

    if loss_name != "CrossEntropyLoss":
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return nn.CrossEntropyLoss(
        weight=class_weights,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
