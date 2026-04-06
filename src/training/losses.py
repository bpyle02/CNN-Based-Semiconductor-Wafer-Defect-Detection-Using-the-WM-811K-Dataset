"""Custom loss functions and builders for training on imbalanced datasets.

References:
    [15] Lin et al. (2017). "Focal Loss for Dense Object Detection". arXiv:1708.02002
    [16] Cui et al. (2019). "Class-Balanced Loss". arXiv:1901.05555
    [19] He & Garcia (2009). "Learning from Imbalanced Data". DOI:10.1109/TKDE.2008.239
    [20] Muller et al. (2019). "When Does Label Smoothing Help?". arXiv:1906.02629
    [73] Cao et al. (2019). "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss". arXiv:1906.07413
    [79] Deng et al. (2019). "ArcFace: Additive Angular Margin Loss". arXiv:1801.07698
    [80] Wang et al. (2018). "CosFace: Large Margin Cosine Loss". arXiv:1801.09414
    [81] Wen et al. (2016). "Center Loss for Face Recognition". ECCV
    [83] Wang et al. (2019). "Multi-Similarity Loss". arXiv:1904.06627
    [84] Kim et al. (2020). "Proxy Anchor Loss for Deep Metric Learning". arXiv:2003.13911
    [131] Sudre et al. (2017). "Generalised Dice Loss". arXiv:1707.03237
    [132] Salehi et al. (2017). "Tversky Loss". arXiv:1706.05721
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):  # Ref [15]: Lin et al. — focal modulation (1-pt)^gamma down-weights easy examples
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
            weight = weight.sqrt()  # Ref [16]: Cui et al. — effective number balancing; sqrt moderates double-compensation
        self.register_buffer("weight", weight.clone().detach() if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(  # Ref [20]: Muller et al. — label smoothing regularization
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


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted softmax cross-entropy for long-tail classification.

    Adds log(class_prior) to logits before computing CE, correcting the
    decision boundary shift caused by class imbalance. Fisher-consistent
    for balanced error rate.

    L = CE(logits + tau * log(pi), y)

    Reference: Menon et al. (2021). "Long-tail learning via logit adjustment". arXiv:2007.07314
    """

    def __init__(
        self,
        class_frequencies: torch.Tensor,
        tau: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        priors = class_frequencies.float() / class_frequencies.float().sum()
        self.register_buffer("log_priors", torch.log(priors + 1e-12))
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_priors
        return F.cross_entropy(adjusted, targets, reduction=self.reduction)


class DiceLoss(nn.Module):
    """Dice loss for multi-class classification.

    Treats classification as soft set-overlap optimization.
    L_c = 1 - (2 * sum(p_c * y_c) + smooth) / (sum(p_c) + sum(y_c) + smooth)

    Reference: [131] Sudre et al. (2017). "Generalised Dice Loss". arXiv:1707.03237
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)  # (B, C)
        one_hot = F.one_hot(targets, num_classes).float()  # (B, C)

        intersection = (probs * one_hot).sum(dim=0)  # (C,)
        union = probs.sum(dim=0) + one_hot.sum(dim=0)  # (C,)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (C,)

        if self.reduction == "none":
            return 1.0 - dice_per_class
        if self.reduction == "sum":
            return (1.0 - dice_per_class).sum()
        return 1.0 - dice_per_class.mean()


class TverskyLoss(nn.Module):
    """Tversky loss with controllable FP/FN tradeoff.

    Generalizes Dice loss with alpha (FP weight) and beta (FN weight).
    Setting alpha=beta=0.5 recovers Dice loss.
    Setting alpha<beta penalizes false negatives more (better for recall on rare classes).

    Reference: [132] Salehi et al. (2017). "Tversky Loss". arXiv:1706.05721
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)  # (B, C)
        one_hot = F.one_hot(targets, num_classes).float()  # (B, C)

        tp = (probs * one_hot).sum(dim=0)  # (C,)
        fp = (probs * (1.0 - one_hot)).sum(dim=0)  # (C,)
        fn = ((1.0 - probs) * one_hot).sum(dim=0)  # (C,)

        tversky_per_class = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )  # (C,)

        if self.reduction == "none":
            return 1.0 - tversky_per_class
        if self.reduction == "sum":
            return (1.0 - tversky_per_class).sum()
        return 1.0 - tversky_per_class.mean()


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature.

    Replaces standard Linear(in, out) with normalized weight vectors and
    cosine similarity, preventing magnitude bias in imbalanced settings.

    logits = temperature * cos(features, W) = temperature * (F/||F|| . W/||W||)

    References: [79] Deng et al. (2019). "ArcFace". arXiv:1801.07698
                [80] Wang et al. (2018). "CosFace". arXiv:1801.09414
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        temperature: float = 16.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        return self.temperature * F.linear(features_norm, weight_norm)


def build_classification_loss(
    loss_name: str = "CrossEntropyLoss",
    *,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    reduction: str = "mean",
    moderate_weights: bool = True,
    class_frequencies: Optional[torch.Tensor] = None,
    logit_adjustment_tau: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
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

    if loss_name == "DiceLoss":
        return DiceLoss(reduction=reduction)

    if loss_name == "TverskyLoss":
        return TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            reduction=reduction,
        )

    if loss_name == "LogitAdjustedLoss":
        if class_frequencies is None:
            raise ValueError(
                "LogitAdjustedLoss requires class_frequencies to be provided"
            )
        return LogitAdjustedLoss(
            class_frequencies=class_frequencies,
            tau=logit_adjustment_tau,
            reduction=reduction,
        )

    if loss_name != "CrossEntropyLoss":
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return nn.CrossEntropyLoss(
        weight=class_weights,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
