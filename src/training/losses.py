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


class CostSensitiveCE(nn.Module):
    """Cross-entropy with a per-class-pair misclassification cost matrix.

    The cost matrix ``cost[true, pred]`` specifies the penalty for predicting
    class ``pred`` when the true class is ``true``. The default (identity on
    diagonal, 1 elsewhere) recovers vanilla cross-entropy. Off-diagonal
    entries can be raised to express domain-specific priorities, e.g. a
    missed Near-full pattern on a wafer is far costlier than confusing two
    visually similar edge-localized patterns.

    Formally, for softmax probabilities p_k(x) and true label y,
        L(x, y) = - sum_k cost[y, k] * log p_k(x)
    when reduction='mean' we average over the batch. This matches the
    cost-sensitive CE of Elkan (2001), "The Foundations of Cost-Sensitive
    Learning", and the "cost-weighted cross-entropy" used in imbalanced
    manufacturing defect classification.

    Args:
        cost_matrix: (C, C) tensor. Entry (i, j) is the cost of predicting
            j when truth is i. Diagonal entries are typically 0 or 1;
            off-diagonals are >= 1 to penalize misclassifications.
        class_weights: Optional (C,) tensor multiplied onto per-sample loss
            based on the true class (for prior re-weighting, e.g. inverse
            frequency). Orthogonal to the cost matrix.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        cost_matrix: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if cost_matrix.ndim != 2 or cost_matrix.size(0) != cost_matrix.size(1):
            raise ValueError(
                f"cost_matrix must be square (C, C); got shape {tuple(cost_matrix.shape)}"
            )
        if (cost_matrix < 0).any():
            raise ValueError("cost_matrix entries must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")

        self.register_buffer("cost_matrix", cost_matrix.float().clone().detach())
        if class_weights is not None:
            self.register_buffer(
                "class_weights", class_weights.float().clone().detach()
            )
        else:
            self.class_weights = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        row_costs = self.cost_matrix[targets]             # (B, C) cost[y_i, :]
        # Per-sample cost-weighted cross-entropy: sum_k cost[y, k] * -log p_k
        per_sample = -(row_costs * log_probs).sum(dim=1)  # (B,)

        if self.class_weights is not None:
            per_sample = per_sample * self.class_weights[targets]

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample


def build_cost_matrix_wm811k(
    class_names,
    near_full_missed: float = 10.0,
    rare_missed: float = 5.0,
    edge_confusion: float = 0.5,
    default: float = 1.0,
) -> torch.Tensor:
    """Construct the defensible WM-811K cost matrix used by cost_sensitive.yaml.

    Convention: cost[true, pred]. Diagonal (correct) = 0. Off-diagonal rules
    are applied in the order below; later rules override earlier ones.

    1. All off-diagonals default to ``default`` (=1).
    2. Rows for Near-full and rare classes (Donut, Random) assign
       ``near_full_missed`` / ``rare_missed`` to their off-diagonal entries
       (missing these is catastrophic / high-impact).
    3. Pairwise confusion within {Loc, Edge-Loc, Edge-Ring} is soft-penalized
       at ``edge_confusion`` < 1 (patterns look similar, penalize less).
    """
    n = len(class_names)
    cost = torch.full((n, n), float(default))
    idx = {name: i for i, name in enumerate(class_names)}

    # Diagonal: no cost for correct classification.
    for i in range(n):
        cost[i, i] = 0.0

    # Rare / catastrophic misses.
    for cls_name, weight in (
        ("Near-full", near_full_missed),
        ("Donut", rare_missed),
        ("Random", rare_missed),
    ):
        if cls_name in idx:
            row = idx[cls_name]
            for j in range(n):
                if j != row:
                    cost[row, j] = float(weight)

    # Soft confusion within edge-localized family.
    edge_family = [idx[n_] for n_ in ("Loc", "Edge-Loc", "Edge-Ring") if n_ in idx]
    for i in edge_family:
        for j in edge_family:
            if i != j:
                cost[i, j] = float(edge_confusion)

    return cost


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
    cost_matrix: Optional[torch.Tensor] = None,
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

    if loss_name == "CostSensitiveCE":
        if cost_matrix is None:
            raise ValueError(
                "CostSensitiveCE requires cost_matrix to be provided"
            )
        return CostSensitiveCE(
            cost_matrix=cost_matrix,
            class_weights=class_weights,
            reduction=reduction,
        )

    if loss_name != "CrossEntropyLoss":
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return nn.CrossEntropyLoss(
        weight=class_weights,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
