"""
Test-Time Augmentation (TTA) for wafer defect classification.

Applies multiple geometric augmentations to each test image and averages
the softmax predictions.  This reduces prediction variance, improves
calibration, and typically boosts minority-class recall because the model
sees each sample from several viewpoints.

Default views (5):
    1. Identity (original image)
    2. Horizontal flip
    3. Vertical flip
    4. Horizontal + vertical flip (180-degree rotation equivalent)
    5. 90-degree rotation

References:
    Simonyan & Zisserman (2015). "Very Deep ConvNets" (TTA at test time).
    Shanmugam et al. (2021). "Better Aggregation in TTA". arXiv:2011.11156
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from torch.utils.data import DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)


def _default_tta_ops() -> List:
    """Return the 8 default TTA transformations as callables on (B,C,H,W).

    Views: identity, hflip, vflip, rot90, rot180, rot270,
    hflip+rot90, vflip+rot90.  All are zero-copy views over torch tensors.
    """
    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])

    def vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])

    def rot90_op(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=[-2, -1])

    def rot180_op(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=[-2, -1])

    def rot270_op(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=3, dims=[-2, -1])

    def hflip_rot90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1])

    def vflip_rot90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(torch.flip(x, dims=[-2]), k=1, dims=[-2, -1])

    return [identity, hflip, vflip, rot90_op, rot180_op, rot270_op,
            hflip_rot90, vflip_rot90]


def predict_with_tta(
    model: nn.Module,
    x: torch.Tensor,
    augmentations: Optional[List] = None,
) -> torch.Tensor:
    """Return averaged softmax probabilities across TTA augmentations.

    Default augmentations (8 total):
      - identity
      - hflip
      - vflip
      - rot90, rot180, rot270
      - hflip + rot90
      - vflip + rot90

    Args:
        model: PyTorch classifier returning logits of shape ``(B, num_classes)``.
        x: Input batch tensor with shape ``(B, C, H, W)``.
        augmentations: Optional list of callables ``(B,C,H,W) -> (B,C,H,W)``.
            Defaults to the 8-view zero-copy rotation/flip set.

    Returns:
        Averaged softmax probabilities of shape ``(B, num_classes)``.

    Notes:
        Softmax outputs for classification are rotation/flip invariant in
        label space, so no inverse transform is needed before averaging.
    """
    if augmentations is None:
        augmentations = _default_tta_ops()

    was_training = model.training
    model.eval()
    probs_sum: Optional[torch.Tensor] = None

    with torch.no_grad():
        for aug in augmentations:
            aug_x = aug(x)
            logits = model(aug_x)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    if was_training:
        model.train()

    assert probs_sum is not None
    return probs_sum / float(len(augmentations))


class TestTimeAugmentation:
    """Test-time augmentation: average predictions over augmented views.

    Applies multiple augmentations to each test image and averages the
    softmax predictions. Improves calibration and minority class recall.

    Args:
        model: Trained PyTorch classifier.
        transforms_list: Optional list of ``transforms.Compose`` pipelines.
            If ``None``, the five default geometric views are used.
        device: Compute device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model: nn.Module,
        transforms_list: Optional[List[transforms.Compose]] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = device
        if transforms_list is None:
            self.transforms_list = self._default_transforms()
        else:
            self.transforms_list = transforms_list

    @staticmethod
    def _default_transforms() -> List[transforms.Compose]:
        """5 TTA views: original, hflip, vflip, hflip+vflip, 90-degree rotation."""
        return [
            transforms.Compose([]),  # identity
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
            transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
            ]),
            transforms.Compose([transforms.RandomRotation(degrees=(90, 90))]),
        ]

    @property
    def num_views(self) -> int:
        """Number of augmented views used for prediction."""
        return len(self.transforms_list)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Average softmax predictions over all TTA views.

        Args:
            images: Batch of images with shape ``(B, C, H, W)``.

        Returns:
            Averaged probability tensor of shape ``(B, num_classes)``.
        """
        self.model.eval()
        all_probs: List[torch.Tensor] = []

        with torch.no_grad():
            for t in self.transforms_list:
                augmented = torch.stack([t(img) for img in images])
                augmented = augmented.to(self.device)
                logits = self.model(augmented)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())

        return torch.stack(all_probs).mean(dim=0)

    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: List[str],
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """Evaluate with TTA on a full test set.

        Iterates over ``test_loader``, applies all TTA views to each batch,
        averages the softmax outputs, and computes standard classification
        metrics.

        Args:
            test_loader: DataLoader yielding ``(images, labels)`` tuples.
            class_names: Human-readable class names for the report.
            model_name: Label used in log messages.

        Returns:
            Dictionary with ``accuracy``, ``macro_f1``, and ``weighted_f1``.
        """
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[np.ndarray] = []

        for images, labels in test_loader:
            avg_probs = self.predict(images)  # (B, num_classes)
            preds = avg_probs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(avg_probs.numpy())

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        acc = accuracy_score(labels_arr, preds_arr)
        macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)

        logger.info(
            "\n%s\n  %s -- TTA Evaluation (%d views)\n%s",
            "=" * 60,
            model_name,
            self.num_views,
            "=" * 60,
        )
        logger.info("  Accuracy    : %.4f", acc)
        logger.info("  Macro F1    : %.4f", macro_f1)
        logger.info("  Weighted F1 : %.4f", weighted_f1)
        logger.info(
            classification_report(
                labels_arr,
                preds_arr,
                labels=list(range(len(class_names))),
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )

        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }
