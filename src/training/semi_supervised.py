"""
FixMatch semi-supervised training for wafer defect detection.

Leverages the ~640K unlabeled wafers in WM-811K alongside the ~172K labeled
ones. High-confidence pseudo-labels on weakly-augmented unlabeled samples
supervise strongly-augmented versions, yielding consistency regularization
without external labeling effort.

References:
    [111] Sohn et al. (2020). "FixMatch: Simplifying Semi-Supervised Learning
          with Consistency and Confidence". arXiv:2001.07685
    [114] Berthelot et al. (2019). "MixMatch: A Holistic Approach to
          Semi-Supervised Learning". arXiv:1905.02249
    [117] Zhu et al. (2005). "Semi-Supervised Learning Literature Survey"
    [118] Zhai et al. (2019). "S4L: Self-Supervised Semi-Supervised Learning".
          arXiv:1905.03670
    [120] Lee (2013). "Pseudo-Label: The Simple and Efficient Semi-Supervised
          Learning Method". ICML Workshop
"""

from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import pandas as pd

import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.transform import resize as skimage_resize
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


class _GaussianNoise:
    """Add Gaussian noise to a tensor (strong augmentation component)."""

    def __init__(self, mean: float = 0.0, std: float = 0.05) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_weak_transform() -> transforms.Compose:
    """Weak augmentation: horizontal flip only.

    FixMatch uses minimal augmentation for the "teacher" branch so that the
    pseudo-label is as clean as possible.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )


def get_strong_transform() -> transforms.Compose:
    """Strong augmentation: flip + rotation + affine + noise + erasing.

    FixMatch uses aggressive augmentation for the "student" branch so that
    the model learns invariance to a wide range of perturbations.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            _GaussianNoise(std=0.05),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        ]
    )


# ---------------------------------------------------------------------------
# Unlabeled dataset
# ---------------------------------------------------------------------------


class UnlabeledWaferDataset(Dataset):
    """Dataset for unlabeled wafer maps (no class labels).

    Each __getitem__ call returns two views of the same wafer map:
    one with weak augmentation and one with strong augmentation, as
    required by the FixMatch consistency loss.

    Args:
        raw_maps: List of raw numpy wafer-map arrays (variable size).
        weak_transform: Transform pipeline for the weak (teacher) view.
        strong_transform: Transform pipeline for the strong (student) view.
        target_size: Spatial size to resize each map to.
    """

    def __init__(
        self,
        raw_maps: List[np.ndarray],
        weak_transform: transforms.Compose,
        strong_transform: transforms.Compose,
        target_size: Tuple[int, int] = (96, 96),
    ) -> None:
        self.maps = raw_maps
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (weak_aug_image, strong_aug_image) for FixMatch.

        Both views start from the same preprocessed base image.
        """
        wm = self.maps[idx]
        arr = wm.astype(np.float32)
        arr = skimage_resize(
            arr,
            self.target_size,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
        arr = arr / 2.0  # WM-811K: {0,1,2} -> [0, 1]

        # Stack grayscale -> 3 channels
        img = np.stack([arr] * 3, axis=0)
        base_tensor = torch.tensor(img, dtype=torch.float32)

        weak_img = self.weak_transform(base_tensor)
        strong_img = self.strong_transform(base_tensor)
        return weak_img, strong_img


# ---------------------------------------------------------------------------
# Infinite data iterator
# ---------------------------------------------------------------------------


class _InfiniteDataIterator:
    """Wraps a DataLoader to iterate indefinitely (restarts on exhaustion)."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._iter: Optional[Iterator] = None

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# ---------------------------------------------------------------------------
# FixMatch trainer
# ---------------------------------------------------------------------------


class FixMatchTrainer:
    """FixMatch semi-supervised training for wafer defect detection.

    Uses labeled data with standard cross-entropy loss combined with unlabeled
    data via a pseudo-label consistency loss.  High-confidence predictions on
    weakly-augmented unlabeled samples become pseudo-labels for the
    corresponding strongly-augmented versions.

    Attributes:
        model: The classification model being trained.
        num_classes: Number of target classes (9 for WM-811K).
        confidence_threshold: Minimum softmax probability to accept a
            pseudo-label (tau in the paper).
        lambda_u: Weight for the unsupervised consistency loss.
        device: Compute device string ("cpu" or "cuda").

    References:
        [111] Sohn et al. (2020). "FixMatch". arXiv:2001.07685
        [114] Berthelot et al. (2019). "MixMatch". arXiv:1905.02249
        [120] Lee (2013). "Pseudo-Label". arXiv:1908.02983
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 9,
        confidence_threshold: float = 0.95,
        lambda_u: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.device = device

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def train(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 50,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        monitored_metric: str = "val_macro_f1",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Train with the FixMatch algorithm.

        Per iteration:
            1. Sample labeled batch ``(x_l, y_l)`` and unlabeled batch
               ``(x_u_weak, x_u_strong)``.
            2. Supervised loss: ``L_s = CE(model(x_l), y_l)``.
            3. Pseudo-labels: ``q = softmax(model(x_u_weak))``,
               ``mask = max(q, dim=1) >= threshold``.
            4. Unsupervised loss: ``L_u = CE(model(x_u_strong[mask]),
               argmax(q[mask]))``.
            5. Total: ``L = L_s + lambda_u * L_u``.

        Args:
            labeled_loader: DataLoader yielding ``(images, labels)``.
            unlabeled_loader: DataLoader yielding ``(weak_images, strong_images)``.
            val_loader: DataLoader for validation (labeled).
            optimizer: Optimizer instance.
            criterion: Loss function for supervised branch.
            epochs: Number of training epochs.
            scheduler: Optional LR scheduler.
            gradient_clip: Optional max gradient norm.
            monitored_metric: Metric used for best-model selection.

        Returns:
            ``(best_model, history_dict)`` where *history_dict* contains
            per-epoch losses, accuracies, F1 scores, and pseudo-label
            statistics.
        """
        self.model.to(self.device)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_metric = float("-inf")
        best_epoch = 0

        history: Dict[str, Any] = {
            "train_loss": [],
            "train_supervised_loss": [],
            "train_unsupervised_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "pseudo_label_ratio": [],
            "pseudo_label_count": [],
            "learning_rate": [],
            "total_time": 0.0,
            "best_epoch": 0,
            "best_metric": 0.0,
            "best_metric_name": monitored_metric,
            "epochs_ran": 0,
        }

        start_time = time.time()

        # Number of training iterations per epoch = length of labeled loader
        # Unlabeled loader cycles infinitely to keep up.
        iters_per_epoch = len(labeled_loader)

        for epoch in range(1, epochs + 1):
            epoch_metrics = self._train_one_epoch(
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                optimizer=optimizer,
                criterion=criterion,
                iters_per_epoch=iters_per_epoch,
                gradient_clip=gradient_clip,
            )

            val_metrics = self._validate(val_loader, criterion)

            current_lr = float(optimizer.param_groups[0]["lr"])
            history["train_loss"].append(epoch_metrics["loss"])
            history["train_supervised_loss"].append(epoch_metrics["sup_loss"])
            history["train_unsupervised_loss"].append(epoch_metrics["unsup_loss"])
            history["train_acc"].append(epoch_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["pseudo_label_ratio"].append(epoch_metrics["pseudo_ratio"])
            history["pseudo_label_count"].append(epoch_metrics["pseudo_count"])
            history["learning_rate"].append(current_lr)

            # Scheduler step
            if scheduler is not None:
                metric_for_scheduler = val_metrics.get("macro_f1", val_metrics["loss"])
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metric_for_scheduler)
                else:
                    scheduler.step()

            # Best model tracking
            current_metric = val_metrics.get("macro_f1", val_metrics["acc"])
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_model_wts = copy.deepcopy(self.model.state_dict())

            logger.info(
                "[FixMatch] Epoch %2d/%d | LR: %.6f | "
                "Train: loss=%.4f (sup=%.4f, unsup=%.4f), acc=%.4f | "
                "Val: loss=%.4f, acc=%.4f, F1=%.4f | "
                "Pseudo: %d (%.1f%%)",
                epoch,
                epochs,
                current_lr,
                epoch_metrics["loss"],
                epoch_metrics["sup_loss"],
                epoch_metrics["unsup_loss"],
                epoch_metrics["acc"],
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["macro_f1"],
                epoch_metrics["pseudo_count"],
                epoch_metrics["pseudo_ratio"] * 100,
            )

            history["epochs_ran"] = epoch

        self.model.load_state_dict(best_model_wts)
        elapsed = time.time() - start_time
        history["total_time"] = elapsed
        history["best_epoch"] = best_epoch
        history["best_metric"] = best_metric

        logger.info(
            "FixMatch training complete in %.1fs (best %s: %.4f at epoch %d)",
            elapsed,
            monitored_metric,
            best_metric,
            best_epoch,
        )

        return self.model, history

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        iters_per_epoch: int,
        gradient_clip: Optional[float],
    ) -> Dict[str, float]:
        """Run one epoch of FixMatch training.

        Returns a dict with keys: loss, sup_loss, unsup_loss, acc,
        pseudo_ratio, pseudo_count.
        """
        self.model.train()
        unlabeled_iter = _InfiniteDataIterator(unlabeled_loader)

        total_loss = 0.0
        total_sup_loss = 0.0
        total_unsup_loss = 0.0
        correct = 0
        total_samples = 0
        total_pseudo_count = 0
        total_unlabeled_count = 0

        for images_l, labels_l in labeled_loader:
            images_l = images_l.to(self.device)
            labels_l = labels_l.to(self.device)

            # Unlabeled batch: (weak_aug, strong_aug)
            weak_u, strong_u = next(unlabeled_iter)
            weak_u = weak_u.to(self.device)
            strong_u = strong_u.to(self.device)

            # --- Supervised loss ---
            logits_l = self.model(images_l)
            loss_s = criterion(logits_l, labels_l)

            # --- Pseudo-label generation (no gradient through teacher) ---
            with torch.no_grad():
                logits_u_weak = self.model(weak_u)
                probs_u_weak = F.softmax(logits_u_weak, dim=1)
                max_probs, pseudo_labels = probs_u_weak.max(dim=1)
                mask = max_probs.ge(self.confidence_threshold)

            # --- Unsupervised loss (only on confident pseudo-labels) ---
            num_above = mask.sum().item()
            total_pseudo_count += num_above
            total_unlabeled_count += weak_u.size(0)

            if num_above > 0:
                logits_u_strong = self.model(strong_u[mask])
                loss_u = F.cross_entropy(logits_u_strong, pseudo_labels[mask])
            else:
                loss_u = torch.tensor(0.0, device=self.device)

            # --- Combined loss ---
            loss = loss_s + self.lambda_u * loss_u

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            optimizer.step()

            batch_size = labels_l.size(0)
            total_loss += loss.item() * batch_size
            total_sup_loss += loss_s.item() * batch_size
            total_unsup_loss += (self.lambda_u * loss_u.item()) * batch_size
            predicted = logits_l.argmax(dim=1)
            correct += (predicted == labels_l).sum().item()
            total_samples += batch_size

        n = max(total_samples, 1)
        pseudo_ratio = total_pseudo_count / max(total_unlabeled_count, 1)
        return {
            "loss": total_loss / n,
            "sup_loss": total_sup_loss / n,
            "unsup_loss": total_unsup_loss / n,
            "acc": correct / n,
            "pseudo_ratio": pseudo_ratio,
            "pseudo_count": total_pseudo_count,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate on validation set (labeled data only)."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                predicted = logits.argmax(dim=1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(predicted.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        n = max(val_total, 1)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        return {
            "loss": val_loss / n,
            "acc": val_correct / n,
            "macro_f1": macro_f1,
        }


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def extract_unlabeled_maps(
    df: "pd.DataFrame",
    known_classes: List[str],
) -> List[np.ndarray]:
    """Extract wafer maps that are NOT in known_classes.

    The WM-811K dataset has ~811K total wafers but only ~172K have a failure
    type label in ``known_classes``. The remaining ~640K are either marked
    as empty/unknown or have no label at all. This function collects those
    unlabeled maps for use in semi-supervised training.

    Args:
        df: Full WM-811K DataFrame with ``failureClass`` and ``waferMap`` columns.
        known_classes: List of valid label strings (e.g., KNOWN_CLASSES).

    Returns:
        List of raw numpy wafer-map arrays (variable sizes).
    """
    unlabeled_mask = ~df["failureClass"].isin(known_classes)
    unlabeled_df = df[unlabeled_mask]

    maps: List[np.ndarray] = []
    for wm in unlabeled_df["waferMap"].values:
        if isinstance(wm, np.ndarray) and wm.size > 0 and wm.ndim == 2:
            maps.append(wm)

    logger.info(
        "Extracted %d unlabeled wafer maps (out of %d non-labeled rows)",
        len(maps),
        len(unlabeled_df),
    )
    return maps
