"""
Domain Adaptation for cross-plant wafer defect detection.

Implements techniques to adapt models trained on one wafer plant
to generalize to different plants with different defect characteristics.

Methods:
    - Fine-tuning: Simple transfer learning on target domain
    - CORAL: Correlation Alignment for domain-invariant features
    - Domain-adversarial training: Adversarial loss to align source and target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CORLAlignmentLoss(nn.Module):
    """
    CORAL (Correlation Alignment) loss for domain adaptation.

    Aligns the second-order statistics (correlations) between source and target domains.

    Args:
        lambda_coral: Weight of CORAL loss (default 1.0)
    """

    def __init__(self, lambda_coral: float = 1.0) -> None:
        super().__init__()
        self.lambda_coral = lambda_coral

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CORAL loss.

        Args:
            source_features: Features from source domain of shape (B_s, feat_dim)
            target_features: Features from target domain of shape (B_t, feat_dim)

        Returns:
            Scalar CORAL loss
        """
        source_features = source_features - source_features.mean(dim=0)
        target_features = target_features - target_features.mean(dim=0)

        # Covariance matrices
        source_cov = torch.mm(
            source_features.T,
            source_features,
        ) / (source_features.shape[0] - 1)
        target_cov = torch.mm(
            target_features.T,
            target_features,
        ) / (target_features.shape[0] - 1)

        # CORAL loss: Frobenius norm of difference
        coral_loss = torch.norm(source_cov - target_cov, p='fro')

        return self.lambda_coral * coral_loss


class DomainAdversarialLoss(nn.Module):
    """
    Domain-adversarial loss for unsupervised domain adaptation.

    Uses an adversarial discriminator to make features domain-invariant.

    Args:
        feature_dim: Dimension of input features
        hidden_dim: Dimension of discriminator hidden layer (default 256)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Domain discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss.

        Args:
            source_features: Features from source domain of shape (B_s, feat_dim)
            target_features: Features from target domain of shape (B_t, feat_dim)

        Returns:
            Scalar adversarial loss
        """
        # Labels: 0=source, 1=target
        source_pred = self.discriminator(source_features)
        target_pred = self.discriminator(target_features)

        source_loss = F.binary_cross_entropy(
            source_pred,
            torch.zeros_like(source_pred),
        )
        target_loss = F.binary_cross_entropy(
            target_pred,
            torch.ones_like(target_pred),
        )

        return source_loss + target_loss


class DomainAdaptationTrainer:
    """
    Domain adaptation training pipeline.

    Supports fine-tuning, CORAL, and adversarial domain adaptation methods.

    Args:
        model: PyTorch model to adapt
        method: Adaptation method ('fine_tuning', 'coral', 'adversarial')
        device: Device to run on (cuda or cpu)
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = 'fine_tuning',
        device: str = 'cuda',
    ) -> None:
        if method not in ['fine_tuning', 'coral', 'adversarial']:
            raise ValueError(f"Unknown method: {method}")

        self.model = model.to(device)
        self.method = method
        self.device = device
        self.discriminator = None

    def prepare_for_adaptation(
        self,
        freeze_backbone: bool = True,
        num_layers_unfreeze: Optional[int] = None,
    ) -> None:
        """
        Prepare model for domain adaptation.

        Args:
            freeze_backbone: Whether to freeze earlier layers
            num_layers_unfreeze: Number of final layers to unfreeze
        """
        if freeze_backbone and num_layers_unfreeze is None:
            # Freeze all but last layer
            for name, param in self.model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
        elif num_layers_unfreeze is not None:
            # Unfreeze last N layers
            params_list = list(self.model.named_parameters())
            for name, param in params_list[:-num_layers_unfreeze]:
                param.requires_grad = False
            for name, param in params_list[-num_layers_unfreeze:]:
                param.requires_grad = True

    def adapt(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        target_labels_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 5,
        lambda_domain: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Run domain adaptation.

        Args:
            source_loader: Source domain training data
            target_loader: Target domain data (unlabeled or labeled)
            target_labels_loader: Optional labeled target data for semi-supervised
            criterion: Classification loss (required for coral/adversarial with labels)
            optimizer: Optimizer for model
            epochs: Number of adaptation epochs
            lambda_domain: Weight for domain loss term

        Returns:
            Dictionary with training history
        """
        if self.method == 'fine_tuning':
            return self._adapt_fine_tuning(
                target_loader,
                criterion,
                optimizer,
                epochs,
            )
        elif self.method == 'coral':
            return self._adapt_coral(
                source_loader,
                target_loader,
                criterion,
                optimizer,
                epochs,
                lambda_domain,
            )
        elif self.method == 'adversarial':
            return self._adapt_adversarial(
                source_loader,
                target_loader,
                criterion,
                optimizer,
                epochs,
                lambda_domain,
            )

    def _adapt_fine_tuning(
        self,
        target_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> Dict[str, Any]:
        """Fine-tuning on target domain."""
        history = defaultdict(list)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for images, labels in target_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            history['loss'].append(avg_loss)

            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(f"Fine-tuning Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return dict(history)

    def _adapt_coral(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        lambda_domain: float,
    ) -> Dict[str, Any]:
        """CORAL domain adaptation."""
        coral_loss_fn = CORLAlignmentLoss(lambda_coral=lambda_domain)
        history = defaultdict(list)

        # Combine loaders (cycle target if shorter)
        max_batches = max(len(source_loader), len(target_loader))
        source_cycle = iter(source_loader)
        target_cycle = iter(target_loader)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_coral_loss = 0.0
            num_batches = 0

            source_cycle = iter(source_loader)
            target_cycle = iter(target_loader)

            for batch_idx in range(max_batches):
                # Get batches (cycle if needed)
                try:
                    source_images, source_labels = next(source_cycle)
                except StopIteration:
                    source_cycle = iter(source_loader)
                    source_images, source_labels = next(source_cycle)

                try:
                    target_images, target_labels = next(target_cycle)
                except StopIteration:
                    target_cycle = iter(target_loader)
                    target_images, target_labels = next(target_cycle)

                source_images = source_images.to(self.device)
                source_labels = source_labels.to(self.device)
                target_images = target_images.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                source_outputs = self.model(source_images)
                target_outputs = self.model(target_images)

                # Classification loss on source
                class_loss = criterion(source_outputs, source_labels)

                # Extract features for CORAL (get intermediate features)
                if hasattr(self.model, 'features'):
                    source_feats = self.model.features(source_images)
                    target_feats = self.model.features(target_images)
                else:
                    # Fallback: use outputs as features
                    source_feats = source_outputs
                    target_feats = target_outputs

                coral_loss = coral_loss_fn(source_feats, target_feats)

                batch_loss = class_loss + coral_loss

                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                total_coral_loss += coral_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_coral = total_coral_loss / num_batches
            history['loss'].append(avg_loss)
            history['coral_loss'].append(avg_coral)

            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    f"CORAL Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {avg_loss:.6f}, CORAL: {avg_coral:.6f}"
                )

        return dict(history)

    def _adapt_adversarial(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        lambda_domain: float,
    ) -> Dict[str, Any]:
        """Domain-adversarial training."""
        # Initialize discriminator
        feature_dim = 512  # Typical feature dimension
        self.discriminator = DomainAdversarialLoss(
            feature_dim=feature_dim,
        ).to(self.device)
        disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4,
        )

        history = defaultdict(list)

        for epoch in range(epochs):
            self.model.train()
            self.discriminator.train()

            total_loss = 0.0
            total_disc_loss = 0.0
            num_batches = 0

            source_cycle = iter(source_loader)
            target_cycle = iter(target_loader)

            for batch_idx in range(max(len(source_loader), len(target_loader))):
                # Get batches
                try:
                    source_images, source_labels = next(source_cycle)
                except StopIteration:
                    source_cycle = iter(source_loader)
                    source_images, source_labels = next(source_cycle)

                try:
                    target_images, _ = next(target_cycle)
                except StopIteration:
                    target_cycle = iter(target_loader)
                    target_images, _ = next(target_cycle)

                source_images = source_images.to(self.device)
                source_labels = source_labels.to(self.device)
                target_images = target_images.to(self.device)

                # Update feature extractor and classifier
                optimizer.zero_grad()

                source_outputs = self.model(source_images)
                class_loss = criterion(source_outputs, source_labels)

                # Get features for discriminator
                if hasattr(self.model, 'features'):
                    source_feats = self.model.features(source_images)
                    target_feats = self.model.features(target_images)
                else:
                    source_feats = source_outputs
                    target_feats = target_outputs

                # Adversarial loss: fool discriminator
                adv_loss = -lambda_domain * self.discriminator(source_feats, target_feats)

                total_model_loss = class_loss + adv_loss
                total_model_loss.backward()
                optimizer.step()

                # Update discriminator
                disc_optimizer.zero_grad()
                disc_loss = self.discriminator(source_feats.detach(), target_feats.detach())
                disc_loss.backward()
                disc_optimizer.step()

                total_loss += total_model_loss.item()
                total_disc_loss += disc_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_disc = total_disc_loss / num_batches
            history['loss'].append(avg_loss)
            history['disc_loss'].append(avg_disc)

            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    f"Adversarial Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {avg_loss:.6f}, Disc: {avg_disc:.6f}"
                )

        return dict(history)
