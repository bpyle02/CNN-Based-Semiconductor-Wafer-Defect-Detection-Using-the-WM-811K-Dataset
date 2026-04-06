"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

Implements SimCLR (Chen et al., ICML 2020) for self-supervised pretraining on unlabeled wafer maps.
Learns useful representations without labels by maximizing agreement between different augmentations.

References:
    [11] Kang et al. (2021). "WaPIRL: Representation Learning for Wafer Maps". DOI:10.1109/TSM.2021.3064435
    [37] Chen et al. (2020). "SimCLR: Contrastive Learning of Visual Representations". arXiv:2002.05709
    [38] Grill et al. (2020). "BYOL: Bootstrap Your Own Latent". arXiv:2006.07733
    [39] He et al. (2020). "MoCo: Momentum Contrast". arXiv:1911.05722
    [53] (2021). "Semi-Supervised Learning for Wafer Map Defect Classification"
    [78] Schroff et al. (2015). "FaceNet: Triplet Loss". arXiv:1503.03832
    [82] Sohn (2016). "Improved Deep Metric Learning with Multi-Class N-Pair Loss". arXiv:1708.01682
    [85] Caron et al. (2020). "SwAV: Unsupervised Multi-Crop Assignments". arXiv:2006.09882
    [111] Sohn et al. (2020). "FixMatch". arXiv:2001.07685
    [114] Berthelot et al. (2019). "MixMatch". arXiv:1905.02249
    [115] Xie et al. (2020). "UDA: Unsupervised Data Augmentation". arXiv:1904.12848
    [119] Tarvainen & Valpola (2017). "Mean Teacher". arXiv:1703.01780
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimCLRProjection(nn.Module):
    """
    Projection head for SimCLR.

    Consists of a non-linear projection head: ReLU -> Linear -> normalized output.

    Args:
        input_dim: Dimension of input features from encoder
        hidden_dim: Dimension of hidden layer (default 2048)
        output_dim: Dimension of output projection (default 128)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, input_dim)

        Returns:
            Normalized projection of shape (B, output_dim)
        """
        x = self.proj(x)
        return F.normalize(x, dim=1)


class SimCLREncoder(nn.Module):
    """
    SimCLR encoder: backbone + projection head.

    Wraps existing model (e.g., ResNet, EfficientNet) and adds SimCLR projection head.

    Args:
        backbone: PyTorch model (e.g., ResNet)
        feature_dim: Dimension of features from backbone (before projection)
        projection_dim: Dimension of projection head output (default 128)
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.projection_head = SimCLRProjection(
            input_dim=feature_dim,
            hidden_dim=2048,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of:
                - features: Backbone output of shape (B, feature_dim)
                - projection: Normalized projection of shape (B, projection_dim)
        """
        # Remove final classification layer
        # Assumes backbone has .features or similar for feature extraction
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features(x)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        else:
            # For ResNet-like models
            features = self.backbone(x)

        projection = self.projection_head(features)

        return features, projection


class SimCLRLoss(nn.Module):
    """
    SimCLR contrastive loss (NT-Xent loss).

    Normalized temperature-scaled cross entropy loss for contrastive learning.

    Args:
        temperature: Temperature parameter for similarity scaling (default 0.07)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss between two augmented views.

        Args:
            z_i: Projections from view i of shape (B, projection_dim)
            z_j: Projections from view j of shape (B, projection_dim)

        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]

        # Concatenate both views: (2B, projection_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix: (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Labels: diagonal pairs are positives
        # Positive pairs: (i, B+i) and (B+i, i)
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])

        # Remove diagonal (self-similarity)
        # Create mask for valid pairs (exclude self-similarity)
        mask = ~torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels, reduction='mean')

        return loss


class SimCLRPretrainer:
    """
    SimCLR pretraining pipeline.

    Handles self-supervised pretraining on unlabeled data.

    Args:
        encoder: SimCLREncoder instance
        optimizer: PyTorch optimizer
        device: Device to run on (cuda or cpu)
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        encoder: SimCLREncoder,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        temperature: float = 0.07,
    ) -> None:
        self.encoder = encoder.to(device)
        self.optimizer = optimizer
        self.device = device
        self.criterion = SimCLRLoss(temperature=temperature)

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Run one training epoch.

        Args:
            train_loader: DataLoader yielding (images, _) pairs

        Returns:
            Average loss over epoch
        """
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            # Handle both single image and paired views
            if isinstance(images, (list, tuple)):
                x_i, x_j = images[0].to(self.device), images[1].to(self.device)
            else:
                # Assume images contains both views concatenated
                batch_size = images.shape[0] // 2
                x_i = images[:batch_size].to(self.device)
                x_j = images[batch_size:].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            _, z_i = self.encoder(x_i)
            _, z_j = self.encoder(x_j)

            # Contrastive loss
            loss = self.criterion(z_i, z_j)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def pretrain(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> list:
        """
        Run full pretraining loop.

        Args:
            train_loader: DataLoader for unlabeled data
            epochs: Number of pretraining epochs
            scheduler: Optional learning rate scheduler

        Returns:
            List of loss values per epoch
        """
        losses = []

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            losses.append(loss)

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {loss:.6f}")

        logger.info(f"Pretraining complete. Final loss: {losses[-1]:.6f}")
        return losses

    def get_backbone(self) -> nn.Module:
        """
        Extract pretrained backbone for downstream tasks.

        Returns:
            Backbone model suitable for fine-tuning
        """
        return self.encoder.backbone


class ContrastiveBYOLLoss(nn.Module):
    """
    BYOL-style contrastive loss (non-contrastive alternative to SimCLR).

    BYOL (Bootstrap Your Own Latent) doesn't require negative pairs.
    Uses momentum encoder to create stability.

    Args:
        temperature: Temperature for similarity (default 0.1)
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        proj_online: torch.Tensor,
        proj_momentum: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BYOL loss (cosine similarity between normalized projections).

        Args:
            proj_online: Online projection of shape (B, projection_dim)
            proj_momentum: Momentum projection of shape (B, projection_dim)

        Returns:
            Scalar loss value
        """
        # Normalize projections
        proj_online = F.normalize(proj_online, dim=1)
        proj_momentum = F.normalize(proj_momentum, dim=1)

        # Cosine similarity loss: MSE between normalized projections
        loss = 2 - 2 * (proj_online * proj_momentum).sum(dim=1).mean()

        return loss
