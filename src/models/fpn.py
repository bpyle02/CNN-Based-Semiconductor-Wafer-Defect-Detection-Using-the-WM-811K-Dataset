"""
Feature Pyramid Network (FPN) variant for wafer defect classification.

Fuses multi-scale features from a 4-stage CNN backbone via top-down
lateral connections, improving detection of defects at varying spatial
scales (e.g. large Edge-Ring vs. thin Scratch vs. small Loc).

References:
    [91] Lin et al. (2017). "Feature Pyramid Networks for Object Detection".
         arXiv:1612.03144
    [47] Ioffe & Szegedy (2015). "Batch Normalization". arXiv:1502.03167
    [48] Srivastava et al. (2014). "Dropout: Preventing Overfitting". JMLR
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FPNBlock(nn.Module):
    """Feature Pyramid Network block.

    Combines a top-down pathway (upsampled coarser features) with
    lateral connections (same-level features) via 1x1 convolutions.

    The lateral connections project each backbone stage to a common
    channel dimension (``out_channels``). Starting from the coarsest
    level, each feature map is upsampled by 2x and element-wise added
    to the next finer lateral output, propagating high-level semantics
    to lower levels while retaining spatial detail.

    Reference: [91] Lin et al. (2017). arXiv:1612.03144

    Args:
        in_channels_list: Number of channels at each backbone stage,
            ordered from finest (largest spatial) to coarsest (smallest
            spatial).  E.g. ``[32, 64, 128, 256]``.
        out_channels: Common channel dimension for all FPN output levels.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions: project each stage to out_channels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list]
        )

        # Optional 3x3 smoothing convolutions after merging (standard FPN)
        self.smooth_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build the feature pyramid from backbone feature maps.

        Args:
            features: List of feature maps from each stage, ordered
                finest-to-coarsest.
                E.g. ``[(B,32,48,48), (B,64,24,24), (B,128,12,12), (B,256,6,6)]``

        Returns:
            List of FPN feature maps (same ordering), all with
            ``out_channels`` channels. Spatial sizes match the input
            features at each level.
        """
        num_levels = len(features)

        # Compute lateral projections
        laterals = [self.lateral_convs[i](features[i]) for i in range(num_levels)]

        # Top-down pathway: upsample coarser level, add to finer lateral
        for i in range(num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode="nearest",
            )
            laterals[i] = laterals[i] + upsampled

        # 3x3 smoothing to reduce aliasing from upsampling
        outputs = [self.smooth_convs[i](laterals[i]) for i in range(num_levels)]

        return outputs


class WaferCNNFPN(nn.Module):
    """Custom CNN with FPN for multi-scale defect detection.

    Extends WaferCNN by extracting intermediate feature maps from each
    of the 4 convolutional stages and fusing them via a Feature Pyramid
    Network. The fused multi-scale features are globally average-pooled
    and concatenated for classification, allowing the model to leverage
    both fine spatial detail (important for small Loc defects and thin
    Scratch patterns) and coarse semantic information (important for
    large Edge-Ring defects).

    Architecture:
        - 4 conv blocks identical to WaferCNN:
          Conv-BN-ReLU-Conv-BN-ReLU-MaxPool per stage
        - Feature extraction at each stage:
          F1(48x48), F2(24x24), F3(12x12), F4(6x6)
        - FPN fuses into P1, P2, P3, P4 (all ``fpn_out_channels``)
        - Global average pool each Pi, concatenate
        - Classifier: Linear(4 * fpn_out_channels, num_classes)

    References:
        [91] Lin et al. (2017). arXiv:1612.03144
        [47] Ioffe & Szegedy (2015). arXiv:1502.03167
        [48] Srivastava et al. (2014). JMLR

    Args:
        num_classes: Number of output classes.
        feature_channels: Output channels per conv stage.
        fpn_out_channels: Common channel dimension for FPN levels.
        dropout_rate: Dropout before the classifier head.
        input_channels: Number of input image channels.
        use_batch_norm: Apply BatchNorm after each convolution.
    """

    def __init__(
        self,
        num_classes: int = 9,
        feature_channels: Sequence[int] = (32, 64, 128, 256),
        fpn_out_channels: int = 128,
        dropout_rate: float = 0.5,
        input_channels: int = 3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if not feature_channels:
            raise ValueError("feature_channels must contain at least one stage")
        feature_channels = tuple(feature_channels)
        if any(ch <= 0 for ch in feature_channels):
            raise ValueError("feature_channels values must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if fpn_out_channels <= 0:
            raise ValueError("fpn_out_channels must be positive")

        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.fpn_out_channels = fpn_out_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build each conv stage as a separate Sequential block so we can
        # extract intermediate features.
        self.stages = nn.ModuleList()
        in_ch = input_channels
        for out_ch in feature_channels:
            block_layers: list[nn.Module] = []
            block_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batch_norm:
                block_layers.append(nn.BatchNorm2d(out_ch))
            block_layers.append(nn.ReLU(inplace=True))
            block_layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            if use_batch_norm:
                block_layers.append(nn.BatchNorm2d(out_ch))
            block_layers.append(nn.ReLU(inplace=True))
            block_layers.append(nn.MaxPool2d(2, 2))
            self.stages.append(nn.Sequential(*block_layers))
            in_ch = out_ch

        # FPN module
        self.fpn = FPNBlock(
            in_channels_list=list(feature_channels),
            out_channels=fpn_out_channels,
        )

        # Global average pool per FPN level then concatenate
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        num_levels = len(feature_channels)
        concat_dim = num_levels * fpn_out_channels
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(concat_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone, FPN, and classifier.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        # Extract features from each backbone stage
        stage_features: list[torch.Tensor] = []
        out = x
        for stage in self.stages:
            out = stage(out)
            stage_features.append(out)

        # FPN fusion
        fpn_features = self.fpn(stage_features)

        # Global average pool each level, flatten, concatenate
        pooled = [torch.flatten(self.gap(feat), 1) for feat in fpn_features]
        fused = torch.cat(pooled, dim=1)

        # Classify
        return self.classifier(fused)
