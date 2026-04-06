"""
RIDE: Routing Diverse Distributed Experts for Long-Tailed Recognition.

Uses K expert classifiers with a shared backbone. A gating network routes
each sample to the appropriate expert based on learned class-frequency
awareness. Experts are encouraged to be diverse via a diversity loss.

Reference: [181] Wang et al. (2022). "RIDE: Routing Diverse Distributed
    Experts". arXiv:2208.09043
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RIDEModel(nn.Module):
    """RIDE: Routing Diverse Distributed Experts for Long-Tailed Recognition.

    Uses K expert classifiers with a shared backbone. A gating network routes
    each sample to the appropriate expert based on learned class-frequency
    awareness. Experts are encouraged to be diverse via a diversity loss.

    Reference: [181] Wang et al. (2022). arXiv:2208.09043
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 9,
        num_experts: int = 3,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")

        self.backbone = backbone
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # K expert classifiers (each is a Linear layer)
        self.experts = nn.ModuleList([
            nn.Linear(feature_dim, num_classes) for _ in range(num_experts)
        ])

        # Gating network: routes each sample to experts
        gate_hidden = max(feature_dim // reduction, 1)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, num_experts),
        )

        self._backbone_type = self._detect_backbone_type()

    def _detect_backbone_type(self) -> str:
        """Detect backbone architecture for feature extraction dispatch."""
        backbone = self.backbone
        cls_name = type(backbone).__name__

        # WaferCNN / WaferCNNFPN: has .features (Sequential) and .avg_pool
        if hasattr(backbone, "features") and hasattr(backbone, "avg_pool"):
            if hasattr(backbone, "classifier"):
                return "wafer_cnn"

        # torchvision ResNet: has layer1..layer4 and fc
        if hasattr(backbone, "layer4") and hasattr(backbone, "fc"):
            return "resnet"

        # torchvision EfficientNet: has .features and .classifier
        if hasattr(backbone, "features") and hasattr(backbone, "classifier"):
            if "EfficientNet" in cls_name or "efficientnet" in cls_name.lower():
                return "efficientnet"
            # Also handles MobileNet and similar torchvision models
            if hasattr(backbone, "avgpool"):
                return "efficientnet"

        # Generic fallback
        return "generic"

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (handle different backbone types).

        Dispatches to the appropriate extraction path based on detected
        backbone architecture. Returns a (B, feature_dim) tensor.
        """
        if self._backbone_type == "wafer_cnn":
            return self._extract_wafer_cnn(x)
        if self._backbone_type == "resnet":
            return self._extract_resnet(x)
        if self._backbone_type == "efficientnet":
            return self._extract_efficientnet(x)
        return self._extract_generic(x)

    def _extract_wafer_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """WaferCNN: features -> avg_pool -> flatten."""
        feat = self.backbone.features(x)
        feat = self.backbone.avg_pool(feat)
        return torch.flatten(feat, 1)

    def _extract_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet: forward through all layers except fc."""
        bb = self.backbone
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)
        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)
        x = bb.layer4(x)
        x = bb.avgpool(x)
        return torch.flatten(x, 1)

    def _extract_efficientnet(self, x: torch.Tensor) -> torch.Tensor:
        """EfficientNet: features -> adaptive_avg_pool -> flatten."""
        feat = self.backbone.features(x)
        feat = F.adaptive_avg_pool2d(feat, 1)
        return torch.flatten(feat, 1)

    def _extract_generic(self, x: torch.Tensor) -> torch.Tensor:
        """Generic fallback: call backbone(x), flatten if spatial."""
        out = self.backbone(x)
        if out.dim() > 2:
            out = F.adaptive_avg_pool2d(out, 1)
            out = torch.flatten(out, 1)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            logits: (B, num_classes) -- gated combination of expert outputs
            expert_outputs: (B, num_experts, num_classes) -- individual expert logits
        """
        features = self._extract_features(x)  # (B, feature_dim)

        # Gate: soft routing weights
        gate_weights = torch.softmax(self.gate(features), dim=1)  # (B, num_experts)

        # Expert predictions
        expert_logits = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )  # (B, num_experts, num_classes)

        # Weighted combination
        # gate_weights: (B, num_experts, 1) * expert_logits: (B, num_experts, num_classes)
        logits = (gate_weights.unsqueeze(-1) * expert_logits).sum(dim=1)  # (B, num_classes)

        return logits, expert_logits


class RIDELoss(nn.Module):
    """RIDE training loss: classification + diversity.

    L = L_cls + lambda_div * L_diversity

    L_cls: standard classification loss on gated output
    L_diversity: encourages experts to make different predictions
                 via KL divergence between expert pairs

    Reference: [181] Wang et al. (2022). arXiv:2208.09043
    """

    def __init__(
        self,
        criterion: nn.Module,
        diversity_weight: float = 0.1,
        num_experts: int = 3,
    ) -> None:
        super().__init__()
        if diversity_weight < 0:
            raise ValueError("diversity_weight must be non-negative")
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")

        self.criterion = criterion
        self.diversity_weight = diversity_weight
        self.num_experts = num_experts

    def forward(
        self,
        logits: torch.Tensor,
        expert_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RIDE loss.

        Args:
            logits: (B, num_classes) gated combined output
            expert_outputs: (B, num_experts, num_classes) individual expert logits
            targets: (B,) integer class labels

        Returns:
            Scalar loss = cls_loss + diversity_weight * diversity_loss
        """
        cls_loss = self.criterion(logits, targets)

        if self.diversity_weight > 0:
            div_loss = self._diversity_loss(expert_outputs)
            return cls_loss + self.diversity_weight * div_loss
        return cls_loss

    def _diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Pairwise KL divergence between experts to encourage diversity.

        Computes the mean KL(p_i || p_j) over all ordered pairs (i, j)
        where i != j, using softmax probabilities from each expert.

        Args:
            expert_outputs: (B, num_experts, num_classes) raw logits

        Returns:
            Scalar diversity loss (negative: we want to maximize divergence,
            so we negate the KL mean to create a loss that decreases when
            experts become more diverse).
        """
        num_experts = expert_outputs.size(1)

        # Convert logits to log-probs and probs
        log_probs = F.log_softmax(expert_outputs, dim=-1)  # (B, K, C)
        probs = F.softmax(expert_outputs, dim=-1)           # (B, K, C)

        # Compute mean pairwise KL divergence
        total_kl = torch.tensor(0.0, device=expert_outputs.device)
        num_pairs = 0

        for i in range(num_experts):
            for j in range(num_experts):
                if i == j:
                    continue
                # KL(p_i || p_j) = sum(p_i * (log p_i - log p_j))
                kl_ij = F.kl_div(
                    log_probs[:, j, :],  # input: log Q
                    probs[:, i, :],       # target: P
                    reduction="batchmean",
                )
                total_kl = total_kl + kl_ij
                num_pairs += 1

        mean_kl = total_kl / max(num_pairs, 1)

        # Negate: high KL means experts are diverse (good), so we want to
        # minimize -KL to encourage diversity.
        return -mean_kl


def _get_feature_dim_for_backbone(backbone_name: str, backbone: nn.Module) -> int:
    """Determine the feature dimensionality for a given backbone.

    Args:
        backbone_name: one of 'cnn', 'cnn_fpn', 'resnet', 'efficientnet',
            'vit', 'swin'
        backbone: the instantiated backbone module

    Returns:
        Integer feature dimension.
    """
    name = backbone_name.lower()

    # WaferCNN: last feature_channels element
    if name in ("cnn", "cnn_fpn"):
        if hasattr(backbone, "feature_channels"):
            return backbone.feature_channels[-1]
        # Fallback: probe with dummy input
        return _probe_feature_dim(backbone)

    # ResNet: fc.in_features (before we replaced fc)
    if name == "resnet":
        if hasattr(backbone, "fc"):
            fc = backbone.fc
            # Could be Sequential (from build_classifier_head) or Linear
            if isinstance(fc, nn.Linear):
                return fc.in_features
            if isinstance(fc, nn.Sequential):
                for layer in fc:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
        return 512  # ResNet-18 default

    # EfficientNet: classifier[1].in_features or features output channels
    if name in ("efficientnet", "effnet"):
        if hasattr(backbone, "classifier"):
            clf = backbone.classifier
            if isinstance(clf, nn.Sequential):
                for layer in clf:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            elif isinstance(clf, nn.Linear):
                return clf.in_features
        return 1280  # EfficientNet-B0 default

    # ViT / Swin: probe
    return _probe_feature_dim(backbone)


def _probe_feature_dim(backbone: nn.Module) -> int:
    """Probe feature dim by running a dummy forward pass.

    Tries to extract features the same way RIDEModel would, then
    measures the output dimension.
    """
    backbone.eval()
    dummy = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        # Try WaferCNN-style extraction
        if hasattr(backbone, "features") and hasattr(backbone, "avg_pool"):
            feat = backbone.features(dummy)
            feat = backbone.avg_pool(feat)
            feat = torch.flatten(feat, 1)
            return feat.size(1)

        # Try ResNet-style
        if hasattr(backbone, "layer4") and hasattr(backbone, "avgpool"):
            x = backbone.conv1(dummy)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            x = backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x.size(1)

        # Try EfficientNet-style
        if hasattr(backbone, "features"):
            feat = backbone.features(dummy)
            feat = F.adaptive_avg_pool2d(feat, 1)
            feat = torch.flatten(feat, 1)
            return feat.size(1)

        # Full forward pass fallback
        out = backbone(dummy)
        if out.dim() > 2:
            out = F.adaptive_avg_pool2d(out, 1)
            out = torch.flatten(out, 1)
        return out.size(1)


def build_ride_model(
    backbone_name: str,
    num_classes: int = 9,
    num_experts: int = 3,
    reduction: int = 4,
    device: str = "cpu",
) -> RIDEModel:
    """Factory to create RIDE model from a backbone name.

    Creates the appropriate backbone (WaferCNN, ResNet, etc.), determines
    feature_dim, and wraps in RIDEModel.

    Args:
        backbone_name: 'cnn', 'resnet', 'efficientnet', 'vit', 'swin'
        num_classes: number of output classes
        num_experts: number of expert classifiers
        reduction: gate hidden-dimension reduction factor
        device: target device

    Returns:
        RIDEModel instance on the specified device.
    """
    from src.models.cnn import WaferCNN
    from src.models.fpn import WaferCNNFPN
    from src.models.vit import get_vit_small
    from src.models.swin import get_swin_tiny

    name = backbone_name.lower()

    if name == "cnn":
        backbone = WaferCNN(num_classes=num_classes)
    elif name == "cnn_fpn":
        backbone = WaferCNNFPN(num_classes=num_classes)
    elif name == "resnet":
        from src.models.pretrained import get_resnet18
        backbone = get_resnet18(num_classes=num_classes, pretrained=True, freeze_until=None)
    elif name in ("efficientnet", "effnet"):
        from src.models.pretrained import get_efficientnet_b0
        backbone = get_efficientnet_b0(num_classes=num_classes, pretrained=True, freeze_until=None)
    elif name == "vit":
        backbone = get_vit_small(num_classes=num_classes, image_size=96, in_channels=3)
    elif name == "swin":
        backbone = get_swin_tiny(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Expected one of: cnn, cnn_fpn, resnet, efficientnet, vit, swin"
        )

    feature_dim = _get_feature_dim_for_backbone(name, backbone)
    logger.info(
        "RIDE: backbone=%s, feature_dim=%d, num_experts=%d, num_classes=%d",
        name, feature_dim, num_experts, num_classes,
    )

    model = RIDEModel(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        reduction=reduction,
    )
    return model.to(device)
