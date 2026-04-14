"""
Supervised Contrastive Learning (SupCon) for wafer defect detection.

Two-stage training: (1) pretrain backbone with SupConLoss to learn discriminative
embeddings that pull same-class samples together and push different-class samples
apart, then (2) fine-tune a linear classifier on the frozen backbone.

Especially effective for imbalanced datasets like WM-811K (150:1 class ratio)
because the contrastive objective treats every same-class pair equally regardless
of class size.

Also includes Parametric Contrastive Learning (PaCo) which extends SupCon with
learnable class-specific prototype centers, giving rare classes more positive
anchors and compensating for imbalance.

References:
    [76] Khosla et al. (2020). "Supervised Contrastive Learning". arXiv:2004.11362
    [37] Chen et al. (2020). "SimCLR: Contrastive Learning of Visual Representations". arXiv:2002.05709
    [182] Cui et al. (2021). "PaCo: Parametric Contrastive Learning". arXiv:2109.01903
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, positives are same-class samples in the batch, negatives are
    different-class samples.  The loss maximises the log-ratio of positive-pair
    similarity to all-pair similarity:

        L = -1/(|P(i)|) * sum_{p in P(i)} log( exp(sim(i,p)/tau) / sum_{a!=i} exp(sim(i,a)/tau) )

    averaged over all anchors *i* in the batch.

    Reference: [76] Khosla et al. (2020). arXiv:2004.11362
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if base_temperature <= 0:
            raise ValueError(f"base_temperature must be positive, got {base_temperature}")
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: (B, D) L2-normalized embedding vectors.
            labels: (B,) integer class labels.

        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure features are L2-normalised (idempotent if already normalised)
        features = F.normalize(features, dim=1)

        # Cosine similarity matrix scaled by temperature: (B, B)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Positive mask: mask[i,j] = 1 iff labels[i] == labels[j] AND i != j
        labels_col = labels.unsqueeze(0)  # (1, B)
        labels_row = labels.unsqueeze(1)  # (B, 1)
        positive_mask = (labels_row == labels_col).float()  # (B, B)

        # Remove self-contrast (diagonal)
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask  # zero out diagonal

        # Mask to exclude self from denominator
        logits_mask = 1.0 - self_mask  # (B, B) — 0 on diagonal, 1 elsewhere

        # Numerical stability: subtract row-wise max before exp
        logits_max, _ = sim_matrix.detach().max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max

        # exp(logits) masked to exclude self
        exp_logits = torch.exp(logits) * logits_mask  # (B, B)

        # log(sum of exp over all negatives + positives, excluding self)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # (B, 1)

        # log-prob of each positive pair
        log_prob = logits - log_sum_exp  # (B, B)

        # Mean log-prob over positives for each anchor
        # Avoid division by zero for anchors with no positives in the batch
        num_positives = positive_mask.sum(dim=1)  # (B,)
        has_positives = (num_positives > 0).float()

        # For anchors with zero positives, set num_positives to 1 to avoid nan
        safe_num_positives = torch.clamp(num_positives, min=1.0)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / safe_num_positives

        # Scale by base_temperature / temperature (as in the paper)
        loss_per_anchor = -(self.base_temperature / self.temperature) * mean_log_prob_pos

        # Average only over anchors that have at least one positive
        total_valid = has_positives.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = (loss_per_anchor * has_positives).sum() / total_valid
        return loss


class PaCoLoss(nn.Module):
    """Parametric Contrastive Loss for long-tailed recognition.

    Extends SupConLoss with learnable class-specific prototype centers.
    Each class has a learnable center vector; these act as additional
    positives in the contrastive objective, ensuring rare classes always
    have sufficient positive pairs.

    Reference: [182] Cui et al. (2021). "PaCo". arXiv:2109.01903
    """

    def __init__(
        self,
        num_classes: int = 9,
        feat_dim: int = 128,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if base_temperature <= 0:
            raise ValueError(f"base_temperature must be positive, got {base_temperature}")
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_classes = num_classes
        # Learnable class centers (prototypes)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute parametric contrastive loss.

        Args:
            features: (B, D) L2-normalized embeddings.
            labels: (B,) integer class labels.

        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2-normalize features and centers
        features = F.normalize(features, dim=1)
        centers = F.normalize(self.centers, dim=1)

        # Concatenate features with class centers
        # extended_features: (B + num_classes, D)
        extended_features = torch.cat([features, centers], dim=0)

        # Create extended labels: original labels + class indices for centers
        center_labels = torch.arange(self.num_classes, device=device)
        extended_labels = torch.cat([labels, center_labels], dim=0)

        # Compute similarity matrix for all pairs
        # (B, B + num_classes)
        sim_matrix = torch.mm(features, extended_features.t()) / self.temperature

        # Create positive mask: same class
        # (B, B + num_classes)
        mask = (labels.unsqueeze(1) == extended_labels.unsqueeze(0)).float()

        # Remove self-similarity (diagonal for the B x B part)
        self_mask = torch.zeros(
            batch_size, batch_size + self.num_classes, device=device,
        )
        self_mask[:, :batch_size] = torch.eye(batch_size, device=device)
        mask = mask * (1 - self_mask)

        # Numerical stability: subtract row-wise max before exp
        logits_max = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        logits = sim_matrix - logits_max

        # exp(logits) masked to exclude self
        exp_logits = torch.exp(logits) * (1 - self_mask)

        # log(sum of exp over all pairs, excluding self)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives for each anchor
        num_positives = mask.sum(dim=1)
        # Avoid division by zero for anchors with no positives
        has_positives = (num_positives > 0).float()
        safe_num_positives = torch.clamp(num_positives, min=1.0)

        log_prob = logits - log_sum_exp
        mean_log_prob = (mask * log_prob).sum(dim=1) / safe_num_positives

        # Scale by base_temperature / temperature (as in the paper)
        loss_per_anchor = -(self.base_temperature / self.temperature) * mean_log_prob

        # Average only over anchors that have at least one positive
        total_valid = has_positives.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = (loss_per_anchor * has_positives).sum() / total_valid
        return loss


class SupConProjectionHead(nn.Module):
    """MLP projection head for supervised contrastive learning.

    Maps backbone features to a lower-dimensional space where the contrastive
    loss is applied.  The architecture follows the standard 2-layer MLP with
    ReLU activation and L2-normalised output.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise features.

        Args:
            x: (B, in_dim) backbone features.

        Returns:
            (B, out_dim) L2-normalised embeddings.
        """
        z = self.projection(x)
        return F.normalize(z, dim=1)


def extract_backbone_features(
    model: nn.Module,
    x: torch.Tensor,
    model_type: str = "auto",
) -> torch.Tensor:
    """Extract features from a backbone model (without the classification head).

    Supported model types:
        - ``"cnn"`` / ``"wafercnn"``: Uses ``model.features`` + ``model.avg_pool``
        - ``"resnet"``: Forward through all layers except ``model.fc``
        - ``"efficientnet"``: Uses ``model.features`` + adaptive pool
        - ``"auto"``: Detect automatically from model class / attributes

    Args:
        model: The backbone model.
        x: (B, C, H, W) input images.
        model_type: Architecture hint.

    Returns:
        (B, D) flattened feature tensor.
    """
    model_type = model_type.lower().strip()

    # Auto-detect
    if model_type == "auto":
        cls_name = type(model).__name__.lower()
        if "wafercnn" in cls_name or "wafercnn" in cls_name:
            model_type = "cnn"
        elif "resnet" in cls_name:
            model_type = "resnet"
        elif "efficientnet" in cls_name:
            model_type = "efficientnet"
        elif hasattr(model, "features") and hasattr(model, "avg_pool"):
            model_type = "cnn"
        elif hasattr(model, "features") and hasattr(model, "classifier"):
            model_type = "efficientnet"
        elif hasattr(model, "layer4") and hasattr(model, "fc"):
            model_type = "resnet"
        else:
            raise ValueError(
                f"Cannot auto-detect model type for {type(model).__name__}. "
                "Pass model_type='cnn', 'resnet', or 'efficientnet' explicitly."
            )

    if model_type in ("cnn", "wafercnn"):
        # WaferCNN: features -> avg_pool -> flatten
        h = model.features(x)
        h = model.avg_pool(h)
        return torch.flatten(h, 1)

    if model_type == "resnet":
        # torchvision ResNet: conv1 -> bn1 -> relu -> maxpool -> layer1..4 -> avgpool
        h = model.conv1(x)
        h = model.bn1(h)
        h = model.relu(h)
        h = model.maxpool(h)
        h = model.layer1(h)
        h = model.layer2(h)
        h = model.layer3(h)
        h = model.layer4(h)
        h = model.avgpool(h)
        return torch.flatten(h, 1)

    if model_type == "efficientnet":
        # torchvision EfficientNet: features -> avgpool -> flatten
        h = model.features(x)
        h = model.avgpool(h)
        return torch.flatten(h, 1)

    raise ValueError(f"Unsupported model_type: {model_type!r}")


def get_backbone_feature_dim(model: nn.Module, model_type: str = "auto") -> int:
    """Infer the feature dimensionality produced by the backbone.

    Args:
        model: The backbone model.
        model_type: Architecture hint (``"cnn"``, ``"resnet"``, ``"efficientnet"``, ``"auto"``).

    Returns:
        Integer feature dimension.
    """
    model_type = model_type.lower().strip()

    if model_type == "auto":
        cls_name = type(model).__name__.lower()
        if "wafercnn" in cls_name:
            model_type = "cnn"
        elif "resnet" in cls_name:
            model_type = "resnet"
        elif "efficientnet" in cls_name:
            model_type = "efficientnet"
        elif hasattr(model, "features") and hasattr(model, "avg_pool"):
            model_type = "cnn"
        elif hasattr(model, "features") and hasattr(model, "classifier"):
            model_type = "efficientnet"
        elif hasattr(model, "layer4") and hasattr(model, "fc"):
            model_type = "resnet"
        else:
            raise ValueError(
                f"Cannot auto-detect model type for {type(model).__name__}. "
                "Pass model_type explicitly."
            )

    if model_type in ("cnn", "wafercnn"):
        return model.feature_channels[-1]

    if model_type == "resnet":
        # model.fc is a Sequential (our custom head) or nn.Linear
        if isinstance(model.fc, nn.Sequential):
            for layer in model.fc:
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        if isinstance(model.fc, nn.Linear):
            return model.fc.in_features
        raise ValueError("Cannot determine ResNet feature dim from model.fc")

    if model_type == "efficientnet":
        if isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        if isinstance(model.classifier, nn.Linear):
            return model.classifier.in_features
        raise ValueError("Cannot determine EfficientNet feature dim from model.classifier")

    raise ValueError(f"Unsupported model_type: {model_type!r}")


class SupConTrainer:
    """Two-stage Supervised Contrastive Learning trainer.

    Stage 1 (pretrain): Train backbone + projection head with ``SupConLoss``.
    Stage 2 (finetune): Freeze backbone, train a linear classifier with CrossEntropy.

    Args:
        backbone: The base model (WaferCNN, ResNet-18, EfficientNet-B0, etc.).
        num_classes: Number of output classes.
        projection_head: Optional custom projection head.  When ``None`` a
            ``SupConProjectionHead`` is built automatically.
        model_type: Architecture hint for feature extraction.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        projection_head: Optional[SupConProjectionHead] = None,
        model_type: str = "auto",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.num_classes = num_classes
        self.model_type = model_type

        # Deep-copy the backbone so we don't mutate the caller's model
        self.backbone = copy.deepcopy(backbone).to(self.device)

        # Determine feature dimension
        self.feature_dim = get_backbone_feature_dim(self.backbone, model_type)

        # Build projection head
        if projection_head is not None:
            self.projection_head = projection_head.to(self.device)
        else:
            self.projection_head = SupConProjectionHead(
                in_dim=self.feature_dim,
            ).to(self.device)

        # Classifier head (built in finetune stage)
        self.classifier: Optional[nn.Module] = None

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone feature extractor."""
        return extract_backbone_features(self.backbone, x, self.model_type)

    def pretrain(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.05,
        temperature: float = 0.07,
        weight_decay: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """Stage 1: Pretrain backbone + projection head with SupConLoss.

        Args:
            train_loader: DataLoader yielding ``(images, labels)`` batches.
            epochs: Number of pretraining epochs.
            lr: Learning rate for SGD optimizer.
            temperature: Temperature for SupConLoss.
            weight_decay: L2 regularisation strength.

        Returns:
            Dictionary with ``"loss"`` key mapping to per-epoch loss values.
        """
        criterion = SupConLoss(temperature=temperature).to(self.device)

        # Unfreeze all backbone parameters for contrastive pretraining
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Combine backbone + projection head parameters
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters())
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.backbone.train()
        self.projection_head.train()

        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Extract features and project
                features = self._extract_features(images)
                projections = self.projection_head(features)

                loss = criterion(projections, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            history["loss"].append(avg_loss)

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "SupCon pretrain epoch %d/%d  loss=%.4f  lr=%.6f",
                    epoch + 1, epochs, avg_loss, current_lr,
                )

        logger.info("SupCon pretraining complete. Final loss: %.4f", history["loss"][-1])
        return history

    def pretrain_paco(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.05,
        temperature: float = 0.07,
        weight_decay: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """Stage 1 (PaCo variant): Pretrain backbone + projection head with PaCoLoss.

        Identical to ``pretrain()`` but uses Parametric Contrastive Loss with
        learnable class-specific prototype centers.  The centers are included
        in the optimizer so they are updated jointly with the backbone and
        projection head.

        Args:
            train_loader: DataLoader yielding ``(images, labels)`` batches.
            epochs: Number of pretraining epochs.
            lr: Learning rate for SGD optimizer.
            temperature: Temperature for PaCoLoss.
            weight_decay: L2 regularisation strength.

        Returns:
            Dictionary with ``"loss"`` key mapping to per-epoch loss values.
        """
        # Infer projection output dim from the projection head
        proj_out_dim: int = 128
        for module in reversed(list(self.projection_head.projection.modules())):
            if isinstance(module, nn.Linear):
                proj_out_dim = module.out_features
                break

        criterion = PaCoLoss(
            num_classes=self.num_classes,
            feat_dim=proj_out_dim,
            temperature=temperature,
        ).to(self.device)

        # Unfreeze all backbone parameters for contrastive pretraining
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Combine backbone + projection head + PaCo center parameters
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(criterion.parameters())
        )
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.backbone.train()
        self.projection_head.train()

        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Extract features and project
                features = self._extract_features(images)
                projections = self.projection_head(features)

                loss = criterion(projections, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            history["loss"].append(avg_loss)

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "PaCo pretrain epoch %d/%d  loss=%.4f  lr=%.6f",
                    epoch + 1, epochs, avg_loss, current_lr,
                )

        logger.info("PaCo pretraining complete. Final loss: %.4f", history["loss"][-1])
        return history

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Stage 2: Train a linear classifier on frozen backbone features.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Number of fine-tuning epochs.
            lr: Learning rate for the linear classifier.
            weight_decay: L2 regularisation strength.
            class_weights: Optional per-class weights for CrossEntropyLoss.

        Returns:
            Tuple of ``(full_model, history)`` where ``full_model`` is an
            ``nn.Sequential(backbone_features, classifier)`` and ``history``
            contains ``"train_loss"``, ``"val_loss"``, and ``"val_acc"`` lists.
        """
        # Freeze backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Build linear classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)

        # Loss and optimizer (only classifier parameters)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = 0.0
        best_state: Optional[Dict[str, Any]] = None

        for epoch in range(epochs):
            # --- Training ---
            self.classifier.train()
            train_loss = 0.0
            num_train = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    features = self._extract_features(images)

                logits = self.classifier(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                num_train += labels.size(0)

            avg_train_loss = train_loss / max(num_train, 1)
            history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            self.classifier.eval()
            val_loss = 0.0
            correct = 0
            num_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    features = self._extract_features(images)
                    logits = self.classifier(features)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    num_val += labels.size(0)

            avg_val_loss = val_loss / max(num_val, 1)
            val_acc = correct / max(num_val, 1)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step(avg_val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "backbone": copy.deepcopy(self.backbone.state_dict()),
                    "classifier": copy.deepcopy(self.classifier.state_dict()),
                }

            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
                logger.info(
                    "SupCon finetune epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                    epoch + 1, epochs, avg_train_loss, avg_val_loss, val_acc,
                )

        # Restore best checkpoint
        if best_state is not None:
            self.backbone.load_state_dict(best_state["backbone"])
            self.classifier.load_state_dict(best_state["classifier"])

        # Assemble final model that outputs class logits
        model = _SupConClassifier(
            backbone=self.backbone,
            classifier=self.classifier,
            model_type=self.model_type,
        )
        model.eval()

        logger.info("SupCon fine-tuning complete. Best val acc: %.4f", best_val_acc)
        return model, history


class _SupConClassifier(nn.Module):
    """Wrapper that combines a frozen backbone with a linear classifier.

    The ``forward`` method produces class logits, making this compatible
    with the rest of the training/evaluation pipeline.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        model_type: str = "auto",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.model_type = model_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = extract_backbone_features(self.backbone, x, self.model_type)
        return self.classifier(features)
