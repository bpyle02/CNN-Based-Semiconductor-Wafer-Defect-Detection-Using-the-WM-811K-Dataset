"""Factories for models, optimizers, and schedulers.

Extracted from ``train.py`` so scripts and the training pipeline can share one
canonical construction path.
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Tuple

import torch.nn as nn
import torch.optim as optim

from src.models import WaferCNN, WaferCNNFPN, get_efficientnet_b0, get_resnet18
from src.models.swin import get_swin_tiny
from src.models.vit import get_vit_small

logger = logging.getLogger(__name__)


DEFAULT_SCHEDULER_CONFIG = SimpleNamespace(
    type="ReduceLROnPlateau",
    mode="auto",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)


def _call_with_supported_kwargs(factory, **kwargs):
    """Call a factory while ignoring unsupported keyword arguments."""
    supported = inspect.signature(factory).parameters
    filtered = {name: value for name, value in kwargs.items() if name in supported}
    return factory(**filtered)


def _apply_attention(model: nn.Module, model_cfg) -> nn.Module:
    """Inject SE or CBAM attention blocks if configured."""
    attention_type = getattr(model_cfg, "attention_type", None) if model_cfg else None
    if not attention_type:
        return model
    reduction = getattr(model_cfg, "attention_reduction", 16)
    if attention_type == "se":
        from src.models.attention import add_se_to_model

        return add_se_to_model(model, reduction=reduction)
    if attention_type == "cbam":
        from src.models.attention import add_cbam_to_model

        return add_cbam_to_model(model, reduction=reduction)
    logger.warning(f"Unknown attention_type '{attention_type}', skipping")
    return model


def build_model(
    model_name: str,
    model_cfg,
    num_classes: int,
    device: str,
) -> Tuple[nn.Module, str]:
    """Construct a model using config-backed settings when supported."""
    common_kwargs = {"num_classes": num_classes}
    dropout_rate = getattr(model_cfg, "dropout_rate", 0.5)
    head_dropout = getattr(model_cfg, "head_dropout", None)
    head_hidden_dim = getattr(model_cfg, "head_hidden_dim", None)
    feature_channels = getattr(model_cfg, "feature_channels", None)
    frozen_prefixes = getattr(model_cfg, "frozen_prefixes", None)

    if model_name == "cnn":
        cnn_kwargs = {
            **common_kwargs,
            "input_channels": getattr(model_cfg, "input_channels", 3),
            "dropout_rate": dropout_rate,
            "use_batch_norm": getattr(model_cfg, "use_batch_norm", True),
        }
        if feature_channels is not None:
            cnn_kwargs["feature_channels"] = feature_channels
        if model_cfg is not None and hasattr(model_cfg, "head_hidden_dim"):
            cnn_kwargs["head_hidden_dim"] = head_hidden_dim
        if (
            model_cfg is not None
            and hasattr(model_cfg, "head_dropout")
            and head_dropout is not None
        ):
            cnn_kwargs["head_dropout"] = head_dropout

        model = _call_with_supported_kwargs(WaferCNN, **cnn_kwargs).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "Custom CNN")

    if model_name == "cnn_fpn":
        fpn_kwargs = {
            **common_kwargs,
            "input_channels": getattr(model_cfg, "input_channels", 3),
            "dropout_rate": dropout_rate,
            "use_batch_norm": getattr(model_cfg, "use_batch_norm", True),
            "fpn_out_channels": getattr(model_cfg, "fpn_out_channels", 128) or 128,
        }
        if feature_channels is not None:
            fpn_kwargs["feature_channels"] = feature_channels
        model = _call_with_supported_kwargs(WaferCNNFPN, **fpn_kwargs).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "Custom CNN-FPN")

    if model_name == "resnet":
        model = _call_with_supported_kwargs(
            get_resnet18,
            **common_kwargs,
            pretrained=getattr(model_cfg, "pretrained", True),
            freeze_until=getattr(model_cfg, "freeze_until", "layer3"),
            frozen_prefixes=frozen_prefixes,
            head_dropout=head_dropout if head_dropout is not None else dropout_rate,
            head_hidden_dim=head_hidden_dim,
            freeze_bn=getattr(model_cfg, "freeze_bn", True),
        ).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "ResNet-18")

    if model_name == "vit":
        model = _call_with_supported_kwargs(
            get_vit_small,
            **common_kwargs,
            image_size=96,
            in_channels=getattr(model_cfg, "input_channels", 3),
            dropout=dropout_rate,
        ).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "ViT-small")

    if model_name == "swin":
        model = _call_with_supported_kwargs(
            get_swin_tiny,
            **common_kwargs,
        ).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "Swin-Tiny")

    if model_name == "ride":
        from src.models.ride import build_ride_model

        ride_backbone = getattr(model_cfg, "backbone", "cnn") if model_cfg else "cnn"
        ride_num_experts = getattr(model_cfg, "num_experts", 3) if model_cfg else 3
        ride_reduction = getattr(model_cfg, "reduction", 4) if model_cfg else 4
        model = build_ride_model(
            backbone_name=ride_backbone,
            num_classes=num_classes,
            num_experts=ride_num_experts,
            reduction=ride_reduction,
            device=device,
        )
        return model, getattr(model_cfg, "name", "RIDE")

    model = _call_with_supported_kwargs(
        get_efficientnet_b0,
        **common_kwargs,
        pretrained=getattr(model_cfg, "pretrained", True),
        freeze_until=getattr(model_cfg, "freeze_until", "features.6"),
        frozen_prefixes=frozen_prefixes,
        head_dropout=head_dropout if head_dropout is not None else dropout_rate,
        head_hidden_dim=head_hidden_dim,
        freeze_bn=getattr(model_cfg, "freeze_bn", True),
    ).to(device)
    return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "EfficientNet-B0")


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
    nesterov: bool = True,
):
    """Create the configured optimizer."""
    name = optimizer_name.lower()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if name == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, scheduler_cfg, epochs: int, monitored_metric: str = "val_macro_f1"):
    """Create the configured scheduler, optionally with linear warmup.

    For ReduceLROnPlateau, the mode is auto-derived from the monitored metric:
    metrics containing 'loss' use 'min', all others use 'max'.  This prevents
    the scheduler from reducing LR when accuracy *increases*.

    When ``warmup_epochs > 0``, a :class:`~torch.optim.lr_scheduler.LinearLR`
    warmup phase is composed with the main scheduler via
    :class:`~torch.optim.lr_scheduler.SequentialLR`.  Warmup is only applied to
    StepLR and CosineAnnealingLR; ReduceLROnPlateau is adaptive by nature and
    skips warmup (a log message is emitted).
    """
    warmup_epochs = getattr(scheduler_cfg, "warmup_epochs", 0)
    warmup_start_factor = getattr(scheduler_cfg, "warmup_start_factor", 0.1)

    scheduler_type = scheduler_cfg.type.lower()
    if scheduler_type in {"none", "off", "disabled"}:
        return None

    if scheduler_type == "reducelronplateau":
        if warmup_epochs > 0:
            logger.info(
                "warmup_epochs=%d ignored for ReduceLROnPlateau (adaptive scheduler)",
                warmup_epochs,
            )
        mode = "min" if "loss" in monitored_metric.lower() else "max"
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=scheduler_cfg.factor,
            patience=scheduler_cfg.patience,
            min_lr=scheduler_cfg.min_lr,
        )

    # Build the main (non-Plateau) scheduler
    main_sched: optim.lr_scheduler.LRScheduler
    if scheduler_type == "steplr":
        main_sched = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, getattr(scheduler_cfg, "step_size", scheduler_cfg.patience)),
            gamma=scheduler_cfg.factor,
        )
    elif scheduler_type == "cosineannealinglr":
        main_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, getattr(scheduler_cfg, "t_max", None) or epochs),
            eta_min=getattr(scheduler_cfg, "min_lr", 0.0),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_cfg.type}")

    if warmup_epochs <= 0:
        return main_sched

    # Compose warmup + main via SequentialLR
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs,
    )
    logger.info(
        "LR warmup enabled: %d epochs, start_factor=%.3f",
        warmup_epochs,
        warmup_start_factor,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, main_sched],
        milestones=[warmup_epochs],
    )


__all__ = [
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "DEFAULT_SCHEDULER_CONFIG",
    "_apply_attention",
    "_call_with_supported_kwargs",
]
