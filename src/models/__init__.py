"""Deep learning model architectures for wafer defect classification."""

from typing import Any, cast

from .attention import (
    CBAMBlock,
    SEBlock,
    SpatialAttention,
    add_cbam_to_model,
    add_se_to_model,
    attention_summary,
)
from .cnn import WaferCNN
from .ensemble import EnsembleModel, LearnedWeightEnsemble, StackingEnsemble
from .fpn import FPNBlock, WaferCNNFPN
from .ride import RIDELoss, RIDEModel, build_ride_model
from .swin import SwinTransformer, get_swin_micro, get_swin_tiny
from .vit import ViT, get_vit_small, get_vit_tiny

_PRETRAINED_IMPORT_ERROR: Exception | None = None

try:
    from .pretrained import get_efficientnet_b0 as _get_efficientnet_b0
    from .pretrained import get_resnet18 as _get_resnet18  # type: ignore[assignment]
except ImportError as exc:
    _PRETRAINED_IMPORT_ERROR = exc

    def _fallback_pretrained(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "torchvision is required to use pretrained models"
        ) from _PRETRAINED_IMPORT_ERROR

    _get_resnet18 = _fallback_pretrained  # type: ignore[assignment]
    _get_efficientnet_b0 = _fallback_pretrained  # type: ignore[assignment]


# Public names typed as Any so the conditional fallback is invisible to mypy.
get_resnet18: Any = cast(Any, _get_resnet18)
get_efficientnet_b0: Any = cast(Any, _get_efficientnet_b0)


__all__ = [
    "WaferCNN",
    "FPNBlock",
    "WaferCNNFPN",
    "get_resnet18",
    "get_efficientnet_b0",
    "ViT",
    "get_vit_small",
    "get_vit_tiny",
    "SwinTransformer",
    "get_swin_tiny",
    "get_swin_micro",
    "EnsembleModel",
    "LearnedWeightEnsemble",
    "StackingEnsemble",
    "SEBlock",
    "SpatialAttention",
    "CBAMBlock",
    "add_se_to_model",
    "add_cbam_to_model",
    "attention_summary",
    "RIDEModel",
    "RIDELoss",
    "build_ride_model",
]
