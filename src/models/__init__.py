"""Deep learning model architectures for wafer defect classification."""

from .cnn import WaferCNN
from .vit import ViT, get_vit_small, get_vit_tiny
from .attention import (
    SEBlock,
    SpatialAttention,
    CBAMBlock,
    add_se_to_model,
    add_cbam_to_model,
    attention_summary,
)

_PRETRAINED_IMPORT_ERROR = None

try:
    from .pretrained import get_resnet18, get_efficientnet_b0
except ImportError as exc:
    _PRETRAINED_IMPORT_ERROR = exc

    def get_resnet18(*args, **kwargs):
        raise ImportError(
            "torchvision is required to use ResNet-18 models"
        ) from _PRETRAINED_IMPORT_ERROR

    def get_efficientnet_b0(*args, **kwargs):
        raise ImportError(
            "torchvision is required to use EfficientNet-B0 models"
        ) from _PRETRAINED_IMPORT_ERROR

__all__ = [
    'WaferCNN',
    'get_resnet18',
    'get_efficientnet_b0',
    'ViT',
    'get_vit_small',
    'get_vit_tiny',
    'SEBlock',
    'SpatialAttention',
    'CBAMBlock',
    'add_se_to_model',
    'add_cbam_to_model',
    'attention_summary',
]
