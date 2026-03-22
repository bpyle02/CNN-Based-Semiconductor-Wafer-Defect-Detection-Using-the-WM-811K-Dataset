"""Deep learning model architectures for wafer defect classification."""

from .cnn import WaferCNN
from .pretrained import get_resnet18, get_efficientnet_b0
from .vit import ViT, get_vit_small, get_vit_tiny
from .attention import (
    SEBlock,
    SpatialAttention,
    CBAMBlock,
    add_se_to_model,
    add_cbam_to_model,
    attention_summary,
)

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
