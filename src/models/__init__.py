"""Deep learning model architectures for wafer defect classification."""

from .cnn import WaferCNN
from .pretrained import get_resnet18, get_efficientnet_b0

__all__ = [
    'WaferCNN',
    'get_resnet18',
    'get_efficientnet_b0',
]
