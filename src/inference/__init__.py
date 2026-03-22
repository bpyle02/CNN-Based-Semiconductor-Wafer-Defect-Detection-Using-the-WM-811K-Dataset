"""Model inference and interpretability utilities."""

from .gradcam import GradCAM
from .visualize import plot_gradcam_grid
from .server import create_app, ModelServer, ModelType

try:
    from .uncertainty import (
        MCDropoutModel,
        UncertaintyEstimator,
        plot_uncertainty_distribution,
        enable_dropout,
        disable_dropout,
        compute_confidence_intervals,
    )
    _uncertainty_available = True
except ImportError:
    _uncertainty_available = False

__all__ = [
    'GradCAM',
    'plot_gradcam_grid',
    'create_app',
    'ModelServer',
    'ModelType',
]

if _uncertainty_available:
    __all__.extend([
        'MCDropoutModel',
        'UncertaintyEstimator',
        'plot_uncertainty_distribution',
        'enable_dropout',
        'disable_dropout',
        'compute_confidence_intervals',
    ])
