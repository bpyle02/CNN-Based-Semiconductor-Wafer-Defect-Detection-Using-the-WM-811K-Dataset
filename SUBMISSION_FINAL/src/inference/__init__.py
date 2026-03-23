"""Model inference and interpretability utilities."""

from .gradcam import GradCAM
from .visualize import plot_gradcam_grid

_SERVER_IMPORT_ERROR = None

try:
    from .server import create_app, ModelServer, ModelType
except ImportError as exc:
    _SERVER_IMPORT_ERROR = exc

    def _raise_server_import_error() -> None:
        raise ImportError(
            "FastAPI and related server dependencies are required for inference server utilities"
        ) from _SERVER_IMPORT_ERROR

    def create_app(*args, **kwargs):
        _raise_server_import_error()

    class ModelServer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            _raise_server_import_error()

    class ModelType:  # type: ignore[override]
        def __new__(cls, *args, **kwargs):
            _raise_server_import_error()

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
