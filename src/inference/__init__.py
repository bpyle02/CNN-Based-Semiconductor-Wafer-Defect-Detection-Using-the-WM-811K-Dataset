"""Model inference and interpretability utilities."""

from typing import Any, NoReturn, cast

from .gradcam import GradCAM
from .tta import TestTimeAugmentation
from .visualize import plot_gradcam_grid

_SERVER_IMPORT_ERROR: Exception | None = None

try:
    from .server import ModelServer as _ModelServer
    from .server import ModelType as _ModelType
    from .server import create_app as _create_app  # type: ignore[assignment]
except ImportError as exc:
    _SERVER_IMPORT_ERROR = exc

    def _raise_server_import_error() -> NoReturn:
        raise ImportError(
            "FastAPI and related server dependencies are required for inference server utilities"
        ) from _SERVER_IMPORT_ERROR

    def _fallback_create(*args: Any, **kwargs: Any) -> Any:
        _raise_server_import_error()

    class _FallbackModelServer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_server_import_error()

    class _FallbackModelType:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            _raise_server_import_error()

    _create_app = _fallback_create  # type: ignore[assignment]
    _ModelServer = _FallbackModelServer  # type: ignore[assignment,misc]
    _ModelType = _FallbackModelType  # type: ignore[assignment,misc]


create_app: Any = cast(Any, _create_app)
ModelServer: Any = cast(Any, _ModelServer)
ModelType: Any = cast(Any, _ModelType)


try:
    from .uncertainty import (
        MCDropoutModel,
        UncertaintyEstimator,
        compute_confidence_intervals,
        disable_dropout,
        enable_dropout,
        plot_uncertainty_distribution,
    )

    _uncertainty_available = True
except ImportError:
    _uncertainty_available = False

__all__ = [
    "GradCAM",
    "TestTimeAugmentation",
    "plot_gradcam_grid",
    "create_app",
    "ModelServer",
    "ModelType",
]

if _uncertainty_available:
    __all__.extend(
        [
            "MCDropoutModel",
            "UncertaintyEstimator",
            "plot_uncertainty_distribution",
            "enable_dropout",
            "disable_dropout",
            "compute_confidence_intervals",
        ]
    )
