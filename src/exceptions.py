"""Custom exception types for wafer defect detection."""


class WaferMapError(Exception):
    """Base exception for wafer map operations."""

    pass


class DataLoadError(WaferMapError):
    """Raised when data cannot be loaded or parsed."""

    pass


class ModelError(WaferMapError):
    """Raised when model initialization/forward pass fails."""

    pass


class TrainingError(WaferMapError):
    """Raised during training/validation."""

    pass


class InferenceError(WaferMapError):
    """Raised during inference."""

    pass


class FederatedError(WaferMapError):
    """Raised during federated learning operations."""

    pass
