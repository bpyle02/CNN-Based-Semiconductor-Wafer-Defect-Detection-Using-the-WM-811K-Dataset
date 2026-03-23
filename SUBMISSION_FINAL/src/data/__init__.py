"""Data loading and preprocessing module."""

from .dataset import load_dataset, extract_failure_label

_PREPROCESSING_IMPORT_ERROR = None

try:
    from .preprocessing import (
        preprocess_wafer_maps,
        preprocess_data,
        get_image_transforms,
        WaferMapDataset,
        get_imagenet_normalize,
    )
except ImportError as exc:
    _PREPROCESSING_IMPORT_ERROR = exc

    def _raise_preprocessing_import_error() -> None:
        raise ImportError(
            "torchvision and preprocessing dependencies are required for data transforms"
        ) from _PREPROCESSING_IMPORT_ERROR

    def preprocess_wafer_maps(*args, **kwargs):
        _raise_preprocessing_import_error()

    def preprocess_data(*args, **kwargs):
        _raise_preprocessing_import_error()

    def get_image_transforms(*args, **kwargs):
        _raise_preprocessing_import_error()

    def get_imagenet_normalize(*args, **kwargs):
        _raise_preprocessing_import_error()

    class WaferMapDataset:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            _raise_preprocessing_import_error()

__all__ = [
    'load_dataset',
    'extract_failure_label',
    'WaferMapDataset',
    'preprocess_wafer_maps',
    'preprocess_data',
    'get_image_transforms',
    'get_imagenet_normalize',
]
