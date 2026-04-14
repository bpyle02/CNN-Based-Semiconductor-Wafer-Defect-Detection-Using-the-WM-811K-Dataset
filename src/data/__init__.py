"""Data loading and preprocessing module."""

from typing import Any, NoReturn

from .dataset import load_dataset, extract_failure_label, KNOWN_CLASSES

_PREPROCESSING_IMPORT_ERROR = None

try:
    from .preprocessing import (
        preprocess_wafer_maps,
        preprocess_data,
        get_image_transforms,
        WaferMapDataset,
        get_imagenet_normalize,
        seed_worker,
        MixupCutmix,
        ClassBalancedSampler,
    )
except ImportError as exc:
    _PREPROCESSING_IMPORT_ERROR = exc

    def _raise_preprocessing_import_error() -> NoReturn:
        raise ImportError(
            "torchvision and preprocessing dependencies are required for data transforms"
        ) from _PREPROCESSING_IMPORT_ERROR

    def preprocess_wafer_maps(*args: Any, **kwargs: Any) -> Any:
        _raise_preprocessing_import_error()

    def preprocess_data(*args: Any, **kwargs: Any) -> Any:
        _raise_preprocessing_import_error()

    def get_image_transforms(*args: Any, **kwargs: Any) -> Any:
        _raise_preprocessing_import_error()

    def get_imagenet_normalize(*args: Any, **kwargs: Any) -> Any:
        _raise_preprocessing_import_error()

    def seed_worker(*args: Any, **kwargs: Any) -> Any:
        _raise_preprocessing_import_error()

    class WaferMapDataset:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_preprocessing_import_error()

    class MixupCutmix:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_preprocessing_import_error()

    class ClassBalancedSampler:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_preprocessing_import_error()

__all__ = [
    'load_dataset',
    'extract_failure_label',
    'KNOWN_CLASSES',
    'WaferMapDataset',
    'preprocess_wafer_maps',
    'preprocess_data',
    'get_image_transforms',
    'get_imagenet_normalize',
    'seed_worker',
    'MixupCutmix',
    'ClassBalancedSampler',
]
