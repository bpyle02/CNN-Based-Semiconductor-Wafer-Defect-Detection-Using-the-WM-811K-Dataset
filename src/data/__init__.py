"""Data loading and preprocessing module."""

from typing import Any, NoReturn, cast

from .dataset import load_dataset, extract_failure_label, KNOWN_CLASSES

_PREPROCESSING_IMPORT_ERROR: Exception | None = None

# Hide the conditional imports from mypy so the "real" typed signatures
# imported at runtime are preserved (we don't want mypy to see both the
# real and the fallback definitions and flag them as incompatible).
try:
    from .preprocessing import (  # type: ignore[assignment]
        preprocess_wafer_maps as _preprocess_wafer_maps,
        preprocess_data as _preprocess_data,
        get_image_transforms as _get_image_transforms,
        WaferMapDataset as _WaferMapDataset,
        get_imagenet_normalize as _get_imagenet_normalize,
        seed_worker as _seed_worker,
        MixupCutmix as _MixupCutmix,
        ClassBalancedSampler as _ClassBalancedSampler,
    )
except ImportError as exc:
    _PREPROCESSING_IMPORT_ERROR = exc

    def _raise_preprocessing_import_error() -> NoReturn:
        raise ImportError(
            "torchvision and preprocessing dependencies are required for data transforms"
        ) from _PREPROCESSING_IMPORT_ERROR

    def _fallback(*args: Any, **kwargs: Any) -> Any:
        _raise_preprocessing_import_error()

    class _FallbackClass:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_preprocessing_import_error()

    _preprocess_wafer_maps = _fallback  # type: ignore[assignment]
    _preprocess_data = _fallback  # type: ignore[assignment]
    _get_image_transforms = _fallback  # type: ignore[assignment]
    _get_imagenet_normalize = _fallback  # type: ignore[assignment]
    _seed_worker = _fallback  # type: ignore[assignment]
    _WaferMapDataset = _FallbackClass  # type: ignore[assignment,misc]
    _MixupCutmix = _FallbackClass  # type: ignore[assignment,misc]
    _ClassBalancedSampler = _FallbackClass  # type: ignore[assignment,misc]


# Re-export with Any-typed public names. At runtime these are the real
# objects when the dependency is available and the error-raising stubs
# when it is not — which is exactly the documented contract.
preprocess_wafer_maps: Any = cast(Any, _preprocess_wafer_maps)
preprocess_data: Any = cast(Any, _preprocess_data)
get_image_transforms: Any = cast(Any, _get_image_transforms)
get_imagenet_normalize: Any = cast(Any, _get_imagenet_normalize)
seed_worker: Any = cast(Any, _seed_worker)
WaferMapDataset: Any = cast(Any, _WaferMapDataset)
MixupCutmix: Any = cast(Any, _MixupCutmix)
ClassBalancedSampler: Any = cast(Any, _ClassBalancedSampler)


__all__ = [
    "load_dataset",
    "extract_failure_label",
    "KNOWN_CLASSES",
    "WaferMapDataset",
    "preprocess_wafer_maps",
    "preprocess_data",
    "get_image_transforms",
    "get_imagenet_normalize",
    "seed_worker",
    "MixupCutmix",
    "ClassBalancedSampler",
]
