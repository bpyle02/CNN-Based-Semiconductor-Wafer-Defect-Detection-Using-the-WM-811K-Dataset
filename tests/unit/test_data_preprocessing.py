"""Unit tests for src/data/preprocessing.py and src/data/dataset.py."""

import numpy as np
import pytest
import torch

from src.data.preprocessing import (
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    preprocess_wafer_maps,
    seed_worker,
)
from src.data.dataset import extract_failure_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_maps():
    """10 random 96x96 float32 arrays."""
    return [np.random.rand(96, 96).astype(np.float32) for _ in range(10)]


@pytest.fixture
def dummy_labels():
    """Integer labels matching dummy_maps length."""
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=np.int64)


@pytest.fixture
def dataset(dummy_maps, dummy_labels):
    return WaferMapDataset(dummy_maps, dummy_labels)


# ---------------------------------------------------------------------------
# WaferMapDataset
# ---------------------------------------------------------------------------

class TestWaferMapDataset:
    def test_len(self, dataset, dummy_maps):
        assert len(dataset) == len(dummy_maps)

    def test_getitem_shapes(self, dataset):
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 96, 96)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long

    def test_getitem_all_channels_identical(self, dataset):
        """Grayscale is replicated across 3 channels."""
        img, _ = dataset[0]
        assert torch.equal(img[0], img[1])
        assert torch.equal(img[1], img[2])

    def test_label_values(self, dataset, dummy_labels):
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            assert label.item() == dummy_labels[idx]

    def test_mismatched_lengths_raises(self, dummy_maps):
        bad_labels = np.array([0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="same length"):
            WaferMapDataset(dummy_maps, bad_labels)

    def test_with_transform(self, dummy_maps, dummy_labels):
        """A simple transform is applied during __getitem__."""
        class ScaleBy2:
            def __call__(self, x):
                return x * 2.0

        ds = WaferMapDataset(dummy_maps, dummy_labels, transform=ScaleBy2())
        img_transformed, _ = ds[0]

        ds_plain = WaferMapDataset(dummy_maps, dummy_labels, transform=None)
        img_plain, _ = ds_plain[0]

        assert torch.allclose(img_transformed, img_plain * 2.0)

    def test_no_transform(self, dataset):
        """Without transform, raw tensor values come through."""
        img, _ = dataset[0]
        assert img.dtype == torch.float32


# ---------------------------------------------------------------------------
# get_image_transforms
# ---------------------------------------------------------------------------

class TestGetImageTransforms:
    def test_returns_callable(self):
        transform = get_image_transforms()
        assert callable(transform)

    def test_augmented_preserves_shape(self):
        transform = get_image_transforms(augment=True)
        dummy = torch.rand(3, 96, 96)
        out = transform(dummy)
        assert out.shape == (3, 96, 96)

    def test_no_augment_preserves_shape(self):
        transform = get_image_transforms(augment=False)
        dummy = torch.rand(3, 96, 96)
        out = transform(dummy)
        assert out.shape == (3, 96, 96)

    def test_no_augment_identity(self):
        """With augment=False the transform is an empty Compose (identity)."""
        transform = get_image_transforms(augment=False)
        dummy = torch.rand(3, 96, 96)
        out = transform(dummy)
        assert torch.equal(out, dummy)


# ---------------------------------------------------------------------------
# get_imagenet_normalize
# ---------------------------------------------------------------------------

class TestGetImagenetNormalize:
    def test_returns_callable(self):
        norm = get_imagenet_normalize()
        assert callable(norm)

    def test_applies_correct_normalization(self):
        norm = get_imagenet_normalize()
        # Constant image with value 0.485 in all channels => after norm channel 0 should be ~0
        img = torch.full((3, 4, 4), 0.485)
        img[1] = 0.456
        img[2] = 0.406
        out = norm(img)
        # Channel 0: (0.485 - 0.485) / 0.229 ~ 0
        assert torch.allclose(out[0], torch.zeros(4, 4), atol=1e-5)
        # Channel 1: (0.456 - 0.456) / 0.224 ~ 0
        assert torch.allclose(out[1], torch.zeros(4, 4), atol=1e-5)
        # Channel 2: (0.406 - 0.406) / 0.225 ~ 0
        assert torch.allclose(out[2], torch.zeros(4, 4), atol=1e-5)

    def test_preserves_shape(self):
        norm = get_imagenet_normalize()
        img = torch.rand(3, 96, 96)
        out = norm(img)
        assert out.shape == (3, 96, 96)


# ---------------------------------------------------------------------------
# preprocess_wafer_maps
# ---------------------------------------------------------------------------

class TestPreprocessWaferMaps:
    def test_resizes_to_target(self):
        small_maps = [np.random.rand(5, 5).astype(np.float32) * 2.0 for _ in range(3)]
        result = preprocess_wafer_maps(small_maps, target_size=(96, 96))
        assert len(result) == 3
        for arr in result:
            assert arr.shape == (96, 96)
            assert arr.dtype == np.float32

    def test_custom_target_size(self):
        maps = [np.random.rand(10, 10).astype(np.float32) * 2.0 for _ in range(2)]
        result = preprocess_wafer_maps(maps, target_size=(48, 48))
        for arr in result:
            assert arr.shape == (48, 48)

    def test_normalization_range(self):
        """Values in [0, 2] are divided by 2.0, so output should be in [0, 1]."""
        maps = [np.full((10, 10), 2.0, dtype=np.float32)]
        result = preprocess_wafer_maps(maps)
        assert result[0].max() <= 1.0 + 1e-6
        assert result[0].min() >= 0.0 - 1e-6

    def test_empty_list(self):
        result = preprocess_wafer_maps([])
        assert result == []


# ---------------------------------------------------------------------------
# seed_worker
# ---------------------------------------------------------------------------

class TestSeedWorker:
    def test_callable(self):
        assert callable(seed_worker)

    def test_runs_without_error(self):
        """Smoke test: calling seed_worker should not raise."""
        seed_worker(0)
        seed_worker(3)


# ---------------------------------------------------------------------------
# extract_failure_label
# ---------------------------------------------------------------------------

class TestExtractFailureLabel:
    def test_ndarray_with_bytes(self):
        label = np.array([[b'Center']])
        assert extract_failure_label(label) == 'Center'

    def test_ndarray_with_string(self):
        label = np.array([['Edge-Loc']])
        assert extract_failure_label(label) == 'Edge-Loc'

    def test_nested_list_bytes(self):
        label = [[b'Scratch']]
        assert extract_failure_label(label) == 'Scratch'

    def test_nested_list_string(self):
        label = [['Donut']]
        assert extract_failure_label(label) == 'Donut'

    def test_flat_list_bytes(self):
        label = [b'none']
        assert extract_failure_label(label) == 'none'

    def test_flat_list_string(self):
        label = ['Random']
        assert extract_failure_label(label) == 'Random'

    def test_direct_bytes(self):
        assert extract_failure_label(b'Edge-Ring') == 'Edge-Ring'

    def test_direct_string(self):
        assert extract_failure_label('Loc') == 'Loc'

    def test_whitespace_stripped(self):
        assert extract_failure_label('  Near-full  ') == 'Near-full'
        assert extract_failure_label(b'  Center  ') == 'Center'

    def test_empty_ndarray_returns_unknown(self):
        assert extract_failure_label(np.array([])) == 'unknown'

    def test_none_returns_unknown(self):
        assert extract_failure_label(None) == 'unknown'

    def test_empty_list_returns_unknown(self):
        assert extract_failure_label([]) == 'unknown'

    def test_numeric_returns_string(self):
        """Numeric values are stringified."""
        assert extract_failure_label([42]) == '42'
