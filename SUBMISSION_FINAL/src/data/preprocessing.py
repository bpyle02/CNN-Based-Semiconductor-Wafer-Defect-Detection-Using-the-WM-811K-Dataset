"""
Data preprocessing and augmentation for wafer map images.

Handles resizing, normalization, and transformation of wafer maps to prepare
them for training and inference with deep learning models.
"""

from typing import List, Tuple, Optional
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform import resize as skimage_resize
from sklearn.preprocessing import LabelEncoder

try:
    import torchvision.transforms as transforms
except ImportError:
    class _Compose:
        def __init__(self, transform_list):
            self.transforms = list(transform_list)

        def __call__(self, x):
            for transform in self.transforms:
                x = transform(x)
            return x

    class _IdentityTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        def __call__(self, x):
            mean = self.mean.to(device=x.device, dtype=x.dtype)
            std = self.std.to(device=x.device, dtype=x.dtype)
            return (x - mean) / std

    class _FallbackTransforms:
        Compose = _Compose
        RandomHorizontalFlip = _IdentityTransform
        RandomVerticalFlip = _IdentityTransform
        RandomRotation = _IdentityTransform
        RandomAffine = _IdentityTransform
        Normalize = _Normalize

    transforms = _FallbackTransforms()

from .dataset import load_dataset, KNOWN_CLASSES

TARGET_SIZE = (96, 96)


class WaferMapDataset(Dataset):
    """
    PyTorch Dataset for WM-811K semiconductor wafer maps.

    Expects pre-resized and normalized wafer maps. Stacks grayscale maps into
    3-channel format for compatibility with pretrained models.

    Attributes:
        maps: List of preprocessed (H, W) numpy arrays
        labels: List of integer class labels
        transform: Optional torchvision transforms to apply
    """

    def __init__(
        self,
        preprocessed_maps: List[np.ndarray],
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Initialize dataset.

        Args:
            preprocessed_maps: List of resized, normalized (H, W) float32 arrays
            labels: 1D array of integer class labels
            transform: Optional torchvision transform pipeline
        """
        self.maps = preprocessed_maps
        self.labels = labels
        self.transform = transform

        if len(self.maps) != len(self.labels):
            raise ValueError(
                f"Maps ({len(self.maps)}) and labels ({len(self.labels)}) "
                "must have same length"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor [C, H, W], label tensor)
        """
        wm = self.maps[idx]

        # Stack to 3 channels: (H, W) -> (3, H, W)
        img = np.stack([wm] * 3, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def preprocess_wafer_maps(
    wafer_maps: List[np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE
) -> List[np.ndarray]:
    """
    Resize all wafer maps to uniform size and normalize to [0, 1].

    Preprocessing is done once upfront rather than per-batch, which dramatically
    speeds up training, especially on CPU.

    Args:
        wafer_maps: List of (H, W) numpy arrays
        target_size: Target (height, width) after resizing

    Returns:
        List of (H, W) float32 arrays normalized to [0, 1]
    """
    preprocessed = []
    for i, wm in enumerate(wafer_maps):
        arr = wm.astype(np.float32)
        arr = skimage_resize(
            arr, target_size, anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        arr = arr / 2.0  # Normalize to [0, 1]
        preprocessed.append(arr)

        if (i + 1) % 25000 == 0:
            print(f"  Preprocessed {i+1:,} / {len(wafer_maps):,} maps...")

    return preprocessed


def preprocess_data(
    dataset_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, transforms.Compose, transforms.Compose]:
    """
    Load raw dataset and filter to known classes.

    Args:
        dataset_path: Optional path to dataset pickle

    Returns:
        Tuple of (filtered DataFrame, train augmentation transform, ImageNet norm)
    """
    data = load_dataset(dataset_path)

    labeled_mask = data['failureClass'].isin(KNOWN_CLASSES)
    data_subset = data[labeled_mask].reset_index(drop=True)

    print(f"\nLabeled wafers: {len(data_subset):,}  (out of {len(data):,} total)")
    print(f"Dropped: {len(data) - len(data_subset):,} unlabeled / unknown wafers")

    class_dist = data_subset['failureClass'].value_counts()
    print("\n--- Failure Class Distribution (After Filtering) ---")
    print(class_dist.to_string())

    majority = class_dist.max()
    minority = class_dist.min()
    print(f"\n--- Class Imbalance Analysis ---")
    print(f"Imbalance ratio (majority / minority): {majority / minority:.1f}x")

    train_aug = get_image_transforms()
    imagenet_norm = get_imagenet_normalize()

    return data_subset, train_aug, imagenet_norm


def get_image_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get data augmentation transforms for training.

    Applies: random H/V flips, rotation (+/- 15°), translation (5%).
    Intensity normalization (x / 2.0) is done during preprocessing.

    Returns:
        Torchvision Compose object with augmentation pipeline
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    else:
        train_transform = transforms.Compose([])

    print("\n--- Image Transform Pipeline ---")
    print(f"Target image size: {TARGET_SIZE}")
    print(f"Channels: 3 (replicated grayscale)")
    print(f"Base normalization: pixel / 2.0 -> [0, 1]")
    print(
        "Augmentations: "
        + ("HFlip, VFlip, Rotation(+/-15), Translate(5%)" if augment else "disabled")
    )
    print(f"Preprocessing: All maps resized ONCE upfront (not per-batch)")

    return train_transform


def get_imagenet_normalize() -> transforms.Normalize:
    """
    Get ImageNet normalization transform for pretrained models.

    ResNet-18 and EfficientNet-B0 were pretrained on ImageNet with these
    specific mean and std values. Must be applied to inputs for these models.

    Returns:
        Normalize transform with ImageNet mean/std
    """
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


if __name__ == "__main__":
    data, aug_transform, imagenet_norm = preprocess_data()
    print(f"\nDataset ready: {len(data)} samples")
    print(f"Augmentation: {aug_transform}")
    print(f"ImageNet norm: {imagenet_norm}")
