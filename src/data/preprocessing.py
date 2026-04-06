"""
Data preprocessing and augmentation for wafer map images.

Handles resizing, normalization, and transformation of wafer maps to prepare
them for training and inference with deep learning models.
"""

from typing import Any, List, Sequence, Tuple, Optional
from pathlib import Path
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform import resize as skimage_resize
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

import torchvision.transforms as transforms

from .dataset import load_dataset, KNOWN_CLASSES

TARGET_SIZE = (96, 96)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers for reproducibility.

    When num_workers > 0, each worker gets its own copy of the dataset
    and random state. This function ensures deterministic behavior by
    seeding each worker based on the global torch seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WaferMapDataset(Dataset):
    """
    PyTorch Dataset for WM-811K semiconductor wafer maps.

    Expects raw wafer maps. Resizes, normalizes, and stacks grayscale maps into
    3-channel format during __getitem__ for lazy loading.

    Attributes:
        maps: List of raw numpy arrays
        labels: List of integer class labels
        transform: Optional torchvision transforms to apply
        target_size: Target (height, width) after resizing
    """

    def __init__(
        self,
        raw_maps: List[np.ndarray],
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = TARGET_SIZE
    ) -> None:
        """
        Initialize dataset.

        Args:
            raw_maps: List of raw numpy arrays
            labels: 1D array of integer class labels
            transform: Optional torchvision transform pipeline
            target_size: Target (height, width) for resizing
        """
        self.maps = raw_maps
        self.labels = labels
        self.transform = transform
        self.target_size = target_size

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
        
        # Preprocess lazily
        arr = wm.astype(np.float32)
        arr = skimage_resize(
            arr, self.target_size, anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        arr = arr / 2.0

        # Stack to 3 channels: (H, W) -> (3, H, W)
        img = np.stack([arr] * 3, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


class PatchWaferDataset(Dataset):
    """
    Dataset for high-resolution wafers that yields patches instead of resizing.
    Useful for preserving resolution-dependent defect signatures.
    """
    def __init__(
        self,
        raw_maps: List[np.ndarray],
        labels: np.ndarray,
        patch_size: int = 32,
        patches_per_wafer: int = 1,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        self.maps = raw_maps
        self.labels = labels
        self.patch_size = patch_size
        self.patches_per_wafer = patches_per_wafer
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels) * self.patches_per_wafer

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wafer_idx = idx // self.patches_per_wafer
        wm = self.maps[wafer_idx]
        h, w = wm.shape
        
        # Randomly sample a patch if wafer is larger than patch_size
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            patch = wm[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Pad or resize if smaller
            patch = skimage_resize(wm, (self.patch_size, self.patch_size), anti_aliasing=True)

        arr = patch.astype(np.float32) / 2.0
        img = np.stack([arr] * 3, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[wafer_idx], dtype=torch.long)
        return img, label


def preprocess_wafer_maps(
    wafer_maps: List[np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE
) -> List[np.ndarray]:
    """
    Resize all wafer maps to uniform size and normalize to [0, 1].
    
    DEPRECATED: Preprocessing is now done lazily in WaferMapDataset.
    This function is kept for backward compatibility or one-off conversions.

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
        # WM-811K wafer maps use pixel values {0, 1, 2} representing
        # background, normal die, and defective die respectively.
        # Dividing by 2.0 normalizes to [0.0, 1.0] range.
        arr = arr / 2.0
        preprocessed.append(arr)

        if (i + 1) % 25000 == 0:
            logger.info(f"  Preprocessed {i+1:,} / {len(wafer_maps):,} maps...")

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

    logger.info(f"\nLabeled wafers: {len(data_subset):,}  (out of {len(data):,} total)")
    logger.info(f"Dropped: {len(data) - len(data_subset):,} unlabeled / unknown wafers")

    class_dist = data_subset['failureClass'].value_counts()
    logger.info("\n--- Failure Class Distribution (After Filtering) ---")
    logger.info(class_dist.to_string())

    majority = class_dist.max()
    minority = class_dist.min()
    logger.info(f"\n--- Class Imbalance Analysis ---")
    logger.info(f"Imbalance ratio (majority / minority): {majority / minority:.1f}x")

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

    logger.info("\n--- Image Transform Pipeline ---")
    logger.info(f"Target image size: {TARGET_SIZE}")
    logger.info(f"Channels: 3 (replicated grayscale)")
    logger.info(f"Base normalization: pixel / 2.0 -> [0, 1]")
    logger.info(
        "Augmentations: "
        + ("HFlip, VFlip, Rotation(+/-15), Translate(5%)" if augment else "disabled")
    )
    logger.info(f"Preprocessing: Maps resized dynamically per-batch (lazy loading)")

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
    logger.info(f"\nDataset ready: {len(data)} samples")
    logger.info(f"Augmentation: {aug_transform}")
    logger.info(f"ImageNet norm: {imagenet_norm}")
