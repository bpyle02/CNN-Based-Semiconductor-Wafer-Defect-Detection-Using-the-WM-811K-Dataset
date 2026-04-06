"""
Data preprocessing and augmentation for wafer map images.

Handles resizing, normalization, and transformation of wafer maps to prepare
them for training and inference with deep learning models.

References:
    [7] Wu et al. (2014). "WM-811K Dataset". DOI:10.1109/TSM.2014.2364237
    [9] Yu et al. (2019). "Wafer Defect Pattern Recognition Based on CNN". DOI:10.1109/TSM.2019.2963656
    [49] Shorten & Khoshgoftaar (2019). "Image Data Augmentation Survey". DOI:10.1186/s40537-019-0197-0
    [58] (2020). "Wafer Map Defect Detection Using Joint Local-Global Features"
    [66] Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization". arXiv:1710.09412
    [67] Yun et al. (2019). "CutMix". arXiv:1905.04899
    [70] Cubuk et al. (2019). "AutoAugment". arXiv:1805.09501
    [71] Cubuk et al. (2020). "RandAugment". arXiv:1909.13719
    [75] DeVries & Taylor (2017). "Cutout". arXiv:1708.04552
    [118] Zhai et al. (2019). "S4L: Self-Supervised Semi-Supervised Learning". arXiv:1905.03670
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

import torch.nn.functional as F
import torchvision.transforms as transforms

from .dataset import load_dataset, KNOWN_CLASSES


class MixupCutmix:
    """Mixup and CutMix batch augmentation for improved generalization.

    Operates on batches (not individual samples). One of mixup or cutmix is
    randomly selected per batch based on their respective probabilities. If
    neither fires, the original batch is returned with lam=1.0.

    References:
        [51] Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization". arXiv:1710.09412
        [52] Yun et al. (2019). "CutMix". arXiv:1905.04899
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        num_classes: int = 9,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes

    @staticmethod
    def _rand_bbox(
        height: int, width: int, lam: float
    ) -> Tuple[int, int, int, int]:
        """Compute a random bounding box whose area ratio equals (1 - lam)."""
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        cy = np.random.randint(height)
        cx = np.random.randint(width)

        y1 = max(0, cy - cut_h // 2)
        y2 = min(height, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(width, cx + cut_w // 2)
        return y1, y2, x1, x2

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup or cutmix to a batch.

        Args:
            images: (B, C, H, W) batch tensor.
            labels: (B,) integer label tensor.

        Returns:
            (mixed_images, labels_a, labels_b, lam) where the mixed loss is
            ``lam * loss(pred, labels_a) + (1 - lam) * loss(pred, labels_b)``.
        """
        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)

        roll = random.random()
        if roll < self.mixup_prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
            mixed = lam * images + (1.0 - lam) * images[indices]
            return mixed, labels, labels[indices], lam

        if roll < self.mixup_prob + self.cutmix_prob:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1.0
            _, _, height, width = images.shape
            y1, y2, x1, x2 = self._rand_bbox(height, width, lam)
            mixed = images.clone()
            mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            # Adjust lambda to actual rectangle area ratio
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)
            return mixed, labels, labels[indices], lam

        # Neither selected — identity pass-through
        return images, labels, labels, 1.0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mixup_alpha={self.mixup_alpha}, cutmix_alpha={self.cutmix_alpha}, "
            f"mixup_prob={self.mixup_prob}, cutmix_prob={self.cutmix_prob})"
        )


class ClassBalancedSampler(torch.utils.data.Sampler):
    """Sampler that oversamples minority classes so each epoch sees balanced data.

    Unlike WeightedRandomSampler (which changes the per-sample probability),
    this sampler replicates minority-class indices to match a target count per
    class, then shuffles the result. The training distribution within each
    epoch is roughly uniform across classes.
    """

    def __init__(
        self,
        labels: np.ndarray,
        num_samples_per_class: Optional[int] = None,
    ) -> None:
        """
        Args:
            labels: 1-D integer array of class labels for every sample.
            num_samples_per_class: Target sample count per class per epoch.
                If ``None``, defaults to the count of the majority class.
        """
        super().__init__()
        self.labels = np.asarray(labels)
        class_indices: dict = {}
        for idx, label in enumerate(self.labels):
            class_indices.setdefault(int(label), []).append(idx)
        self.class_indices = class_indices

        if num_samples_per_class is None:
            num_samples_per_class = max(len(v) for v in class_indices.values())
        self.num_samples_per_class = num_samples_per_class
        self._length = num_samples_per_class * len(class_indices)

    def __iter__(self):
        indices = []
        for cls_indices in self.class_indices.values():
            n = len(cls_indices)
            if n == 0:
                continue
            # Repeat then truncate to get exactly num_samples_per_class
            repeats = self.num_samples_per_class // n
            remainder = self.num_samples_per_class % n
            expanded = cls_indices * repeats + list(
                np.random.choice(cls_indices, size=remainder, replace=False)
                if remainder <= n
                else np.random.choice(cls_indices, size=remainder, replace=True)
            )
            indices.extend(expanded)
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self._length


class GaussianNoise:
    """Add Gaussian noise to simulate sensor measurement variability."""
    # Domain-specific: simulates sensor measurement variability in wafer imaging

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RadialDistortion:
    # Domain-specific: barrel distortion simulates thermal wafer warping
    """Apply radial distortion to simulate wafer warping effects.

    Wafers are circular substrates that can warp during thermal processing.
    Barrel distortion (radial displacement proportional to r^2) models this
    physical effect, making it a domain-appropriate augmentation.
    """

    def __init__(self, strength: float = 0.1, p: float = 0.3):
        self.strength = strength
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        C, H, W = tensor.shape
        # Create coordinate grid centered at image center
        cy, cx = H / 2.0, W / 2.0
        y_coords = torch.arange(H, dtype=torch.float32) - cy
        x_coords = torch.arange(W, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        r_max = torch.sqrt(torch.tensor(cx**2 + cy**2))
        r_norm = r / r_max
        # Barrel distortion factor
        factor = 1.0 + self.strength * r_norm**2
        # Distorted coordinates
        xx_dist = (xx * factor + cx).long().clamp(0, W - 1)
        yy_dist = (yy * factor + cy).long().clamp(0, H - 1)
        # Apply distortion per channel
        result = tensor[:, yy_dist, xx_dist]
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strength={self.strength}, p={self.p})"


class RadialJitter:
    """Apply small radial distortion centered on the wafer.

    Wafers are circular, so radial scaling (zoom in/out from center)
    is more physically meaningful than arbitrary translation. Uses
    affine_grid + grid_sample for smooth, differentiable distortion.
    """

    def __init__(self, scale_range: tuple = (0.95, 1.05)):
        self.scale_range = scale_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        # tensor is (C, H, W); affine_grid expects (N, C, H, W)
        x = tensor.unsqueeze(0)
        # Build 2x3 affine matrix: uniform scaling centered at origin
        theta = torch.tensor(
            [[scale, 0.0, 0.0],
             [0.0, scale, 0.0]],
            dtype=tensor.dtype
        ).unsqueeze(0)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.squeeze(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range})"

TARGET_SIZE = (96, 96)  # 96x96 preserves defect patterns while keeping computation tractable [7]


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


def get_image_transforms(augment: bool = True, domain_augment: bool = True) -> transforms.Compose:
    """
    Get data augmentation transforms for training.

    Applies generic geometric augmentations (H/V flips, rotation, translation)
    plus optional domain-specific augmentations for semiconductor wafer maps:
    - RadialDistortion: barrel distortion simulating wafer warping
    - GaussianBlur: simulates sensor blur in wafer imaging equipment
    - GaussianNoise: simulates sensor measurement variability
    - RandomErasing: simulates partial defect occlusion or missing data

    Args:
        augment: Enable augmentation pipeline (False = identity transform)
        domain_augment: Include domain-specific augmentations. Only applies
            when augment=True.

    Intensity normalization (x / 2.0) is done during preprocessing.

    Returns:
        Torchvision Compose object with augmentation pipeline
    """
    if augment:
        # Base geometric augmentations
        augmentation_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
        if domain_augment:
            # Domain-specific augmentations for semiconductor wafer maps
            augmentation_list.extend([
                RadialDistortion(strength=0.1, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                GaussianNoise(std=0.02),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            ])
        train_transform = transforms.Compose(augmentation_list)
    else:
        train_transform = transforms.Compose([])

    logger.info("\n--- Image Transform Pipeline ---")
    logger.info(f"Target image size: {TARGET_SIZE}")
    logger.info(f"Channels: 3 (replicated grayscale)")
    logger.info(f"Base normalization: pixel / 2.0 -> [0, 1]")
    aug_desc = "disabled"
    if augment:
        aug_desc = "HFlip, VFlip, Rotation(+/-15), Translate(5%)"
        if domain_augment:
            aug_desc += ", RadialDistortion(s=0.1), GaussianBlur(k=3), GaussianNoise(std=0.02), RandomErasing(p=0.3)"
    logger.info(f"Augmentations: {aug_desc}")
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
