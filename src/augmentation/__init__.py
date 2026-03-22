"""
Synthetic data augmentation module for wafer defect detection.

Provides GAN-based and rule-based generators for creating synthetic wafer maps
to address class imbalance and data scarcity issues.
"""

from .synthetic import (
    SimpleWaferGAN,
    SyntheticDataAugmenter,
    WaferMapGenerator,
    balance_dataset_with_synthetic,
)

__all__ = [
    "SimpleWaferGAN",
    "SyntheticDataAugmenter",
    "WaferMapGenerator",
    "balance_dataset_with_synthetic",
]
