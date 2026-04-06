#!/usr/bin/env python3
"""
Synthetic data augmentation example script.

Demonstrates how to use the synthetic augmentation module to:
1. Generate synthetic wafer maps using rule-based or GAN-based methods
2. Balance class distributions by augmenting rare classes
3. Visualize generated samples
4. Train a GAN on wafer map data (optional, CPU-intensive)

Usage:
    python synthetic_augment.py --generator rule-based  # Fast, rule-based
    python synthetic_augment.py --generator gan         # Slower, GAN-based
    python synthetic_augment.py --visualize-only        # Just visualize
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from collections import Counter

from src.data.dataset import load_dataset, KNOWN_CLASSES
from src.data.preprocessing import preprocess_wafer_maps
from src.augmentation.synthetic import (
    SyntheticDataAugmenter,
    WaferMapGenerator,
    balance_dataset_with_synthetic,
)
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic wafer data augmentation"
    )
    parser.add_argument(
        '--generator',
        choices=['rule-based', 'gan'],
        default='rule-based',
        help='Generator type (default: rule-based)'
    )
    parser.add_argument(
        '--gan-epochs',
        type=int,
        default=5,
        help='Number of GAN training epochs (default: 5)'
    )
    parser.add_argument(
        '--gan-batch-size',
        type=int,
        default=32,
        help='GAN training batch size (default: 32)'
    )
    parser.add_argument(
        '--target-samples-per-class',
        type=int,
        default=5000,
        help='Target samples per class after augmentation (default: 5000)'
    )
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Only visualize generated samples, skip augmentation'
    )
    parser.add_argument(
        '--num-samples-per-class',
        type=int,
        default=3,
        help='Number of samples to visualize per class (default: 3)'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to use (default: cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    return parser.parse_args()


def setup_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main() -> None:
    """Main execution."""
    args = parse_args()
    setup_seeds(args.seed)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Generator type: {args.generator}")

    # ---- Load and preprocess data ----
    logger.info("\n=== Loading Dataset ===")
    try:
        df = load_dataset()
    except FileNotFoundError:
        logger.warning("ERROR: Dataset not found. Place LSWMD_new.pkl in data/ directory.")
        return

    # Filter to known classes
    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_filtered = df[labeled_mask].reset_index(drop=True)
    logger.info(f"Loaded {len(df_filtered)} labeled samples")

    # Extract wafer maps and labels
    wafer_maps = np.array(df_filtered['waferMap'].tolist())
    labels = np.array([KNOWN_CLASSES.index(c) for c in df_filtered['failureClass']])

    logger.info(f"Wafer maps shape: {wafer_maps.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    # Count samples per class
    class_counts = Counter(labels)
    logger.info("\n--- Original Class Distribution ---")
    for class_idx in sorted(class_counts.keys()):
        class_name = KNOWN_CLASSES[class_idx]
        count = class_counts[class_idx]
        pct = 100 * count / len(labels)
        logger.info(f"  {class_name:12s}: {count:6d} ({pct:5.1f}%)")

    # For demonstration, use a subset (preprocessing is slow)
    logger.info("\n=== Subsampling for Demo (first 1000 samples) ===")
    subsample_size = min(1000, len(wafer_maps))
    indices = np.random.choice(len(wafer_maps), subsample_size, replace=False)
    wafer_maps = wafer_maps[indices]
    labels = labels[indices]

    # Preprocess (resize and normalize)
    logger.info("\nPreprocessing wafer maps (resize to 96x96)...")
    preprocessed_maps = preprocess_wafer_maps(wafer_maps, target_size=(96, 96))
    preprocessed_maps = np.array(preprocessed_maps)
    logger.info(f"Preprocessed shape: {preprocessed_maps.shape}")

    if args.visualize_only:
        # ---- Visualization Only ----
        logger.info("\n=== Generating and Visualizing Samples ===")
        augmenter = SyntheticDataAugmenter(
            generator_type=args.generator,
            image_size=96,
            device=device
        )

        augmenter.visualize_generated_samples(
            class_names=KNOWN_CLASSES,
            num_samples_per_class=args.num_samples_per_class
        )
        return

    # ---- Train GAN (if applicable) ----
    if args.generator == 'gan':
        logger.info("\n=== Training GAN ===")
        augmenter = SyntheticDataAugmenter(
            generator_type='gan',
            image_size=96,
            device=device
        )

        augmenter.train_generator(
            preprocessed_maps,
            epochs=args.gan_epochs,
            batch_size=args.gan_batch_size,
            verbose=True
        )

        # Save trained GAN
        gan_path = 'wafer_gan_checkpoint.pth'
        augmenter.save_gan(gan_path)
        logger.info(f"GAN saved to {gan_path}")

    # ---- Augment Dataset ----
    logger.info("\n=== Augmenting Dataset ===")
    augmented_maps, augmented_labels = balance_dataset_with_synthetic(
        preprocessed_maps,
        labels,
        class_names=KNOWN_CLASSES,
        generator_type=args.generator,
        strategy='oversample_to_max'
    )

    # Show augmented distribution
    aug_counts = Counter(augmented_labels)
    logger.info("\n--- Augmented Class Distribution ---")
    for class_idx in sorted(aug_counts.keys()):
        class_name = KNOWN_CLASSES[class_idx]
        count = aug_counts[class_idx]
        pct = 100 * count / len(augmented_labels)
        logger.info(f"  {class_name:12s}: {count:6d} ({pct:5.1f}%)")

    # ---- Visualization ----
    logger.info("\n=== Visualizing Generated Samples ===")
    augmenter = SyntheticDataAugmenter(
        generator_type=args.generator,
        image_size=96,
        device=device
    )

    augmenter.visualize_generated_samples(
        class_names=KNOWN_CLASSES,
        num_samples_per_class=args.num_samples_per_class
    )

    logger.info("\n✓ Augmentation complete!")
    logger.info(f"  Original dataset: {len(preprocessed_maps)} samples")
    logger.info(f"  Augmented dataset: {len(augmented_maps)} samples")
    logger.info(f"  Synthetic samples generated: {len(augmented_maps) - len(preprocessed_maps)}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
