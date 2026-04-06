#!/usr/bin/env python3
"""Out-of-distribution detection analysis on trained wafer defect models.

Loads a trained checkpoint, extracts features from train and test sets,
fits a Mahalanobis-based OOD detector, and reports detection metrics.

Usage:
    python scripts/ood_analysis.py --checkpoint checkpoints/best_cnn.pth --model-type cnn
    python scripts/ood_analysis.py --checkpoint checkpoints/best_resnet.pth --model-type resnet
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_model(model_type: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a model from checkpoint."""
    from src.models import WaferCNN, get_resnet18, get_efficientnet_b0

    if model_type == "cnn":
        model = WaferCNN(num_classes=9)
    elif model_type == "resnet":
        model = get_resnet18(num_classes=9, pretrained=False, freeze_until=None)
    elif model_type == "efficientnet":
        model = get_efficientnet_b0(num_classes=9, pretrained=False, freeze_until=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"])
    model.to(device).eval()
    return model


def prepare_data(data_path: str = None, batch_size: int = 64):
    """Load dataset and create train/test loaders with label arrays."""
    from src.data.dataset import load_dataset, KNOWN_CLASSES
    from src.data.preprocessing import WaferMapDataset, get_image_transforms

    df = load_dataset(Path(data_path) if data_path else None)
    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    labels = le.transform(df["failureClass"].values)
    maps = df["waferMap"].tolist()

    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.15, stratify=labels, random_state=42
    )

    train_maps = [maps[i] for i in idx_train]
    test_maps = [maps[i] for i in idx_test]
    train_labels = labels[idx_train]
    test_labels = labels[idx_test]

    transform = get_image_transforms(augment=False)
    train_ds = WaferMapDataset(train_maps, train_labels, transform=transform)
    test_ds = WaferMapDataset(test_maps, test_labels, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_labels


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="OOD detection on wafer defect models")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", required=True, choices=["cnn", "resnet", "efficientnet"])
    parser.add_argument("--data-path", default=None, help="Path to LSWMD_new.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        from src.detection.ood import OutOfDistributionDetector
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}. Install with: pip install scikit-learn")
        return 1

    logger.info(f"Loading {args.model_type} model from {args.checkpoint}")
    model = load_model(args.model_type, args.checkpoint, args.device)

    logger.info("Loading and splitting dataset...")
    train_loader, test_loader, test_labels = prepare_data(args.data_path, args.batch_size)

    logger.info("Initializing OOD detector...")
    ood = OutOfDistributionDetector(model, device=args.device)

    logger.info("Fitting detector on training features (batched)...")
    train_feats = []
    for images, _ in train_loader:
        train_feats.append(ood.extract_features(images.to(args.device)))
    train_features = np.concatenate(train_feats, axis=0)
    ood.detector.fit(train_features)

    logger.info("Detecting OOD on test set...")
    test_feats = []
    for images, _ in test_loader:
        test_feats.append(ood.extract_features(images.to(args.device)))
    test_features = np.concatenate(test_feats, axis=0)
    distances, is_ood = ood.detector.detect(test_features)

    n_ood = int(is_ood.sum())
    n_total = len(is_ood)
    logger.info(f"\n{'='*50}")
    logger.info(f"OOD Detection Results")
    logger.info(f"{'='*50}")
    logger.info(f"Test samples:        {n_total}")
    logger.info(f"OOD flagged:         {n_ood} ({100*n_ood/n_total:.1f}%)")
    logger.info(f"In-distribution:     {n_total - n_ood}")
    logger.info(f"Distance (mean):     {distances.mean():.4f}")
    logger.info(f"Distance (std):      {distances.std():.4f}")
    logger.info(f"Threshold (95th):    {ood.detector.threshold:.4f}")
    logger.info(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
