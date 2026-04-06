#!/usr/bin/env python3
"""Anomaly detection analysis on trained wafer defect models.

Fits an anomaly detector on 'none' (normal) class features extracted from a
trained model, then scores all test data and reports AUROC.

Usage:
    python scripts/anomaly_analysis.py --checkpoint checkpoints/best_cnn.pth --model-type cnn
    python scripts/anomaly_analysis.py --checkpoint checkpoints/best_resnet.pth --model-type resnet --method ocsvm
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

METHOD_CHOICES = ["isolation_forest", "ocsvm", "autoencoder", "mahalanobis"]
METHOD_MAP = {"ocsvm": "one_class_svm", "isolation_forest": "isolation_forest",
              "autoencoder": "autoencoder", "mahalanobis": "mahalanobis"}


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
    """Load dataset and split into normal-train, normal-test, anomaly-test loaders."""
    from src.data.dataset import load_dataset, KNOWN_CLASSES
    from src.data.preprocessing import WaferMapDataset, get_image_transforms

    df = load_dataset(Path(data_path) if data_path else None)
    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    labels = le.transform(df["failureClass"].values)
    maps = df["waferMap"].tolist()
    none_idx = le.transform(["none"])[0]

    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.15, stratify=labels, random_state=42
    )

    # Normal train: 'none' class from training split
    normal_train_idx = [i for i in idx_train if labels[i] == none_idx]
    # Test split: separate normal vs anomaly
    normal_test_idx = [i for i in idx_test if labels[i] == none_idx]
    anomaly_test_idx = [i for i in idx_test if labels[i] != none_idx]

    transform = get_image_transforms(augment=False)
    full_ds = WaferMapDataset(maps, labels, transform=transform)

    normal_train_loader = DataLoader(Subset(full_ds, normal_train_idx), batch_size=batch_size, shuffle=False)
    normal_test_loader = DataLoader(Subset(full_ds, normal_test_idx), batch_size=batch_size, shuffle=False)
    anomaly_test_loader = DataLoader(Subset(full_ds, anomaly_test_idx), batch_size=batch_size, shuffle=False)

    return normal_train_loader, normal_test_loader, anomaly_test_loader


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Anomaly detection on wafer defect models")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", required=True, choices=["cnn", "resnet", "efficientnet"])
    parser.add_argument("--method", default="isolation_forest", choices=METHOD_CHOICES,
                        help="Anomaly detection method (default: isolation_forest)")
    parser.add_argument("--data-path", default=None, help="Path to LSWMD_new.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        from src.analysis.anomaly import AnomalyDetector
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}. Install with: pip install scikit-learn")
        return 1

    logger.info(f"Loading {args.model_type} model from {args.checkpoint}")
    model = load_model(args.model_type, args.checkpoint, args.device)

    logger.info("Loading and splitting dataset...")
    normal_train_loader, normal_test_loader, anomaly_test_loader = prepare_data(
        args.data_path, args.batch_size
    )
    logger.info(f"Normal train batches: {len(normal_train_loader)}, "
                f"Normal test batches: {len(normal_test_loader)}, "
                f"Anomaly test batches: {len(anomaly_test_loader)}")

    internal_method = METHOD_MAP[args.method]
    logger.info(f"Fitting anomaly detector (method={internal_method})...")
    detector = AnomalyDetector(method=internal_method, device=args.device)
    detector.fit(model, normal_train_loader)

    logger.info("Evaluating on test set...")
    metrics = detector.evaluate(model, normal_test_loader, anomaly_test_loader)

    logger.info(f"\n{'='*50}")
    logger.info(f"Anomaly Detection Results ({args.method})")
    logger.info(f"{'='*50}")
    logger.info(f"AUROC:               {metrics['auroc']:.4f}")
    logger.info(f"FPR @ 95% TPR:       {metrics['fpr_at_95_tpr']:.4f}")
    logger.info(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
