"""
Model evaluation metrics and parameter counting.

Computes accuracy, macro/weighted F1, and detailed per-class metrics.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
)
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str = "Model",
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model on test set and return metrics.

    Computes:
        - Accuracy (overall correct predictions)
        - Macro F1 (unweighted average across classes)
        - Weighted F1 (weighted by support across classes)
        - Per-class precision, recall, F1 (printed as classification report)

    Args:
        model: PyTorch model in eval mode
        test_loader: DataLoader for test set
        class_names: List of class names for reporting
        model_name: Name for logging
        device: Compute device

    Returns:
        Tuple of (predictions, labels, metrics_dict) where metrics_dict contains:
            - accuracy: Float in [0, 1]
            - macro_f1: Float in [0, 1]
            - weighted_f1: Float in [0, 1]
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} -- Test Set Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy    : {acc:.4f}")
    logger.info(f"  Macro F1    : {macro_f1:.4f}")
    logger.info(f"  Weighted F1 : {weighted_f1:.4f}")
    logger.info(f"{'='*60}")
    logger.info()
    logger.info(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    metrics = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
    }

    return all_preds, all_labels, metrics


def get_per_class_f1(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Compute per-class F1 scores.

    Args:
        labels: True labels
        preds: Predicted labels

    Returns:
        Array of F1 scores (one per class)
    """
    _, _, f1s, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    return f1s


def count_params(model: nn.Module) -> int:
    """
    Count total parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Trainable parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    logger.info("Evaluation module loaded.")
