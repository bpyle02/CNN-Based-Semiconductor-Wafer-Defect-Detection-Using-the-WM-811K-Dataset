"""
Model evaluation metrics and parameter counting.

Computes accuracy, macro/weighted F1, calibration metrics, and detailed
per-class metrics.
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


def _compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
) -> List[Dict[str, float | int]]:
    """Compute reliability bin statistics and expected calibration error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats: List[Dict[str, float | int]] = []
    total_count = float(len(confidences))
    ece = 0.0
    mce = 0.0

    for index in range(n_bins):
        lower = float(bins[index])
        upper = float(bins[index + 1])
        if index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum())
        if count == 0:
            bin_stats.append(
                {
                    "bin_index": index,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        gap = abs(bin_accuracy - bin_confidence)
        ece += gap * (count / total_count)
        mce = max(mce, gap)
        bin_stats.append(
            {
                "bin_index": index,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "gap": gap,
            }
        )

    return bin_stats, float(ece), float(mce)


def compute_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute calibration and reliability metrics for a multiclass classifier.

    Args:
        probabilities: Array of shape (n_samples, n_classes) with softmax scores.
        labels: Ground-truth class indices.
        n_bins: Number of reliability bins for ECE/MCE computation.

    Returns:
        JSON-serializable metrics dictionary.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (n_samples, n_classes)")
    if probs.shape[0] != labels_arr.shape[0]:
        raise ValueError("probabilities and labels must contain the same number of samples")
    if probs.shape[0] == 0:
        raise ValueError("probabilities must contain at least one sample")

    predicted = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (predicted == labels_arr).astype(np.float64)

    true_one_hot = np.zeros_like(probs)
    true_one_hot[np.arange(len(labels_arr)), labels_arr] = 1.0

    eps = np.finfo(np.float64).eps
    true_class_prob = probs[np.arange(len(labels_arr)), labels_arr]

    bin_stats, ece, mce = _compute_ece(confidences, correct, n_bins=n_bins)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": float(np.mean(np.sum((probs - true_one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(-np.mean(np.log(np.clip(true_class_prob, eps, 1.0)))),
        "mean_confidence": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "accuracy_from_confidence": float(np.mean(correct)),
        "reliability_bins": bin_stats,
        "n_bins": int(n_bins),
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str = "Model",
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluate model on test set and return metrics.

    Computes:
        - Accuracy (overall correct predictions)
        - Macro F1 (unweighted average across classes)
        - Weighted F1 (weighted by support across classes)
        - ECE, MCE, Brier score, and negative log-likelihood
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
            - ece: Expected calibration error
            - brier_score: Multiclass Brier score
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = probabilities.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    calibration = compute_calibration_metrics(all_probabilities, all_labels)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} -- Test Set Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy    : {acc:.4f}")
    logger.info(f"  Macro F1    : {macro_f1:.4f}")
    logger.info(f"  Weighted F1 : {weighted_f1:.4f}")
    logger.info(f"  ECE         : {calibration['ece']:.4f}")
    logger.info(f"  Brier score : {calibration['brier_score']:.4f}")
    logger.info(f"  NLL         : {calibration['negative_log_likelihood']:.4f}")
    logger.info(f"{'='*60}")
    logger.info(classification_report(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    metrics = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        **calibration,
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
