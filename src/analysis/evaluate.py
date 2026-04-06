"""
Model evaluation metrics and parameter counting.

Computes accuracy, macro/weighted F1, calibration metrics, and detailed
per-class metrics.  Includes post-hoc calibration via temperature scaling
and asymmetric balanced calibration (ABC) for long-tailed distributions.

References:
    [13] Alam et al. (2022). "Semiconductor Defect Detection by Hybrid Classical-Quantum DL". arXiv:2206.09912
    [29] Guo et al. (2017). "On Calibration of Modern Neural Networks". arXiv:1706.04599
    [32] Nixon et al. (2019). "Measuring Calibration in Deep Learning". arXiv:1904.01685
    [57] (2019). "Automated Visual Inspection of Semiconductor Wafers"
    [62] (2021). "Statistical Process Control for Semiconductor Manufacturing"
    [188] Ma et al. (2022). "ABC: Asymmetric Balanced Calibration". arXiv:2203.14395
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
)
import logging

from src.inference.uncertainty import TemperatureScaler

logger = logging.getLogger(__name__)


def _compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
) -> Tuple[List[Dict[str, float | int]], float, float]:
    """Compute reliability bin statistics and expected calibration error."""
    # Ref [29, 32]: Expected Calibration Error — bin-based calibration metric
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


def _collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model on *loader* in eval mode and return (logits, labels) tensors."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def _metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute classification + calibration metrics from raw logits.

    When *temperature* != 1.0 the logits are scaled before softmax.
    """
    scaled = logits / temperature
    probs_t = F.softmax(scaled, dim=1)
    probs = probs_t.numpy()
    labels_np = labels.numpy()

    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels_np, preds)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels_np, preds, average="weighted", zero_division=0)
    calibration = compute_calibration_metrics(probs, labels_np)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **calibration,
    }
    return preds, labels_np, metrics


def calibrate_and_evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str = "Model",
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    """
    Evaluate model with post-hoc temperature scaling calibration.

    Fits temperature T on validation set, then evaluates on test set
    with calibrated probabilities. Returns (preds, labels, metrics, temperature).

    The returned metrics dict contains both ``raw_*`` and ``calibrated_*``
    prefixed metrics so callers can compare the effect of calibration.  The
    top-level (un-prefixed) accuracy / f1 / calibration keys correspond to
    the **calibrated** predictions.

    Args:
        model: PyTorch model (will be set to eval mode).
        val_loader: DataLoader for validation/calibration set.
        test_loader: DataLoader for test set.
        class_names: List of class names for reporting.
        model_name: Name used in log messages.
        device: Compute device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        Tuple of (calibrated_predictions, labels, combined_metrics_dict, temperature).
    """
    # 1. Collect validation logits and fit temperature
    val_logits, val_labels = _collect_logits_and_labels(model, val_loader, device)
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)
    logger.info(
        f"Temperature scaling fitted on validation set: T = {temperature:.4f}"
    )

    # 2. Collect test logits
    test_logits, test_labels = _collect_logits_and_labels(model, test_loader, device)

    # 3. Raw (uncalibrated) metrics
    raw_preds, labels_np, raw_metrics = _metrics_from_logits(test_logits, test_labels, temperature=1.0)

    # 4. Calibrated metrics
    cal_preds, _, cal_metrics = _metrics_from_logits(test_logits, test_labels, temperature=temperature)

    # 5. Build combined dict
    combined: Dict[str, Any] = {}
    for key, value in raw_metrics.items():
        combined[f"raw_{key}"] = value
    for key, value in cal_metrics.items():
        combined[f"calibrated_{key}"] = value
    # Top-level keys use calibrated values (they are the "better" estimates)
    combined.update(cal_metrics)
    combined["temperature"] = temperature

    # 6. Log comparison
    ece_delta = raw_metrics.get("ece", 0.0) - cal_metrics.get("ece", 0.0)
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} -- Calibrated Evaluation (T={temperature:.4f})")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Metric':<20} {'Raw':>10} {'Calibrated':>12}")
    logger.info(f"  {'-'*44}")
    for key in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score", "negative_log_likelihood"):
        raw_val = raw_metrics.get(key, 0.0)
        cal_val = cal_metrics.get(key, 0.0)
        logger.info(f"  {key:<20} {raw_val:>10.4f} {cal_val:>12.4f}")
    logger.info(f"  ECE improvement: {ece_delta:+.4f}")
    logger.info(f"{'='*60}")
    logger.info(
        classification_report(
            labels_np,
            cal_preds,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    return cal_preds, labels_np, combined, temperature


class AsymmetricCalibrator:
    """Asymmetric Balanced Calibration for long-tailed recognition.

    Learns per-class temperature parameters instead of a single global
    temperature.  Rare classes receive higher temperatures (softer
    predictions) to improve calibration on the tail of the distribution.

    The per-class temperature is optimised by minimising the negative
    log-likelihood on the subset of validation samples belonging to each
    class, using bounded scalar optimization.

    Reference:
        [188] Ma et al. (2022). "ABC: Asymmetric Balanced Calibration".
        arXiv:2203.14395
    """

    def __init__(self, num_classes: int = 9, device: str = "cpu") -> None:
        self.num_classes = num_classes
        self.device = device
        self.class_temperatures: torch.Tensor = torch.ones(num_classes)

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> np.ndarray:
        """Fit per-class temperatures on the validation set.

        For each class *c*, extracts the logits of all validation samples
        whose true label is *c* and finds the temperature that minimises
        the NLL on those samples via bounded scalar optimisation.

        Args:
            model: Trained model (set to eval mode internally).
            val_loader: Validation ``DataLoader`` yielding ``(images, labels)``.

        Returns:
            NumPy array of shape ``(num_classes,)`` with the learned
            temperatures.
        """
        logits, labels = _collect_logits_and_labels(model, val_loader, self.device)

        temperatures = torch.ones(self.num_classes)

        for cls_idx in range(self.num_classes):
            mask = labels == cls_idx
            if mask.sum() == 0:
                # No samples for this class -- keep default T=1
                continue

            cls_logits = logits[mask]  # (N_c, C)
            cls_labels = labels[mask]  # (N_c,)

            def nll_for_class(temp: float) -> float:
                """Negative log-likelihood at temperature ``temp``."""
                scaled = cls_logits / max(temp, 1e-4)
                log_probs = F.log_softmax(scaled, dim=1)
                return -log_probs[
                    torch.arange(cls_labels.size(0)), cls_labels
                ].mean().item()

            result = minimize_scalar(
                nll_for_class,
                bounds=(0.01, 10.0),
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 200},
            )
            temperatures[cls_idx] = result.x

        self.class_temperatures = temperatures
        logger.info(
            "Asymmetric calibration fitted: T_per_class = %s",
            ", ".join(f"{t:.3f}" for t in temperatures.tolist()),
        )
        return temperatures.numpy()

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply per-class temperature scaling to logits.

        Args:
            logits: Tensor of shape ``(B, C)`` with raw model logits.

        Returns:
            Tensor of shape ``(B, C)`` with temperature-scaled logits.
        """
        temps = self.class_temperatures.to(logits.device)
        return logits / temps.unsqueeze(0)


def calibrate_and_evaluate_asymmetric(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str = "Model",
    device: str = "cpu",
    num_classes: int = 9,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """Evaluate a model with asymmetric balanced calibration (ABC).

    Fits per-class temperatures on the validation set, then evaluates
    on the test set with the asymmetrically calibrated probabilities.

    Args:
        model: Trained PyTorch model.
        val_loader: DataLoader for validation/calibration set.
        test_loader: DataLoader for test set.
        class_names: Human-readable class names for the report.
        model_name: Label for log messages.
        device: Compute device (``"cpu"`` or ``"cuda"``).
        num_classes: Number of output classes.

    Returns:
        Tuple of ``(calibrated_predictions, labels, combined_metrics,
        class_temperatures)``.
    """
    calibrator = AsymmetricCalibrator(num_classes=num_classes, device=device)
    class_temperatures = calibrator.fit(model, val_loader)

    # Collect test logits
    test_logits, test_labels = _collect_logits_and_labels(model, test_loader, device)

    # Raw metrics (no calibration)
    raw_preds, labels_np, raw_metrics = _metrics_from_logits(
        test_logits, test_labels, temperature=1.0
    )

    # Asymmetrically calibrated metrics
    cal_logits = calibrator.calibrate_logits(test_logits)
    cal_probs = F.softmax(cal_logits, dim=1).numpy()
    cal_preds = cal_probs.argmax(axis=1)
    cal_acc = accuracy_score(labels_np, cal_preds)
    cal_macro_f1 = f1_score(labels_np, cal_preds, average="macro", zero_division=0)
    cal_weighted_f1 = f1_score(labels_np, cal_preds, average="weighted", zero_division=0)
    cal_calibration = compute_calibration_metrics(cal_probs, labels_np)
    cal_metrics: Dict[str, Any] = {
        "accuracy": cal_acc,
        "macro_f1": cal_macro_f1,
        "weighted_f1": cal_weighted_f1,
        **cal_calibration,
    }

    # Build combined dict
    combined: Dict[str, Any] = {}
    for key, value in raw_metrics.items():
        combined[f"raw_{key}"] = value
    for key, value in cal_metrics.items():
        combined[f"asymmetric_{key}"] = value
    combined.update(cal_metrics)
    combined["class_temperatures"] = class_temperatures.tolist()

    # Log comparison
    ece_delta = raw_metrics.get("ece", 0.0) - cal_metrics.get("ece", 0.0)
    logger.info("\n%s", "=" * 60)
    logger.info(
        "  %s -- Asymmetric Calibrated Evaluation", model_name
    )
    logger.info("%s", "=" * 60)
    logger.info("  %-20s %10s %12s", "Metric", "Raw", "Asymmetric")
    logger.info("  %s", "-" * 44)
    for key in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score", "negative_log_likelihood"):
        raw_val = raw_metrics.get(key, 0.0)
        cal_val = cal_metrics.get(key, 0.0)
        logger.info("  %-20s %10.4f %12.4f", key, raw_val, cal_val)
    logger.info("  ECE improvement: %+.4f", ece_delta)
    logger.info("%s", "=" * 60)
    logger.info(
        classification_report(
            labels_np,
            cal_preds,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    return cal_preds, labels_np, combined, class_temperatures


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


def evaluate_with_tta(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str = "Model",
    device: str = "cpu",
    transforms_list: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluate a model using Test-Time Augmentation.

    Averages softmax predictions over multiple augmented views of each
    test image and computes classification metrics on the averaged
    predictions.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        class_names: List of human-readable class names.
        model_name: Label used in log messages.
        device: Compute device (``"cpu"`` or ``"cuda"``).
        transforms_list: Optional list of ``torchvision.transforms.Compose``
            augmentations.  ``None`` uses the default 5 geometric views
            (identity, hflip, vflip, hflip+vflip, 90-deg rotation).

    Returns:
        Tuple of ``(predictions, labels, metrics_dict)`` where
        ``metrics_dict`` contains ``accuracy``, ``macro_f1``,
        ``weighted_f1``, and calibration metrics.
    """
    from src.inference.tta import TestTimeAugmentation

    tta = TestTimeAugmentation(
        model=model,
        transforms_list=transforms_list,
        device=device,
    )

    model.eval()
    all_preds: list = []
    all_labels: list = []
    all_probabilities: list = []

    for images, labels in test_loader:
        avg_probs = tta.predict(images)  # (B, num_classes) on cpu
        preds = avg_probs.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probabilities.extend(avg_probs.numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probabilities)

    acc = accuracy_score(labels_arr, preds_arr)
    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)
    calibration = compute_calibration_metrics(probs_arr, labels_arr)

    logger.info("\n%s", "=" * 60)
    logger.info("  %s -- TTA Evaluation (%d views)", model_name, tta.num_views)
    logger.info("%s", "=" * 60)
    logger.info("  Accuracy    : %.4f", acc)
    logger.info("  Macro F1    : %.4f", macro_f1)
    logger.info("  Weighted F1 : %.4f", weighted_f1)
    logger.info("  ECE         : %.4f", calibration["ece"])
    logger.info("  Brier score : %.4f", calibration["brier_score"])
    logger.info("%s", "=" * 60)
    logger.info(
        classification_report(
            labels_arr,
            preds_arr,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "tta_num_views": tta.num_views,
        **calibration,
    }

    return preds_arr, labels_arr, metrics


if __name__ == "__main__":
    logger.info("Evaluation module loaded.")
