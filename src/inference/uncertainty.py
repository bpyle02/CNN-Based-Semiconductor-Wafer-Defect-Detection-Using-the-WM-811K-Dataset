"""
Uncertainty quantification using Monte Carlo Dropout.

Implements Bayesian approximation via dropout (Gal & Ghahramani, ICML 2016)
for uncertainty estimation in deep neural networks. Enables confidence intervals,
reliability analysis, and active learning strategies.

Key concepts:
- MC Dropout: Run inference T times with dropout enabled, treat predictions
  as samples from a posterior distribution
- Aleatoric uncertainty: Data noise (parameter uncertainty, irreducible)
- Epistemic uncertainty: Model uncertainty (reducible with more data)
- Prediction entropy and variance as uncertainty measures
"""

from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss


class MCDropoutModel:
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.

    Enables dropout during inference and runs multiple forward passes
    to generate a posterior distribution over predictions. This approximates
    Bayesian inference via stochastic forward passes.

    Attributes:
        model: PyTorch model (any architecture)
        num_iterations: Number of forward passes (T samples from posterior)
        device: Compute device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model: nn.Module,
        num_iterations: int = 50,
        device: str = "cpu",
    ) -> None:
        """
        Initialize MC Dropout wrapper.

        Args:
            model: PyTorch model with Dropout layers
            num_iterations: Number of stochastic forward passes (default 50)
            device: Compute device ('cpu' or 'cuda')

        Raises:
            ValueError: If num_iterations < 1
        """
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")

        self.model = model
        self.num_iterations = num_iterations
        self.device = device
        self.model.to(device)

    def _enable_dropout(self) -> None:
        """Enable dropout during inference (Bayesian approximation)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        """Disable dropout (standard inference)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_dist: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Run MC Dropout inference T times and compute uncertainty estimates.

        Algorithm:
        1. Run model forward pass T times with dropout enabled
        2. Collect logits/probabilities from each pass
        3. Compute mean and variance across T samples
        4. Variance approximates epistemic uncertainty (model confidence)

        Args:
            x: Input tensor (B, C, H, W) or (B, features...)
            return_dist: If True, also return raw distribution of predictions

        Returns:
            - Tuple of (mean_pred, uncertainty) where:
              - mean_pred: Mean prediction across T passes, shape (B, num_classes)
              - uncertainty: Standard deviation (epistemic uncertainty), shape (B,)
              - If return_dist=True: Also returns (B, T, num_classes) distribution

        Shape:
            Input: (B, ...)
            Output: mean (B, C), uncertainty (B,)
        """
        self.model.eval()
        self._enable_dropout()

        B = x.shape[0]
        num_classes = None
        logits_dist = []

        with torch.no_grad():
            for _ in range(self.num_iterations):
                logits = self.model(x)
                logits_dist.append(logits.cpu().numpy())
                if num_classes is None:
                    num_classes = logits.shape[1]

        # Stack distributions: (T, B, C) -> (B, T, C)
        logits_dist = np.array(logits_dist)  # (T, B, C)
        logits_dist = np.transpose(logits_dist, (1, 0, 2))  # (B, T, C)

        # Convert logits to probabilities via softmax
        probs_dist = self._softmax_numpy(logits_dist)  # (B, T, C)

        # Compute mean and variance across T samples
        mean_probs = probs_dist.mean(axis=1)  # (B, C)
        var_probs = probs_dist.var(axis=1)  # (B, C)

        # Epistemic uncertainty: variance in predicted class probabilities
        # For multiclass: use variance of max probability or entropy
        max_prob = probs_dist.max(axis=2)  # (B, T)
        epistemic_unc = max_prob.std(axis=1)  # (B,)

        self._disable_dropout()

        if return_dist:
            return mean_probs, epistemic_unc, probs_dist
        return mean_probs, epistemic_unc

    def predict_proba_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get class probabilities with per-class uncertainty.

        Returns mean and std of class probabilities across T passes.

        Args:
            x: Input tensor (B, ...)

        Returns:
            - mean_probs: Mean probabilities (B, C)
            - std_probs: Standard deviation of probabilities (B, C)
            - entropy: Predictive entropy as uncertainty (B,)
        """
        mean_probs, _, probs_dist = self.predict_with_uncertainty(
            x, return_dist=True
        )
        std_probs = probs_dist.std(axis=1)  # (B, C)

        # Entropy as predictive uncertainty
        # H(y|x) = -sum_c p(c|x) * log p(c|x)
        entropy = -np.sum(
            mean_probs * np.log(np.clip(mean_probs, 1e-10, 1.0)),
            axis=1
        )  # (B,)

        return mean_probs, std_probs, entropy

    def confidence_intervals(
        self,
        x: torch.Tensor,
        percentiles: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute percentile-based confidence intervals for predictions.

        For each sample and class, computes the percentile bounds across
        the T stochastic forward passes.

        Args:
            x: Input tensor (B, ...)
            percentiles: Lower and upper percentile bounds (default 95% CI)

        Returns:
            - point_pred: Best estimate (median), shape (B, C)
            - lower_bound: Lower percentile, shape (B, C)
            - upper_bound: Upper percentile, shape (B, C)
        """
        _, _, probs_dist = self.predict_with_uncertainty(x, return_dist=True)

        lower_p, upper_p = percentiles
        lower = np.percentile(probs_dist, lower_p, axis=1)  # (B, C)
        upper = np.percentile(probs_dist, upper_p, axis=1)  # (B, C)
        median = np.percentile(probs_dist, 50, axis=1)  # (B, C)

        return median, lower, upper

    @staticmethod
    def _softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax in NumPy."""
        x = x - x.max(axis=axis, keepdims=True)
        return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)


class UncertaintyEstimator:
    """
    High-level API for uncertainty analysis on datasets.

    Provides methods for:
    - Uncertainty estimation across full datasets
    - Identifying most uncertain predictions (active learning)
    - Calibration analysis (uncertainty vs. correctness)
    - Confidence intervals and reliability metrics
    """

    def __init__(
        self,
        model: nn.Module,
        num_iterations: int = 50,
        device: str = "cpu",
    ) -> None:
        """
        Initialize UncertaintyEstimator.

        Args:
            model: PyTorch model
            num_iterations: MC Dropout samples (default 50)
            device: Compute device
        """
        self.mc_model = MCDropoutModel(model, num_iterations, device)
        self.device = device

    def estimate_dataset_uncertainty(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty for all samples in a dataset.

        Args:
            dataloader: PyTorch DataLoader
            return_predictions: If True, also return class predictions

        Returns:
            Dictionary with keys:
            - 'uncertainty': Shape (N,), epistemic uncertainty per sample
            - 'mean_probs': Shape (N, C), mean class probabilities
            - 'entropy': Shape (N,), predictive entropy
            - 'predictions': Shape (N,), argmax predictions (if return_predictions=True)
            - 'true_labels': Shape (N,), ground truth labels (if available)
        """
        uncertainties = []
        mean_probs_list = []
        entropies = []
        predictions = []
        true_labels = []
        has_labels = False

        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
                has_labels = True
                true_labels.extend(y.numpy())
            else:
                x = batch

            x = x.to(self.device)

            # Get MC estimates
            mean_p, std_p, entropy = (
                self.mc_model.predict_proba_with_uncertainty(x)
            )

            # Epistemic uncertainty from variance
            eps_unc = std_p.mean(axis=1)  # Average across classes

            uncertainties.extend(eps_unc)
            mean_probs_list.extend(mean_p)
            entropies.extend(entropy)
            predictions.extend(mean_p.argmax(axis=1))

        result = {
            'uncertainty': np.array(uncertainties),
            'mean_probs': np.array(mean_probs_list),
            'entropy': np.array(entropies),
            'predictions': np.array(predictions),
        }

        if has_labels:
            result['true_labels'] = np.array(true_labels)

        return result

    def get_uncertain_samples(
        self,
        dataloader: DataLoader,
        k: int = 100,
        metric: str = 'entropy',
    ) -> Dict[str, np.ndarray]:
        """
        Get top-K most uncertain predictions (for active learning).

        Args:
            dataloader: PyTorch DataLoader
            k: Number of uncertain samples to return
            metric: Uncertainty metric ('entropy', 'variance', or 'margin')
                - 'entropy': Predictive entropy H(y|x)
                - 'variance': Mean class probability variance
                - 'margin': Margin between top-2 predicted classes

        Returns:
            Dictionary with:
            - 'indices': Shape (K,), indices into dataset
            - 'uncertainties': Shape (K,), uncertainty scores
            - 'predictions': Shape (K, C), predicted probabilities
            - 'top2_margin': Shape (K,), margin between top 2 classes
        """
        if metric not in ['entropy', 'variance', 'margin']:
            raise ValueError(f"metric must be 'entropy', 'variance', or 'margin'")

        results = self.estimate_dataset_uncertainty(dataloader)
        mean_probs = results['mean_probs']

        # Compute requested metric
        if metric == 'entropy':
            scores = results['entropy']
        elif metric == 'variance':
            scores = mean_probs.var(axis=1)
        elif metric == 'margin':
            # Margin: difference between top-2 classes
            top2_probs = np.partition(mean_probs, -2, axis=1)[:, -2:]
            scores = 1.0 - (top2_probs.max(axis=1) - top2_probs.min(axis=1))

        # Get top-K uncertain
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return {
            'indices': top_k_indices,
            'uncertainties': scores[top_k_indices],
            'predictions': mean_probs[top_k_indices],
            'top2_margin': 1.0 - np.diff(
                np.partition(mean_probs[top_k_indices], -2, axis=1)[:, -2:],
                axis=1
            ).flatten(),
        }

    def uncertainty_calibration(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics: uncertainty vs. correctness.

        Well-calibrated model: high uncertainty -> low accuracy on those samples.
        Poorly calibrated: no correlation between uncertainty and correctness.

        Metrics:
        - Brier Score: Mean squared error of probability predictions
        - ECE (Expected Calibration Error): Max absolute difference between
          predicted confidence and empirical accuracy across confidence bins
        - AUROC: Area under receiver operating characteristic curve

        Args:
            dataloader: PyTorch DataLoader with (images, labels)

        Returns:
            Dictionary with calibration metrics
        """
        results = self.estimate_dataset_uncertainty(
            dataloader, return_predictions=True
        )

        if 'true_labels' not in results:
            raise ValueError("DataLoader must provide (x, y) pairs for calibration")

        true_labels = results['true_labels']
        mean_probs = results['mean_probs']
        uncertainties = results['uncertainty']
        predictions = results['predictions']

        # Accuracy per sample
        correct = (predictions == true_labels).astype(np.float32)

        # Brier score: MSE of probability predictions
        # Compute as mean squared error between true one-hot and predicted probs
        num_classes = mean_probs.shape[1]
        targets_onehot = np.eye(num_classes)[true_labels]
        brier = ((targets_onehot - mean_probs) ** 2).mean()

        # Expected Calibration Error (ECE)
        # Bin predictions by confidence, compare avg confidence to accuracy
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        confidences = mean_probs.max(axis=1)

        ece = 0.0
        for i in range(num_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_confidence = confidences[mask].mean()
                avg_accuracy = correct[mask].mean()
                ece += np.abs(avg_confidence - avg_accuracy) * mask.sum()
        ece /= len(correct)

        # Correlation: uncertainty vs. correctness
        # High negative correlation = well-calibrated
        correlation = np.corrcoef(uncertainties, correct)[0, 1]

        return {
            'brier_score': float(brier),
            'ece': float(ece),
            'uncertainty_accuracy_correlation': float(correlation),
            'mean_uncertainty': float(uncertainties.mean()),
            'std_uncertainty': float(uncertainties.std()),
        }


def plot_uncertainty_distribution(
    uncertainties: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Visualize uncertainty distribution across samples and classes.

    Creates subplots showing:
    1. Histogram of uncertainties
    2. Uncertainty vs. prediction confidence (calibration)
    3. Per-class uncertainty distribution
    4. Uncertainty vs. correctness (if labels provided)

    Args:
        uncertainties: Array of uncertainty scores, shape (N,)
        predictions: Predicted class probabilities, shape (N, C)
        true_labels: Ground truth labels, shape (N,)
        class_names: List of class names (optional)
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Monte Carlo Dropout Uncertainty Analysis', fontsize=16, fontweight='bold')

    # 1. Uncertainty histogram
    ax = axes[0, 0]
    ax.hist(uncertainties, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(uncertainties.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(uncertainties), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Epistemic Uncertainty')
    ax.set_ylabel('Frequency')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Uncertainty vs. confidence (calibration)
    if predictions is not None:
        ax = axes[0, 1]
        confidences = predictions.max(axis=1)
        ax.scatter(confidences, uncertainties, alpha=0.5, s=20)
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Epistemic Uncertainty')
        ax.set_title('Uncertainty vs. Confidence (Calibration)')
        ax.grid(True, alpha=0.3)

        # Add correlation
        corr = np.corrcoef(confidences, uncertainties)[0, 1]
        ax.text(
            0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    else:
        axes[0, 1].text(0.5, 0.5, 'Provide predictions for calibration plot',
                        ha='center', va='center', transform=axes[0, 1].transAxes)

    # 3. Per-class uncertainty
    if predictions is not None and class_names is not None:
        ax = axes[1, 0]
        predicted_classes = predictions.argmax(axis=1)
        class_uncertainties = [
            uncertainties[predicted_classes == i] for i in range(predictions.shape[1])
        ]
        bp = ax.boxplot(class_uncertainties, labels=class_names)
        ax.set_ylabel('Epistemic Uncertainty')
        ax.set_title('Uncertainty by Predicted Class')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'Provide predictions and class names',
                        ha='center', va='center', transform=axes[1, 0].transAxes)

    # 4. Uncertainty vs. correctness
    if true_labels is not None and predictions is not None:
        ax = axes[1, 1]
        predicted_classes = predictions.argmax(axis=1)
        correct = (predicted_classes == true_labels).astype(np.float32)

        # Separate by correctness
        correct_unc = uncertainties[correct == 1]
        incorrect_unc = uncertainties[correct == 0]

        parts = ax.violinplot(
            [correct_unc, incorrect_unc],
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.set_ylabel('Epistemic Uncertainty')
        ax.set_title('Uncertainty by Prediction Correctness')
        ax.grid(True, alpha=0.3, axis='y')

        # Correlation
        corr = np.corrcoef(correct, uncertainties)[0, 1]
        ax.text(
            0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    else:
        axes[1, 1].text(0.5, 0.5, 'Provide true labels and predictions',
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.show()


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference (utility function).

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    """
    Disable dropout layers (standard inference mode).

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def compute_confidence_intervals(
    predictions: np.ndarray,
    percentiles: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence intervals from prediction distribution.

    Args:
        predictions: Array of shape (N, T, C) where:
                    N = samples, T = MC samples, C = classes
        percentiles: (lower, upper) percentile bounds

    Returns:
        - point_est: Best estimate (median), shape (N, C)
        - lower_bound: Lower percentile, shape (N, C)
        - upper_bound: Upper percentile, shape (N, C)
    """
    lower_p, upper_p = percentiles
    lower = np.percentile(predictions, lower_p, axis=1)
    upper = np.percentile(predictions, upper_p, axis=1)
    median = np.percentile(predictions, 50, axis=1)
    return median, lower, upper
