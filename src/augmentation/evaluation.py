"""Evaluation metrics for synthetic data quality (FID, IS)."""

from typing import List, Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import entropy


class FIDScorer:
    """Fréchet Inception Distance (FID) scorer.

    Measures distributional similarity between real and synthetic data
    using Inception network features.
    """

    @staticmethod
    def compute_fid(
        real_features: np.ndarray,
        synthetic_features: np.ndarray,
    ) -> float:
        """Compute FID between real and synthetic features.

        Args:
            real_features: Real sample features (n_real, feature_dim)
            synthetic_features: Synthetic sample features (n_synthetic, feature_dim)

        Returns:
            FID score (lower is better, 0 is perfect)
        """
        mu_real = np.mean(real_features, axis=0)
        mu_synthetic = np.mean(synthetic_features, axis=0)

        sigma_real = np.cov(real_features.T)
        sigma_synthetic = np.cov(synthetic_features.T)

        # Fréchet distance
        diff = mu_real - mu_synthetic
        covmean = sqrtm(sigma_real @ sigma_synthetic)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_synthetic - 2 * covmean)

        return float(fid)


class InceptionScorer:
    """Inception Score (IS) evaluator.

    Measures quality and diversity of synthetic samples. Higher is better.
    """

    @staticmethod
    def compute_is(
        synthetic_probs: np.ndarray,
        splits: int = 10,
    ) -> Tuple[float, float]:
        """Compute Inception Score with variance.

        Args:
            synthetic_probs: Predicted class probabilities (n_samples, n_classes)
            splits: Number of splits for variance estimation

        Returns:
            Tuple of (IS mean, IS std)
        """
        n_samples = synthetic_probs.shape[0]
        split_size = n_samples // splits

        scores = []

        for i in range(splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else n_samples

            split_probs = synthetic_probs[start_idx:end_idx]

            # Marginal distribution
            p_y = np.mean(split_probs, axis=0)

            # KL divergence: D_KL(p(y|x) || p(y))
            kl_divs = []
            for prob in split_probs:
                kl = entropy(prob, p_y)
                kl_divs.append(kl)

            # Inception Score is exp(mean(KL))
            is_score = np.exp(np.mean(kl_divs))
            scores.append(is_score)

        return float(np.mean(scores)), float(np.std(scores))
