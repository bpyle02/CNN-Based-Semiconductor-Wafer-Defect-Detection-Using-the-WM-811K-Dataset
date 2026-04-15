"""Out-of-distribution detection via free-energy scoring.

Implements the energy-based OOD detector of Liu et al. (2020),
"Energy-based Out-of-distribution Detection" (NeurIPS), arXiv:2010.03759.

A classifier's logits define a free energy
    E(x; T) = -T * logsumexp(f(x) / T, dim=-1)
which is lower for in-distribution samples (sharp, confident logits)
and higher for OOD samples. Thresholding on E gives a lightweight,
post-hoc OOD detector requiring no auxiliary data or retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Return the free energy per sample: -T * logsumexp(logits / T).

    Args:
        logits: Tensor of shape (..., num_classes).
        T: Temperature. Higher T smooths the energy surface.

    Returns:
        Energy tensor of shape logits.shape[:-1]. Lower values indicate
        in-distribution samples; higher values indicate OOD.
    """
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got T={T}")
    return -T * torch.logsumexp(logits / T, dim=-1)


def fit_ood_threshold(
    val_logits: torch.Tensor,
    val_labels: Optional[torch.Tensor] = None,
    target_fpr: float = 0.05,
    T: float = 1.0,
) -> float:
    """Pick the energy threshold whose FPR on in-dist val data equals target_fpr.

    The threshold is the (1 - target_fpr) quantile of in-distribution
    energies: by construction, only target_fpr of clean samples exceed it
    and are (falsely) flagged as OOD.

    Args:
        val_logits: (N, C) logits on a clean, in-distribution validation set.
        val_labels: Unused; accepted for API symmetry (the threshold is
            label-free because all val samples are assumed in-distribution).
        target_fpr: Desired false-positive rate on in-dist val data.
        T: Temperature passed to ``energy_score``.

    Returns:
        Scalar threshold such that energies above it are predicted OOD.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    energies = energy_score(val_logits, T=T).detach().cpu().numpy()
    # (1 - target_fpr) quantile: target_fpr fraction of in-dist energies
    # lie above the threshold, giving FPR = target_fpr.
    threshold = float(np.quantile(energies, 1.0 - target_fpr))
    return threshold


@dataclass
class OODPrediction:
    """Structured OOD prediction output."""

    pred: torch.Tensor  # (N,) argmax class indices
    confidence: torch.Tensor  # (N,) softmax max probability
    is_ood: torch.Tensor  # (N,) bool flags (True if OOD)


class OODDetector:
    """Wraps a trained classifier with post-hoc energy-based OOD detection.

    Usage:
        detector = OODDetector(model, device='cuda')
        detector.fit(val_loader, target_fpr=0.05)
        pred, conf, is_ood = detector.predict(x)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        T: float = 1.0,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.T = float(T)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = x.to(self.device)
        return self.model(x)

    @torch.no_grad()
    def fit(self, val_loader, target_fpr: float = 0.05) -> float:
        """Collect val-set logits and fit the energy threshold.

        Args:
            val_loader: Iterable yielding (images, labels) batches, all in-dist.
            target_fpr: Desired FPR on in-dist val data.

        Returns:
            The fitted threshold (also stored on ``self.threshold``).
        """
        all_logits = []
        for batch in val_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            all_logits.append(self._logits(images).cpu())
        logits = torch.cat(all_logits, dim=0)
        self.threshold = fit_ood_threshold(logits, target_fpr=target_fpr, T=self.T)
        return self.threshold

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pred, confidence, is_ood) for a batch of inputs.

        Args:
            x: Input tensor of shape (N, ...).

        Returns:
            Tuple of three tensors, each of length N:
              pred        - argmax class index
              confidence  - softmax max probability
              is_ood      - boolean OOD flag (True iff energy > threshold)
        """
        if self.threshold is None:
            raise RuntimeError("OODDetector threshold not fitted. Call .fit(val_loader) first.")
        logits = self._logits(x)
        probs = F.softmax(logits, dim=-1)
        confidence, pred = probs.max(dim=-1)
        energies = energy_score(logits, T=self.T)
        is_ood = energies > self.threshold
        return pred, confidence, is_ood

    def __call__(self, x: torch.Tensor):
        return self.predict(x)
