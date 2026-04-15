"""Temperature scaling for post-hoc probability calibration.

Fits a single scalar ``T`` on a held-out validation set by minimizing
negative log-likelihood of softmax(logits / T).  Scaling logits by a
learned temperature does not change ``argmax`` predictions but it does
repair over/under-confident softmax outputs, which matters for ECE,
threshold-based decisioning, and downstream ensembling.

Reference:
    Guo, Pleiss, Sun, Weinberger (2017). "On Calibration of Modern
    Neural Networks." ICML. arXiv:1706.04599
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


def _nll(temperature: float, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Cross-entropy NLL of softmax(logits / T) against integer labels."""
    t = max(float(temperature), 1e-6)
    scaled = logits / t
    return float(F.cross_entropy(scaled, labels, reduction="mean").item())


class TemperatureScaling:
    """Post-hoc calibration by scaling logits by learned temperature T.

    Fit T by minimizing NLL on a held-out validation set. T > 1 softens,
    T < 1 sharpens. Does not change argmax predictions but makes softmax
    probabilities trustworthy.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.fitted: bool = False

    def fit(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Fit temperature by minimizing NLL on (logits, labels).

        Args:
            logits: ``(N, num_classes)`` raw model outputs (pre-softmax).
            labels: ``(N,)`` integer class labels.

        Returns:
            The fitted temperature scalar.
        """
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        else:
            logits = logits.detach().float()
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        else:
            labels = labels.detach().long()

        # Bounded scalar optimization on T in (0, 10].
        result = minimize_scalar(
            lambda t: _nll(t, logits, labels),
            bounds=(1e-2, 10.0),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 200},
        )
        self.temperature = float(result.x)
        self.fitted = True
        logger.info(
            "TemperatureScaling fitted: T=%.4f (NLL=%.6f)", self.temperature, float(result.fun)
        )
        return self.temperature

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Return logits divided by the fitted temperature."""
        return logits / self.temperature

    def transform_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        return torch.softmax(self.transform(logits), dim=-1)
