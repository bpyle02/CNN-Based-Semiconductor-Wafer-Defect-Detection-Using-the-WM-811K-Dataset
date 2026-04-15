"""Unit tests for energy-based out-of-distribution detection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.ood import OODDetector, energy_score, fit_ood_threshold


def test_energy_is_negative_logsumexp():
    """energy_score(logits, T) must equal -T * logsumexp(logits / T)."""
    torch.manual_seed(0)
    logits = torch.randn(32, 9)

    for T in (1.0, 0.5, 2.0):
        expected = -T * torch.logsumexp(logits / T, dim=-1)
        actual = energy_score(logits, T=T)
        assert torch.allclose(actual, expected, atol=1e-6), f"mismatch at T={T}"
        # Shape: one energy per sample.
        assert actual.shape == (logits.shape[0],)


def test_threshold_achieves_target_fpr():
    """Fitted threshold must produce the requested FPR on in-dist val data."""
    torch.manual_seed(42)
    # Large, smooth val set so quantile is well-defined.
    val_logits = torch.randn(10_000, 9)

    for target_fpr in (0.01, 0.05, 0.10):
        thr = fit_ood_threshold(val_logits, target_fpr=target_fpr)
        energies = energy_score(val_logits).numpy()
        empirical_fpr = float((energies > thr).mean())
        # NumPy quantile interpolation can nudge the empirical rate by ~1/N.
        assert abs(empirical_fpr - target_fpr) < 0.005, (
            f"target={target_fpr}, got empirical_fpr={empirical_fpr}"
        )


def test_detector_returns_three_tuple():
    """OODDetector.predict returns (pred, confidence, is_ood) of correct types."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    torch.manual_seed(7)
    model = TinyModel().eval()

    # Tiny in-distribution "val" batch for threshold fitting.
    val_x = torch.randn(256, 4)
    val_loader = [(val_x[i : i + 32], torch.zeros(32, dtype=torch.long))
                  for i in range(0, 256, 32)]

    detector = OODDetector(model, T=1.0, device="cpu")

    # Must fail if used before fitting.
    with pytest.raises(RuntimeError):
        detector.predict(torch.randn(4, 4))

    detector.fit(val_loader, target_fpr=0.05)
    assert detector.threshold is not None

    x = torch.randn(16, 4)
    out = detector.predict(x)
    assert isinstance(out, tuple) and len(out) == 3

    pred, confidence, is_ood = out
    assert pred.shape == (16,)
    assert pred.dtype in (torch.int64, torch.int32)
    assert confidence.shape == (16,)
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))
    assert is_ood.shape == (16,)
    assert is_ood.dtype == torch.bool
