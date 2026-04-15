"""Unit tests for ``src.inference.calibration.TemperatureScaling``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.inference.calibration import TemperatureScaling


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def test_temperature_greater_than_one_when_overconfident() -> None:
    """Overconfident logits (large magnitude, some wrong) should yield T > 1."""
    torch.manual_seed(0)
    num_classes = 4
    n = 400
    labels = torch.randint(0, num_classes, (n,))
    # Correct-class logit boosted hugely so model is very confident,
    # but flip ~30% of the "peaks" to a wrong class so many confident
    # predictions are incorrect -> NLL high -> T should soften (>1).
    base = torch.randn(n, num_classes) * 0.1
    boost_targets = labels.clone()
    wrong_mask = torch.rand(n) < 0.3
    boost_targets[wrong_mask] = (
        labels[wrong_mask] + torch.randint(1, num_classes, (int(wrong_mask.sum().item()),))
    ) % num_classes
    logits = base.clone()
    logits[torch.arange(n), boost_targets] += 15.0

    ts = TemperatureScaling()
    t = ts.fit(logits, labels)
    assert ts.fitted is True
    assert t > 1.0, f"expected T > 1 for overconfident logits, got T={t:.4f}"


def test_transform_preserves_argmax() -> None:
    torch.manual_seed(1)
    logits = torch.randn(64, 7) * 3.0
    labels = torch.randint(0, 7, (64,))
    ts = TemperatureScaling()
    ts.fit(logits, labels)

    scaled = ts.transform(logits)
    assert torch.equal(scaled.argmax(dim=1), logits.argmax(dim=1))


def test_nll_after_calibration_not_greater_than_before() -> None:
    torch.manual_seed(2)
    # Overconfident synthetic classifier (large-magnitude logits).
    n, c = 300, 5
    labels = torch.randint(0, c, (n,))
    logits = torch.randn(n, c) * 0.2
    # Make it confident but with noise, so calibration helps.
    peak = torch.where(torch.rand(n) < 0.7, labels, torch.randint(0, c, (n,)))
    logits[torch.arange(n), peak] += 10.0

    nll_before = _nll(logits, labels)
    ts = TemperatureScaling()
    ts.fit(logits, labels)
    nll_after = _nll(ts.transform(logits), labels)
    # Fitted T minimizes NLL, so equality only when T==1 is optimal.
    assert nll_after <= nll_before + 1e-6, (
        f"NLL should not increase after calibration: before={nll_before:.6f}, "
        f"after={nll_after:.6f}, T={ts.temperature:.4f}"
    )


def test_default_temperature_is_identity() -> None:
    ts = TemperatureScaling()
    assert ts.temperature == 1.0
    logits = torch.randn(3, 4)
    assert torch.allclose(ts.transform(logits), logits)


def test_transform_probs_is_softmax_of_scaled_logits() -> None:
    torch.manual_seed(3)
    logits = torch.randn(10, 6)
    labels = torch.randint(0, 6, (10,))
    ts = TemperatureScaling()
    ts.fit(logits, labels)

    expected = torch.softmax(logits / ts.temperature, dim=-1)
    assert torch.allclose(ts.transform_probs(logits), expected, atol=1e-6)
