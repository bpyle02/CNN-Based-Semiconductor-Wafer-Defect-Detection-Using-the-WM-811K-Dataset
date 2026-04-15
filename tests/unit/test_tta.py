"""Unit tests for ``src.inference.tta.predict_with_tta``."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.tta import predict_with_tta, _default_tta_ops


class _TinyNet(nn.Module):
    """Deterministic classifier: one conv + global mean + linear head."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.head = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x).mean(dim=(-2, -1))  # (B, 4)
        return self.head(feats)


def test_default_tta_has_eight_views() -> None:
    ops = _default_tta_ops()
    assert len(ops) == 8


def test_tta_output_shape_matches_batch() -> None:
    torch.manual_seed(0)
    model = _TinyNet(num_classes=5)
    x = torch.randn(4, 1, 16, 16)
    probs = predict_with_tta(model, x)
    assert probs.shape == (4, 5)
    # Output is a proper probability distribution per row.
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_tta_is_deterministic() -> None:
    torch.manual_seed(0)
    model = _TinyNet(num_classes=3)
    model.eval()
    x = torch.randn(2, 1, 12, 12)
    a = predict_with_tta(model, x)
    b = predict_with_tta(model, x)
    assert torch.allclose(a, b, atol=0.0)


def test_tta_averages_across_eight_augmentations() -> None:
    """Manually average 8 softmaxes and compare to ``predict_with_tta``."""
    torch.manual_seed(0)
    model = _TinyNet(num_classes=4)
    model.eval()
    x = torch.randn(3, 1, 10, 10)

    ops = _default_tta_ops()
    with torch.no_grad():
        manual = torch.stack(
            [torch.softmax(model(op(x)), dim=1) for op in ops], dim=0
        ).mean(dim=0)

    tta = predict_with_tta(model, x)
    assert torch.allclose(tta, manual, atol=1e-6)


def test_tta_accepts_custom_augmentations() -> None:
    torch.manual_seed(0)
    model = _TinyNet(num_classes=2)
    model.eval()
    x = torch.randn(1, 1, 8, 8)
    # Two-view identity average must equal a single forward pass softmax.
    identity = lambda t: t  # noqa: E731
    probs = predict_with_tta(model, x, augmentations=[identity, identity])
    with torch.no_grad():
        expected = torch.softmax(model(x), dim=1)
    assert torch.allclose(probs, expected, atol=1e-6)


def test_tta_restores_training_mode() -> None:
    model = _TinyNet(num_classes=2)
    model.train()
    _ = predict_with_tta(model, torch.randn(1, 1, 8, 8))
    assert model.training is True
