"""Tests for src/training/domain_adaptation.py.

Focus on loss function output shapes/types and basic trainer wiring.
"""

import pytest
import torch
import torch.nn as nn

from src.training.domain_adaptation import (
    CORLAlignmentLoss,
    DomainAdversarialLoss,
    DomainAdaptationTrainer,
)


def test_coral_loss_returns_scalar():
    loss_fn = CORLAlignmentLoss(lambda_coral=1.0)
    source = torch.randn(16, 32)
    target = torch.randn(16, 32)
    loss = loss_fn(source, target)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_coral_loss_zero_for_identical_distributions():
    loss_fn = CORLAlignmentLoss(lambda_coral=1.0)
    torch.manual_seed(0)
    features = torch.randn(64, 16)
    loss = loss_fn(features, features)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_coral_loss_lambda_scales_output():
    torch.manual_seed(0)
    source = torch.randn(32, 16)
    target = torch.randn(32, 16) + 2.0
    l1 = CORLAlignmentLoss(lambda_coral=1.0)(source, target).item()
    l2 = CORLAlignmentLoss(lambda_coral=2.0)(source, target).item()
    assert l2 == pytest.approx(2.0 * l1, rel=1e-5)


def test_coral_loss_is_differentiable():
    loss_fn = CORLAlignmentLoss()
    source = torch.randn(8, 16, requires_grad=True)
    target = torch.randn(8, 16)
    loss = loss_fn(source, target)
    loss.backward()
    assert source.grad is not None
    assert source.grad.shape == source.shape


def test_domain_adversarial_loss_returns_scalar():
    loss_fn = DomainAdversarialLoss(feature_dim=32, hidden_dim=16)
    source = torch.randn(8, 32)
    target = torch.randn(8, 32)
    loss = loss_fn(source, target)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_domain_adversarial_loss_is_differentiable():
    loss_fn = DomainAdversarialLoss(feature_dim=16, hidden_dim=8)
    source = torch.randn(4, 16, requires_grad=True)
    target = torch.randn(4, 16, requires_grad=True)
    loss = loss_fn(source, target)
    loss.backward()
    assert source.grad is not None
    assert target.grad is not None


def test_domain_adversarial_discriminator_output_shape():
    loss_fn = DomainAdversarialLoss(feature_dim=10)
    source = torch.randn(5, 10)
    out = loss_fn.discriminator(source)
    assert out.shape == (5, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_trainer_rejects_unknown_method():
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError):
        DomainAdaptationTrainer(model, method="bogus", device="cpu")


def test_trainer_prepare_freezes_backbone():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    # Name the last layer "fc" so prepare_for_adaptation keeps it trainable.
    model.fc = nn.Linear(2, 2)  # type: ignore[attr-defined]
    trainer = DomainAdaptationTrainer(model, method="fine_tuning", device="cpu")
    trainer.prepare_for_adaptation(freeze_backbone=True)
    # fc params trainable; others frozen
    fc_params = [p.requires_grad for n, p in model.named_parameters() if "fc" in n]
    other_params = [p.requires_grad for n, p in model.named_parameters() if "fc" not in n]
    assert all(fc_params)
    assert not any(other_params)


def test_trainer_prepare_unfreeze_last_n_layers():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 2))
    trainer = DomainAdaptationTrainer(model, method="fine_tuning", device="cpu")
    trainer.prepare_for_adaptation(freeze_backbone=False, num_layers_unfreeze=2)
    params = list(model.named_parameters())
    # last 2 params trainable, rest frozen
    assert all(p.requires_grad for _, p in params[-2:])
    assert not any(p.requires_grad for _, p in params[:-2])
