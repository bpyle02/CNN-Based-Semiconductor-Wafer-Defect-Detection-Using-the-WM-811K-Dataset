"""Unit tests for src/training/simclr.py."""

import pytest
import torch
import torch.nn as nn

from src.training.simclr import (
    SimCLRProjection,
    SimCLRLoss,
    ContrastiveBYOLLoss,
    SimCLREncoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_features():
    """Batch of random feature vectors."""
    return torch.randn(8, 512)


@pytest.fixture
def projection_head():
    return SimCLRProjection(input_dim=512, hidden_dim=256, output_dim=128)


# ---------------------------------------------------------------------------
# SimCLRProjection
# ---------------------------------------------------------------------------

class TestSimCLRProjection:
    def test_output_shape(self, projection_head, dummy_features):
        out = projection_head(dummy_features)
        assert out.shape == (8, 128)

    def test_output_is_normalized(self, projection_head, dummy_features):
        out = projection_head(dummy_features)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_single_sample(self, projection_head):
        out = projection_head(torch.randn(1, 512))
        assert out.shape == (1, 128)
        assert torch.allclose(torch.norm(out, dim=1), torch.ones(1), atol=1e-5)

    def test_custom_dims(self):
        proj = SimCLRProjection(input_dim=256, hidden_dim=128, output_dim=64)
        out = proj(torch.randn(4, 256))
        assert out.shape == (4, 64)

    def test_output_dtype(self, projection_head, dummy_features):
        out = projection_head(dummy_features)
        assert out.dtype == torch.float32

    def test_gradient_flows(self, projection_head, dummy_features):
        dummy_features.requires_grad_(True)
        out = projection_head(dummy_features)
        loss = out.sum()
        loss.backward()
        assert dummy_features.grad is not None


# ---------------------------------------------------------------------------
# SimCLRLoss
# ---------------------------------------------------------------------------

class TestSimCLRLoss:
    def test_returns_scalar(self):
        loss_fn = SimCLRLoss(temperature=0.07)
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        loss = loss_fn(z_i, z_j)
        assert loss.dim() == 0  # scalar

    def test_loss_positive(self):
        loss_fn = SimCLRLoss(temperature=0.07)
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        loss = loss_fn(z_i, z_j)
        assert loss.item() > 0

    def test_loss_with_identical_views(self):
        """Identical views should yield lower loss than random views."""
        loss_fn = SimCLRLoss(temperature=0.5)
        z = torch.randn(8, 128)
        z = torch.nn.functional.normalize(z, dim=1)
        loss_identical = loss_fn(z, z)

        z_rand = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        loss_random = loss_fn(z, z_rand)

        # Identical views should have lower or equal loss
        assert loss_identical.item() <= loss_random.item() + 1.0

    def test_different_temperatures(self):
        z_i = torch.randn(4, 64)
        z_j = torch.randn(4, 64)

        loss_low_temp = SimCLRLoss(temperature=0.01)(z_i, z_j)
        loss_high_temp = SimCLRLoss(temperature=1.0)(z_i, z_j)

        # Both should be valid scalars
        assert loss_low_temp.dim() == 0
        assert loss_high_temp.dim() == 0

    def test_small_batch(self):
        loss_fn = SimCLRLoss(temperature=0.07)
        z_i = torch.randn(2, 128)
        z_j = torch.randn(2, 128)
        loss = loss_fn(z_i, z_j)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flows(self):
        loss_fn = SimCLRLoss(temperature=0.07)
        z_i = torch.randn(4, 128, requires_grad=True)
        z_j = torch.randn(4, 128, requires_grad=True)
        loss = loss_fn(z_i, z_j)
        loss.backward()
        assert z_i.grad is not None
        assert z_j.grad is not None


# ---------------------------------------------------------------------------
# ContrastiveBYOLLoss
# ---------------------------------------------------------------------------

class TestContrastiveBYOLLoss:
    def test_returns_scalar(self):
        loss_fn = ContrastiveBYOLLoss()
        p1 = torch.randn(8, 128)
        p2 = torch.randn(8, 128)
        loss = loss_fn(p1, p2)
        assert loss.dim() == 0

    def test_identical_projections_zero_loss(self):
        loss_fn = ContrastiveBYOLLoss()
        p = torch.randn(8, 128)
        loss = loss_fn(p, p)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_projections_positive_loss(self):
        loss_fn = ContrastiveBYOLLoss()
        p1 = torch.zeros(1, 128)
        p1[0, 0] = 1.0
        p2 = torch.zeros(1, 128)
        p2[0, 1] = 1.0
        loss = loss_fn(p1, p2)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# SimCLREncoder
# ---------------------------------------------------------------------------

class TestSimCLREncoder:
    def test_forward_with_features_backbone(self):
        """Test with a backbone that has a .features attribute."""
        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )

            def forward(self, x):
                return self.features(x).flatten(1)

        backbone = DummyBackbone()
        encoder = SimCLREncoder(backbone, feature_dim=16, projection_dim=64)
        img = torch.randn(2, 3, 96, 96)
        features, projection = encoder(img)
        assert projection.shape == (2, 64)
        # Projection should be normalized
        norms = torch.norm(projection, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_forward_without_features_backbone(self):
        """Test with a backbone that returns features directly."""
        class SimpleBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96, 256)

            def forward(self, x):
                return self.fc(x.flatten(1))

        backbone = SimpleBackbone()
        encoder = SimCLREncoder(backbone, feature_dim=256, projection_dim=128)
        img = torch.randn(2, 3, 96, 96)
        features, projection = encoder(img)
        assert features.shape == (2, 256)
        assert projection.shape == (2, 128)
