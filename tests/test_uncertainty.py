"""
Unit tests for uncertainty quantification module.

Tests:
- MCDropoutModel forward passes and uncertainty estimation
- UncertaintyEstimator dataset-level analysis
- Calibration metrics computation
- Confidence interval calculation
- Active learning sample selection
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.inference.uncertainty import (
    MCDropoutModel,
    UncertaintyEstimator,
    enable_dropout,
    disable_dropout,
    compute_confidence_intervals,
)


class SimpleDropoutNet(nn.Module):
    """Simple network with dropout for testing."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class TestMCDropoutModel:
    """Tests for MCDropoutModel class."""

    @pytest.fixture
    def model_and_device(self):
        """Create model and device for testing."""
        model = SimpleDropoutNet(num_classes=3)
        device = 'cpu'
        return model, device

    def test_initialization(self, model_and_device):
        """Test MCDropoutModel initialization."""
        model, device = model_and_device
        mc_model = MCDropoutModel(model, num_iterations=10, device=device)

        assert mc_model.num_iterations == 10
        assert mc_model.device == device
        assert mc_model.model == model

    def test_initialization_validation(self, model_and_device):
        """Test invalid num_iterations raises error."""
        model, device = model_and_device
        with pytest.raises(ValueError):
            MCDropoutModel(model, num_iterations=0, device=device)

    def test_predict_with_uncertainty_shape(self, model_and_device):
        """Test output shapes of predict_with_uncertainty."""
        model, device = model_and_device
        mc_model = MCDropoutModel(model, num_iterations=5, device=device)

        x = torch.randn(4, 10).to(device)  # Batch of 4 samples
        mean_probs, uncertainty = mc_model.predict_with_uncertainty(x)

        assert mean_probs.shape == (4, 3), "Mean probs shape mismatch"
        assert uncertainty.shape == (4,), "Uncertainty shape mismatch"
        assert np.all(mean_probs >= 0) and np.all(mean_probs <= 1), "Probs not in [0, 1]"

    def test_predict_with_uncertainty_distribution(self, model_and_device):
        """Test return_dist parameter."""
        model, device = model_and_device
        mc_model = MCDropoutModel(model, num_iterations=5, device=device)

        x = torch.randn(2, 10).to(device)
        mean_probs, unc, probs_dist = mc_model.predict_with_uncertainty(
            x, return_dist=True
        )

        assert probs_dist.shape == (2, 5, 3), "Distribution shape mismatch"
        assert np.allclose(mean_probs, probs_dist.mean(axis=1), atol=1e-5)

    def test_predict_proba_with_uncertainty(self, model_and_device):
        """Test predict_proba_with_uncertainty method."""
        model, device = model_and_device
        mc_model = MCDropoutModel(model, num_iterations=10, device=device)

        x = torch.randn(4, 10).to(device)
        mean_probs, std_probs, entropy = mc_model.predict_proba_with_uncertainty(x)

        assert mean_probs.shape == (4, 3)
        assert std_probs.shape == (4, 3)
        assert entropy.shape == (4,)
        assert np.all(entropy >= 0), "Entropy should be non-negative"
        assert np.all(entropy <= np.log(3)), "Entropy exceeds max for 3 classes"

    def test_confidence_intervals(self, model_and_device):
        """Test confidence_intervals method."""
        model, device = model_and_device
        mc_model = MCDropoutModel(model, num_iterations=20, device=device)

        x = torch.randn(4, 10).to(device)
        median, lower, upper = mc_model.confidence_intervals(
            x, percentiles=(2.5, 97.5)
        )

        assert median.shape == (4, 3)
        assert lower.shape == (4, 3)
        assert upper.shape == (4, 3)
        assert np.all(lower <= median), "Lower bound > median"
        assert np.all(median <= upper), "Median > upper bound"

    def test_variability_across_iterations(self, model_and_device):
        """Test that MC samples vary (dropout is active)."""
        model, device = model_and_device
        mc_model = MCDropoutModel(model, num_iterations=20, device=device)

        x = torch.randn(1, 10).to(device)
        _, _, probs_dist = mc_model.predict_with_uncertainty(x, return_dist=True)

        # Check that variance is non-zero (dropout is varying predictions)
        var = probs_dist.var(axis=1)[0]
        assert var.sum() > 1e-5, "Predictions not varying across MC samples"


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator class."""

    @pytest.fixture
    def estimator_and_loader(self):
        """Create estimator and test data loader."""
        model = SimpleDropoutNet(num_classes=3)
        estimator = UncertaintyEstimator(model, num_iterations=5, device='cpu')

        # Create synthetic dataset
        x = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        return estimator, loader

    def test_estimate_dataset_uncertainty(self, estimator_and_loader):
        """Test dataset uncertainty estimation."""
        estimator, loader = estimator_and_loader
        results = estimator.estimate_dataset_uncertainty(loader)

        assert 'uncertainty' in results
        assert 'mean_probs' in results
        assert 'entropy' in results
        assert 'predictions' in results
        assert 'true_labels' in results

        assert len(results['uncertainty']) == 20
        assert results['mean_probs'].shape == (20, 3)
        assert results['entropy'].shape == (20,)

    def test_get_uncertain_samples(self, estimator_and_loader):
        """Test uncertain sample selection."""
        estimator, loader = estimator_and_loader

        for metric in ['entropy', 'variance', 'margin']:
            uncertain = estimator.get_uncertain_samples(loader, k=5, metric=metric)

            assert 'indices' in uncertain
            assert 'uncertainties' in uncertain
            assert 'predictions' in uncertain

            assert len(uncertain['indices']) == 5
            assert len(uncertain['uncertainties']) == 5

    def test_get_uncertain_samples_invalid_metric(self, estimator_and_loader):
        """Test invalid metric raises error."""
        estimator, loader = estimator_and_loader
        with pytest.raises(ValueError):
            estimator.get_uncertain_samples(loader, k=5, metric='invalid')

    def test_uncertainty_calibration(self, estimator_and_loader):
        """Test calibration metrics."""
        estimator, loader = estimator_and_loader
        metrics = estimator.uncertainty_calibration(loader)

        assert 'brier_score' in metrics
        assert 'ece' in metrics
        assert 'uncertainty_accuracy_correlation' in metrics

        assert 0 <= metrics['brier_score'] <= 1
        assert 0 <= metrics['ece'] <= 1
        assert -1 <= metrics['uncertainty_accuracy_correlation'] <= 1


class TestDropoutHelpers:
    """Tests for enable/disable dropout helpers."""

    def test_enable_dropout(self):
        """Test enable_dropout sets training mode."""
        model = SimpleDropoutNet()
        model.eval()  # Start in eval mode

        enable_dropout(model)

        # Check dropout layers are in training mode
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training

    def test_disable_dropout(self):
        """Test disable_dropout sets eval mode."""
        model = SimpleDropoutNet()
        model.train()  # Start in training mode

        disable_dropout(model)

        # Check dropout layers are in eval mode
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert not module.training


class TestConfidenceIntervals:
    """Tests for confidence_intervals helper function."""

    def test_compute_confidence_intervals(self):
        """Test confidence interval computation."""
        # Create mock prediction distribution: (N, T, C) shape
        # N=samples, T=MC iterations, C=classes
        np.random.seed(42)
        predictions = np.random.random((10, 15, 3))
        # Normalize to probabilities
        predictions = predictions / predictions.sum(axis=2, keepdims=True)

        median, lower, upper = compute_confidence_intervals(
            predictions, percentiles=(2.5, 97.5)
        )

        assert median.shape == (10, 3)
        assert lower.shape == (10, 3)
        assert upper.shape == (10, 3)
        assert np.all(lower <= median)
        assert np.all(median <= upper)


class TestIntegration:
    """Integration tests for uncertainty quantification pipeline."""

    def test_full_pipeline(self):
        """Test complete uncertainty quantification workflow."""
        # Create model and data
        model = SimpleDropoutNet(num_classes=4)
        x = torch.randn(32, 10)
        y = torch.randint(0, 4, (32,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Initialize estimator
        estimator = UncertaintyEstimator(model, num_iterations=10, device='cpu')

        # Estimate uncertainties
        results = estimator.estimate_dataset_uncertainty(loader)
        assert len(results['uncertainty']) == 32

        # Get calibration metrics
        metrics = estimator.uncertainty_calibration(loader)
        assert all(k in metrics for k in ['brier_score', 'ece'])

        # Get uncertain samples
        uncertain = estimator.get_uncertain_samples(loader, k=5, metric='entropy')
        assert len(uncertain['indices']) == 5

    def test_entropy_properties(self):
        """Test entropy computation properties."""
        model = SimpleDropoutNet(num_classes=2)
        estimator = UncertaintyEstimator(model, num_iterations=20, device='cpu')

        x = torch.randn(20, 10)
        mean_probs, _, entropy = estimator.mc_model.predict_proba_with_uncertainty(x)

        # Entropy should be in valid range for 2 classes
        assert np.all(entropy >= 0), "Entropy should be non-negative"
        assert np.all(entropy <= np.log(2)), f"Entropy exceeds max for binary case: {entropy.max()} > {np.log(2)}"

        # With 20 samples, entropy should vary across samples
        assert entropy.std() > 1e-5, "Entropy should have some variance across samples"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
