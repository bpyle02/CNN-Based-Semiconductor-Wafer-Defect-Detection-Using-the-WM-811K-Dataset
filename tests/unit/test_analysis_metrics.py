"""Tests for calibration and evaluation metrics."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis import compute_calibration_metrics, evaluate_model


def test_compute_calibration_metrics_returns_expected_keys():
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1], dtype=np.int64)

    metrics = compute_calibration_metrics(probabilities, labels, n_bins=2)

    assert metrics["ece"] == pytest.approx(0.15, abs=1e-6)
    assert metrics["brier_score"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["negative_log_likelihood"] > 0
    assert metrics["mean_confidence"] == pytest.approx(0.85, abs=1e-6)
    assert len(metrics["reliability_bins"]) == 2


def test_evaluate_model_includes_calibration_metrics():
    class FixedLogitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "logits",
                torch.tensor(
                    [
                        [3.0, 1.0, 0.0],
                        [0.5, 2.0, 0.1],
                    ],
                    dtype=torch.float32,
                ),
            )

        def forward(self, x):
            return self.logits[: x.shape[0]]

    inputs = torch.zeros(2, 1)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    model = FixedLogitModel()

    preds, y_true, metrics = evaluate_model(
        model,
        loader,
        class_names=["A", "B", "C"],
        model_name="Fixed",
        device="cpu",
    )

    assert preds.shape == (2,)
    assert y_true.shape == (2,)
    assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)
    assert "ece" in metrics
    assert "brier_score" in metrics
    assert "negative_log_likelihood" in metrics
    assert len(metrics["reliability_bins"]) == 15
