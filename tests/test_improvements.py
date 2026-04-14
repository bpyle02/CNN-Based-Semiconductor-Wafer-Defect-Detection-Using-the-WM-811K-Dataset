import pytest
import torch
import numpy as np
from src.model_registry import ModelRegistry, ModelMetadata
from src.federated.fed_avg import ByzantineRobustAggregator
from src.analysis.anomaly import OODDetector


def test_model_registry(workspace_tmp_path):
    """Test model registration and retrieval."""
    registry = ModelRegistry(str(workspace_tmp_path))
    
    model = torch.nn.Linear(10, 2)
    metadata = ModelMetadata(
        model_name="test_model",
        architecture="Linear",
        num_classes=2,
        training_config={"lr": 0.01},
        metrics={"accuracy": 0.95},
        dataset_version="v1"
    )
    
    model_id = registry.register(model, metadata, version="v1.0")
    assert model_id in registry.list_models()
    
    loaded_state, loaded_meta = registry.load(model_id)
    assert "weight" in loaded_state
    assert loaded_meta["metrics"]["accuracy"] == 0.95

def test_byzantine_aggregation():
    """Test robust federated aggregation."""
    aggregator_median = ByzantineRobustAggregator(method='median')
    aggregator_trimmed = ByzantineRobustAggregator(method='trimmed_mean')
    
    weights_list = [
        {"layer1": torch.tensor([1.0, 2.0])},
        {"layer1": torch.tensor([1.1, 2.1])},
        {"layer1": torch.tensor([10.0, 20.0])} # Poisoned
    ]
    
    median_result = aggregator_median.aggregate(weights_list)
    assert torch.allclose(median_result["layer1"], torch.tensor([1.1, 2.1]))

def test_ood_detection():
    """Test out-of-distribution detection."""
    detector = OODDetector(method='mahalanobis', threshold=0.90)

    # Normal distribution
    rng = np.random.default_rng(42)
    normal_features = rng.normal(0, 1, (100, 10))
    detector.fit(normal_features)

    # In-distribution sample at the learned center
    id_sample = detector.feature_mean[None, :]
    # Out-of-distribution sample
    ood_sample = np.full((1, 10), 10.0)
    
    id_pred = detector.detect_ood(id_sample)
    ood_pred = detector.detect_ood(ood_sample)
    
    assert not id_pred[0]
    assert ood_pred[0]
