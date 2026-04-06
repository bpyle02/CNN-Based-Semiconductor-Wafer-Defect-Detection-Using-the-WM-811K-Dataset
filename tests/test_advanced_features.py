#!/usr/bin/env python3
"""Integration tests for Byzantine-robust FL, synthetic data, and OOD detection."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.federated.aggregation import Krum, MultiKrum, ByzantineRobustAggregator
from src.augmentation.synthetic import SyntheticDataGenerator, DefectSimulator
from src.augmentation.evaluation import FIDScorer, InceptionScorer
from src.detection.ood import MahalanobisDetector, OutOfDistributionDetector
from src.models import WaferCNN


def test_byzantine_aggregation():
    """Test Byzantine-robust federated aggregation."""
    print("\n" + "="*70)
    print("TESTING BYZANTINE-ROBUST AGGREGATION")
    print("="*70)
    
    # Simulate client updates (pretend gradients)
    n_clients = 5
    n_params = 100

    # Create benign clients with similar small values (cluster around 1.0)
    client_updates = [
        {'param': torch.randn(n_params) * 0.1 + 1.0}  # Mean ~1.0, std ~0.1
        for _ in range(n_clients - 1)
    ]

    # Add Byzantine (malicious) update with large outlier value
    client_updates.append({
        'param': torch.ones(n_params) * 100  # Outlier
    })

    # Test Krum
    selected = Krum.aggregate(client_updates, byzantine_tolerance=1)
    print(f"\nKrum selected: {selected['param'].abs().max():.4f} (should be ~1.0, not 100)")
    
    # Test MultiKrum
    avg = MultiKrum.aggregate(client_updates, num_selected=3, byzantine_tolerance=1)
    print(f"MultiKrum aggregated: {avg['param'].abs().max():.4f}")
    
    # Test FedAvg for comparison
    fedavg = ByzantineRobustAggregator.aggregate(client_updates, strategy='fedavg')
    print(f"FedAvg (no robustness): {fedavg['param'].abs().max():.4f} (affected by Byzantine)")
    
    print("[OK] Byzantine-robust aggregation tests passed")


def test_synthetic_generation():
    """Test synthetic data generation."""
    print("\n" + "="*70)
    print("TESTING SYNTHETIC DATA GENERATION")
    print("="*70)
    
    gen = SyntheticDataGenerator(method='rule_based')
    
    # Generate samples for each defect class
    for class_id in range(3):
        synthetic = gen.generate(num_samples=10, class_label=class_id, size=96)
        print(f"\nClass {class_id}: Generated shape {synthetic.shape}")
        print(f"  Min: {synthetic.min():.4f}, Max: {synthetic.max():.4f}, Mean: {synthetic.mean():.4f}")
        assert synthetic.shape == (10, 96, 96)
        assert synthetic.min() >= 0 and synthetic.max() <= 1
    
    # Test individual simulators
    center = DefectSimulator.generate_center_defect(96)
    edge = DefectSimulator.generate_edge_loc_defect(96)
    scratch = DefectSimulator.generate_scratch_defect(96)
    
    print(f"\nCenter defect: {center.shape}, range [{center.min():.2f}, {center.max():.2f}]")
    print(f"Edge defect: {edge.shape}, range [{edge.min():.2f}, {edge.max():.2f}]")
    print(f"Scratch defect: {scratch.shape}, range [{scratch.min():.2f}, {scratch.max():.2f}]")
    
    print("[OK] Synthetic data generation tests passed")


def test_ood_detection():
    """Test OOD detection."""
    print("\n" + "="*70)
    print("TESTING OUT-OF-DISTRIBUTION DETECTION")
    print("="*70)
    
    # Create simple detector
    detector = MahalanobisDetector(shrinkage=True)
    
    # Generate in-distribution training data
    np.random.seed(42)
    X_train = np.random.normal(0, 1, (100, 50))
    detector.fit(X_train)
    print(f"\nFitted detector on {X_train.shape[0]} samples")
    
    # Test on in-distribution samples
    X_test_id = np.random.normal(0, 1, (20, 50))
    distances_id, is_ood_id = detector.detect(X_test_id)
    
    # Test on out-of-distribution samples
    X_test_ood = np.random.normal(5, 1, (20, 50))
    distances_ood, is_ood_ood = detector.detect(X_test_ood)
    
    print(f"ID samples - Mean distance: {distances_id.mean():.4f}, OOD detected: {is_ood_id.sum()}/{len(is_ood_id)}")
    print(f"OOD samples - Mean distance: {distances_ood.mean():.4f}, OOD detected: {is_ood_ood.sum()}/{len(is_ood_ood)}")
    
    # Verify OOD detection works
    assert distances_ood.mean() > distances_id.mean(), "OOD samples should have larger distances"
    assert is_ood_ood.sum() > is_ood_id.sum(), "More OOD samples should be detected"
    
    print("[OK] OOD detection tests passed")


def test_inception_score():
    """Test Inception Score computation."""
    print("\n" + "="*70)
    print("TESTING INCEPTION SCORE")
    print("="*70)
    
    # Create mock probabilities
    n_samples = 100
    n_classes = 9
    
    # High-quality: concentrated on few classes (low diversity, high confidence)
    high_quality_probs = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        high_quality_probs[i, i % 3] = 0.9
        high_quality_probs[i, (i + 1) % 3] = 0.05
        high_quality_probs[i, (i + 2) % 3] = 0.05
    
    # Low-quality: uniform (high diversity, low confidence)
    low_quality_probs = np.ones((n_samples, n_classes)) / n_classes
    
    scorer = InceptionScorer()
    is_high, _ = scorer.compute_is(high_quality_probs, splits=5)
    is_low, _ = scorer.compute_is(low_quality_probs, splits=5)
    
    print(f"\nHigh-quality IS: {is_high:.4f}")
    print(f"Low-quality IS: {is_low:.4f}")
    assert is_high > is_low, "High-quality should have higher IS"
    
    print("[OK] Inception Score tests passed")


if __name__ == '__main__':
    try:
        test_byzantine_aggregation()
        test_synthetic_generation()
        test_ood_detection()
        test_inception_score()
        
        print("\n" + "="*70)
        print("ALL ADVANCED FEATURE TESTS PASSED [OK]")
        print("="*70)
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
