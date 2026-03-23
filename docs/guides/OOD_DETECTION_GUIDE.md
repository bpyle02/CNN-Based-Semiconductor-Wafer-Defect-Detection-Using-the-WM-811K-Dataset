# Out-of-Distribution (OOD) Detection Guide

## Overview
Semiconductor fabrication frequently produces novel defect patterns that the model has never seen during training. OOD detection is critical to flag these anomalies rather than confidently assigning them to an incorrect known class.

## Methods Available
The `OODDetector` class supports two primary mathematical frameworks:
1. **Mahalanobis Distance**: Fits a Gaussian distribution to the normal feature space and calculates the standard deviations from the mean for new samples.
2. **ODIN (Out-of-DIstribution Network)**: Uses temperature scaling on logits to expose unseen distributions via softmax confidence depression.

## Quick Start
```python
import numpy as np
from src.analysis.anomaly import OODDetector

# 1. Instantiate the detector
detector = OODDetector(method='mahalanobis', threshold=0.95)

# 2. Fit on known 'normal' training features
# features shape: (N_samples, feature_dim)
detector.fit(train_features)

# 3. Detect anomalies in production
predictions = detector.detect_ood(new_production_features)

for is_ood in predictions:
    if is_ood:
        print("Alert: Novel Defect Pattern Detected!")
```