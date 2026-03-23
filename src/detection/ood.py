"""Out-of-Distribution (OOD) detection for wafer maps.

Detects samples that deviate from training distribution, useful for
identifying anomalous or corrupted wafer maps.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import ledoit_wolf


class MahalanobisDetector:
    """Mahalanobis distance-based OOD detection.
    
    Computes Mahalanobis distance from sample to training data distribution
    mean. High distance indicates OOD (anomalous) sample.
    """

    def __init__(self, shrinkage: bool = True, percentile: float = 95.0):
        """Initialize detector.

        Args:
            shrinkage: Use Ledoit-Wolf shrinkage for covariance estimation
            percentile: Percentile for threshold computation (default: 95)
        """
        self.mean = None
        self.inv_cov = None
        self.shrinkage = shrinkage
        self.percentile = percentile
        self.threshold = None

    def fit(self, features: np.ndarray) -> None:
        """Fit detector on training features.

        Args:
            features: Training feature matrix (n_samples, feature_dim)
        """
        self.mean = np.mean(features, axis=0)

        if self.shrinkage:
            cov, _ = ledoit_wolf(features)
        else:
            cov = np.cov(features.T)

        # Use pseudo-inverse for numerical stability
        self.inv_cov = np.linalg.pinv(cov)

        # Compute threshold on training data
        diff = features - self.mean
        distances = np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))
        self.threshold = np.percentile(distances, self.percentile)

    def detect(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect OOD samples.

        Args:
            features: Test feature matrix (n_samples, feature_dim)

        Returns:
            Tuple of (distances, is_ood) where is_ood is boolean array
        """
        if self.mean is None or self.inv_cov is None or self.threshold is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        # Mahalanobis distance
        diff = features - self.mean
        distances = np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))

        # Use fixed threshold from training
        is_ood = distances > self.threshold

        return distances, is_ood


class OutOfDistributionDetector:
    """High-level OOD detection interface."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """Initialize detector.
        
        Args:
            model: Trained classifier (features extracted from intermediate layer)
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.feature_extractor = None
        self.detector = MahalanobisDetector(shrinkage=True)

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> np.ndarray:
        """Extract features from model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
        
        Returns:
            Feature matrix (batch_size, feature_dim)
        """
        self.model.eval()
        with torch.no_grad():
            # Extract features before final FC layer
            if hasattr(self.model, 'avgpool'):
                # ResNet-like
                features = self.model.avgpool(self.model.layer4(self.model.layer3(
                    self.model.layer2(self.model.layer1(self.model.conv1(x))))))
            else:
                # Custom CNN or other
                features = x
                for module in self.model.modules():
                    if isinstance(module, nn.Linear):
                        break
                    features = module(features) if hasattr(module, '__call__') else features
        
        # Flatten
        batch_size = features.shape[0]
        return features.view(batch_size, -1).cpu().numpy()

    def fit(self, x_train: torch.Tensor) -> None:
        """Fit OOD detector on training data.
        
        Args:
            x_train: Training data tensor
        """
        features = self.extract_features(x_train)
        self.detector.fit(features)

    def detect(self, x_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Detect OOD samples in test data.
        
        Args:
            x_test: Test data tensor
        
        Returns:
            Tuple of (distances, is_ood)
        """
        features = self.extract_features(x_test)
        distances, is_ood = self.detector.detect(features)
        return distances, is_ood
