"""
Anomaly Detection for wafer defect classification.

Implements one-class and novelty detection methods to identify defects as anomalies
relative to a "normal" class (e.g., 'none' class in wafer detection).

Methods:
    - Isolation Forest: Tree-based anomaly scoring
    - One-Class SVM: Support vector method for single-class learning
    - Autoencoder-based: Reconstruction error for anomaly detection
    - Statistical: Mahalanobis distance from normal class distribution
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class AnomalyDetectionAutoencoder(nn.Module):
    """
    Simple autoencoder for wafer map reconstruction.

    Learns to reconstruct "normal" wafers and uses reconstruction error
    as anomaly score.

    Args:
        input_channels: Number of input channels (3)
        latent_dim: Dimension of latent space (default 32)
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Flattened dim: 128 * (96//8) * (96//8) = 128 * 12 * 12 = 18432
        self.fc_encode = nn.Linear(128 * 12 * 12, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 12 * 12)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Constrain to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        x = self.encoder(x)
        x = x.flatten(1)
        z = self.fc_encode(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image."""
        x = self.fc_decode(z)
        x = x.reshape(-1, 128, 12, 12)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (reconstruction, latent_code)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class AnomalyDetector:
    """
    High-level anomaly detection interface.

    Supports multiple detection methods: Isolation Forest, One-Class SVM,
    Autoencoder reconstruction error, Mahalanobis distance.

    Args:
        method: Detection method ('isolation_forest', 'one_class_svm', 'autoencoder', 'mahalanobis')
        device: Device to run on (cuda or cpu)
    """

    def __init__(
        self,
        method: str = 'isolation_forest',
        device: str = 'cuda',
    ) -> None:
        if method not in ['isolation_forest', 'one_class_svm', 'autoencoder', 'mahalanobis']:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        self.normal_mean = None
        self.normal_cov = None
        self.normal_cov_inv = None

    def extract_features(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> np.ndarray:
        """
        Extract features from model for anomaly detection.

        Args:
            model: PyTorch model (features extracted from final layer before classification)
            data_loader: DataLoader for extracting features

        Returns:
            Feature matrix of shape (N, feature_dim)
        """
        model.eval()
        features_list = []

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)

                # Extract features from model
                # Assumes model has a method to get features
                if hasattr(model, 'features'):
                    feats = model.features(images)
                    feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
                    feats = feats.flatten(1)
                else:
                    # For models like ResNet, get intermediate features
                    # This is a simplification; may need to be model-specific
                    feats = model(images)

                features_list.append(feats.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        return features

    def fit_isolation_forest(
        self,
        features: np.ndarray,
        contamination: float = 0.05,
    ) -> None:
        """
        Fit Isolation Forest on normal class features.

        Args:
            features: Feature matrix from normal class
            contamination: Expected proportion of anomalies
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(features)

    def fit_one_class_svm(
        self,
        features: np.ndarray,
        nu: float = 0.05,
    ) -> None:
        """
        Fit One-Class SVM on normal class features.

        Args:
            features: Feature matrix from normal class
            nu: Expected proportion of outliers
        """
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)

        self.model = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=nu,
        )
        self.model.fit(features_normalized)

    def fit_mahalanobis(
        self,
        features: np.ndarray,
    ) -> None:
        """
        Fit Mahalanobis distance detector.

        Args:
            features: Feature matrix from normal class
        """
        self.normal_mean = features.mean(axis=0)
        self.normal_cov = np.cov(features.T)

        # Add small regularization to avoid singular matrix
        self.normal_cov += np.eye(self.normal_cov.shape[0]) * 1e-6
        self.normal_cov_inv = np.linalg.inv(self.normal_cov)

    def fit_autoencoder(
        self,
        train_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> None:
        """
        Train autoencoder on normal class data.

        Args:
            train_loader: DataLoader for normal class
            epochs: Number of training epochs
            lr: Learning rate
        """
        autoencoder = AnomalyDetectionAutoencoder().to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for images, _ in train_loader:
                images = images.to(self.device)

                optimizer.zero_grad()

                recon, _ = autoencoder(images)
                loss = criterion(recon, images)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0:
                print(f"Autoencoder Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        self.model = autoencoder

    def predict_isolation_forest(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (0=normal, -1=anomaly).

        Returns: Array of predictions
        """
        return self.model.predict(features)

    def predict_one_class_svm(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (1=normal, -1=anomaly).

        Returns: Array of predictions
        """
        features_normalized = self.scaler.transform(features)
        return self.model.predict(features_normalized)

    def predict_mahalanobis(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance as anomaly score.

        Higher score = more anomalous.

        Returns: Array of Mahalanobis distances
        """
        diff = features - self.normal_mean
        distances = np.sqrt(np.sum(diff @ self.normal_cov_inv * diff, axis=1))
        return distances

    def predict_autoencoder(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> np.ndarray:
        """
        Compute reconstruction error as anomaly score.

        Args:
            model: Original model for feature extraction
            data_loader: DataLoader for inference

        Returns:
            Array of reconstruction errors
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)

                recon, _ = self.model(images)
                error = torch.mean((recon - images) ** 2, dim=[1, 2, 3])

                errors.extend(error.cpu().numpy())

        return np.array(errors)

    def fit(
        self,
        model: nn.Module,
        normal_loader: DataLoader,
        **kwargs: Any,
    ) -> None:
        """
        Fit anomaly detector on normal class data.

        Args:
            model: PyTorch model for feature extraction
            normal_loader: DataLoader for normal class
            **kwargs: Method-specific arguments
        """
        if self.method == 'autoencoder':
            self.fit_autoencoder(normal_loader, **kwargs)
        else:
            # Extract features for other methods
            features = self.extract_features(model, normal_loader)

            if self.method == 'isolation_forest':
                self.fit_isolation_forest(features, **kwargs)
            elif self.method == 'one_class_svm':
                self.fit_one_class_svm(features, **kwargs)
            elif self.method == 'mahalanobis':
                self.fit_mahalanobis(features, **kwargs)

    def score(
        self,
        model: Optional[nn.Module],
        data_loader: DataLoader,
    ) -> np.ndarray:
        """
        Compute anomaly scores for data.

        Args:
            model: PyTorch model (None for autoencoder method)
            data_loader: DataLoader for scoring

        Returns:
            Array of anomaly scores
        """
        if self.method == 'autoencoder':
            return self.predict_autoencoder(model, data_loader)

        features = self.extract_features(model, data_loader)

        if self.method == 'isolation_forest':
            return self.predict_isolation_forest(features)
        elif self.method == 'one_class_svm':
            return self.predict_one_class_svm(features)
        elif self.method == 'mahalanobis':
            return self.predict_mahalanobis(features)

    def evaluate(
        self,
        model: Optional[nn.Module],
        normal_loader: DataLoader,
        anomaly_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate anomaly detector on normal vs. anomalous data.

        Args:
            model: PyTorch model
            normal_loader: DataLoader for normal samples
            anomaly_loader: DataLoader for anomalous samples

        Returns:
            Dictionary with metrics (AUROC, AUPR, etc.)
        """
        normal_scores = self.score(model, normal_loader)
        anomaly_scores = self.score(model, anomaly_loader)

        # For Isolation Forest and OCSVM: -1=anomaly, 1=normal
        # Flip so higher=more anomalous
        if self.method in ['isolation_forest', 'one_class_svm']:
            normal_scores = -normal_scores
            anomaly_scores = -anomaly_scores

        # Combine labels: 0=normal, 1=anomaly
        y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        y_scores = np.concatenate([normal_scores, anomaly_scores])

        # Compute metrics
        auroc = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        metrics = {
            'auroc': auroc,
            'fpr_at_95_tpr': np.min(fpr[tpr >= 0.95]) if np.any(tpr >= 0.95) else 1.0,
        }

        return metrics


class OODDetector:
    """Out-of-distribution detection."""

    def __init__(self, method: str = 'mahalanobis', threshold: float = 0.95):
        self.method = method
        self.threshold = threshold  # Percentile for anomaly threshold
        self.feature_mean = None
        self.feature_cov_inv = None
        self.ood_threshold = None

    def fit(self, features: np.ndarray) -> None:
        """Fit OOD detector on in-distribution features."""
        self.feature_mean = features.mean(axis=0)
        cov = np.cov(features.T)
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularization
        self.feature_cov_inv = np.linalg.inv(cov)
        if self.method == 'mahalanobis':
            train_scores = self._mahalanobis_distance(features)
        elif self.method == 'odin':
            train_scores = self._odin_score(features)
        else:
            raise ValueError(f"Unknown OOD method: {self.method}")
        self.ood_threshold = np.percentile(train_scores, self.threshold * 100)

    def detect_ood(self, features: np.ndarray) -> np.ndarray:
        """Return True for OOD samples."""
        if self.ood_threshold is None:
            raise RuntimeError("OODDetector must be fitted before calling detect_ood.")
        if self.method == 'mahalanobis':
            scores = self._mahalanobis_distance(features)
        elif self.method == 'odin':
            scores = self._odin_score(features)
        else:
            raise ValueError(f"Unknown OOD method: {self.method}")

        return scores > self.ood_threshold

    def _mahalanobis_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance from mean."""
        if self.feature_mean is None or self.feature_cov_inv is None:
            raise RuntimeError("OODDetector must be fitted before calling detect_ood.")
        diff = features - self.feature_mean
        distances = np.sqrt(np.sum(diff @ self.feature_cov_inv * diff, axis=1))
        return distances

    def _odin_score(self, logits: np.ndarray, temperature: float = 1000.0) -> np.ndarray:
        """ODIN score - based on softmax confidence."""
        # Higher ODIN score = more likely OOD
        exp_logits = np.exp(logits / temperature)
        softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        max_confidence = np.max(softmax, axis=1)
        return -max_confidence  # Negate so high = OOD
