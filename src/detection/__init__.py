"""Anomaly and out-of-distribution detection module."""

from src.detection.ood import MahalanobisDetector, OutOfDistributionDetector

__all__ = ["OutOfDistributionDetector", "MahalanobisDetector"]
