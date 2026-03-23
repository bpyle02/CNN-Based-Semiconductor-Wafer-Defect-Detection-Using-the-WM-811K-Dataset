"""Anomaly and out-of-distribution detection module."""

from src.detection.ood import OutOfDistributionDetector, MahalanobisDetector

__all__ = ['OutOfDistributionDetector', 'MahalanobisDetector']
