"""Augmentation module with synthetic data generation."""

from src.augmentation.evaluation import FIDScorer, InceptionScorer
from src.augmentation.synthetic import SyntheticDataGenerator

__all__ = ["SyntheticDataGenerator", "FIDScorer", "InceptionScorer"]
