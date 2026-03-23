"""Augmentation module with synthetic data generation."""

from src.augmentation.synthetic import SyntheticDataGenerator
from src.augmentation.evaluation import FIDScorer, InceptionScorer

__all__ = ['SyntheticDataGenerator', 'FIDScorer', 'InceptionScorer']
