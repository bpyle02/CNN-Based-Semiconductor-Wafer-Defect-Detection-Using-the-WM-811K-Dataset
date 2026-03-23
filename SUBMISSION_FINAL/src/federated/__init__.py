"""Federated learning module with Byzantine-robust aggregation."""

from src.federated.aggregation import ByzantineRobustAggregator, Krum, MultiKrum

__all__ = [
    'ByzantineRobustAggregator',
    'Krum',
    'MultiKrum',
]
