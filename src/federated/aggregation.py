"""Byzantine-robust aggregation methods for federated learning.

Implements robust aggregation strategies that can tolerate malicious or faulty
clients in federated learning settings.
"""

from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn


class Krum:
    """Krum aggregation: selects parameter vector closest to most others.

    Filters out potential Byzantine updates by selecting the gradient vector
    that is closest (in Euclidean distance) to the maximum number of other
    gradient vectors. Robust to f malicious clients where n >= 2f + 3
    (per Blanchard et al., 2017).
    """

    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        byzantine_tolerance: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate using Krum algorithm.
        
        Args:
            client_updates: List of client parameter dicts
            byzantine_tolerance: Number of Byzantine clients to tolerate
        
        Returns:
            Aggregated parameter dict (single selected update)
        """
        # Flatten all updates to 1D vectors
        n_clients = len(client_updates)
        flat_updates = []
        
        for update_dict in client_updates:
            flat_vec = torch.cat([
                v.flatten() for v in update_dict.values()
            ])
            flat_updates.append(flat_vec)
        
        flat_updates = torch.stack(flat_updates)  # (n_clients, n_params)
        
        # Compute pairwise distances
        distances = torch.cdist(flat_updates, flat_updates)  # (n_clients, n_clients)

        # Exclude self-distances by setting diagonal to infinity
        distances.fill_diagonal_(float('inf'))

        # For each client, find distance to k-closest others (where k = n - f - 2)
        k = n_clients - byzantine_tolerance - 2
        closest_distances = torch.topk(distances, k, largest=False)[0]

        # Sum of distances to k-closest neighbors (excluding self)
        neighbor_sums = closest_distances.sum(dim=1)
        
        # Select client with minimum sum of distances
        selected_idx = neighbor_sums.argmin()
        
        # Return the selected client's update
        return client_updates[selected_idx]


class MultiKrum:
    """Multi-Krum: selects m parameter vectors via repeated Krum.
    
    Applies Krum m times, removing the selected vector each iteration.
    More robust than single Krum as it combines multiple diverse updates.
    """

    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        num_selected: int = 2,
        byzantine_tolerance: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate using Multi-Krum algorithm.
        
        Args:
            client_updates: List of client parameter dicts
            num_selected: Number of vectors to select
            byzantine_tolerance: Number of Byzantine clients to tolerate
        
        Returns:
            Averaged aggregation of top-m selected vectors
        """
        selected_updates = []
        remaining_updates = list(client_updates)

        for _ in range(min(num_selected, len(client_updates))):
            if not remaining_updates:
                break
            # Apply Krum to remaining updates
            selected = Krum.aggregate(remaining_updates, byzantine_tolerance)
            selected_updates.append(selected)

            # Remove selected update by identity (safe: Krum returns original ref)
            remaining_updates = [
                u for u in remaining_updates
                if u is not selected
            ]
        
        # Average the selected updates
        return ByzantineRobustAggregator._average_dicts(selected_updates)


class ByzantineRobustAggregator:
    """Byzantine-robust aggregation with multiple strategies."""

    STRATEGIES = {
        'krum': Krum.aggregate,
        'multi_krum': MultiKrum.aggregate,
        'median': None,  # Placeholder
        'trimmed_mean': None,  # Placeholder
    }

    @staticmethod
    def aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        strategy: str = 'multi_krum',
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using specified strategy.
        
        Args:
            client_updates: List of client parameter dicts
            strategy: Aggregation strategy ('krum', 'multi_krum', etc.)
            **kwargs: Strategy-specific parameters
        
        Returns:
            Aggregated parameters
        
        Raises:
            ValueError: If strategy not supported
        """
        if strategy == 'krum':
            return Krum.aggregate(
                client_updates,
                byzantine_tolerance=kwargs.get('byzantine_tolerance', 1),
            )
        elif strategy == 'multi_krum':
            return MultiKrum.aggregate(
                client_updates,
                num_selected=kwargs.get('num_selected', 2),
                byzantine_tolerance=kwargs.get('byzantine_tolerance', 1),
            )
        elif strategy == 'fedavg':
            # Standard FedAvg (no Byzantine robustness)
            return ByzantineRobustAggregator._average_dicts(client_updates)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _average_dicts(
        dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Average a list of parameter dicts element-wise.
        
        Args:
            dicts: List of parameter dicts with same structure
        
        Returns:
            Averaged parameter dict
        """
        if not dicts:
            raise ValueError("No dicts to average")
        
        averaged = {}
        keys = dicts[0].keys()
        
        for key in keys:
            stacked = torch.stack([d[key] for d in dicts])
            averaged[key] = stacked.mean(dim=0)
        
        return averaged
