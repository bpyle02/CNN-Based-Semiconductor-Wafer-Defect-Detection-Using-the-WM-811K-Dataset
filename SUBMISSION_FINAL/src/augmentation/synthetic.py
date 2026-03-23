"""Synthetic wafer map generation for data augmentation."""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class DefectSimulator:
    """Rule-based defect pattern simulator."""

    @staticmethod
    def generate_center_defect(size: int = 96, intensity: float = 0.5) -> np.ndarray:
        """Generate synthetic center defect pattern."""
        wafer = np.ones((size, size), dtype=np.float32) * 0.1
        center = size // 2
        radius = size // 8
        
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        wafer[mask] = 0.5 + intensity * 0.5
        
        return wafer

    @staticmethod
    def generate_edge_loc_defect(size: int = 96, intensity: float = 0.5) -> np.ndarray:
        """Generate synthetic edge localization defect."""
        wafer = np.ones((size, size), dtype=np.float32) * 0.1
        edge_width = size // 4
        
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            wafer[:edge_width, :] = 0.5 + intensity * 0.5
        elif edge == 'bottom':
            wafer[-edge_width:, :] = 0.5 + intensity * 0.5
        elif edge == 'left':
            wafer[:, :edge_width] = 0.5 + intensity * 0.5
        else:
            wafer[:, -edge_width:] = 0.5 + intensity * 0.5
        
        return wafer

    @staticmethod
    def generate_scratch_defect(size: int = 96, intensity: float = 0.5) -> np.ndarray:
        """Generate synthetic scratch (linear) defect."""
        wafer = np.ones((size, size), dtype=np.float32) * 0.1
        thickness = np.random.randint(2, 5)
        x0, y0 = np.random.randint(0, size, 2)

        # Create 2D coordinate grids
        yy, xx = np.ogrid[:size, :size]
        # Draw a diagonal scratch line with thickness
        dist_to_line = np.abs((xx - x0) - (yy - y0))
        mask = dist_to_line < thickness
        wafer[mask] = 0.5 + intensity * 0.5

        return np.clip(wafer, 0, 1)


class SyntheticDataGenerator:
    """High-level interface for synthetic data generation."""

    def __init__(self, method: str = 'rule_based'):
        """Initialize generator."""
        self.method = method

    def generate(
        self,
        num_samples: int,
        class_label: int,
        size: int = 96,
    ) -> np.ndarray:
        """Generate synthetic samples."""
        if self.method == 'rule_based':
            return self._generate_rule_based(num_samples, class_label, size)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _generate_rule_based(
        self,
        num_samples: int,
        class_label: int,
        size: int = 96,
    ) -> np.ndarray:
        """Generate synthetic samples using rule-based approach."""
        samples = []
        
        for _ in range(num_samples):
            if class_label == 0:  # Center
                sample = DefectSimulator.generate_center_defect(size)
            elif class_label in [1, 3]:  # Edge-Loc, Edge-Ring
                sample = DefectSimulator.generate_edge_loc_defect(size)
            elif class_label == 7:  # Scratch
                sample = DefectSimulator.generate_scratch_defect(size)
            else:
                sample = np.random.uniform(0.1, 0.9, (size, size)).astype(np.float32)
            
            samples.append(sample)
        
        return np.array(samples, dtype=np.float32)
