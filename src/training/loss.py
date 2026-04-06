"""
Custom loss functions for training models on imbalanced datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Dynamically scales the loss based on prediction confidence, focusing more
    on hard examples (minority classes).
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # compute probabilities
        pt = torch.exp(-ce_loss)
        
        # compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
