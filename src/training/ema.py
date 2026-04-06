"""
Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of model weights that is updated as an
exponential moving average of the training weights after each
optimizer step.  The EMA model typically generalizes better than
the final training checkpoint because it smooths over the noisy
trajectory of SGD / Adam.

    shadow = decay * shadow + (1 - decay) * current

Reference:
    Polyak & Juditsky (1992). "Acceleration of Stochastic Approximation
    by Averaging". SIAM Journal on Control and Optimization.
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * current

    The EMA model typically generalizes better than the final checkpoint.

    Args:
        model: The source model whose parameters will be tracked.
        decay: Smoothing factor in [0, 1). Higher values produce smoother
            averages (0.999 is typical for medium-length training runs).

    Example::

        ema = EMAModel(model, decay=0.999)
        for batch in train_loader:
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
            ema.update(model)

        # For evaluation:
        ema.apply_shadow(model)
        val_metrics = evaluate(model, val_loader)
        ema.restore(model)  # return model to training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.shadow: Dict[str, "torch.Tensor"] = {}  # noqa: F821
        self.backup: Dict[str, "torch.Tensor"] = {}  # noqa: F821

        # Initialize shadow weights from model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights.

        Should be called once after each ``optimizer.step()``.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model weights with shadow weights (for evaluation).

        The original weights are saved internally so they can be restored
        afterwards with :meth:`restore`.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore model weights from backup (after evaluation).

        Must be preceded by a call to :meth:`apply_shadow`.
        """
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
