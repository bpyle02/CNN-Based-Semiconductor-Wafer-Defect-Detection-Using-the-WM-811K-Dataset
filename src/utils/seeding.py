"""Centralized seed management.

A single entry point (:func:`seed_everything`) that sets seeds across every
RNG source the project touches: Python's ``random`` module, NumPy,
PyTorch (CPU and CUDA), and the ``PYTHONHASHSEED`` environment variable.

Idempotent: calling twice with the same seed produces the same state.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

_MIN_SEED: Final[int] = 0
_MAX_SEED: Final[int] = 2**32 - 1


def seed_everything(seed: int = 42) -> None:
    """Set seeds for python ``random``, numpy, torch (CPU + CUDA), and ``PYTHONHASHSEED``.

    Logs the seed at INFO. Idempotent. Returns nothing.

    Args:
        seed: Seed value; must fit in an unsigned 32-bit integer
            (``0 <= seed <= 2**32 - 1``). Defaults to 42.

    Raises:
        ValueError: If ``seed`` is outside the valid range.
    """
    if not isinstance(seed, int) or seed < _MIN_SEED or seed > _MAX_SEED:
        raise ValueError(f"seed must be an int in [{_MIN_SEED}, {_MAX_SEED}], got {seed!r}")

    # PYTHONHASHSEED affects the hash of str/bytes/datetime in child processes.
    # Must be set before child process spawn to take effect there.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Torch is imported lazily so this module stays importable without torch.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not available, skipping torch seed")

    logger.info("seed_everything: seed=%d", seed)
