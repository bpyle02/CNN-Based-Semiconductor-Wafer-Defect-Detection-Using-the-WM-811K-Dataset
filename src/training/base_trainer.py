"""Base trainer class for uniform configuration management."""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict, Any, Union
from src.config import load_config
from src.exceptions import TrainingError
from src.model_registry import save_checkpoint_with_hash, verify_checkpoint

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Base trainer that ensures consistent configuration across all training scripts.

    Provides:
    - Config loading from YAML with validation
    - Reproducibility (seed setting across all RNGs)
    - Device management (CPU/GPU)
    - Checkpoint saving/loading with metadata
    """

    def __init__(self, config_path: str = 'config.yaml') -> None:
        """
        Initialize trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = load_config(config_path)
        self.device = torch.device(self.config.device)
        self.seed = self.config.seed
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.set_seed()

    def set_seed(self) -> None:
        """
        Set random seed for reproducibility across all libraries.

        Sets seeds for: torch, numpy, random, and CUDA (if available)
        Also disables non-deterministic algorithms for CUDA.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        filename: str = 'checkpoint.pth',
    ) -> str:
        """
        Save training checkpoint with model, optimizer, and metrics.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer instance
            epoch: Current training epoch
            metrics: Dictionary of metrics to save
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename
        try:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'seed': self.seed,
                'device': str(self.device),
            }
            file_hash = save_checkpoint_with_hash(checkpoint_data, checkpoint_path)
            logger.info(
                f"Checkpoint saved to {checkpoint_path} "
                f"(SHA-256: {file_hash[:16]}...)"
            )
        except Exception as e:
            raise TrainingError(f"Failed to save checkpoint to {checkpoint_path}: {e}") from e
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Load model and optimizer state from checkpoint.

        Args:
            model: PyTorch model to load state into
            optimizer: Optional optimizer to restore state
            checkpoint_path: Path to checkpoint. If None, uses latest.

        Returns:
            Dictionary with epoch and metrics

        Raises:
            TrainingError: If checkpoint not found or cannot be loaded
        """
        if checkpoint_path is None:
            checkpoints = list(self.checkpoint_dir.glob('*.pth'))
            if not checkpoints:
                raise TrainingError(f"No checkpoints in {self.checkpoint_dir}")
            checkpoint_path = sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-1]

        checkpoint_path = Path(checkpoint_path)

        # Verify checkpoint integrity before loading
        if not verify_checkpoint(checkpoint_path):
            logger.warning(
                f"Checkpoint integrity verification FAILED for {checkpoint_path}. "
                "File may be corrupted or tampered with."
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint['model_state_dict'])

            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            raise TrainingError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e

        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
        }

    def get_training_params(self) -> Dict[str, Any]:
        """
        Get training hyperparameters from config.

        Returns:
            Dictionary with commonly used training parameters
        """
        training_cfg = self.config.training
        data_cfg = self.config.data

        learning_rate = training_cfg.learning_rate
        if isinstance(learning_rate, dict):
            learning_rate = learning_rate.get(
                training_cfg.default_model,
                next(iter(learning_rate.values())),
            )

        return {
            'device': str(self.device),
            'seed': self.seed,
            'learning_rate': learning_rate,
            'batch_size': training_cfg.batch_size,
            'epochs': training_cfg.epochs,
            'weight_decay': training_cfg.weight_decay,
            'optimizer': training_cfg.optimizer,
            'scheduler': getattr(training_cfg, 'scheduler', 'plateau'),
        }
