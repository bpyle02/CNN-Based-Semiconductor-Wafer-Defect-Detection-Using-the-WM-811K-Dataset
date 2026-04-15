"""
Training configuration and hyperparameters.

Centralizes all training settings in a single dataclass for reproducibility
and easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    """
    Training configuration for wafer defect detection models.

    Attributes:
        num_epochs: Number of training epochs (default: 5 for CPU feasibility)
        batch_size: Mini-batch size (default: 64)
        learning_rate: Initial learning rate (default: 1e-3 for CNN, 1e-4 for pretrained)
        weight_decay: L2 regularization coefficient (default: 1e-4)
        num_classes: Number of output classes (default: 9)
        seed: Random seed for reproducibility (default: 42)
        device: Compute device ('cuda' or 'cpu')
        scheduler_patience: Patience for ReduceLROnPlateau (default: 3)
        scheduler_factor: LR reduction factor (default: 0.5)
        loss_weights: Per-class weights for CrossEntropyLoss (set during data prep)
    """

    num_epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_classes: int = 9
    seed: int = 42
    device: str = "cpu"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    loss_weights: Optional[Any] = None  # Set to torch.Tensor during training
    model_name: str = "model"

    # Advanced options (rarely changed)
    gradient_clip: Optional[float] = None
    warmup_epochs: int = 0
    mixed_precision: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None and k != "loss_weights"}

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class AnalysisConfig:
    """Configuration for model analysis and visualization."""

    num_samples_per_class: int = 1
    confusion_matrix_normalize: bool = True
    plot_dpi: int = 120
    font_size: int = 11
    style: str = "seaborn-v0_8-whitegrid"
