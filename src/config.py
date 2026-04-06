"""
Configuration management for wafer defect detection training.

Provides YAML-based configuration loading with type validation and defaults.
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


def _as_int(value: Any, default: int) -> int:
    """Coerce YAML/config values to int when possible."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return int(str(value).strip())


def _as_float(value: Any, default: float) -> float:
    """Coerce YAML/config values to float when possible."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce YAML/config values to bool when possible."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _as_learning_rate(
    value: Any,
    default: Union[Dict[str, float], float],
) -> Union[Dict[str, float], float]:
    """Support either a global scalar LR or per-model mapping."""
    if value is None:
        return default
    if isinstance(value, dict):
        return {
            str(key): _as_float(val, 0.0)
            for key, val in value.items()
        }
    return _as_float(value, float(default) if isinstance(default, (int, float)) else 1e-4)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_path: str = "data/LSWMD_new.pkl"
    image_size: int = 96
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.image_size = _as_int(self.image_size, 96)
        self.train_size = _as_float(self.train_size, 0.70)
        self.val_size = _as_float(self.val_size, 0.15)
        self.test_size = _as_float(self.test_size, 0.15)
        self.num_workers = _as_int(self.num_workers, 0)
        self.pin_memory = _as_bool(self.pin_memory, False)
        assert 0 < self.train_size < 1, "train_size must be between 0 and 1"
        assert 0 < self.val_size < 1, "val_size must be between 0 and 1"
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert abs(self.train_size + self.val_size + self.test_size - 1.0) < 1e-6, "Sizes must sum to 1"


@dataclass
class TrainingConfig:
    """Training configuration."""
    default_model: str = "all"
    epochs: int = 5
    batch_size: int = 64
    learning_rate: Union[Dict[str, float], float] = field(default_factory=lambda: {
        "cnn": 1e-3,
        "resnet": 1e-4,
        "efficientnet": 1e-4
    })
    weight_decay: float = 1e-4
    gradient_clip_max_norm: float = 1.0
    optimizer: str = "adam"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.epochs = _as_int(self.epochs, 5)
        self.batch_size = _as_int(self.batch_size, 64)
        self.learning_rate = _as_learning_rate(
            self.learning_rate,
            {
                "cnn": 1e-3,
                "resnet": 1e-4,
                "efficientnet": 1e-4,
            },
        )
        self.weight_decay = _as_float(self.weight_decay, 1e-4)
        self.gradient_clip_max_norm = _as_float(self.gradient_clip_max_norm, 1.0)
        self.seed = _as_int(self.seed, 42)
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        if isinstance(self.learning_rate, dict):
            assert self.learning_rate, "learning_rate mapping cannot be empty"
            assert all(value > 0 for value in self.learning_rate.values()), "learning_rate values must be positive"
        else:
            assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.optimizer in ["adam", "sgd", "adamw"], f"Unknown optimizer: {self.optimizer}"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    architecture: str
    pretrained: bool = False
    num_classes: int = 9
    dropout_rate: float = 0.5
    freeze_until: Optional[str] = None


@dataclass
class Config:
    """Master configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary."""
        cfg = cls()
        config_dict = config_dict or {}

        # Load data config
        if 'data' in config_dict:
            data_cfg = config_dict['data'] or {}
            cfg.data = DataConfig(
                dataset_path=data_cfg.get('dataset_path', cfg.data.dataset_path),
                image_size=data_cfg.get('image_size', cfg.data.image_size),
                train_size=data_cfg.get('train_size', cfg.data.train_size),
                val_size=data_cfg.get('val_size', cfg.data.val_size),
                test_size=data_cfg.get('test_size', cfg.data.test_size),
                num_workers=data_cfg.get('num_workers', cfg.data.num_workers),
                pin_memory=data_cfg.get('pin_memory', cfg.data.pin_memory),
            )

        # Load training config
        if 'training' in config_dict:
            train_cfg = config_dict['training'] or {}
            cfg.training = TrainingConfig(
                default_model=train_cfg.get('default_model', cfg.training.default_model),
                epochs=train_cfg.get('epochs', cfg.training.epochs),
                batch_size=train_cfg.get('batch_size', cfg.training.batch_size),
                learning_rate=train_cfg.get('learning_rate', cfg.training.learning_rate),
                weight_decay=train_cfg.get('weight_decay', cfg.training.weight_decay),
                gradient_clip_max_norm=train_cfg.get('gradient_clip_max_norm', cfg.training.gradient_clip_max_norm),
                optimizer=train_cfg.get('optimizer', cfg.training.optimizer),
                seed=train_cfg.get('seed', cfg.training.seed),
            )

        # Load device and other settings
        device_cfg = config_dict.get('device')
        if isinstance(device_cfg, dict):
            cfg.device = device_cfg.get('type', cfg.device)
            cfg.seed = _as_int(device_cfg.get('seed', config_dict.get('seed', cfg.seed)), cfg.seed)
        elif device_cfg is not None:
            cfg.device = str(device_cfg)
            cfg.seed = _as_int(config_dict.get('seed', cfg.seed), cfg.seed)
        else:
            cfg.seed = _as_int(config_dict.get('seed', cfg.seed), cfg.seed)

        paths_cfg = config_dict.get('paths')
        if not isinstance(paths_cfg, dict):
            paths_cfg = {}

        cfg.log_dir = str(config_dict.get('log_dir', paths_cfg.get('log_dir', cfg.log_dir)))
        cfg.checkpoint_dir = str(
            config_dict.get('checkpoint_dir', paths_cfg.get('checkpoint_dir', cfg.checkpoint_dir))
        )

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.to_dict(), indent=2)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file or use defaults."""
    config_path = Path(config_path)

    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        return Config.from_yaml(str(config_path))
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return Config()


if __name__ == "__main__":
    # Example usage
    cfg = load_config("config.yaml")
    logger.info(cfg)
