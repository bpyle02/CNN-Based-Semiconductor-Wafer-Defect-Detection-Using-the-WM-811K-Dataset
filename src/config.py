"""
Configuration management for wafer defect detection training.

The checked-in ``config.yaml`` is treated as the canonical schema for this
repository. The models below intentionally cover that file instead of silently
ignoring most sections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

MODEL_ALIASES = {"effnet": "efficientnet", "vit_small": "vit", "vit_tiny": "vit", "swin_tiny": "swin", "swin_micro": "swin"}
SUPPORTED_MODELS = {"cnn", "cnn_fpn", "resnet", "efficientnet", "vit", "swin", "ride"}


def canonicalize_model_name(value: str, allow_all: bool = False) -> str:
    """Normalize supported model identifiers and legacy aliases."""
    normalized = MODEL_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    allowed = set(SUPPORTED_MODELS)
    if allow_all:
        allowed.add("all")
    if normalized not in allowed:
        allowed_display = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown model name '{value}'. Expected one of: {allowed_display}")
    return normalized


class StrictConfigModel(BaseModel):
    """Base model that rejects undeclared keys."""

    model_config = ConfigDict(extra="forbid")


class SyntheticAugConfig(StrictConfigModel):
    """Config for synthetic defect generation to balance rare classes."""
    enabled: bool = False
    target_per_class: Optional[int] = None  # None = match max class count


class AugmentationConfig(StrictConfigModel):
    enabled: bool = True
    random_rotation: float = 15.0
    horizontal_flip: bool = True
    vertical_flip: bool = False
    gaussian_noise: float = 0.01
    gaussian_noise_std: float = 0.02
    gaussian_blur: bool = True
    random_erasing: bool = True
    radial_jitter: bool = True
    domain_specific: bool = True
    random_erasing_prob: float = 0.3
    radial_distortion_strength: float = 0.1
    brightness: float = 0.1
    contrast: float = 0.1
    synthetic: SyntheticAugConfig = Field(default_factory=SyntheticAugConfig)


class DataConfig(StrictConfigModel):
    """Data loading and preprocessing configuration."""

    dataset_path: str = "data/LSWMD_new.pkl"
    image_size: int = 96
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    num_workers: int = 0
    pin_memory: bool = False
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @model_validator(mode="after")
    def validate_sizes(self) -> "DataConfig":
        total = self.train_size + self.val_size + self.test_size
        if not 0 < self.train_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        if not 0 < self.val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if abs(total - 1.0) >= 1e-6:
            raise ValueError("train_size + val_size + test_size must sum to 1.0")
        return self


class ModelParameterConfig(StrictConfigModel):
    total: int = 0
    trainable: int = 0


class ModelConfig(StrictConfigModel):
    """Model architecture configuration."""

    name: str
    architecture: str
    pretrained: bool = False
    input_channels: int = 3
    num_classes: int = 9
    dropout_rate: float = 0.5
    use_batch_norm: bool = False
    feature_channels: List[int] = Field(default_factory=list)
    head_hidden_dim: Optional[int] = None
    head_dropout: Optional[float] = None
    freeze_until: Optional[str] = None
    frozen_prefixes: Optional[List[str]] = None
    fpn_out_channels: Optional[int] = None
    attention_type: Optional[str] = None  # None, "se", or "cbam"
    attention_reduction: int = 16
    # Swin Transformer fields
    embed_dim: Optional[int] = None
    depths: Optional[List[int]] = None
    num_heads: Optional[List[int]] = None
    window_size: Optional[int] = None
    mlp_ratio: Optional[float] = None
    attention_dropout: Optional[float] = None
    drop_path: Optional[float] = None
    # RIDE fields
    backbone: Optional[str] = None
    num_experts: Optional[int] = None
    reduction: Optional[int] = None
    diversity_weight: Optional[float] = None
    parameters: ModelParameterConfig = Field(default_factory=ModelParameterConfig)


class ModelCollectionConfig(StrictConfigModel):
    cnn: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="Custom CNN",
            architecture="custom",
            input_channels=3,
            use_batch_norm=True,
            feature_channels=[32, 64, 128, 256],
            head_hidden_dim=128,
            head_dropout=0.3,
        )
    )
    resnet: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="ResNet-18",
            architecture="resnet18",
            pretrained=True,
            freeze_until="layer3",
        )
    )
    efficientnet: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="EfficientNet-B0",
            architecture="efficientnet_b0",
            pretrained=True,
            freeze_until="features.6",
        )
    )
    cnn_fpn: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="Custom CNN-FPN",
            architecture="custom_fpn",
            input_channels=3,
            use_batch_norm=True,
            feature_channels=[32, 64, 128, 256],
            fpn_out_channels=128,
            dropout_rate=0.5,
        )
    )
    vit: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="ViT-small",
            architecture="vit_small",
            input_channels=3,
            dropout_rate=0.1,
        )
    )
    swin: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="Swin-Tiny",
            architecture="swin_tiny",
            input_channels=3,
            dropout_rate=0.0,
        )
    )
    ride: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="RIDE",
            architecture="ride",
            input_channels=3,
            dropout_rate=0.5,
        )
    )


class SchedulerConfig(StrictConfigModel):
    type: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-6
    step_size: int = 5
    t_max: Optional[int] = None
    eta_min: float = 0.0
    warmup_epochs: int = Field(default=0, ge=0)
    warmup_start_factor: float = Field(default=0.1, gt=0.0, le=1.0)
    verbose: bool = True

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = str(value).strip()
        aliases = {
            "plateau": "ReduceLROnPlateau",
            "reducelronplateau": "ReduceLROnPlateau",
            "step": "StepLR",
            "steplr": "StepLR",
            "cosine": "CosineAnnealingLR",
            "cosineannealinglr": "CosineAnnealingLR",
            "none": "none",
            "off": "none",
            "disabled": "none",
        }
        canonical = aliases.get(normalized.lower(), normalized)
        if canonical not in {"ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "none"}:
            raise ValueError(
                "scheduler.type must be one of: ReduceLROnPlateau, StepLR, CosineAnnealingLR, none"
            )
        return canonical

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"min", "max", "auto"}:
            raise ValueError("scheduler.mode must be one of: min, max, auto")
        return normalized


class MixupConfig(StrictConfigModel):
    """Mixup / CutMix batch augmentation configuration."""
    enabled: bool = False
    mixup_alpha: float = Field(default=0.2, ge=0.0)
    cutmix_alpha: float = Field(default=1.0, ge=0.0)
    mixup_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    cutmix_prob: float = Field(default=0.5, ge=0.0, le=1.0)


class EMAConfig(StrictConfigModel):
    """Exponential Moving Average configuration.

    Reference: Polyak & Juditsky (1992). "Acceleration of Stochastic Approximation"
    """
    enabled: bool = False
    decay: float = Field(default=0.999, ge=0.0, lt=1.0)


class TTAConfig(StrictConfigModel):
    """Test-Time Augmentation configuration.

    Reference: Shanmugam et al. (2021). "Better Aggregation in TTA". arXiv:2011.11156
    """
    enabled: bool = False
    num_views: int = Field(default=5, ge=1)


class SemiSupervisedConfig(StrictConfigModel):
    """Semi-supervised learning (FixMatch) configuration.

    Reference: [111] Sohn et al. (2020). "FixMatch". arXiv:2001.07685
    """
    enabled: bool = False
    method: str = "fixmatch"
    confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    lambda_u: float = Field(default=1.0, ge=0.0)
    strong_augment: bool = True
    unlabeled_batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=50, ge=1)


class LossConfig(StrictConfigModel):
    type: str = "CrossEntropyLoss"
    weighted: bool = True
    class_weights: Optional[List[float]] = None
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)
    reduction: str = "mean"
    tversky_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    tversky_beta: float = Field(default=0.7, ge=0.0, le=1.0)
    logit_adjustment_tau: float = Field(default=1.0, gt=0.0)
    cost_matrix: Optional[List[List[float]]] = None
    cost_near_full_missed: float = Field(default=10.0, ge=0.0)
    cost_rare_missed: float = Field(default=5.0, ge=0.0)
    cost_edge_confusion: float = Field(default=0.5, ge=0.0)

    @field_validator("type")
    @classmethod
    def validate_loss_type(cls, value: str) -> str:
        normalized = str(value).strip()
        allowed = {
            "CrossEntropyLoss",
            "FocalLoss",
            "DiceLoss",
            "TverskyLoss",
            "LogitAdjustedLoss",
            "CostSensitiveCE",
        }
        if normalized not in allowed:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(f"loss.type must be one of: {allowed_display}")
        return normalized

    @field_validator("reduction")
    @classmethod
    def validate_reduction(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"mean", "sum", "none"}:
            raise ValueError("loss.reduction must be one of: mean, sum, none")
        return normalized


class EarlyStoppingConfig(StrictConfigModel):
    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4


class CheckpointingConfig(StrictConfigModel):
    enabled: bool = True
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    metric: str = "val_loss"

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized not in {"val_loss", "val_acc", "val_macro_f1"}:
            raise ValueError("checkpointing.metric must be one of: val_loss, val_acc, val_macro_f1")
        return normalized


class TrainingConfig(StrictConfigModel):
    """Training configuration."""

    default_model: str = "all"
    epochs: int = Field(default=5, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: Union[Dict[str, float], float] = Field(
        default_factory=lambda: {
            "cnn": 1e-3,
            "resnet": 1e-4,
            "efficientnet": 1e-4,
        }
    )
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_clip_max_norm: float = 1.0
    optimizer: str = "adam"
    momentum: float = Field(default=0.9, ge=0.0)
    nesterov: bool = True
    use_focal_loss: bool = False
    mixed_precision: bool = False
    seed: int = 42
    balanced_sampling: bool = False
    drw_epoch: int = Field(default=0, ge=0)
    adaptive_rebalance: bool = False
    mixup: MixupConfig = Field(default_factory=MixupConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, value: str) -> str:
        return canonicalize_model_name(value, allow_all=True)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"adam", "sgd", "adamw"}:
            raise ValueError("optimizer must be one of: adam, sgd, adamw")
        return normalized

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(
        cls,
        value: Union[Dict[str, float], float],
    ) -> Union[Dict[str, float], float]:
        if isinstance(value, dict):
            if not value:
                raise ValueError("learning_rate mapping cannot be empty")
            normalized: Dict[str, float] = {}
            for key, raw_lr in value.items():
                model_name = canonicalize_model_name(key)
                lr = float(raw_lr)
                if lr <= 0:
                    raise ValueError("learning_rate values must be positive")
                normalized[model_name] = lr
            return normalized

        scalar_lr = float(value)
        if scalar_lr <= 0:
            raise ValueError("learning_rate must be positive")
        return scalar_lr


class ValidationConfig(StrictConfigModel):
    eval_every_n_epochs: int = 1
    compute_confusion_matrix: bool = True
    compute_per_class_metrics: bool = True
    compute_gradcam: bool = True
    num_gradcam_samples: int = 9


class GradCAMConfig(StrictConfigModel):
    enabled: bool = True
    target_layer: Optional[str] = None
    colormap: str = "jet"


class InferenceConfig(StrictConfigModel):
    model_path: str = "checkpoints/best_model.pth"
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    use_half_precision: bool = False
    gradcam: GradCAMConfig = Field(default_factory=GradCAMConfig)
    tta: TTAConfig = Field(default_factory=TTAConfig)


class EnsembleConfig(StrictConfigModel):
    enabled: bool = False
    models: List[str] = Field(default_factory=lambda: ["cnn", "resnet", "efficientnet"])
    aggregation: str = "voting"
    weights: List[float] = Field(default_factory=lambda: [0.33, 0.33, 0.34])

    @field_validator("models")
    @classmethod
    def validate_models(cls, values: List[str]) -> List[str]:
        return [canonicalize_model_name(value) for value in values]


class SearchSpaceConfig(StrictConfigModel):
    learning_rate: List[float] = Field(default_factory=lambda: [1e-5, 1e-2])
    batch_size: List[int] = Field(default_factory=lambda: [32, 64, 128])
    dropout_rate: List[float] = Field(default_factory=lambda: [0.3, 0.7])
    weight_decay: List[float] = Field(default_factory=lambda: [1e-5, 1e-3])


class OptimizationConfig(StrictConfigModel):
    metric: str = "macro_f1"
    direction: str = "maximize"


class TuningConfig(StrictConfigModel):
    enabled: bool = False
    method: str = "optuna"
    n_trials: int = 100
    n_jobs: int = 4
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


class QuantizationConfig(StrictConfigModel):
    enabled: bool = False
    dtype: str = "int8"


class PruningConfig(StrictConfigModel):
    enabled: bool = False
    percentage: float = 0.3


class DistillationConfig(StrictConfigModel):
    enabled: bool = False
    teacher_model: str = "checkpoints/best_model.pth"
    temperature: float = 4.0
    alpha: float = 0.7


class CompressionConfig(StrictConfigModel):
    enabled: bool = False
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)


class ActiveLearningConfig(StrictConfigModel):
    enabled: bool = False
    strategy: str = "uncertainty"
    initial_labeled: float = 0.1
    acquisition_size: int = 100
    max_iterations: int = 10
    uncertainty_method: str = "entropy"


class WandbConfig(StrictConfigModel):
    enabled: bool = False
    project: str = "wafer-defect-detection"
    entity: Optional[str] = None
    tags: List[str] = Field(default_factory=lambda: ["production", "v1.0"])


class MLflowConfig(StrictConfigModel):
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "wafer_defect"


class LoggingConfig(StrictConfigModel):
    level: str = "INFO"
    log_dir: str = "logs"
    save_configs: bool = True


class MLOpsConfig(StrictConfigModel):
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class CrossValidationConfig(StrictConfigModel):
    enabled: bool = False
    n_splits: int = 5
    method: str = "stratified_kfold"
    random_state: int = 42


class ProgressiveStageConfig(StrictConfigModel):
    image_size: int
    epochs: int
    learning_rate_factor: float


class ProgressiveTrainingConfig(StrictConfigModel):
    enabled: bool = False
    stages: List[ProgressiveStageConfig] = Field(default_factory=list)


class DistributedConfig(StrictConfigModel):
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500


class CalibrationConfig(StrictConfigModel):
    """Post-hoc probability calibration via temperature scaling or asymmetric."""
    enabled: bool = False
    method: str = "temperature_scaling"

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        allowed = {"temperature_scaling", "asymmetric"}
        if normalized not in allowed:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(f"calibration.method must be one of: {allowed_display}")
        return normalized


class UncertaintyConfig(StrictConfigModel):
    enabled: bool = False
    method: str = "monte_carlo_dropout"
    n_samples: int = 10
    dropout_rate: float = 0.5


class EvaluationConfig(StrictConfigModel):
    metrics: List[str] = Field(default_factory=list)
    visualizations: List[str] = Field(default_factory=list)
    save_predictions: bool = True
    output_dir: str = "results"


class DeviceRuntimeConfig(StrictConfigModel):
    type: str = "cuda"
    device_ids: List[int] = Field(default_factory=lambda: [0])
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False


class PathsConfig(StrictConfigModel):
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    result_dir: str = "results"
    config_dir: str = "configs"


class ExportConfig(StrictConfigModel):
    enabled: bool = False
    format: str = "onnx"
    output_dir: str = "exported_models"
    optimize: bool = True
    quantize_for_export: bool = False


class SupConConfig(StrictConfigModel):
    """Supervised Contrastive Learning configuration.

    Reference: Khosla et al. (2020). "Supervised Contrastive Learning". arXiv:2004.11362
    """
    enabled: bool = False
    temperature: float = Field(default=0.07, gt=0.0)
    projection_dim: int = Field(default=128, gt=0)
    pretrain_epochs: int = Field(default=50, gt=0)
    pretrain_lr: float = Field(default=0.05, gt=0.0)
    finetune_epochs: int = Field(default=25, gt=0)
    finetune_lr: float = Field(default=0.01, gt=0.0)


class SemiSupervisedConfig(StrictConfigModel):
    """Semi-supervised learning (FixMatch) configuration.

    Reference: Sohn et al. (2020). "FixMatch". arXiv:2001.07685
    """
    enabled: bool = False
    method: str = "fixmatch"
    confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    lambda_u: float = Field(default=1.0, ge=0.0)
    strong_augment: bool = True
    unlabeled_batch_size: int = Field(default=128, gt=0)
    epochs: int = Field(default=50, gt=0)


class Config(StrictConfigModel):
    """Master configuration class."""

    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelCollectionConfig = Field(default_factory=ModelCollectionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    progressive_training: ProgressiveTrainingConfig = Field(default_factory=ProgressiveTrainingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    supcon: SupConConfig = Field(default_factory=SupConConfig)
    semi_supervised: SemiSupervisedConfig = Field(default_factory=SemiSupervisedConfig)
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    device_settings: DeviceRuntimeConfig = Field(
        default_factory=DeviceRuntimeConfig,
        exclude=True,
    )

    @model_validator(mode="after")
    def sync_runtime_fields(self) -> "Config":
        """Keep compatibility convenience fields aligned with structured sections."""
        self.device = self.device_settings.type
        self.seed = self.device_settings.seed
        self.log_dir = self.paths.log_dir
        self.checkpoint_dir = self.paths.checkpoint_dir
        return self

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle)

        return cls.from_dict(config_dict)

    @classmethod
    def from_files(cls, config_paths: Iterable[Union[str, Path]]) -> "Config":
        """Load and merge multiple YAML config files in order."""
        merged: Dict[str, Any] = {}
        loaded_any = False

        for config_path in config_paths:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}

            merged = deep_merge_dicts(merged, payload)
            loaded_any = True

        if not loaded_any:
            raise ValueError("At least one config file path must be provided")

        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Load configuration from a dictionary with compatibility shims."""
        config_dict = dict(config_dict or {})

        device_cfg = config_dict.get("device")
        if isinstance(device_cfg, dict):
            config_dict["device_settings"] = device_cfg
            config_dict["device"] = device_cfg.get("type", "cuda")
            config_dict.setdefault("seed", device_cfg.get("seed", 42))

        paths_cfg = config_dict.get("paths")
        if isinstance(paths_cfg, dict):
            config_dict.setdefault("checkpoint_dir", paths_cfg.get("checkpoint_dir", "checkpoints"))
            config_dict.setdefault("log_dir", paths_cfg.get("log_dir", "logs"))

        training_cfg = config_dict.get("training")
        if isinstance(training_cfg, dict):
            checkpointing_cfg = training_cfg.get("checkpointing")
            if isinstance(checkpointing_cfg, dict):
                config_dict.setdefault("checkpoint_dir", checkpointing_cfg.get("save_dir", "checkpoints"))
            config_dict.setdefault("seed", training_cfg.get("seed", config_dict.get("seed", 42)))

        mlops_cfg = config_dict.get("mlops")
        if isinstance(mlops_cfg, dict):
            logging_cfg = mlops_cfg.get("logging")
            if isinstance(logging_cfg, dict):
                config_dict.setdefault("log_dir", logging_cfg.get("log_dir", "logs"))

        paths_payload = dict(config_dict.get("paths") or {})
        if "checkpoint_dir" in config_dict:
            paths_payload["checkpoint_dir"] = config_dict["checkpoint_dir"]
        if "log_dir" in config_dict:
            paths_payload["log_dir"] = config_dict["log_dir"]
        if paths_payload:
            config_dict["paths"] = paths_payload

        device_payload = dict(config_dict.get("device_settings") or {})
        device_payload.setdefault("type", config_dict.get("device", "cuda"))
        device_payload.setdefault("seed", config_dict.get("seed", 42))
        config_dict["device_settings"] = device_payload

        return cls.model_validate(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a repository-friendly dictionary."""
        payload = self.model_dump(exclude={"device_settings"})
        payload["device"] = self.device_settings.model_dump()
        return payload

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.to_dict(), indent=2)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file or use defaults."""
    config_path = Path(config_path)

    if config_path.exists():
        logger.info("Loading configuration from %s", config_path)
        return Config.from_yaml(str(config_path))

    logger.warning("Config file not found: %s, using defaults", config_path)
    return Config()


def load_merged_config(config_paths: Iterable[Union[str, Path]]) -> Config:
    """Load a merged configuration from multiple YAML files."""
    config_paths = list(config_paths)
    logger.info("Loading merged configuration from %s", ", ".join(str(path) for path in config_paths))
    return Config.from_files(config_paths)


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    logger.info(cfg)
