#!/usr/bin/env python3
"""
CLI entry point for training wafer defect models.

Usage:
    python train.py --model cnn --epochs 5 --device cuda
    python train.py --model all --epochs 5 --device cpu
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_recall_fscore_support,
)
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.model_registry import save_checkpoint_with_hash

from src.config import canonicalize_model_name, load_config, load_merged_config
from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset, get_image_transforms, get_imagenet_normalize, seed_worker
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.models.vit import get_vit_small
from src.analysis import evaluate_model
from src.mlops import MLFlowLogger, WandBLogger
from src.training.losses import build_classification_loss


from src.data.dataset import KNOWN_CLASSES
SEED = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_data(
    dataset_path,
    train_size=0.70,
    test_size=0.15,
    val_size=0.15,
    seed=SEED,
    synthetic=False,
):
    """Load and split data."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    logger.info(f"\n{'='*70}")
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info(f"{'='*70}")

    logger.info("Loading dataset...")
    df = load_dataset(dataset_path)

    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    logger.info(f"Total samples: {len(labels):,}")
    logger.info(f"Class distribution:")
    for i, cls in enumerate(KNOWN_CLASSES):
        count = (labels == i).sum()
        pct = 100 * count / len(labels)
        logger.info(f"  {cls:12s}: {count:6,} ({pct:5.1f}%)")

    # Split: train (70%), val (15%), test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)), labels, test_size=test_size,
        stratify=labels, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size),
        stratify=y_temp, random_state=seed
    )

    train_maps = np.array([wafer_maps[i] for i in X_train])
    val_maps = np.array([wafer_maps[i] for i in X_val])
    test_maps = np.array([wafer_maps[i] for i in X_test])

    # Optional: balance training set with synthetic augmentation
    if synthetic:
        from src.augmentation.synthetic import balance_dataset_with_synthetic
        logger.info("Applying synthetic augmentation to balance training set...")
        train_maps, y_train = balance_dataset_with_synthetic(
            train_maps, y_train, target_per_class=None, size=target_size[0]
        )
        logger.info(f"  Training set after augmentation: {len(y_train):,} samples")

    # Compute class weights from training set
    class_counts_train = Counter(y_train)
    total_train = len(y_train)
    loss_weights = torch.tensor(
        [total_train / (len(KNOWN_CLASSES) * class_counts_train[c]) for c in range(len(KNOWN_CLASSES))],
        dtype=torch.float32
    )
    logger.info(f"\nClass weights (from training set):")
    logger.info(f"  {[f'{w:.2f}' for w in loss_weights.tolist()]}")

    return {
        'train_maps': train_maps, 'y_train': y_train,
        'val_maps': val_maps, 'y_val': y_val,
        'test_maps': test_maps, 'y_test': y_test,
        'loss_weights': loss_weights,
        'class_names': KNOWN_CLASSES,
    }


from src.training.trainer import train_model

DEFAULT_SCHEDULER_CONFIG = SimpleNamespace(
    type="ReduceLROnPlateau",
    mode="auto",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)


def _call_with_supported_kwargs(factory, **kwargs):
    """Call a factory while ignoring unsupported keyword arguments."""
    supported = inspect.signature(factory).parameters
    filtered = {name: value for name, value in kwargs.items() if name in supported}
    return factory(**filtered)


def _apply_attention(model: nn.Module, model_cfg) -> nn.Module:
    """Inject SE or CBAM attention blocks if configured."""
    attention_type = getattr(model_cfg, "attention_type", None) if model_cfg else None
    if not attention_type:
        return model
    reduction = getattr(model_cfg, "attention_reduction", 16)
    if attention_type == "se":
        from src.models.attention import add_se_to_model
        return add_se_to_model(model, reduction=reduction)
    if attention_type == "cbam":
        from src.models.attention import add_cbam_to_model
        return add_cbam_to_model(model, reduction=reduction)
    logger.warning(f"Unknown attention_type '{attention_type}', skipping")
    return model


def build_model(model_name: str, model_cfg, num_classes: int, device: str) -> tuple[nn.Module, str]:
    """Construct a model using config-backed settings when supported."""
    common_kwargs = {"num_classes": num_classes}
    dropout_rate = getattr(model_cfg, "dropout_rate", 0.5)
    head_dropout = getattr(model_cfg, "head_dropout", None)
    head_hidden_dim = getattr(model_cfg, "head_hidden_dim", None)
    feature_channels = getattr(model_cfg, "feature_channels", None)
    frozen_prefixes = getattr(model_cfg, "frozen_prefixes", None)

    if model_name == "cnn":
        cnn_kwargs = {
            **common_kwargs,
            "input_channels": getattr(model_cfg, "input_channels", 3),
            "dropout_rate": dropout_rate,
            "use_batch_norm": getattr(model_cfg, "use_batch_norm", True),
        }
        if feature_channels is not None:
            cnn_kwargs["feature_channels"] = feature_channels
        if model_cfg is not None and hasattr(model_cfg, "head_hidden_dim"):
            cnn_kwargs["head_hidden_dim"] = head_hidden_dim
        if model_cfg is not None and hasattr(model_cfg, "head_dropout") and head_dropout is not None:
            cnn_kwargs["head_dropout"] = head_dropout

        model = _call_with_supported_kwargs(WaferCNN, **cnn_kwargs).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "Custom CNN")

    if model_name == "resnet":
        model = _call_with_supported_kwargs(
            get_resnet18,
            **common_kwargs,
            pretrained=getattr(model_cfg, "pretrained", True),
            freeze_until=getattr(model_cfg, "freeze_until", "layer3"),
            frozen_prefixes=frozen_prefixes,
            head_dropout=head_dropout if head_dropout is not None else dropout_rate,
            head_hidden_dim=head_hidden_dim,
        ).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "ResNet-18")

    if model_name == "vit":
        model = _call_with_supported_kwargs(
            get_vit_small,
            **common_kwargs,
            image_size=96,
            in_channels=getattr(model_cfg, "input_channels", 3),
            dropout=dropout_rate,
        ).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "ViT-small")

    model = _call_with_supported_kwargs(
        get_efficientnet_b0,
        **common_kwargs,
        pretrained=getattr(model_cfg, "pretrained", True),
        freeze_until=getattr(model_cfg, "freeze_until", "features.6"),
        frozen_prefixes=frozen_prefixes,
        head_dropout=head_dropout if head_dropout is not None else dropout_rate,
        head_hidden_dim=head_hidden_dim,
    ).to(device)
    return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "EfficientNet-B0")


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
    nesterov: bool = True,
):
    """Create the configured optimizer."""
    name = optimizer_name.lower()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if name == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, scheduler_cfg, epochs: int, monitored_metric: str = "val_macro_f1"):
    """Create the configured scheduler, optionally with linear warmup.

    For ReduceLROnPlateau, the mode is auto-derived from the monitored metric:
    metrics containing 'loss' use 'min', all others use 'max'.  This prevents
    the scheduler from reducing LR when accuracy *increases*.

    When ``warmup_epochs > 0``, a :class:`~torch.optim.lr_scheduler.LinearLR`
    warmup phase is composed with the main scheduler via
    :class:`~torch.optim.lr_scheduler.SequentialLR`.  Warmup is only applied to
    StepLR and CosineAnnealingLR; ReduceLROnPlateau is adaptive by nature and
    skips warmup (a log message is emitted).
    """
    warmup_epochs = getattr(scheduler_cfg, "warmup_epochs", 0)
    warmup_start_factor = getattr(scheduler_cfg, "warmup_start_factor", 0.1)

    scheduler_type = scheduler_cfg.type.lower()
    if scheduler_type in {"none", "off", "disabled"}:
        return None

    if scheduler_type == "reducelronplateau":
        if warmup_epochs > 0:
            logger.info(
                "warmup_epochs=%d ignored for ReduceLROnPlateau (adaptive scheduler)",
                warmup_epochs,
            )
        mode = "min" if "loss" in monitored_metric.lower() else "max"
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=scheduler_cfg.factor,
            patience=scheduler_cfg.patience,
            min_lr=scheduler_cfg.min_lr,
        )

    # Build the main (non-Plateau) scheduler
    main_sched: optim.lr_scheduler.LRScheduler
    if scheduler_type == "steplr":
        main_sched = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, getattr(scheduler_cfg, "step_size", scheduler_cfg.patience)),
            gamma=scheduler_cfg.factor,
        )
    elif scheduler_type == "cosineannealinglr":
        main_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, getattr(scheduler_cfg, "t_max", None) or epochs),
            eta_min=getattr(scheduler_cfg, "min_lr", 0.0),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_cfg.type}")

    if warmup_epochs <= 0:
        return main_sched

    # Compose warmup + main via SequentialLR
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs,
    )
    logger.info(
        "LR warmup enabled: %d epochs, start_factor=%.3f",
        warmup_epochs,
        warmup_start_factor,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, main_sched],
        milestones=[warmup_epochs],
    )

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train wafer defect detection models')
    parser.add_argument('--model', choices=['cnn', 'resnet', 'efficientnet', 'effnet', 'vit', 'all'], default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-path', type=Path, default=None)
    parser.add_argument('--synthetic', action='store_true', help='Balance rare classes with synthetic augmentation')
    parser.add_argument('--distributed', action='store_true', help='Enable DataParallel multi-GPU')
    parser.add_argument('--uncertainty', action='store_true', help='Run MC Dropout uncertainty estimation after training')
    parser.add_argument('--pretrained-checkpoint', type=Path, default=None, help='Load pretrained backbone (e.g. from SimCLR)')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow logging')
    parser.add_argument(
        '--config',
        type=Path,
        action='append',
        default=None,
        help='Configuration file path. Repeat to merge overlays in order.',
    )
    args = parser.parse_args()

    # Load config.yaml defaults if available
    config = None
    config_paths = args.config
    if config_paths:
        if len(config_paths) == 1:
            config = load_config(str(config_paths[0]))
        else:
            config = load_merged_config(config_paths)
        logger.info(
            "Loaded defaults from %s",
            ", ".join(str(path) for path in config_paths),
        )
    elif Path('config.yaml').exists():
        config = load_config('config.yaml')
        logger.info("Loaded defaults from config.yaml")
    else:
        logger.info("No config.yaml found, using hardcoded defaults")

    # Resolve parameters: CLI > config.yaml > hardcoded defaults
    hw_default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_choice = args.model if args.model is not None else (
        config.training.default_model if config else 'cnn')
    model_choice = canonicalize_model_name(model_choice, allow_all=True)

    epochs = args.epochs if args.epochs is not None else (
        config.training.epochs if config else 25)
    batch_size = args.batch_size if args.batch_size is not None else (
        config.training.batch_size if config else 64)
    device = args.device if args.device is not None else (
        config.device if config else hw_default_device)
    seed = args.seed if args.seed is not None else (
        config.seed if config else SEED)
    data_path = args.data_path if args.data_path is not None else (
        Path(config.data.dataset_path) if config else Path('data/LSWMD_new.pkl'))

    # Store resolved values back into args for downstream use
    args.model = model_choice
    args.epochs = epochs
    args.batch_size = batch_size
    args.device = device
    args.seed = seed
    args.data_path = data_path

    set_seed(args.seed)
    device = args.device
    logger.info(f"Device: {device}")
    logger.info(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"model={args.model}, seed={args.seed}")

    # Load data
    use_synthetic = args.synthetic or bool(config and config.data.augmentation.synthetic.enabled)
    data = load_and_preprocess_data(
        args.data_path,
        train_size=(config.data.train_size if config else 0.70),
        val_size=(config.data.val_size if config else 0.15),
        test_size=(config.data.test_size if config else 0.15),
        seed=args.seed,
        synthetic=use_synthetic,
    )
    train_maps = data['train_maps']
    y_train = data['y_train']
    val_maps = data['val_maps']
    y_val = data['y_val']
    test_maps = data['test_maps']
    y_test = data['y_test']
    loss_weights = data['loss_weights'].to(device)
    class_names = data['class_names']

    training_cfg = config.training if config else None
    data_cfg = config.data if config else None

    loss_cfg = training_cfg.loss if training_cfg else None
    loss_name = loss_cfg.type if loss_cfg else "CrossEntropyLoss"
    if training_cfg and training_cfg.use_focal_loss:
        loss_name = "FocalLoss"

    configured_class_weights = None
    if loss_cfg and loss_cfg.class_weights:
        if len(loss_cfg.class_weights) != len(class_names):
            raise ValueError(
                f"Expected {len(class_names)} class weights, got {len(loss_cfg.class_weights)}"
            )
        configured_class_weights = torch.tensor(
            loss_cfg.class_weights,
            dtype=torch.float32,
            device=device,
        )

    criterion_weights = None
    if configured_class_weights is not None:
        criterion_weights = configured_class_weights
    elif loss_cfg is None or loss_cfg.weighted:
        criterion_weights = configured_class_weights if configured_class_weights is not None else loss_weights

    criterion = build_classification_loss(
        loss_name,
        class_weights=criterion_weights,
        label_smoothing=(loss_cfg.label_smoothing if loss_cfg else 0.0),
        focal_gamma=(loss_cfg.focal_gamma if loss_cfg else 2.0),
        reduction=(loss_cfg.reduction if loss_cfg else "mean"),
    )
    logger.info("Using loss function: %s", loss_name)

    # Create transforms
    try:
        import torchvision.transforms as tv_transforms
    except ImportError:
        tv_transforms = None

    train_aug = get_image_transforms(augment=(data_cfg.augmentation.enabled if data_cfg else True))
    imagenet_norm = get_imagenet_normalize()

    # Compose augmentation + ImageNet norm for pretrained training
    if tv_transforms is not None:
        pretrained_train_transform = tv_transforms.Compose([
            *train_aug.transforms,
            imagenet_norm,
        ])
        pretrained_val_transform = tv_transforms.Compose([imagenet_norm])
    else:
        pretrained_train_transform = imagenet_norm
        pretrained_val_transform = imagenet_norm

    # Output directories
    ckpt_dir = Path(config.checkpoint_dir if config else 'checkpoints')
    results_dir = Path(config.paths.result_dir if config else 'results')
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    models_to_train = ['cnn', 'resnet', 'efficientnet', 'vit'] if args.model == 'all' else [args.model]
    results = {}

    # Resolve learning rate: CLI --lr > config.yaml per-model > hardcoded default
    def _resolve_lr(model_key: str, hardcoded: float) -> float:
        if args.lr is not None:
            return args.lr
        if config and isinstance(config.training.learning_rate, dict):
            return config.training.learning_rate.get(model_key, hardcoded)
        if config and isinstance(config.training.learning_rate, (int, float)):
            return float(config.training.learning_rate)
        return hardcoded

    # Initialize loggers
    wb_logger = None
    wandb_enabled = args.wandb or bool(config and config.mlops.wandb.enabled)
    if wandb_enabled:
        wb_logger = WandBLogger(
            name=f"wafer-train-{int(time.time())}",
            project=config.mlops.wandb.project if config else "wafer-defect-detection",
            entity=config.mlops.wandb.entity if config else None,
            config=vars(args),
        )
    
    mf_logger = None
    mlflow_enabled = args.mlflow or bool(config and config.mlops.mlflow.enabled)
    if mlflow_enabled:
        mf_logger = MLFlowLogger(
            experiment_name=config.mlops.mlflow.experiment_name if config else "wafer-defect-detection",
            tracking_uri=config.mlops.mlflow.tracking_uri if config else "http://localhost:5000",
        )
        mf_logger.log_params(vars(args))

    for model_name in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info(f"{'='*70}")

        model_cfg = getattr(config.models, model_name) if config else None
        model, display_name = build_model(model_name, model_cfg, len(class_names), device)

        # Load pretrained backbone if provided (e.g. from SimCLR)
        if args.pretrained_checkpoint:
            state = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            logger.info(f"  Loaded pretrained weights from {args.pretrained_checkpoint}")

        # Wrap with DataParallel if distributed
        if args.distributed and torch.cuda.device_count() > 1:
            from src.training.distributed import wrap_model_dataparallel
            model = wrap_model_dataparallel(model)
            logger.info(f"  Wrapped model with DataParallel ({torch.cuda.device_count()} GPUs)")

        default_lr = {"cnn": 1e-3, "vit": 3e-4}.get(model_name, 1e-4)
        lr = _resolve_lr(model_name, default_lr)
        if model_name in ('cnn', 'vit'):
            transforms_train = train_aug
            transforms_val = None
        else:
            transforms_train = pretrained_train_transform
            transforms_val = pretrained_val_transform

        # Create loaders
        train_dataset = WaferMapDataset(train_maps, y_train, transform=transforms_train)
        val_dataset = WaferMapDataset(val_maps, y_val, transform=transforms_val)
        test_dataset = WaferMapDataset(test_maps, y_test, transform=transforms_val)

        g = torch.Generator().manual_seed(args.seed)
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": data_cfg.num_workers if data_cfg else 0,
            "pin_memory": bool(data_cfg.pin_memory if data_cfg else False) and str(device).startswith("cuda"),
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        logger.info(f"Model: {display_name}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Batch size: {args.batch_size}")

        # Train
        optimizer = build_optimizer(
            model,
            optimizer_name=(training_cfg.optimizer if training_cfg else "adam"),
            learning_rate=lr,
            weight_decay=(training_cfg.weight_decay if training_cfg else 1e-4),
            momentum=(training_cfg.momentum if training_cfg else 0.9),
            nesterov=(training_cfg.nesterov if training_cfg else True),
        )
        monitored_metric = (training_cfg.checkpointing.metric if training_cfg else "val_macro_f1")
        scheduler = build_scheduler(
            optimizer,
            scheduler_cfg=(training_cfg.scheduler if training_cfg else DEFAULT_SCHEDULER_CONFIG),
            epochs=args.epochs,
            monitored_metric=monitored_metric,
        ) if training_cfg else None
        t0 = time.time()
        model, epoch_history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            model_name=display_name,
            device=device,
            gradient_clip=(training_cfg.gradient_clip_max_norm if training_cfg else None),
            mixed_precision=(training_cfg.mixed_precision if training_cfg else False),
            early_stopping_enabled=(training_cfg.early_stopping.enabled if training_cfg else False),
            early_stopping_patience=(training_cfg.early_stopping.patience if training_cfg else None),
            early_stopping_min_delta=(training_cfg.early_stopping.min_delta if training_cfg else 0.0),
            monitored_metric=monitored_metric,
        )
        train_time = time.time() - t0

        # Save checkpoint with integrity hash
        ckpt_path = ckpt_dir / f'best_{model_name}.pth'
        file_hash = save_checkpoint_with_hash(model.state_dict(), ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path} (SHA-256: {file_hash[:16]}...)")

        # Evaluate
        logger.info(f"\nEvaluating on test set...")
        preds, labels_arr, metrics = evaluate_model(model, test_loader, class_names, display_name, device)

        # Log to W&B / MLflow
        if wb_logger:
            wb_logger.log_metrics(metrics)
            wb_logger.log_confusion_matrix(labels_arr, preds, class_names)
        if mf_logger:
            mf_logger.log_metrics(metrics)

        # Per-class metrics
        prec, rec, f1_per, support = precision_recall_fscore_support(
            labels_arr, preds, average=None, zero_division=0
        )
        per_class = {}
        for i, cls in enumerate(class_names):
            per_class[cls] = {
                'precision': float(prec[i]),
                'recall': float(rec[i]),
                'f1': float(f1_per[i]),
                'support': int(support[i]),
            }

        results[model_name] = {
            'display_name': display_name,
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1']),
            'ece': float(metrics.get('ece', 0.0)),
            'negative_log_likelihood': float(metrics.get('negative_log_likelihood', 0.0)),
            'brier_score': float(metrics.get('brier_score', 0.0)),
            'time_sec': float(train_time),
            'per_class': per_class,
            'epoch_history': epoch_history,
        }

        logger.info(f"  Accuracy    : {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1    : {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
        if 'ece' in metrics:
            logger.info(f"  ECE         : {metrics['ece']:.4f}")
        logger.info(f"  Time        : {train_time:.1f}s")

        # Optional: MC Dropout uncertainty estimation
        if args.uncertainty:
            try:
                from src.inference.uncertainty import UncertaintyEstimator
                unc_cfg = config.uncertainty if config else None
                n_samples = getattr(unc_cfg, 'n_samples', 10) if unc_cfg else 10
                unc_est = UncertaintyEstimator(model, num_iterations=n_samples, device=device)
                unc_results = unc_est.estimate_dataset_uncertainty(test_loader, return_predictions=True)
                mean_unc = float(unc_results['uncertainty'].mean())
                mean_ent = float(unc_results['entropy'].mean())
                results[model_name]['uncertainty'] = {
                    'mean_uncertainty': mean_unc,
                    'mean_entropy': mean_ent,
                }
                logger.info(f"  MC Dropout uncertainty: {mean_unc:.4f}, entropy: {mean_ent:.4f}")
            except Exception as e:
                logger.warning(f"  Uncertainty estimation failed: {e}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"\n{'Model':<18} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Time (s)':<10}")
    logger.info("-" * 70)
    for model_name in models_to_train:
        r = results[model_name]
        logger.info(f"{model_name:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f} {r['time_sec']:<10.1f}")

    # Save metrics JSON
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")

    if wb_logger:
        wb_logger.finish()
    if mf_logger:
        mf_logger.finish()

    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(main())
