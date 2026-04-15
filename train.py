#!/usr/bin/env python3
"""
CLI entry point for training wafer defect models.

Usage:
    python train.py --model cnn --epochs 5 --device cuda
    python train.py --model all --epochs 5 --device cpu

References:
    [10] Kim et al. (2020). "DL Based Wafer Map Defect Pattern Classification". DOI:10.1109/ACCESS.2020.3040684
    [17] Buda et al. (2018). "Class Imbalance Problem in CNNs". arXiv:1710.05381
    [55] (2021). "Smart Semiconductor Manufacturing Overview"
    [74] Kornblith et al. (2019). "Do Better ImageNet Models Transfer Better?". arXiv:1805.08974
    [93] Dai et al. (2017). "Deformable Convolutional Networks". arXiv:1703.06211
    [95] Caruana (1997). "Multitask Learning". Machine Learning 28(1)
    [100] Ruder (2017). "Overview of Multi-Task Learning". arXiv:1706.05098
    [112] Finn et al. (2017). "MAML: Model-Agnostic Meta-Learning". arXiv:1703.03400
    [113] Snell et al. (2017). "Prototypical Networks". arXiv:1703.05175
    [117] Zhu et al. (2005). "Semi-Supervised Learning Literature Survey"
    [120] Lee (2013). "Pseudo-Label". arXiv:1908.02983
    [133] Loshchilov & Hutter (2017). "SGDR: Warm Restarts". arXiv:1608.03983
    [135] Goyal et al. (2017). "Large Minibatch SGD". arXiv:1706.02677
    [137] Zhang et al. (2019). "Lookahead Optimizer". arXiv:1907.08610
    [138] Liu et al. (2020). "RAdam". arXiv:1908.03265
    [139] Foret et al. (2021). "SAM: Sharpness-Aware Minimization". arXiv:2010.01412
    [140] You et al. (2020). "LAMB: Large Batch Optimization". arXiv:1904.00962
    [145] Parisi et al. (2019). "Continual Lifelong Learning". arXiv:1802.07569
    [146] Li & Hoiem (2017). "Learning without Forgetting". arXiv:1606.09282
    [150] Lu et al. (2018). "Concept Drift Adaptation". arXiv:1810.02822
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
from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset, get_image_transforms, get_imagenet_normalize, seed_worker, MixupCutmix, ClassBalancedSampler
from src.models import WaferCNN, WaferCNNFPN, get_resnet18, get_efficientnet_b0
from src.models.vit import get_vit_small
from src.models.swin import get_swin_tiny
from src.analysis import evaluate_model, calibrate_and_evaluate
from src.mlops import MLFlowLogger, WandBLogger
from src.training.losses import build_classification_loss


from src.data.dataset import KNOWN_CLASSES
SEED = 42


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    By default we enable ``torch.backends.cudnn.benchmark`` (autotuner picks
    the fastest conv algorithm per input shape) and leave determinism off —
    this gives ~8-15% speedup on Ampere+ GPUs like Colab Pro's A100. Pass
    ``deterministic=True`` to trade that for bit-for-bit reproducibility
    between runs on the same hardware (needed for CI metric gating).
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def load_and_preprocess_data(
    dataset_path,
    train_size=0.70,
    test_size=0.15,
    val_size=0.15,
    seed=SEED,
    synthetic=False,
    target_size=(96, 96),
):
    """Load and split data."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    logger.info(f"\n{'='*70}")
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info(f"{'='*70}")

    # Fast path: pre-resized cache from scripts/precompute_tensors.py.
    # We intentionally do NOT auto-build the cache from inside train.py
    # anymore. The cache-build spawns a multiprocessing Pool, and on
    # Colab T4 (13.6 GB) the combined footprint of (raw df) + (pool fork
    # copies) + (imminent training setup) exceeds RAM and the kernel
    # gets SIGKILLed (exit -9). Build the cache separately ahead of time:
    #     python scripts/precompute_tensors.py
    # or run the notebook's Cell 5b. The Colab quickstart Cell 6 also
    # runs a preflight that builds the cache as its own subprocess if
    # missing — that process exits cleanly and releases RAM before
    # train.py starts.
    cache_path = Path(dataset_path).parent / "LSWMD_cache.npz"
    maps_npy_path = cache_path.with_suffix(".maps.npy")
    if cache_path.exists():
        logger.info(f"Using pre-resized cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        cached_labels_str = cache["labels"]

        # Two cache layouts supported:
        #   1. Current: sidecar .npz (this file) + .maps.npy (memmap). RAM-lean.
        #   2. Legacy:  single .npz whose "maps" key holds the full array.
        #      Still readable but loads the whole 3.2 GB array into RAM.
        if maps_npy_path.exists():
            logger.info(f"Memory-mapping {maps_npy_path} (mmap_mode='r')")
            cached_maps = np.load(maps_npy_path, mmap_mode="r")
        elif "maps" in cache.files:
            logger.info("Legacy cache layout: loading full maps array into RAM")
            cached_maps = cache["maps"]
        else:
            raise RuntimeError(
                f"{cache_path} has no 'maps' key and {maps_npy_path} is missing. "
                "Regenerate the cache with scripts/precompute_tensors.py."
            )

        le = LabelEncoder().fit(np.array(KNOWN_CLASSES))
        labels = le.transform(cached_labels_str)
        # Hand wafer_maps as an object array so the existing downstream code
        # path (index + object-array assignment) works unchanged. The per-item
        # shape is already target_size, so WaferMapDataset takes its fast path.
        # With the memmap layout, each wafer_maps[i] is a memmap slice — the
        # backing data only pages into RAM when the DataLoader actually reads
        # it per-batch, which keeps training-time RSS bounded.
        wafer_maps = np.empty(len(cached_maps), dtype=object)
        for i in range(len(cached_maps)):
            wafer_maps[i] = cached_maps[i]
    else:
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

    # WM-811K wafer maps have heterogeneous shapes; keep as object arrays so the
    # WaferMapDataset's lazy per-item resize handles them. Collapsing into a dense
    # np.array() fails with "inhomogeneous shape" for the raw dataset.
    train_maps = np.empty(len(X_train), dtype=object)
    train_maps[:] = [wafer_maps[i] for i in X_train]
    val_maps = np.empty(len(X_val), dtype=object)
    val_maps[:] = [wafer_maps[i] for i in X_val]
    test_maps = np.empty(len(X_test), dtype=object)
    test_maps[:] = [wafer_maps[i] for i in X_test]

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

    if model_name == "cnn_fpn":
        fpn_kwargs = {
            **common_kwargs,
            "input_channels": getattr(model_cfg, "input_channels", 3),
            "dropout_rate": dropout_rate,
            "use_batch_norm": getattr(model_cfg, "use_batch_norm", True),
            "fpn_out_channels": getattr(model_cfg, "fpn_out_channels", 128) or 128,
        }
        if feature_channels is not None:
            fpn_kwargs["feature_channels"] = feature_channels
        model = _call_with_supported_kwargs(WaferCNNFPN, **fpn_kwargs).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "Custom CNN-FPN")

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

    if model_name == "swin":
        model = _call_with_supported_kwargs(
            get_swin_tiny,
            **common_kwargs,
        ).to(device)
        return _apply_attention(model, model_cfg), getattr(model_cfg, "name", "Swin-Tiny")

    if model_name == "ride":
        from src.models.ride import build_ride_model
        ride_backbone = getattr(model_cfg, "backbone", "cnn") if model_cfg else "cnn"
        ride_num_experts = getattr(model_cfg, "num_experts", 3) if model_cfg else 3
        ride_reduction = getattr(model_cfg, "reduction", 4) if model_cfg else 4
        model = build_ride_model(
            backbone_name=ride_backbone,
            num_classes=num_classes,
            num_experts=ride_num_experts,
            reduction=ride_reduction,
            device=device,
        )
        return model, getattr(model_cfg, "name", "RIDE")

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

class TrainingPipeline:
    """Orchestrates the full training pipeline with clear phase methods."""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.data = None
        self.criterion = None
        self.train_aug = None
        self.pretrained_train_transform = None
        self.pretrained_val_transform = None
        self.training_cfg = None
        self.data_cfg = None
        self.ckpt_dir = None
        self.results_dir = None
        self.wb_logger = None
        self.mf_logger = None
        self.results = {}

    def configure(self) -> None:
        """Resolve CLI args vs config.yaml vs hardcoded defaults."""
        args = self.args
        config = self.config

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

    def prepare_data(self) -> None:
        """Call load_and_preprocess_data(), create transforms, set up loss function."""
        args = self.args
        config = self.config
        device = args.device

        use_synthetic = args.synthetic or bool(config and config.data.augmentation.synthetic.enabled)
        self.data = load_and_preprocess_data(
            args.data_path,
            train_size=(config.data.train_size if config else 0.70),
            val_size=(config.data.val_size if config else 0.15),
            test_size=(config.data.test_size if config else 0.15),
            seed=args.seed,
            synthetic=use_synthetic,
        )
        loss_weights = self.data['loss_weights'].to(device)
        class_names = self.data['class_names']

        self.training_cfg = config.training if config else None
        self.data_cfg = config.data if config else None

        loss_cfg = self.training_cfg.loss if self.training_cfg else None
        loss_name = loss_cfg.type if loss_cfg else "CrossEntropyLoss"
        if self.training_cfg and self.training_cfg.use_focal_loss:
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

        self.criterion = build_classification_loss(
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

        self.train_aug = get_image_transforms(augment=(self.data_cfg.augmentation.enabled if self.data_cfg else True))
        imagenet_norm = get_imagenet_normalize()

        # Compose augmentation + ImageNet norm for pretrained training
        if tv_transforms is not None:
            self.pretrained_train_transform = tv_transforms.Compose([
                *self.train_aug.transforms,
                imagenet_norm,
            ])
            self.pretrained_val_transform = tv_transforms.Compose([imagenet_norm])
        else:
            self.pretrained_train_transform = imagenet_norm
            self.pretrained_val_transform = imagenet_norm

        # Output directories
        self.ckpt_dir = Path(config.checkpoint_dir if config else 'checkpoints')
        self.results_dir = Path(config.paths.result_dir if config else 'results')
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def _resolve_lr(self, model_key: str, hardcoded: float) -> float:
        """Resolve learning rate: CLI --lr > config.yaml per-model > hardcoded default."""
        if self.args.lr is not None:
            return self.args.lr
        if self.config and isinstance(self.config.training.learning_rate, dict):
            return self.config.training.learning_rate.get(model_key, hardcoded)
        if self.config and isinstance(self.config.training.learning_rate, (int, float)):
            return float(self.config.training.learning_rate)
        return hardcoded

    def build_model(self, model_name: str) -> tuple[nn.Module, str]:
        """Construct model, optimizer, scheduler for a given model name.

        Also loads pretrained checkpoint and wraps with DataParallel if configured.
        Returns (model, display_name).
        """
        args = self.args
        config = self.config
        device = args.device
        class_names = self.data['class_names']

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

        return model, display_name

    def train_and_evaluate(self, model_name: str) -> dict:
        """Train one model, evaluate on test set, return results dict."""
        args = self.args
        config = self.config
        device = args.device
        training_cfg = self.training_cfg
        data_cfg = self.data_cfg
        class_names = self.data['class_names']

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info(f"{'='*70}")

        model, display_name = self.build_model(model_name)

        default_lr = {"cnn": 1e-3, "cnn_fpn": 1e-3, "vit": 3e-4}.get(model_name, 1e-4)
        lr = self._resolve_lr(model_name, default_lr)
        if model_name in ('cnn', 'cnn_fpn', 'vit', 'ride'):
            transforms_train = self.train_aug
            transforms_val = None
        else:
            transforms_train = self.pretrained_train_transform
            transforms_val = self.pretrained_val_transform

        # Create loaders
        train_maps = self.data['train_maps']
        y_train = self.data['y_train']
        val_maps = self.data['val_maps']
        y_val = self.data['y_val']
        test_maps = self.data['test_maps']
        y_test = self.data['y_test']

        train_dataset = WaferMapDataset(train_maps, y_train, transform=transforms_train)
        val_dataset = WaferMapDataset(val_maps, y_val, transform=transforms_val)
        test_dataset = WaferMapDataset(test_maps, y_test, transform=transforms_val)

        g = torch.Generator().manual_seed(args.seed)
        num_workers = data_cfg.num_workers if data_cfg else 0
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": num_workers,
            "pin_memory": bool(data_cfg.pin_memory if data_cfg else False) and str(device).startswith("cuda"),
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        # persistent_workers + prefetch_factor require num_workers > 0. They
        # cut per-epoch worker-respawn overhead (~10-20% on Colab Pro) so the
        # GPU doesn't stall between epochs waiting for DataLoader warmup.
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        # Class-balanced sampling: CLI flag or config
        use_balanced = args.balanced_sampling or bool(
            training_cfg and training_cfg.balanced_sampling
        )
        if use_balanced:
            sampler = ClassBalancedSampler(y_train)
            logger.info("Using ClassBalancedSampler (oversampling minority classes)")
            train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
        else:
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
        # Mixup / CutMix batch augmentation: CLI flag or config
        use_mixup = args.mixup or bool(
            training_cfg and training_cfg.mixup.enabled
        )
        batch_transform = None
        if use_mixup:
            mixup_cfg = training_cfg.mixup if training_cfg else None
            batch_transform = MixupCutmix(
                mixup_alpha=(mixup_cfg.mixup_alpha if mixup_cfg else 0.2),
                cutmix_alpha=(mixup_cfg.cutmix_alpha if mixup_cfg else 1.0),
                mixup_prob=(mixup_cfg.mixup_prob if mixup_cfg else 0.5),
                cutmix_prob=(mixup_cfg.cutmix_prob if mixup_cfg else 0.5),
                num_classes=len(class_names),
            )
            logger.info("Using MixupCutmix batch augmentation: %s", batch_transform)

        t0 = time.time()
        model, epoch_history = train_model(
            model,
            train_loader,
            val_loader,
            self.criterion,
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
            batch_transform=batch_transform,
        )
        train_time = time.time() - t0

        # Save checkpoint with integrity hash
        ckpt_path = self.ckpt_dir / f'best_{model_name}.pth'
        file_hash = save_checkpoint_with_hash(model.state_dict(), ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path} (SHA-256: {file_hash[:16]}...)")

        # Evaluate
        logger.info(f"\nEvaluating on test set...")
        preds, labels_arr, metrics = evaluate_model(model, test_loader, class_names, display_name, device)

        # Log to W&B / MLflow
        if self.wb_logger:
            self.wb_logger.log_metrics(metrics)
            self.wb_logger.log_confusion_matrix(labels_arr, preds, class_names)
        if self.mf_logger:
            self.mf_logger.log_metrics(metrics)

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

        result = {
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

        # Optional: calibrated evaluation with temperature scaling
        try:
            from src.analysis.evaluate import calibrate_and_evaluate
            cal_preds, cal_labels, cal_metrics, temperature = calibrate_and_evaluate(
                model, val_loader, test_loader, class_names, display_name, device
            )
            result['calibrated_metrics'] = {
                'temperature': temperature,
                'ece': cal_metrics.get('ece', 0.0),
                'brier_score': cal_metrics.get('brier_score', 0.0),
            }
            logger.info(f"  Calibrated ECE: {cal_metrics.get('ece', 0.0):.4f} (T={temperature:.3f})")
        except Exception as e:
            logger.warning(f"  Calibrated evaluation failed: {e}")

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
                result['uncertainty'] = {
                    'mean_uncertainty': mean_unc,
                    'mean_entropy': mean_ent,
                }
                logger.info(f"  MC Dropout uncertainty: {mean_unc:.4f}, entropy: {mean_ent:.4f}")
            except Exception as e:
                logger.warning(f"  Uncertainty estimation failed: {e}")

        return result

    def _run_ensemble_evaluation(self, models_to_train: list) -> None:
        """Load trained model checkpoints and evaluate learned-weight and stacking ensembles."""
        from src.models.ensemble import LearnedWeightEnsemble, StackingEnsemble

        args = self.args
        device = args.device
        class_names = self.data['class_names']
        data_cfg = self.data_cfg

        logger.info(f"\n{'='*70}")
        logger.info("ENSEMBLE EVALUATION (learned weights + stacking)")
        logger.info(f"{'='*70}")

        # Reload each model from its saved checkpoint
        loaded_models: list[nn.Module] = []
        for model_name in models_to_train:
            ckpt_path = self.ckpt_dir / f'best_{model_name}.pth'
            if not ckpt_path.exists():
                logger.warning("Checkpoint %s not found, skipping ensemble", ckpt_path)
                return
            model, _ = self.build_model(model_name)
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            loaded_models.append(model)

        # Build val/test loaders using the right transform per model
        # For ensemble we need a single loader; use no-augmentation transforms.
        # Pretrained models need ImageNet norm; CNN/ViT don't.
        # Strategy: use pretrained_val_transform (ImageNet norm) if any pretrained
        # model is present; otherwise None. This matches the evaluation-time transform.
        has_pretrained = any(m in ('resnet', 'efficientnet') for m in models_to_train)
        val_transform = self.pretrained_val_transform if has_pretrained else None
        test_transform = self.pretrained_val_transform if has_pretrained else None

        val_dataset = WaferMapDataset(self.data['val_maps'], self.data['y_val'], transform=val_transform)
        test_dataset = WaferMapDataset(self.data['test_maps'], self.data['y_test'], transform=test_transform)

        g = torch.Generator().manual_seed(args.seed)
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": data_cfg.num_workers if data_cfg else 0,
            "pin_memory": bool(data_cfg.pin_memory if data_cfg else False) and str(device).startswith("cuda"),
            "worker_init_fn": seed_worker,
            "generator": g,
            "shuffle": False,
        }
        val_loader = DataLoader(val_dataset, **loader_kwargs)
        test_loader = DataLoader(test_dataset, **loader_kwargs)

        # --- LearnedWeightEnsemble ---
        try:
            t0 = time.time()
            lwe = LearnedWeightEnsemble(loaded_models, device=device)
            lwe.optimize_weights(val_loader, class_names)
            lwe_metrics = lwe.evaluate(test_loader, class_names)
            lwe_time = time.time() - t0
            self.results['learned_ensemble'] = {
                'display_name': 'Learned-Weight Ensemble',
                'accuracy': lwe_metrics['accuracy'],
                'macro_f1': lwe_metrics['macro_f1'],
                'weighted_f1': lwe_metrics['weighted_f1'],
                'time_sec': lwe_time,
                'weights': lwe.weights.cpu().tolist(),
            }
            logger.info("Learned-Weight Ensemble: Acc=%.4f, F1=%.4f (%.1fs)",
                        lwe_metrics['accuracy'], lwe_metrics['macro_f1'], lwe_time)
        except Exception as exc:
            logger.warning("LearnedWeightEnsemble failed: %s", exc)

        # --- StackingEnsemble ---
        try:
            t0 = time.time()
            stacker = StackingEnsemble(loaded_models, num_classes=len(class_names), device=device)
            stacker.fit(val_loader, epochs=50, lr=0.01)
            stk_metrics = stacker.evaluate(test_loader, class_names)
            stk_time = time.time() - t0
            self.results['stacking_ensemble'] = {
                'display_name': 'Stacking Ensemble',
                'accuracy': stk_metrics['accuracy'],
                'macro_f1': stk_metrics['macro_f1'],
                'weighted_f1': stk_metrics['weighted_f1'],
                'time_sec': stk_time,
            }
            logger.info("Stacking Ensemble: Acc=%.4f, F1=%.4f (%.1fs)",
                        stk_metrics['accuracy'], stk_metrics['macro_f1'], stk_time)
        except Exception as exc:
            logger.warning("StackingEnsemble failed: %s", exc)

    def run(self) -> int:
        """Orchestrate the full pipeline: configure, prepare_data, loop over models, print summary, save metrics JSON."""
        self.configure()
        self.prepare_data()

        args = self.args
        config = self.config

        models_to_train = ['cnn', 'cnn_fpn', 'resnet', 'efficientnet', 'vit', 'swin', 'ride'] if args.model == 'all' else [args.model]

        # Initialize loggers
        wandb_enabled = args.wandb or bool(config and config.mlops.wandb.enabled)
        if wandb_enabled:
            self.wb_logger = WandBLogger(
                name=f"wafer-train-{int(time.time())}",
                project=config.mlops.wandb.project if config else "wafer-defect-detection",
                entity=config.mlops.wandb.entity if config else None,
                config=vars(args),
            )

        mlflow_enabled = args.mlflow or bool(config and config.mlops.mlflow.enabled)
        if mlflow_enabled:
            self.mf_logger = MLFlowLogger(
                experiment_name=config.mlops.mlflow.experiment_name if config else "wafer-defect-detection",
                tracking_uri=config.mlops.mlflow.tracking_uri if config else "http://localhost:5000",
            )
            self.mf_logger.log_params(vars(args))

        for model_name in models_to_train:
            self.results[model_name] = self.train_and_evaluate(model_name)

        # Learned-weight and stacking ensemble when 2+ models trained
        if len(models_to_train) >= 2:
            self._run_ensemble_evaluation(models_to_train)

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"\n{'Model':<18} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Time (s)':<10}")
        logger.info("-" * 70)
        for model_name in models_to_train:
            r = self.results[model_name]
            logger.info(f"{model_name:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f} {r['time_sec']:<10.1f}")
        # Print ensemble rows if present
        for ens_key in ("learned_ensemble", "stacking_ensemble"):
            if ens_key in self.results:
                r = self.results[ens_key]
                logger.info(f"{ens_key:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f} {r.get('time_sec', 0.0):<10.1f}")

        # Save metrics JSON — merge with existing file so subprocess-per-model
        # invocations (notebook Cell 6) accumulate metrics across runs instead
        # of each subprocess clobbering the last model's results.
        metrics_path = self.results_dir / 'metrics.json'
        merged: dict = {}
        if metrics_path.exists():
            try:
                merged = json.loads(metrics_path.read_text(encoding='utf-8'))
                if not isinstance(merged, dict):
                    merged = {}
            except (json.JSONDecodeError, OSError):
                merged = {}
        merged.update(self.results)
        with open(metrics_path, 'w') as f:
            json.dump(merged, f, indent=2)
        logger.info(f"\nMetrics saved to {metrics_path} ({len(merged)} model(s) total)")

        if self.wb_logger:
            self.wb_logger.finish()
        if self.mf_logger:
            self.mf_logger.finish()

        return 0


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train wafer defect detection models')
    parser.add_argument('--model', choices=['cnn', 'cnn_fpn', 'resnet', 'efficientnet', 'effnet', 'vit', 'swin', 'ride', 'all'], default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-path', type=Path, default=None)
    parser.add_argument('--synthetic', action='store_true', help='Balance rare classes with synthetic augmentation')
    parser.add_argument('--mixup', action='store_true', help='Enable Mixup/CutMix batch augmentation')
    parser.add_argument('--balanced-sampling', action='store_true', help='Enable class-balanced batch sampling')
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

    pipeline = TrainingPipeline(args, config)
    return pipeline.run()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(main())
