"""TrainingPipeline orchestrator.

The pipeline resolves CLI args vs config, prepares data, builds models one at
a time, runs the inner training loop (``core_training_loop``), evaluates, and
optionally runs ensemble evaluation + MLflow/W&B logging.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.analysis import evaluate_model
from src.config import canonicalize_model_name
from src.data import (
    ClassBalancedSampler,
    MixupCutmix,
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    seed_worker,
)
from src.data.loaders import create_test_loader, create_train_loader, create_val_loader
from src.data.pipeline import SEED, load_and_preprocess_data
from src.mlops import MLFlowLogger, WandBLogger
from src.model_registry import save_checkpoint_with_hash
from src.training.builders import (
    DEFAULT_SCHEDULER_CONFIG,
    build_model,
    build_optimizer,
    build_scheduler,
)
from src.training.losses import build_classification_loss
from src.training.trainer import core_training_loop
from src.utils.reproducibility import compute_manifest

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    By default we enable ``torch.backends.cudnn.benchmark`` (autotuner picks
    the fastest conv algorithm per input shape) and leave determinism off —
    this gives ~8-15% speedup on Ampere+ GPUs like Colab Pro's A100. Pass
    ``deterministic=True`` to trade that for bit-for-bit reproducibility
    between runs on the same hardware (needed for CI metric gating).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


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
        self.results: Dict[str, Any] = {}

    def configure(self) -> None:
        """Resolve CLI args vs config.yaml vs hardcoded defaults."""
        args = self.args
        config = self.config

        hw_default_device = "cuda" if torch.cuda.is_available() else "cpu"

        model_choice = (
            args.model
            if args.model is not None
            else (config.training.default_model if config else "cnn")
        )
        model_choice = canonicalize_model_name(model_choice, allow_all=True)

        epochs = (
            args.epochs if args.epochs is not None else (config.training.epochs if config else 25)
        )
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else (config.training.batch_size if config else 64)
        )
        device = (
            args.device
            if args.device is not None
            else (config.device if config else hw_default_device)
        )
        seed = args.seed if args.seed is not None else (config.seed if config else SEED)
        data_path = (
            args.data_path
            if args.data_path is not None
            else (Path(config.data.dataset_path) if config else Path("data/LSWMD_new.pkl"))
        )

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
        logger.info(
            f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
            f"model={args.model}, seed={args.seed}"
        )

    def prepare_data(self) -> None:
        """Call load_and_preprocess_data(), create transforms, set up loss function."""
        args = self.args
        config = self.config
        device = args.device

        use_synthetic = args.synthetic or bool(
            config and config.data.augmentation.synthetic.enabled
        )
        self.data = load_and_preprocess_data(
            args.data_path,
            train_size=(config.data.train_size if config else 0.70),
            val_size=(config.data.val_size if config else 0.15),
            test_size=(config.data.test_size if config else 0.15),
            seed=args.seed,
            synthetic=use_synthetic,
        )
        loss_weights = self.data["loss_weights"].to(device)
        class_names = self.data["class_names"]

        self.training_cfg = config.training if config else None
        self.data_cfg = config.data if config else None

        loss_cfg = self.training_cfg.loss if self.training_cfg else None
        loss_name = loss_cfg.type if loss_cfg else "CrossEntropyLoss"
        if self.training_cfg and self.training_cfg.use_focal_loss:
            loss_name = "FocalLoss"
        if getattr(args, "cost_sensitive", False):
            loss_name = "CostSensitiveCE"

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
            criterion_weights = (
                configured_class_weights if configured_class_weights is not None else loss_weights
            )

        cost_matrix_tensor = None
        if loss_name == "CostSensitiveCE":
            from src.training.losses import build_cost_matrix_wm811k

            if loss_cfg is not None and loss_cfg.cost_matrix is not None:
                provided = torch.tensor(loss_cfg.cost_matrix, dtype=torch.float32, device=device)
                if provided.shape != (len(class_names), len(class_names)):
                    raise ValueError(
                        f"loss.cost_matrix must be {len(class_names)}x{len(class_names)}, "
                        f"got {tuple(provided.shape)}"
                    )
                cost_matrix_tensor = provided
            else:
                cost_matrix_tensor = build_cost_matrix_wm811k(
                    class_names,
                    near_full_missed=(loss_cfg.cost_near_full_missed if loss_cfg else 10.0),
                    rare_missed=(loss_cfg.cost_rare_missed if loss_cfg else 5.0),
                    edge_confusion=(loss_cfg.cost_edge_confusion if loss_cfg else 0.5),
                ).to(device)

        self.criterion = build_classification_loss(
            loss_name,
            class_weights=criterion_weights,
            label_smoothing=(loss_cfg.label_smoothing if loss_cfg else 0.0),
            focal_gamma=(loss_cfg.focal_gamma if loss_cfg else 2.0),
            reduction=(loss_cfg.reduction if loss_cfg else "mean"),
            cost_matrix=cost_matrix_tensor,
        )
        logger.info("Using loss function: %s", loss_name)

        # Create transforms
        try:
            import torchvision.transforms as tv_transforms
        except ImportError:
            tv_transforms = None

        self.train_aug = get_image_transforms(
            augment=(self.data_cfg.augmentation.enabled if self.data_cfg else True)
        )
        imagenet_norm = get_imagenet_normalize()

        # Compose augmentation + ImageNet norm for pretrained training
        if tv_transforms is not None:
            self.pretrained_train_transform = tv_transforms.Compose(
                [
                    *self.train_aug.transforms,
                    imagenet_norm,
                ]
            )
            self.pretrained_val_transform = tv_transforms.Compose([imagenet_norm])
        else:
            self.pretrained_train_transform = imagenet_norm
            self.pretrained_val_transform = imagenet_norm

        # Output directories
        self.ckpt_dir = Path(config.checkpoint_dir if config else "checkpoints")
        self.results_dir = Path(config.paths.result_dir if config else "results")
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

    def build_model(self, model_name: str) -> Tuple[nn.Module, str]:
        """Construct model, optimizer, scheduler for a given model name.

        Also loads pretrained checkpoint and wraps with DataParallel if configured.
        Returns (model, display_name).
        """
        args = self.args
        config = self.config
        device = args.device
        class_names = self.data["class_names"]

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
        class_names = self.data["class_names"]

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info(f"{'='*70}")

        model, display_name = self.build_model(model_name)

        default_lr = {"cnn": 1e-3, "cnn_fpn": 1e-3, "vit": 3e-4}.get(model_name, 1e-4)
        lr = self._resolve_lr(model_name, default_lr)
        if model_name in ("cnn", "cnn_fpn", "vit", "ride"):
            transforms_train = self.train_aug
            transforms_val = None
        else:
            transforms_train = self.pretrained_train_transform
            transforms_val = self.pretrained_val_transform

        # Create loaders via canonical factories
        train_maps = self.data["train_maps"]
        y_train = self.data["y_train"]
        val_maps = self.data["val_maps"]
        y_val = self.data["y_val"]
        test_maps = self.data["test_maps"]
        y_test = self.data["y_test"]

        g = torch.Generator().manual_seed(args.seed)

        # Class-balanced sampling: CLI flag or config
        use_balanced = args.balanced_sampling or bool(
            training_cfg and training_cfg.balanced_sampling
        )
        sampler = None
        if use_balanced:
            sampler = ClassBalancedSampler(y_train)
            logger.info("Using ClassBalancedSampler (oversampling minority classes)")

        train_loader = create_train_loader(
            train_maps,
            y_train,
            config=config,
            batch_size=args.batch_size,
            transform=transforms_train,
            sampler=sampler,
            generator=g,
            device=device,
        )
        val_loader = create_val_loader(
            val_maps,
            y_val,
            config=config,
            batch_size=args.batch_size,
            transform=transforms_val,
            generator=g,
            device=device,
        )
        test_loader = create_test_loader(
            test_maps,
            y_test,
            config=config,
            batch_size=args.batch_size,
            transform=transforms_val,
            generator=g,
            device=device,
        )

        train_dataset = train_loader.dataset

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
        monitored_metric = training_cfg.checkpointing.metric if training_cfg else "val_macro_f1"
        scheduler = (
            build_scheduler(
                optimizer,
                scheduler_cfg=(
                    training_cfg.scheduler if training_cfg else DEFAULT_SCHEDULER_CONFIG
                ),
                epochs=args.epochs,
                monitored_metric=monitored_metric,
            )
            if training_cfg
            else None
        )
        # Mixup / CutMix batch augmentation: CLI flag or config
        use_mixup = args.mixup or bool(training_cfg and training_cfg.mixup.enabled)
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
        model, epoch_history = core_training_loop(
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
            early_stopping_patience=(
                training_cfg.early_stopping.patience if training_cfg else None
            ),
            early_stopping_min_delta=(
                training_cfg.early_stopping.min_delta if training_cfg else 0.0
            ),
            monitored_metric=monitored_metric,
            batch_transform=batch_transform,
        )
        train_time = time.time() - t0

        # Save checkpoint with integrity hash
        ckpt_path = self.ckpt_dir / f"best_{model_name}.pth"
        file_hash = save_checkpoint_with_hash(model.state_dict(), ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path} (SHA-256: {file_hash[:16]}...)")

        # Evaluate
        logger.info(f"\nEvaluating on test set...")
        preds, labels_arr, metrics = evaluate_model(
            model, test_loader, class_names, display_name, device
        )

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
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1_per[i]),
                "support": int(support[i]),
            }

        result = {
            "display_name": display_name,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "ece": float(metrics.get("ece", 0.0)),
            "negative_log_likelihood": float(metrics.get("negative_log_likelihood", 0.0)),
            "brier_score": float(metrics.get("brier_score", 0.0)),
            "time_sec": float(train_time),
            "per_class": per_class,
            "epoch_history": epoch_history,
        }

        logger.info(f"  Accuracy    : {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1    : {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
        if "ece" in metrics:
            logger.info(f"  ECE         : {metrics['ece']:.4f}")
        logger.info(f"  Time        : {train_time:.1f}s")

        # Optional: calibrated evaluation with temperature scaling
        try:
            from src.analysis.evaluate import calibrate_and_evaluate

            cal_preds, cal_labels, cal_metrics, temperature = calibrate_and_evaluate(
                model, val_loader, test_loader, class_names, display_name, device
            )
            result["calibrated_metrics"] = {
                "temperature": temperature,
                "ece": cal_metrics.get("ece", 0.0),
                "brier_score": cal_metrics.get("brier_score", 0.0),
            }
            logger.info(
                f"  Calibrated ECE: {cal_metrics.get('ece', 0.0):.4f} (T={temperature:.3f})"
            )
        except Exception as e:
            logger.warning(f"  Calibrated evaluation failed: {e}")

        # Optional: MC Dropout uncertainty estimation
        if args.uncertainty:
            try:
                from src.inference.uncertainty import UncertaintyEstimator

                unc_cfg = config.uncertainty if config else None
                n_samples = getattr(unc_cfg, "n_samples", 10) if unc_cfg else 10
                unc_est = UncertaintyEstimator(model, num_iterations=n_samples, device=device)
                unc_results = unc_est.estimate_dataset_uncertainty(
                    test_loader, return_predictions=True
                )
                mean_unc = float(unc_results["uncertainty"].mean())
                mean_ent = float(unc_results["entropy"].mean())
                result["uncertainty"] = {
                    "mean_uncertainty": mean_unc,
                    "mean_entropy": mean_ent,
                }
                logger.info(f"  MC Dropout uncertainty: {mean_unc:.4f}, entropy: {mean_ent:.4f}")
            except Exception as e:
                logger.warning(f"  Uncertainty estimation failed: {e}")

        return result

    def _run_ensemble_evaluation(self, models_to_train: List[str]) -> None:
        """Load trained model checkpoints and evaluate learned-weight and stacking ensembles."""
        from src.models.ensemble import LearnedWeightEnsemble, StackingEnsemble

        args = self.args
        device = args.device
        class_names = self.data["class_names"]
        data_cfg = self.data_cfg

        logger.info(f"\n{'='*70}")
        logger.info("ENSEMBLE EVALUATION (learned weights + stacking)")
        logger.info(f"{'='*70}")

        # Reload each model from its saved checkpoint
        loaded_models: List[nn.Module] = []
        for model_name in models_to_train:
            ckpt_path = self.ckpt_dir / f"best_{model_name}.pth"
            if not ckpt_path.exists():
                logger.warning("Checkpoint %s not found, skipping ensemble", ckpt_path)
                return
            model, _ = self.build_model(model_name)
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            loaded_models.append(model)

        # For ensemble we use no-augmentation transforms. Pretrained models need
        # ImageNet norm; CNN/ViT don't. If any pretrained model is present, use
        # pretrained_val_transform; otherwise None.
        has_pretrained = any(m in ("resnet", "efficientnet") for m in models_to_train)
        val_transform = self.pretrained_val_transform if has_pretrained else None
        test_transform = self.pretrained_val_transform if has_pretrained else None

        g = torch.Generator().manual_seed(args.seed)
        val_loader = create_val_loader(
            self.data["val_maps"],
            self.data["y_val"],
            config=self.config,
            batch_size=args.batch_size,
            transform=val_transform,
            generator=g,
            device=device,
        )
        test_loader = create_test_loader(
            self.data["test_maps"],
            self.data["y_test"],
            config=self.config,
            batch_size=args.batch_size,
            transform=test_transform,
            generator=g,
            device=device,
        )

        # --- LearnedWeightEnsemble ---
        try:
            t0 = time.time()
            lwe = LearnedWeightEnsemble(loaded_models, device=device)
            lwe.optimize_weights(val_loader, class_names)
            lwe_metrics = lwe.evaluate(test_loader, class_names)
            lwe_time = time.time() - t0
            self.results["learned_ensemble"] = {
                "display_name": "Learned-Weight Ensemble",
                "accuracy": lwe_metrics["accuracy"],
                "macro_f1": lwe_metrics["macro_f1"],
                "weighted_f1": lwe_metrics["weighted_f1"],
                "time_sec": lwe_time,
                "weights": lwe.weights.cpu().tolist(),
            }
            logger.info(
                "Learned-Weight Ensemble: Acc=%.4f, F1=%.4f (%.1fs)",
                lwe_metrics["accuracy"],
                lwe_metrics["macro_f1"],
                lwe_time,
            )
        except Exception as exc:
            logger.warning("LearnedWeightEnsemble failed: %s", exc)

        # --- StackingEnsemble ---
        try:
            t0 = time.time()
            stacker = StackingEnsemble(loaded_models, num_classes=len(class_names), device=device)
            stacker.fit(val_loader, epochs=50, lr=0.01)
            stk_metrics = stacker.evaluate(test_loader, class_names)
            stk_time = time.time() - t0
            self.results["stacking_ensemble"] = {
                "display_name": "Stacking Ensemble",
                "accuracy": stk_metrics["accuracy"],
                "macro_f1": stk_metrics["macro_f1"],
                "weighted_f1": stk_metrics["weighted_f1"],
                "time_sec": stk_time,
            }
            logger.info(
                "Stacking Ensemble: Acc=%.4f, F1=%.4f (%.1fs)",
                stk_metrics["accuracy"],
                stk_metrics["macro_f1"],
                stk_time,
            )
        except Exception as exc:
            logger.warning("StackingEnsemble failed: %s", exc)

    def run(self) -> int:
        """Orchestrate the full pipeline: configure, prepare_data, loop over models,
        print summary, save metrics JSON."""
        self.configure()
        self.prepare_data()

        args = self.args
        config = self.config

        models_to_train = (
            ["cnn", "cnn_fpn", "resnet", "efficientnet", "vit", "swin", "ride"]
            if args.model == "all"
            else [args.model]
        )

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
                experiment_name=(
                    config.mlops.mlflow.experiment_name if config else "wafer-defect-detection"
                ),
                tracking_uri=(
                    config.mlops.mlflow.tracking_uri if config else "http://localhost:5000"
                ),
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
        logger.info(
            f"\n{'Model':<18} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Time (s)':<10}"
        )
        logger.info("-" * 70)
        for model_name in models_to_train:
            r = self.results[model_name]
            logger.info(
                f"{model_name:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} "
                f"{r['weighted_f1']:<12.4f} {r['time_sec']:<10.1f}"
            )
        for ens_key in ("learned_ensemble", "stacking_ensemble"):
            if ens_key in self.results:
                r = self.results[ens_key]
                logger.info(
                    f"{ens_key:<18} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} "
                    f"{r['weighted_f1']:<12.4f} {r.get('time_sec', 0.0):<10.1f}"
                )

        # Save metrics JSON — merge with existing file so subprocess-per-model
        # invocations (notebook Cell 6) accumulate metrics across runs instead
        # of each subprocess clobbering the last model's results.
        metrics_path = self.results_dir / "metrics.json"
        merged: dict = {}
        if metrics_path.exists():
            try:
                merged = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not isinstance(merged, dict):
                    merged = {}
            except (json.JSONDecodeError, OSError):
                merged = {}
        merged.update(self.results)

        # Embed reproducibility manifest (hashes of data/config/code + env).
        try:
            cfg_path = getattr(self.args, "config", None) or "config.yaml"
            merged["reproducibility"] = compute_manifest(
                data_path=self.args.data_path,
                config_path=cfg_path,
            )
        except Exception as exc:  # pragma: no cover — manifest must never kill a run
            logger.warning("Reproducibility manifest failed: %s", exc)

        with open(metrics_path, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info(f"\nMetrics saved to {metrics_path} ({len(merged)} model(s) total)")

        if self.wb_logger:
            self.wb_logger.finish()
        if self.mf_logger:
            self.mf_logger.finish()

        return 0


__all__ = ["TrainingPipeline", "set_seed"]
