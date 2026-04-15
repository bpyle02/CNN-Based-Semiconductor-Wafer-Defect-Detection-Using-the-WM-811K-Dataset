"""
Training loop for wafer defect classification models.

Implements a supervised training loop with validation, learning rate
scheduling, early stopping, optional mixed precision, metric tracking,
Deferred Re-Weighting (DRW), Adaptive Re-Balancing (AREA), and
Exponential Moving Average (EMA).

References:
    [45] Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization". arXiv:1412.6980
    [46] Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization". arXiv:1711.05101
    [133] Loshchilov & Hutter (2017). "SGDR: Warm Restarts". arXiv:1608.03983
    [134] Gotmare et al. (2019). "A Closer Look at Deep Learning Heuristics". arXiv:1811.03716
    [135] Goyal et al. (2017). "Large Minibatch SGD". arXiv:1706.02677
    [136] Smith & Topin (2019). "Super-Convergence". arXiv:1708.07120
    [137] Cao et al. (2019). "LDAM + DRW". arXiv:1906.07413
    [138] Polyak & Juditsky (1992). "Acceleration of Stochastic Approximation"
    [190] Chen et al. (2022). "AREA: Adaptive Re-balancing via an Effective Approach". arXiv:2206.02841
"""

from __future__ import annotations

import copy
import logging
import math
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .ema import EMAModel
from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class AdaptiveRebalancer:
    """Adaptive re-balancing: gradually increase class weights from uniform to
    inverse-frequency based on per-class validation performance.

    Instead of a hard DRW cutoff, smoothly interpolates::

        w_c(t) = (1 - alpha(t)) * uniform + alpha(t) * inverse_freq

    where ``alpha(t)`` increases from 0 to 1 over training using a cosine
    schedule.  When per-class F1 scores are supplied the weights are further
    modulated so that poorly-performing classes receive larger weights.

    Reference:
        [190] Chen et al. (2022). "AREA: Adaptive Re-balancing via an
        Effective Approach". arXiv:2206.02841
    """

    def __init__(
        self,
        num_classes: int,
        class_frequencies: torch.Tensor,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")

        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # Uniform weights
        self.uniform = torch.ones(num_classes) / num_classes

        # Inverse-frequency weights, normalized so they sum to ``num_classes``
        # (same scale as ``torch.ones(num_classes)``).
        freq = class_frequencies.float()
        inv_freq = 1.0 / (freq + 1e-8)
        self.target_weights = inv_freq / inv_freq.sum() * num_classes

    def get_weights(
        self,
        epoch: int,
        per_class_f1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return class weights for the current epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            per_class_f1: Optional tensor of per-class F1 scores from the
                previous validation pass.  When provided, classes with low
                F1 receive a proportionally higher weight boost.

        Returns:
            Tensor of shape ``(num_classes,)`` with non-negative weights.
        """
        if epoch < self.warmup_epochs:
            return self.uniform.clone()

        # Cosine interpolation from uniform to target weights
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        alpha = 0.5 * (1.0 - math.cos(math.pi * min(progress, 1.0)))

        base_weights = (1.0 - alpha) * self.uniform + alpha * self.target_weights

        # Optional F1-based modulation: boost under-performing classes
        if per_class_f1 is not None:
            f1_factor = 1.0 / (per_class_f1.float() + 0.1)
            f1_factor = f1_factor / f1_factor.mean()
            base_weights = base_weights * f1_factor

        return base_weights


MONITORED_METRICS = {"val_loss", "val_acc", "val_macro_f1"}


def _is_loss_metric(metric_name: str) -> bool:
    return "loss" in metric_name.lower()


def _get_metric_value(metric_name: str, values: Dict[str, float]) -> float:
    if metric_name not in values:
        supported = ", ".join(sorted(MONITORED_METRICS))
        raise ValueError(
            f"Unsupported monitored metric '{metric_name}'. Expected one of: {supported}"
        )
    return float(values[metric_name])


def _is_improvement(
    metric_name: str,
    current_value: float,
    best_value: float,
    min_delta: float,
) -> bool:
    if _is_loss_metric(metric_name):
        return current_value < (best_value - min_delta)
    return current_value > (best_value + min_delta)


def _step_scheduler(
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    monitored_value: float,
) -> None:
    if scheduler is None:
        return

    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitored_value)
        return

    scheduler.step()


def core_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    epochs: int = 25,
    model_name: str = "Model",
    device: str = "cpu",
    gradient_clip: Optional[float] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
    metrics_output_path: Optional[str] = None,
    mixed_precision: bool = False,
    early_stopping_enabled: bool = False,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    monitored_metric: str = "val_macro_f1",
    batch_transform: Optional[Any] = None,
    drw_epoch: Optional[int] = None,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    adaptive_rebalance: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model and return the best checkpoint according to ``monitored_metric``.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
        optimizer: Optimizer instance.
        scheduler: Optional learning-rate scheduler.
        epochs: Number of epochs to train.
        model_name: Human-readable name for logging.
        device: Compute device.
        gradient_clip: Optional gradient norm clipping threshold.
        metrics_tracker: Optional metrics tracker.
        metrics_output_path: Optional JSON path for tracked metrics.
        mixed_precision: Enable CUDA mixed precision if available.
        early_stopping_enabled: Whether to stop after repeated non-improving epochs.
        early_stopping_patience: Number of bad epochs to tolerate.
        early_stopping_min_delta: Minimum metric improvement to reset patience.
        monitored_metric: One of ``val_loss``, ``val_acc``, or ``val_macro_f1``.
        batch_transform: Optional callable (e.g. MixupCutmix) applied to each
            training batch. Must return ``(mixed_images, labels_a, labels_b, lam)``.
        drw_epoch: Deferred Re-Weighting schedule epoch. When set, epochs 1
            through ``drw_epoch`` use unweighted cross-entropy (representation
            learning phase), and epochs after ``drw_epoch`` switch to the
            weighted ``criterion`` (classifier adaptation phase). ``None`` or
            ``0`` disables DRW entirely. Ref: Cao et al. (2019) arXiv:1906.07413
        use_ema: If ``True``, maintain an Exponential Moving Average of the
            model weights and use EMA weights for validation. The returned
            best model will have EMA weights applied. Ref: Polyak &
            Juditsky (1992).
        ema_decay: Decay factor for EMA (only used when ``use_ema=True``).
            Typical range: 0.99 -- 0.9999.
        adaptive_rebalance: If ``True``, use an :class:`AdaptiveRebalancer`
            to smoothly interpolate class weights from uniform to
            inverse-frequency over training instead of the hard DRW
            cutoff.  Requires the ``criterion`` to have a ``weight``
            attribute (e.g. ``CrossEntropyLoss``).  Mutually exclusive
            with DRW (``drw_epoch`` is ignored when enabled).
            Ref: Chen et al. (2022) arXiv:2206.02841

    Returns:
        A tuple of ``(best_model, history_dict)``.
    """
    model.to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = mixed_precision and device_type == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_metric = float("inf") if _is_loss_metric(monitored_metric) else float("-inf")
    epochs_without_improvement = 0

    # --- DRW flag (needed before history init) ---
    drw_active = drw_epoch is not None and drw_epoch > 0

    # --- Adaptive rebalancing (overrides DRW when enabled) ---
    rebalancer: Optional[AdaptiveRebalancer] = None
    if adaptive_rebalance:
        # Compute class frequencies from the training loader
        label_counts: Dict[int, int] = {}
        for _, batch_labels in train_loader:
            for label in batch_labels.tolist():
                label_counts[label] = label_counts.get(label, 0) + 1
        num_classes = len(label_counts)
        class_freqs = torch.zeros(num_classes)
        for cls_idx, count in label_counts.items():
            class_freqs[cls_idx] = count
        rebalancer = AdaptiveRebalancer(
            num_classes=num_classes,
            class_frequencies=class_freqs,
            warmup_epochs=max(epochs // 10, 1),
            total_epochs=epochs,
        )
        # Disable DRW when adaptive rebalancing is active
        drw_active = False
        logger.info(
            "[%s] Adaptive rebalancing enabled (warmup=%d, total=%d)",
            model_name,
            rebalancer.warmup_epochs,
            rebalancer.total_epochs,
        )

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
        "total_time": 0.0,
        "best_epoch": 0,
        "best_metric": best_metric,
        "best_metric_name": monitored_metric,
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
        "epochs_ran": 0,
        "stopped_early": False,
        "mixed_precision": amp_enabled,
        "drw_epoch": drw_epoch if drw_active else None,
        "adaptive_rebalance": adaptive_rebalance,
        "use_ema": use_ema,
        "ema_decay": ema_decay if use_ema else None,
    }

    # --- EMA initialisation ---
    ema: Optional[EMAModel] = None
    if use_ema:
        ema = EMAModel(model, decay=ema_decay)
        logger.info("[%s] EMA enabled (decay=%.6f)", model_name, ema_decay)

    if drw_active:
        logger.info(
            "[%s] DRW schedule: unweighted loss for epochs 1-%d, weighted loss after",
            model_name,
            drw_epoch,
        )

    start_time = time.time()

    # Track per-class F1 for adaptive rebalancing feedback
    _prev_per_class_f1: Optional[torch.Tensor] = None

    for epoch in range(1, epochs + 1):
        # --- Adaptive rebalancing: update criterion weights each epoch ---
        if rebalancer is not None and hasattr(criterion, "weight"):
            new_weights = rebalancer.get_weights(epoch, per_class_f1=_prev_per_class_f1)
            criterion.weight = new_weights.to(device)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        current_lr = float(optimizer.param_groups[0]["lr"])

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.amp.autocast(device_type=device_type, dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )

            # DRW: use unweighted CE during representation-learning phase
            use_unweighted = drw_active and epoch <= drw_epoch

            with autocast_context:
                if batch_transform is not None:
                    images, labels_a, labels_b, lam = batch_transform(images, labels)
                    outputs = model(images)
                    if use_unweighted:
                        loss = lam * F.cross_entropy(outputs, labels_a) + (
                            1.0 - lam
                        ) * F.cross_entropy(outputs, labels_b)
                    else:
                        loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(
                            outputs, labels_b
                        )
                else:
                    outputs = model(images)
                    if use_unweighted:
                        loss = F.cross_entropy(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)

            if amp_enabled:
                scaler.scale(loss).backward()
                if gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            # EMA: update shadow weights after each optimizer step
            if ema is not None:
                ema.update(model)

            train_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # EMA: swap in shadow weights for validation
        if ema is not None:
            ema.apply_shadow(model)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                autocast_context = (
                    torch.amp.autocast(device_type=device_type, dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_context:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                predicted = outputs.argmax(dim=1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_predictions.extend(predicted.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_macro_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)

        # Capture per-class F1 for adaptive rebalancing feedback
        if rebalancer is not None:
            per_class_f1_arr = f1_score(val_targets, val_predictions, average=None, zero_division=0)
            _prev_per_class_f1 = torch.tensor(per_class_f1_arr, dtype=torch.float32)

        # EMA: restore training weights after validation
        if ema is not None:
            ema.restore(model)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["learning_rate"].append(current_lr)

        if metrics_tracker is not None:
            metrics_tracker.update("train_loss", train_loss, step=epoch)
            metrics_tracker.update("train_accuracy", train_acc, step=epoch)
            metrics_tracker.update("val_loss", val_loss, step=epoch)
            metrics_tracker.update("val_accuracy", val_acc, step=epoch)
            metrics_tracker.update("val_macro_f1", val_macro_f1, step=epoch)
            metrics_tracker.update("learning_rate", current_lr, step=epoch)

        metric_values = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        }
        current_metric = _get_metric_value(monitored_metric, metric_values)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        _step_scheduler(scheduler, current_metric)

        if _is_improvement(monitored_metric, current_metric, best_metric, early_stopping_min_delta):
            best_metric = current_metric
            best_epoch = epoch
            # When EMA is active, save the EMA (shadow) weights as the
            # best checkpoint since validation was performed with them.
            if ema is not None:
                ema.apply_shadow(model)
                best_model_wts = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
            history["best_metric"] = current_metric
            history["best_epoch"] = epoch
            history["best_val_acc"] = val_acc
            history["best_val_loss"] = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "[%s] Epoch %2d/%d | LR: %.6f | Train Loss: %.4f, Acc: %.4f | "
            "Val Loss: %.4f, Acc: %.4f, Macro F1: %.4f",
            model_name,
            epoch,
            epochs,
            current_lr,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if metrics_tracker is not None and epoch >= 3:
            trend = metrics_tracker.get_trend("val_loss", lookback=min(epoch, 10))
            if trend == "degrading":
                logger.warning(
                    "[%s] Validation loss trend is degrading at epoch %d.",
                    model_name,
                    epoch,
                )

        history["epochs_ran"] = epoch

        if (
            early_stopping_enabled
            and early_stopping_patience is not None
            and early_stopping_patience >= 0
            and epochs_without_improvement > early_stopping_patience
        ):
            history["stopped_early"] = True
            logger.info(
                "[%s] Early stopping triggered at epoch %d after %d non-improving epochs.",
                model_name,
                epoch,
                epochs_without_improvement,
            )
            break

    model.load_state_dict(best_model_wts)
    elapsed = time.time() - start_time
    history["total_time"] = elapsed
    history["best_epoch"] = best_epoch
    history["best_metric"] = best_metric
    history["best_val_acc"] = (
        best_val_acc if history["best_epoch"] == 0 else history["best_val_acc"]
    )
    history["best_val_loss"] = (
        best_val_loss if history["best_epoch"] == 0 else history["best_val_loss"]
    )

    logger.info(
        "%s training complete in %.1fs (best %s: %.4f at epoch %d)",
        model_name,
        elapsed,
        monitored_metric,
        best_metric,
        best_epoch,
    )

    if metrics_tracker is not None and metrics_output_path is not None:
        metrics_tracker.to_json(metrics_output_path)

    if metrics_tracker is not None:
        summary = metrics_tracker.get_summary()
        for name, info in summary.items():
            logger.info(
                "[%s] %s: last=%.4f, best=%.4f (step %s), MA=%.4f, trend=%s",
                model_name,
                name,
                info["last"],
                info["best"],
                info["best_step"],
                info["moving_avg"],
                info["trend"],
            )

    return model, history


# Backward-compat alias. The inner training loop was renamed from
# ``train_model`` to ``core_training_loop`` for clarity (the pipeline is
# the orchestrator, this is the inner loop). Existing scripts and tests
# continue to import ``train_model``.
train_model = core_training_loop


if __name__ == "__main__":
    logger.info("Training module loaded. Use core_training_loop() from training pipeline.")
