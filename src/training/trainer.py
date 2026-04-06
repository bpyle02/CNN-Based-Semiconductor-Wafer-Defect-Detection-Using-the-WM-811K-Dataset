"""
Training loop for wafer defect classification models.

Implements a supervised training loop with validation, learning rate
scheduling, early stopping, optional mixed precision, and metric tracking.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)

MONITORED_METRICS = {"val_loss", "val_acc", "val_macro_f1"}


def _is_loss_metric(metric_name: str) -> bool:
    return "loss" in metric_name.lower()


def _get_metric_value(metric_name: str, values: Dict[str, float]) -> float:
    if metric_name not in values:
        supported = ", ".join(sorted(MONITORED_METRICS))
        raise ValueError(f"Unsupported monitored metric '{metric_name}'. Expected one of: {supported}")
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    epochs: int = 5,
    model_name: str = "Model",
    device: str = "cpu",
    gradient_clip: Optional[float] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
    metrics_output_path: Optional[str] = None,
    mixed_precision: bool = False,
    early_stopping_enabled: bool = False,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    monitored_metric: str = "val_acc",
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
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
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

            with autocast_context:
                outputs = model(images)
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

            train_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

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
    history["best_val_acc"] = best_val_acc if history["best_epoch"] == 0 else history["best_val_acc"]
    history["best_val_loss"] = best_val_loss if history["best_epoch"] == 0 else history["best_val_loss"]

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


if __name__ == "__main__":
    logger.info("Training module loaded. Use train_model() from training pipeline.")
