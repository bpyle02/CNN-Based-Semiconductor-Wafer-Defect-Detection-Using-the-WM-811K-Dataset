"""
Training loop for wafer defect classification models.

Implements a standard supervised learning loop with validation, learning rate
scheduling, and history tracking. Optionally integrates MetricsTracker for
moving average analysis and trend detection.
"""

import time
from typing import Tuple, Dict, Any, Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)

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
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model and return best checkpoint by validation accuracy.

    Training procedure:
        1. Forward pass on batch
        2. Compute loss (weighted for class imbalance)
        3. Backward pass and optimizer step
        4. Optional gradient clipping
        5. Validation at end of each epoch
        6. Learning rate scheduling based on val loss
        7. Save best checkpoint (by val accuracy)
        8. (Optional) Update MetricsTracker with epoch metrics

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        criterion: Loss function (e.g., CrossEntropyLoss with weights)
        optimizer: Optimizer (e.g., Adam)
        scheduler: Optional learning rate scheduler
        epochs: Number of training epochs
        model_name: Name for logging
        device: Compute device ('cuda' or 'cpu')
        gradient_clip: Optional gradient norm clipping value
        metrics_tracker: Optional MetricsTracker for moving average tracking
            and trend detection. If None, training proceeds without tracking.
        metrics_output_path: Optional path to save metrics JSON at end of
            training. Only used if metrics_tracker is provided.

    Returns:
        Tuple of (best_model, history_dict) where history_dict contains:
            - train_loss, train_acc: Lists of per-epoch training metrics
            - val_loss, val_acc: Lists of per-epoch validation metrics
            - total_time: Total training time in seconds
            - best_epoch: Epoch with best validation accuracy
    """
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'total_time': 0,
        'best_epoch': 0,
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Optional gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update metrics tracker if provided
        if metrics_tracker is not None:
            current_lr = optimizer.param_groups[0]['lr']
            metrics_tracker.update('train_loss', train_loss, step=epoch)
            metrics_tracker.update('val_loss', val_loss, step=epoch)
            metrics_tracker.update('val_accuracy', val_acc, step=epoch)
            metrics_tracker.update('learning_rate', current_lr, step=epoch)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        # Log progress
        logger.info(
            f"[{model_name}] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # Check for degrading trend and warn
        if metrics_tracker is not None and epoch >= 3:
            trend = metrics_tracker.get_trend('val_loss', lookback=min(epoch, 10))
            if trend == 'degrading':
                logger.warning(
                    f"[{model_name}] Validation loss trend is DEGRADING "
                    f"at epoch {epoch}. Consider early stopping or LR reduction."
                )

    # Load best model
    model.load_state_dict(best_model_wts)
    elapsed = time.time() - start_time
    history['total_time'] = elapsed
    history['best_epoch'] = best_epoch

    logger.info(
        f"\n{model_name} training complete in {elapsed:.1f}s "
        f"(best val acc: {best_val_acc:.4f} at epoch {best_epoch})\n"
    )

    # Export metrics tracker data if path provided
    if metrics_tracker is not None and metrics_output_path is not None:
        metrics_tracker.to_json(metrics_output_path)

    # Log final summary from tracker
    if metrics_tracker is not None:
        summary = metrics_tracker.get_summary()
        for name, info in summary.items():
            logger.info(
                f"[{model_name}] {name}: last={info['last']:.4f}, "
                f"best={info['best']:.4f} (step {info['best_step']}), "
                f"MA={info['moving_avg']:.4f}, trend={info['trend']}"
            )

    return model, history


if __name__ == "__main__":
    logger.info("Training module loaded. Use train_model() from training pipeline.")
