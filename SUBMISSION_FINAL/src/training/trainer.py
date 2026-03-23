"""
Training loop for wafer defect classification models.

Implements a standard supervised learning loop with validation, learning rate
scheduling, and history tracking.
"""

import time
from typing import Tuple, Dict, Any, Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        # Log progress
        print(
            f"[{model_name}] Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

    # Load best model
    model.load_state_dict(best_model_wts)
    elapsed = time.time() - start_time
    history['total_time'] = elapsed
    history['best_epoch'] = best_epoch

    print(
        f"\n{model_name} training complete in {elapsed:.1f}s "
        f"(best val acc: {best_val_acc:.4f} at epoch {best_epoch})\n"
    )

    return model, history


if __name__ == "__main__":
    print("Training module loaded. Use train_model() from training pipeline.")
