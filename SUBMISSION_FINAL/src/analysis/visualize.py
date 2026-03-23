"""
Visualization utilities for model analysis.

Produces plots for training curves, confusion matrices, model comparison,
and per-class performance analysis.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    figsize: Tuple[int, int] = (18, 10),
) -> None:
    """
    Plot training and validation loss/accuracy curves for multiple models.

    Args:
        histories: Dict mapping model names to history dicts with keys:
                   'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size (width, height)
    """
    palette = {
        'Custom CNN': '#1f77b4',
        'ResNet-18': '#ff7f0e',
        'EfficientNet-B0': '#2ca02c',
    }

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Training & Validation Curves', fontsize=16, fontweight='bold')

    for col, (name, hist) in enumerate(histories.items()):
        epochs_range = range(1, len(hist['train_loss']) + 1)
        color = palette.get(name, '#1f77b4')

        # Loss curves
        ax_loss = axes[0, col]
        ax_loss.plot(epochs_range, hist['train_loss'], '-', color=color, alpha=0.8, label='Train')
        ax_loss.plot(epochs_range, hist['val_loss'], '--', color=color, alpha=0.8, label='Val')
        ax_loss.set_title(f'{name} -- Loss', fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy curves
        ax_acc = axes[1, col]
        ax_acc.plot(epochs_range, hist['train_acc'], '-', color=color, alpha=0.8, label='Train')
        ax_acc.plot(epochs_range, hist['val_acc'], '--', color=color, alpha=0.8, label='Val')
        ax_acc.set_title(f'{name} -- Accuracy', fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: List[str],
    figsize: Tuple[int, int] = (22, 7),
) -> None:
    """
    Plot confusion matrices for multiple models.

    Args:
        results: Dict mapping model names to (predictions, labels) tuples
        class_names: List of class names
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(1, len(results), figsize=figsize)
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')

    for idx, (name, (preds, labels)) in enumerate(results.items()):
        cm = confusion_matrix(labels, preds)
        # Normalize by row (true label)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        ax = axes[idx] if len(results) > 1 else axes
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=True, square=True, linewidths=0.5,
            annot_kws={'size': 8}
        )
        ax.set_title(name, fontweight='bold', fontsize=13)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    summary_data: Dict[str, list],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot grouped bar chart comparing model performance metrics.

    Args:
        summary_data: Dict with keys like 'Accuracy', 'Macro F1', 'Weighted F1'
                     and values as lists per model
        figsize: Figure size (width, height)
    """
    palette = {
        'Custom CNN': '#1f77b4',
        'ResNet-18': '#ff7f0e',
        'EfficientNet-B0': '#2ca02c',
    }

    fig, ax = plt.subplots(figsize=figsize)

    metrics_names = ['Accuracy', 'Macro F1', 'Weighted F1']
    model_names = ['Custom CNN', 'ResNet-18', 'EfficientNet-B0']
    x = np.arange(len(metrics_names))
    width = 0.25

    values = [
        summary_data['Accuracy'],
        summary_data['Macro F1'],
        summary_data['Weighted F1'],
    ]

    for i, model_name in enumerate(model_names):
        model_values = [values[j][i] for j in range(len(values))]
        bars = ax.bar(
            x + i * width, model_values, width, label=model_name,
            color=palette.get(model_name, '#1f77b4'), edgecolor='black', linewidth=0.5
        )
        for bar, v in zip(bars, model_values):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_per_class_f1(
    f1_scores: Dict[str, np.ndarray],
    class_names: List[str],
    figsize: Tuple[int, int] = (16, 7),
) -> None:
    """
    Plot per-class F1 scores for multiple models.

    Args:
        f1_scores: Dict mapping model names to F1 score arrays
        class_names: List of class names
        figsize: Figure size (width, height)
    """
    palette = {
        'Custom CNN': '#1f77b4',
        'ResNet-18': '#ff7f0e',
        'EfficientNet-B0': '#2ca02c',
    }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    width = 0.25

    for i, (model_name, f1s) in enumerate(f1_scores.items()):
        ax.bar(
            x + i * width, f1s, width, label=model_name,
            color=palette.get(model_name, '#1f77b4'),
            edgecolor='black', linewidth=0.5
        )

    ax.set_xlabel('Defect Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded.")
