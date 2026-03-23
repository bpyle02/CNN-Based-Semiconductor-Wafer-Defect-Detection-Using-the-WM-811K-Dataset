"""Model evaluation and visualization utilities."""

from .evaluate import evaluate_model, count_params, count_trainable
from .visualize import (
    plot_training_curves,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_per_class_f1,
)

__all__ = [
    'evaluate_model',
    'count_params',
    'count_trainable',
    'plot_training_curves',
    'plot_confusion_matrices',
    'plot_model_comparison',
    'plot_per_class_f1',
]
