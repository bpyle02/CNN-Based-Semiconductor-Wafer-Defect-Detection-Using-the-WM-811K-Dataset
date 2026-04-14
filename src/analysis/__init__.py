"""Model evaluation and visualization utilities."""

from .evaluate import (
    evaluate_model,
    evaluate_with_tta,
    calibrate_and_evaluate,
    calibrate_and_evaluate_asymmetric,
    AsymmetricCalibrator,
    compute_calibration_metrics,
    count_params,
    count_trainable,
    get_per_class_f1,
)
from .artifacts import (
    build_experiment_manifest,
    detect_latest_checkpoint,
    compute_file_hash,
    compute_tree_hash,
    hash_path,
    write_manifest,
)
from .visualize import (
    plot_training_curves,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_per_class_f1,
)

__all__ = [
    'evaluate_model',
    'evaluate_with_tta',
    'calibrate_and_evaluate',
    'calibrate_and_evaluate_asymmetric',
    'AsymmetricCalibrator',
    'compute_calibration_metrics',
    'count_params',
    'count_trainable',
    'get_per_class_f1',
    'build_experiment_manifest',
    'detect_latest_checkpoint',
    'compute_file_hash',
    'compute_tree_hash',
    'hash_path',
    'write_manifest',
    'plot_training_curves',
    'plot_confusion_matrices',
    'plot_model_comparison',
    'plot_per_class_f1',
]
