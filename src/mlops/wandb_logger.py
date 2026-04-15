"""
Weights & Biases integration for experiment tracking.

Logs training metrics, hyperparameters, model checkpoints, and evaluation results.

References:
    [147] Biewald (2020). "Experiment Tracking with Weights and Biases"
    [148] Zaharia et al. (2018). "Accelerating the ML Lifecycle with MLflow". DOI:10.1109/DSAA.2018.00032
    [149] Amershi et al. (2019). "Software Engineering for ML". arXiv:1904.07204
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandBLogger:
    """Weights & Biases experiment logger."""

    def __init__(
        self,
        project: str = "wafer-defect-detection",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity/team name
            name: Experiment name
            config: Hyperparameter dictionary
            enabled: Enable logging
        """
        self.enabled = enabled and HAS_WANDB
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}

        if self.enabled:
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=["wafer-defect", "production"],
            )
            logger.info(f"W&B initialized: {project}/{entity or 'default'}")
        else:
            if enabled and not HAS_WANDB:
                logger.warning("Warning: wandb not installed, logging disabled")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_model(self, model: torch.nn.Module, checkpoint_path: str) -> None:
        """Log model artifact."""
        if self.enabled:
            run = wandb.run
            if run is None:
                logger.warning("wandb.run is None; cannot log model artifact")
                return
            artifact = wandb.Artifact(
                name=f"model-{run.id}",
                type="model",
                description=f"Model checkpoint from step {run.step}",
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    def log_confusion_matrix(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        class_names: List[str],
    ) -> None:
        """Log confusion matrix."""
        if self.enabled:
            # sklearn.metrics.confusion_matrix accepts array-like; coerce to
            # numpy ndarray so the type signature is satisfied uniformly.
            from sklearn.metrics import confusion_matrix as cm

            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            cm(y_true_arr, y_pred_arr)
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=y_true_arr.tolist(),
                        preds=y_pred_arr.tolist(),
                        class_names=class_names,
                    )
                }
            )

    def finish(self) -> None:
        """Finish logging run."""
        if self.enabled:
            wandb.finish()


class MLFlowLogger:
    """MLflow experiment logger."""

    def __init__(
        self,
        experiment_name: str = "wafer-defect-detection",
        tracking_uri: str = "http://localhost:5000",
        enabled: bool = True,
    ) -> None:
        """
        Initialize MLFlow logger.

        Args:
            experiment_name: Experiment name
            tracking_uri: MLflow tracking server URI
            enabled: Enable logging
        """
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        try:
            import mlflow

            self.mlflow = mlflow
            self.mlflow.set_tracking_uri(tracking_uri)
            self.mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow initialized: {experiment_name}")
        except ImportError:
            if enabled:
                logger.warning("Warning: mlflow not installed, logging disabled")
            self.enabled = False

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if self.enabled:
            for key, value in params.items():
                self.mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if self.enabled:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)

    def log_model(self, model: torch.nn.Module, artifact_path: str = "model") -> None:
        """Log model."""
        if self.enabled:
            self.mlflow.pytorch.log_model(model, artifact_path)

    def finish(self) -> None:
        """End logging run."""
        if self.enabled:
            self.mlflow.end_run()
