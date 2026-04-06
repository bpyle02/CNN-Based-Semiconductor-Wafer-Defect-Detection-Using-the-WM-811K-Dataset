"""
Model ensembling for improved predictions.

Supports multiple aggregation strategies:
- Voting: Majority vote from predictions
- Averaging: Average of softmax probabilities
- Weighted Averaging: Weighted combination of probabilities
"""

from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple neural network models.

    Combines predictions from multiple models using various aggregation strategies.

    Attributes:
        models: List of trained PyTorch models
        aggregation: Strategy for combining predictions ("voting", "averaging", "weighted_averaging")
        weights: Weights for each model (used in weighted_averaging)
    """

    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = "averaging",
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize ensemble.

        Args:
            models: List of trained models
            aggregation: How to combine predictions ("voting", "averaging", "weighted_averaging")
            weights: Weights for each model (summed to 1.0)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.num_models = len(models)

        if aggregation == "weighted_averaging":
            if weights is None:
                weights = [1.0 / self.num_models] * self.num_models
            else:
                total = sum(weights)
                weights = [w / total for w in weights]
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        elif weights is not None:
            logger.warning(f"Warning: weights ignored for aggregation={aggregation}")

        # Validate aggregation method
        assert aggregation in ["voting", "averaging", "weighted_averaging"], \
            f"Unknown aggregation method: {aggregation}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ensemble aggregation.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Aggregated logits (batch_size, num_classes)
        """
        logits_list = []

        # Get predictions from all models
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                logits_list.append(logits)

        # Aggregate predictions
        if self.aggregation == "voting":
            return self._aggregate_voting(logits_list)
        elif self.aggregation == "averaging":
            return self._aggregate_averaging(logits_list)
        elif self.aggregation == "weighted_averaging":
            return self._aggregate_weighted_averaging(logits_list)

    def _aggregate_voting(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Majority voting aggregation."""
        # Get predictions (argmax)
        preds = torch.stack([logits.argmax(dim=1) for logits in logits_list])  # (num_models, batch_size)

        # Mode (most common prediction per sample)
        ensemble_pred = torch.mode(preds, dim=0)[0]  # (batch_size,)

        # Return one-hot encoded as logits (set voted class to 1.0, others to 0.0)
        batch_size = ensemble_pred.size(0)
        num_classes = logits_list[0].size(1)
        logits = torch.zeros(batch_size, num_classes, device=ensemble_pred.device)
        logits.scatter_(1, ensemble_pred.unsqueeze(1), 1.0)

        return logits

    def _aggregate_averaging(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Average of softmax probabilities."""
        probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (batch_size, num_classes)
        return torch.log(avg_probs + 1e-10)  # Return log-probabilities

    def _aggregate_weighted_averaging(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Weighted average of softmax probabilities."""
        probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
        weighted_probs = sum(w * p for w, p in zip(self.weights, probs_list))
        return torch.log(weighted_probs + 1e-10)  # Return log-probabilities

    def to(self, device: torch.device) -> "EnsembleModel":
        """Move all models to device."""
        super().to(device)
        for model in self.models:
            model.to(device)
        return self

    def eval(self) -> "EnsembleModel":
        """Set all models to eval mode."""
        super().eval()
        for model in self.models:
            model.eval()
        return self

    def train(self, mode: bool = True) -> "EnsembleModel":
        """Set all models to train mode."""
        super().train(mode)
        for model in self.models:
            model.train(mode)
        return self


class EnsembleEvaluator:
    """
    Evaluate ensemble model performance.

    Provides metrics and analysis for ensemble predictions compared to individual models.
    """

    def __init__(self, ensemble: EnsembleModel, device: str = "cpu") -> None:
        """
        Initialize evaluator.

        Args:
            ensemble: EnsembleModel instance
            device: Device to run evaluation on
        """
        self.ensemble = ensemble
        self.device = torch.device(device)

    def evaluate(
        self,
        data_loader: DataLoader,
        num_classes: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble on test data.

        Args:
            data_loader: Test data loader
            num_classes: Number of classes

        Returns:
            Dictionary with metrics for ensemble and individual models
        """
        self.ensemble.to(self.device)
        self.ensemble.eval()

        ensemble_preds = []
        individual_preds = [[] for _ in range(self.ensemble.num_models)]
        targets = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                targets.append(labels.cpu().numpy())

                # Ensemble prediction
                ensemble_logits = self.ensemble(inputs)
                ensemble_preds.append(ensemble_logits.argmax(dim=1).cpu().numpy())

                # Individual model predictions
                for i, model in enumerate(self.ensemble.models):
                    logits = model(inputs)
                    individual_preds[i].append(logits.argmax(dim=1).cpu().numpy())

        # Concatenate all predictions
        ensemble_preds = np.concatenate(ensemble_preds)
        individual_preds = [np.concatenate(p) for p in individual_preds]
        targets = np.concatenate(targets)

        # Compute metrics
        metrics = {
            "ensemble": self._compute_metrics(ensemble_preds, targets, "Ensemble"),
        }

        for i, preds in enumerate(individual_preds):
            metrics[f"model_{i}"] = self._compute_metrics(preds, targets, f"Model {i}")

        return metrics

    @staticmethod
    def _compute_metrics(preds: np.ndarray, targets: np.ndarray, name: str) -> Dict[str, float]:
        """Compute accuracy and F1 score."""
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)

        return {
            "name": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def get_agreement_matrix(self, data_loader: DataLoader) -> np.ndarray:
        """
        Compute agreement between models.

        Returns:
            Agreement matrix showing how often each pair of models agrees
        """
        self.ensemble.eval()

        individual_preds = [[] for _ in range(self.ensemble.num_models)]

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)

                for i, model in enumerate(self.ensemble.models):
                    logits = model(inputs)
                    individual_preds[i].append(logits.argmax(dim=1).cpu().numpy())

        individual_preds = [np.concatenate(p) for p in individual_preds]

        # Compute pairwise agreement
        agreement = np.zeros((self.ensemble.num_models, self.ensemble.num_models))
        for i in range(self.ensemble.num_models):
            for j in range(self.ensemble.num_models):
                agreement[i, j] = np.mean(individual_preds[i] == individual_preds[j])

        return agreement


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_constructors: List[Callable[[], nn.Module]],
    device: str = "cpu",
    aggregation: str = "averaging",
) -> EnsembleModel:
    """
    Load models from checkpoints and create ensemble.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        model_constructors: List of functions that create models
        device: Device to load models on
        aggregation: Aggregation strategy

    Returns:
        EnsembleModel instance
    """
    models = []
    for checkpoint_path, constructor in zip(checkpoint_paths, model_constructors):
        model = constructor()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    return EnsembleModel(models, aggregation=aggregation)


if __name__ == "__main__":
    logger.info("Ensemble module loaded successfully")
