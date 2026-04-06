"""
Model ensembling for improved predictions.

Supports multiple aggregation strategies:
- Voting: Majority vote from predictions
- Averaging: Average of softmax probabilities
- Weighted Averaging: Weighted combination of probabilities
- Learned Weights: Optimized per-model weights via validation macro F1
- Stacking: Meta-learner trained on base model softmax outputs

References:
    [30] Lakshminarayanan et al. (2017). "Deep Ensembles". arXiv:1612.01474
"""

from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
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


class LearnedWeightEnsemble(nn.Module):
    """Ensemble with weights learned to maximize macro F1 on validation data.

    Optimizes per-model weights using the validation set's macro F1 score.
    Uses scipy.optimize.minimize to find optimal weights under a simplex constraint.

    Reference: [30] Lakshminarayanan et al. (2017). arXiv:1612.01474
    """

    def __init__(self, models: List[nn.Module], device: str = "cpu") -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.device = torch.device(device)
        # Initialize uniform weights
        self.register_buffer(
            "weights",
            torch.ones(self.num_models, dtype=torch.float32) / self.num_models,
        )

    @torch.no_grad()
    def _collect_predictions(
        self, loader: DataLoader
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Collect softmax predictions from every base model on a data loader.

        Returns:
            all_probs: list of length num_models, each (N, C) numpy array of softmax probs
            all_labels: (N,) numpy array of ground-truth labels
        """
        self.eval()
        self.to(self.device)

        per_model_probs: List[List[np.ndarray]] = [[] for _ in range(self.num_models)]
        all_labels: List[np.ndarray] = []

        for images, labels in loader:
            images = images.to(self.device)
            all_labels.append(labels.cpu().numpy())

            for i, model in enumerate(self.models):
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                per_model_probs[i].append(probs)

        all_probs = [np.concatenate(p, axis=0) for p in per_model_probs]
        labels_arr = np.concatenate(all_labels, axis=0)
        return all_probs, labels_arr

    def optimize_weights(
        self, val_loader: DataLoader, class_names: List[str]
    ) -> np.ndarray:
        """Find weights that maximize macro F1 on validation set.

        Uses Nelder-Mead optimization over the probability simplex.
        The objective function computes the negative macro F1 of the
        weighted-average softmax predictions.

        Args:
            val_loader: Validation data loader.
            class_names: List of class name strings (used for logging).

        Returns:
            Optimal weight vector as a numpy array of shape (num_models,).
        """
        all_probs, labels = self._collect_predictions(val_loader)

        def _softmax_weights(raw: np.ndarray) -> np.ndarray:
            """Project raw parameters onto the probability simplex via softmax."""
            shifted = raw - raw.max()
            exp_w = np.exp(shifted)
            return exp_w / exp_w.sum()

        def objective(raw: np.ndarray) -> float:
            """Negative macro F1 of weighted-average predictions."""
            w = _softmax_weights(raw)
            blended = sum(w[i] * all_probs[i] for i in range(self.num_models))
            preds = blended.argmax(axis=1)
            score = f1_score(labels, preds, average="macro", zero_division=0)
            return -score

        # Start from uniform (raw zeros -> softmax -> uniform)
        x0 = np.zeros(self.num_models)
        result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-6})
        optimal_weights = _softmax_weights(result.x)

        # Store optimized weights in the buffer
        self.weights.copy_(torch.tensor(optimal_weights, dtype=torch.float32))

        best_f1 = -result.fun
        logger.info(
            "LearnedWeightEnsemble optimized weights: %s (macro F1=%.4f)",
            [f"{w:.4f}" for w in optimal_weights],
            best_f1,
        )
        return optimal_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted ensemble forward pass.

        Args:
            x: Input tensor (batch_size, channels, height, width).

        Returns:
            Log-probabilities (batch_size, num_classes).
        """
        probs_list = []
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                probs_list.append(torch.softmax(logits, dim=1))

        weighted_probs = sum(
            self.weights[i] * probs_list[i] for i in range(self.num_models)
        )
        return torch.log(weighted_probs + 1e-10)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Weighted ensemble prediction (class indices).

        Args:
            images: Input tensor (batch_size, channels, height, width).

        Returns:
            Predicted class indices (batch_size,).
        """
        log_probs = self.forward(images)
        return log_probs.argmax(dim=1)

    def evaluate(
        self, test_loader: DataLoader, class_names: List[str]
    ) -> Dict[str, float]:
        """Evaluate ensemble on a test set.

        Args:
            test_loader: Test data loader.
            class_names: List of class name strings.

        Returns:
            Dictionary with accuracy, macro_f1, and weighted_f1.
        """
        from sklearn.metrics import accuracy_score

        self.eval()
        self.to(self.device)

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                preds = self.predict(images).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)

        accuracy = accuracy_score(labels_arr, preds_arr)
        macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)

        logger.info(
            "LearnedWeightEnsemble test — Accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
            accuracy,
            macro_f1,
            weighted_f1,
        )
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }


class StackingEnsemble(nn.Module):
    """Stacking ensemble: train a meta-learner on base model predictions.

    Each base model produces softmax outputs on validation data.
    A small neural network (meta-learner) is trained to combine these predictions.

    Architecture: Linear(num_models * num_classes, num_classes).
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int = 9,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.base_models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes
        self.device = torch.device(device)

        input_dim = self.num_models * num_classes
        self.meta_learner = nn.Linear(input_dim, num_classes)

    @torch.no_grad()
    def _collect_base_features(
        self, loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run base models on a loader, concatenate softmax outputs.

        Returns:
            features: (N, num_models * num_classes) tensor
            labels: (N,) tensor
        """
        self.eval()
        for m in self.base_models:
            m.to(self.device)
            m.eval()

        feat_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            batch_feats = []
            for model in self.base_models:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                batch_feats.append(probs)
            # Concatenate along class dimension -> (batch, num_models * num_classes)
            combined = torch.cat(batch_feats, dim=1)
            feat_chunks.append(combined.cpu())
            label_chunks.append(labels)

        return torch.cat(feat_chunks, dim=0), torch.cat(label_chunks, dim=0)

    def fit(
        self,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> List[float]:
        """Train meta-learner on validation predictions.

        Args:
            val_loader: Validation data loader.
            epochs: Training epochs for the meta-learner.
            lr: Learning rate for the meta-learner optimizer.

        Returns:
            List of per-epoch training losses.
        """
        features, labels = self._collect_base_features(val_loader)

        meta_dataset = TensorDataset(features, labels)
        meta_loader = DataLoader(meta_dataset, batch_size=256, shuffle=True)

        self.meta_learner.to(self.device)
        self.meta_learner.train()
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        epoch_losses: List[float] = []
        for epoch in range(epochs):
            running_loss = 0.0
            n_batches = 0
            for feat_batch, label_batch in meta_loader:
                feat_batch = feat_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.meta_learner(feat_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1
            avg_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        logger.info(
            "StackingEnsemble meta-learner trained for %d epochs (final loss=%.4f)",
            epochs,
            epoch_losses[-1] if epoch_losses else float("nan"),
        )
        return epoch_losses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stacked ensemble forward pass.

        Args:
            x: Input tensor (batch_size, channels, height, width).

        Returns:
            Logits from meta-learner (batch_size, num_classes).
        """
        batch_feats = []
        with torch.no_grad():
            for model in self.base_models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                batch_feats.append(probs)
        combined = torch.cat(batch_feats, dim=1)
        return self.meta_learner(combined)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Stacked ensemble prediction (class indices).

        Args:
            images: Input tensor (batch_size, channels, height, width).

        Returns:
            Predicted class indices (batch_size,).
        """
        logits = self.forward(images)
        return logits.argmax(dim=1)

    def evaluate(
        self, test_loader: DataLoader, class_names: List[str]
    ) -> Dict[str, float]:
        """Evaluate stacking ensemble on a test set.

        Args:
            test_loader: Test data loader.
            class_names: List of class name strings.

        Returns:
            Dictionary with accuracy, macro_f1, and weighted_f1.
        """
        from sklearn.metrics import accuracy_score

        self.to(self.device)
        for m in self.base_models:
            m.to(self.device)
            m.eval()
        self.meta_learner.eval()

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                preds = self.predict(images).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)

        accuracy = accuracy_score(labels_arr, preds_arr)
        macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)

        logger.info(
            "StackingEnsemble test — Accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
            accuracy,
            macro_f1,
            weighted_f1,
        )
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }


if __name__ == "__main__":
    logger.info("Ensemble module loaded successfully")
