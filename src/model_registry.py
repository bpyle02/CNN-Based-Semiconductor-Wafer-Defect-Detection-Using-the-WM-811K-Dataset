import json
import hashlib
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from src.exceptions import ModelError

class ModelMetadata:
    """Metadata for saved models."""

    def __init__(
        self,
        model_name: str,
        architecture: str,
        num_classes: int,
        training_config: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_version: str,
    ):
        self.model_name = model_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.training_config = training_config
        self.metrics = metrics
        self.dataset_version = dataset_version
        self.timestamp = datetime.now().isoformat()
        self.model_hash = ""
        self.num_params = 0


class ModelRegistry:
    """Centralized model storage and versioning."""

    def __init__(self, registry_path: str = 'model_registry'):
        self.registry_path = Path(registry_path).resolve()
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / 'registry.json'
        self.models = self._load_registry()

    def register(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        version: str = 'v1.0',
    ) -> str:
        """Register and save model with metadata."""
        state_dict = model.state_dict()
        metadata_dict = {
            **metadata.__dict__,
            'timestamp': datetime.now().isoformat(),
        }
        checkpoint_payload = io.BytesIO()
        torch.save({
            'state_dict': state_dict,
            'metadata': metadata_dict,
        }, checkpoint_payload)

        checkpoint_bytes = checkpoint_payload.getvalue()
        metadata.model_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        metadata.num_params = sum(param.numel() for param in model.parameters())
        metadata_dict['model_hash'] = metadata.model_hash
        metadata_dict['num_params'] = metadata.num_params

        timestamp_slug = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_model_id = f"{metadata.model_name}_{version}_{timestamp_slug}"
        model_id = base_model_id
        suffix = 1
        while model_id in self.models:
            model_id = f"{base_model_id}_{suffix}"
            suffix += 1

        # Save model
        model_path = (self.registry_path / f"{model_id}.pth").resolve()
        model_path.write_bytes(checkpoint_bytes)

        # Update registry
        self.models[model_id] = {
            'path': str(model_path),
            'metadata': metadata_dict,
            'registered_at': datetime.now().isoformat(),
        }
        self._save_registry()

        return model_id

    def load(self, model_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load model state dict and metadata by ID."""
        if model_id not in self.models:
            raise ModelError(f"Model {model_id} not found in registry")

        model_path = Path(self.models[model_id]['path']).resolve()
        try:
            checkpoint_bytes = model_path.read_bytes()
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location='cpu',
                weights_only=False,
            )
        except OSError as exc:
            raise ModelError(f"Failed to read model artifact for {model_id}") from exc
        except RuntimeError as exc:
            raise ModelError(f"Failed to deserialize model artifact for {model_id}") from exc
        return checkpoint['state_dict'], checkpoint['metadata']

    def list_models(self) -> list:
        """List all registered models."""
        return list(self.models.keys())

    def compare(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare metrics between two models."""
        if model_id1 not in self.models or model_id2 not in self.models:
            raise ModelError("One or both model IDs not found in registry")

        meta1 = self.models[model_id1]['metadata']
        meta2 = self.models[model_id2]['metadata']

        return {
            'model1': model_id1,
            'model2': model_id2,
            'accuracy_diff': meta1.get('metrics', {}).get('accuracy', 0) - meta2.get('metrics', {}).get('accuracy', 0),
            'f1_diff': meta1.get('metrics', {}).get('weighted_f1', 0) - meta2.get('metrics', {}).get('weighted_f1', 0),
            'params_ratio': meta1.get('num_params', 1) / meta2.get('num_params', 1) if meta2.get('num_params', 0) > 0 else 1.0,
        }

    def _load_registry(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=2)
