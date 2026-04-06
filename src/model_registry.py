import json
import hashlib
import io
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.exceptions import ModelError

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUSTED_CHECKPOINT_DIRS = (
    REPO_ROOT / "checkpoints",
    REPO_ROOT / "model_registry",
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_checkpoint_path(path: Union[str, Path]) -> Path:
    """Resolve and validate a checkpoint path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ModelError(f"Checkpoint not found: {resolved}")
    if resolved.suffix != ".pth":
        raise ModelError(f"Expected .pth checkpoint, got {resolved.suffix or '<none>'}")
    if resolved.is_symlink():
        raise ModelError("Symlinked checkpoint paths are not allowed")
    return resolved


def resolve_trusted_checkpoint_path(
    path: Union[str, Path],
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
) -> Path:
    """Resolve a checkpoint path and ensure it is inside a trusted root."""
    resolved = resolve_checkpoint_path(path)
    trusted_roots = tuple(
        Path(root).expanduser().resolve() for root in (allowed_roots or DEFAULT_TRUSTED_CHECKPOINT_DIRS)
    )
    if any(_is_relative_to(resolved, root) for root in trusted_roots):
        return resolved

    trusted_display = ", ".join(str(root) for root in trusted_roots)
    raise ModelError(
        f"Checkpoint path {resolved} is outside the trusted checkpoint roots: {trusted_display}"
    )


def compute_checkpoint_hash(path: Path) -> str:
    """Compute SHA-256 hash of a checkpoint file.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_checkpoint_with_hash(state_dict: dict, path: Path) -> str:
    """Save checkpoint and write a companion .sha256 hash file.

    Args:
        state_dict: Data to pass to ``torch.save``.
        path: Destination file path.

    Returns:
        Hex-encoded SHA-256 digest of the saved file.
    """
    torch.save(state_dict, path)
    file_hash = compute_checkpoint_hash(path)
    hash_path = path.with_suffix('.sha256')
    hash_path.write_text(file_hash)
    logger.info(f"Checkpoint saved to {path} (SHA-256: {file_hash[:16]}...)")
    return file_hash


def verify_checkpoint(path: Path) -> bool:
    """Verify checkpoint integrity against its stored SHA-256 hash.

    If no companion ``.sha256`` file exists the check is skipped for
    backwards compatibility and the function returns ``True``.

    Args:
        path: Path to the checkpoint file.

    Returns:
        ``True`` if the file matches its hash or no hash file exists,
        ``False`` if the hash does not match.
    """
    hash_path = path.with_suffix('.sha256')
    if not hash_path.exists():
        return True  # No hash file = skip verification (backwards compatible)
    stored_hash = hash_path.read_text().strip()
    actual_hash = compute_checkpoint_hash(path)
    if stored_hash != actual_hash:
        logger.warning(
            f"Checkpoint integrity check FAILED for {path}: "
            f"expected {stored_hash[:16]}..., got {actual_hash[:16]}..."
        )
        return False
    logger.info(f"Checkpoint integrity verified for {path}")
    return True

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
    ) -> None:
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

    def __init__(self, registry_path: str = 'model_registry') -> None:
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
        checkpoint_payload = {
            'state_dict': state_dict,
            'metadata': metadata_dict,
        }
        metadata.num_params = sum(param.numel() for param in model.parameters())
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
        file_hash = save_checkpoint_with_hash(checkpoint_payload, model_path)
        metadata.model_hash = file_hash
        metadata_dict['model_hash'] = file_hash

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
        if not verify_checkpoint(model_path):
            raise ModelError(f"Checkpoint integrity verification failed for {model_path}")

        try:
            checkpoint_bytes = model_path.read_bytes()
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location='cpu',
                weights_only=True,
            )
        except OSError as exc:
            raise ModelError(f"Failed to read model artifact for {model_id}") from exc
        except RuntimeError as exc:
            raise ModelError(f"Failed to deserialize model artifact for {model_id}") from exc
        return checkpoint['state_dict'], checkpoint['metadata']

    def list_models(self) -> List[str]:
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

    def _save_registry(self) -> None:
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=2)
