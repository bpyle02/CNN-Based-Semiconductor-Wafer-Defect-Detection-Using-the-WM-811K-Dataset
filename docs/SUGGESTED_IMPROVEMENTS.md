# Suggested Improvements & Fixes

**Based on**: Comprehensive Review of 23 Fully Implemented Improvements
**Date**: 2026-03-22

---

## Quick Fixes (1-2 hours)

### 1. Fix ViT Patch Size for 96x96 Images

**File**: `src/models/vit.py`

**Problem**: Patch size of 16x16 creates only 6x6 grid for 96x96 images, which is too coarse.

**Current Code** (lines 20-30):
```python
def __init__(self, image_size: int = 96, patch_size: int = 16, ...):
    assert image_size % patch_size == 0, "Image size must be divisible by patch size"
    self.num_patches = (image_size // patch_size) ** 2  # = 36 patches
```

**Recommended Fix**:
```python
def __init__(self, image_size: int = 96, patch_size: int = 8, ...):
    """
    Use patch_size=8 for 96x96 images -> 12x12 grid (144 patches).
    Better balance of local-global context vs sequence length.
    """
    assert image_size % patch_size == 0, "Image size must be divisible by patch size"
    self.num_patches = (image_size // patch_size) ** 2  # = 144 patches

    # Add MLP head with hidden layer (better than single linear)
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Linear(hidden_dim * 2, num_classes),
    )
```

**Expected Impact**: 5-10% accuracy improvement on ViT-small/tiny.

---

### 2. Add Custom Exceptions for Better Error Handling

**Create**: `src/exceptions.py`

```python
"""Custom exception types for wafer defect detection."""

class WaferMapError(Exception):
    """Base exception for wafer map operations."""
    pass

class DataLoadError(WaferMapError):
    """Raised when data cannot be loaded or parsed."""
    pass

class ModelError(WaferMapError):
    """Raised when model initialization/forward pass fails."""
    pass

class TrainingError(WaferMapError):
    """Raised during training/validation."""
    pass

class InferenceError(WaferMapError):
    """Raised during inference."""
    pass

class FederatedError(WaferMapError):
    """Raised during federated learning operations."""
    pass
```

**Update all modules to use these**:
```python
# In src/data/dataset.py
try:
    data = pickle.load(f)
except Exception as e:
    raise DataLoadError(f"Failed to load dataset from {path}: {e}") from e

# In src/inference/server.py
try:
    output = self.model(images)
except RuntimeError as e:
    raise InferenceError(f"Model inference failed on batch shape {images.shape}: {e}") from e
```

---

### 3. Fix Configuration Consistency Across All Scripts

**Problem**: `progressive_train.py`, `active_learn.py`, `compress_model.py` have hardcoded hyperparameters instead of using `Config`.

**Create**: `src/training/base_trainer.py`

```python
from src.config import Config

class BaseTrainer:
    """Base trainer that loads config from YAML."""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config.from_yaml(config_path)
        self.device = self.config.device
        self.seed = self.config.seed
        self.set_seed()

    def set_seed(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
```

**Update all scripts** to inherit from `BaseTrainer`:
```python
# progressive_train.py
from src.training.base_trainer import BaseTrainer

class ProgressiveTrainer(BaseTrainer):
    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        # Use self.config for all hyperparameters
        self.learning_rate = self.config.training.learning_rate
        self.batch_size = self.config.data.batch_size
```

---

### 4. Add Type Hints to Remaining Functions

**Files to update**:
- `dashboard.py` - Missing return type annotations on ~15 functions
- `active_learn.py` - Missing type hints on loop variables
- `progressive_train.py` - Missing return types on 5-8 functions

**Example fix** (dashboard.py):
```python
# Before
def get_model_info(model):
    return {
        'num_params': sum(p.numel() for p in model.parameters()),
        ...
    }

# After
def get_model_info(model: nn.Module) -> Dict[str, int]:
    """Return model parameter statistics."""
    return {
        'num_params': sum(p.numel() for p in model.parameters()),
        ...
    }
```

---

## Medium Priority Fixes (2-4 hours)

### 5. Implement Model Registry with Versioning

**Create**: `src/model_registry.py` (250+ lines)

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

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

class ModelRegistry:
    """Centralized model storage and versioning."""

    def __init__(self, registry_path: str = 'model_registry'):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / 'registry.json'
        self.models = self._load_registry()

    def register(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        version: str = 'v1.0',
    ) -> str:
        """Register and save model with metadata."""
        model_id = f"{metadata.model_name}_{version}_{metadata.timestamp[:10]}"

        # Save model
        model_path = self.registry_path / f"{model_id}.pth"
        torch.save({
            'state_dict': model.state_dict(),
            'metadata': metadata.__dict__,
        }, model_path)

        # Update registry
        self.models[model_id] = {
            'path': str(model_path),
            'metadata': metadata.__dict__,
            'registered_at': datetime.now().isoformat(),
        }
        self._save_registry()

        return model_id

    def load(self, model_id: str) -> tuple[nn.Module, Dict]:
        """Load model and metadata by ID."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        checkpoint = torch.load(self.models[model_id]['path'])
        return checkpoint['state_dict'], checkpoint['metadata']

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self.models.keys())

    def compare(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare metrics between two models."""
        meta1 = self.models[model_id1]['metadata']
        meta2 = self.models[model_id2]['metadata']

        return {
            'model1': model_id1,
            'model2': model_id2,
            'accuracy_diff': meta1['metrics'].get('accuracy', 0) - meta2['metrics'].get('accuracy', 0),
            'f1_diff': meta1['metrics'].get('weighted_f1', 0) - meta2['metrics'].get('weighted_f1', 0),
            'params_ratio': meta1.get('num_params', 0) / meta2.get('num_params', 1),
        }

    def _load_registry(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.models, f, indent=2)
```

**Integration with training**:
```python
# In train.py
registry = ModelRegistry()

# After training
metadata = ModelMetadata(
    model_name=args.model,
    architecture=model.__class__.__name__,
    num_classes=9,
    training_config=asdict(config),
    metrics={'accuracy': acc, 'weighted_f1': f1},
    dataset_version='wm811k_v1',
)

model_id = registry.register(model, metadata, version=f'v{epoch+1}.0')
print(f"Model saved as: {model_id}")
```

---

### 6. Complete Synthetic Data Pipeline

**File**: `src/augmentation/train_generator.py` (250+ lines)

```python
"""Training script for GAN-based synthetic wafer map generation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from typing import Tuple
import numpy as np

def compute_fid_score(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    inception_model: nn.Module,
    device: str = 'cuda',
) -> float:
    """
    Compute Frechet Inception Distance (FID) between real and fake images.
    Lower FID = better quality synthetic images.
    """
    inception_model.eval()

    with torch.no_grad():
        real_features = inception_model(real_images).cpu().numpy()
        fake_features = inception_model(fake_images).cpu().numpy()

    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # FID score
    diff = mu_real - mu_fake
    covmean = np.linalg.sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2*covmean)
    return float(fid)

def train_generator(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    epochs: int = 50,
    device: str = 'cuda',
) -> Tuple[float, float]:
    """Train GAN on wafer map data."""

    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    inception = inception_v3(pretrained=True, transform_input=False).to(device)

    for epoch in range(epochs):
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # Train discriminator
            d_optimizer.zero_grad()

            real_output = discriminator(images)
            real_loss = criterion(real_output, torch.ones(batch_size, 1).to(device))

            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, torch.zeros(batch_size, 1).to(device))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()

            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, torch.ones(batch_size, 1).to(device))

            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                noise = torch.randn(100, 100).to(device)
                fake_batch = generator(noise)
                fid = compute_fid_score(images[:100], fake_batch, inception, device)

            print(f"Epoch {epoch+1}/{epochs}, D_Loss: {d_loss:.4f}, "
                  f"G_Loss: {g_loss:.4f}, FID: {fid:.2f}")

    return d_loss.item(), g_loss.item()
```

**Integration with training**:
```python
# In src/augmentation/synthetic.py, add this method:
def augment_training_data(
    synthetic_generator,
    normal_class_count: int,
    rare_class_counts: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic samples to balance training data."""

    synthetic_maps = []
    synthetic_labels = []

    # Oversample rare classes
    for class_idx, count in rare_class_counts.items():
        target_count = normal_class_count
        needed = max(0, target_count - count)

        if needed > 0:
            noise = torch.randn(needed, 100)
            synthetic = synthetic_generator(noise)
            synthetic_maps.append(synthetic)
            synthetic_labels.extend([class_idx] * needed)

    if synthetic_maps:
        return torch.cat(synthetic_maps), torch.tensor(synthetic_labels)
    return None, None
```

---

### 7. Add Byzantine-Resistant Federated Learning

**File**: Update `src/federated/fed_avg.py`

```python
class ByzantineRobustAggregator:
    """Robust aggregation resistant to model poisoning attacks."""

    def __init__(self, method: str = 'median'):
        """
        method: 'median' (robust to 50% malicious),
                'trimmed_mean' (robust to 30% malicious),
                'krum' (robust to n-f clients, where f < n/3)
        """
        self.method = method

    def aggregate(self, client_weights: list[Dict]) -> Dict:
        """Aggregate client weights robustly."""

        if self.method == 'median':
            return self._median_aggregation(client_weights)
        elif self.method == 'trimmed_mean':
            return self._trimmed_mean_aggregation(client_weights, trim_ratio=0.2)
        elif self.method == 'krum':
            return self._krum_aggregation(client_weights)

    def _median_aggregation(self, weights_list: list[Dict]) -> Dict:
        """Median aggregation - robust to 50% poisoned clients."""
        aggregated = {}

        for key in weights_list[0].keys():
            values = torch.stack([w[key] for w in weights_list])
            aggregated[key] = torch.median(values, dim=0)[0]

        return aggregated

    def _trimmed_mean_aggregation(self, weights_list: list[Dict], trim_ratio: float = 0.2) -> Dict:
        """Trimmed mean aggregation - removes outlier updates."""
        aggregated = {}

        for key in weights_list[0].keys():
            values = torch.stack([w[key] for w in weights_list])

            # Sort and remove extremes
            sorted_vals = torch.sort(values, dim=0)[0]
            trim_count = int(len(weights_list) * trim_ratio)

            trimmed = sorted_vals[trim_count:-trim_count]
            aggregated[key] = torch.mean(trimmed, dim=0)

        return aggregated

    def _krum_aggregation(self, weights_list: list[Dict]) -> Dict:
        """Krum aggregation - selects parameter update with minimum euclidean distance."""
        # Compute pairwise distances
        distances = torch.zeros(len(weights_list))

        for i in range(len(weights_list)):
            for j in range(len(weights_list)):
                if i != j:
                    dist = 0
                    for key in weights_list[i].keys():
                        dist += torch.norm(weights_list[i][key] - weights_list[j][key])**2
                    distances[i] += dist

        # Select minimum
        best_idx = torch.argmin(distances)
        return weights_list[best_idx]
```

---

### 8. Add OOD Detection to Anomaly Module

**File**: Update `src/analysis/anomaly.py`

```python
class OODDetector:
    """Out-of-distribution detection."""

    def __init__(self, method: str = 'mahalanobis', threshold: float = 0.95):
        self.method = method
        self.threshold = threshold  # Percentile for anomaly threshold
        self.feature_mean = None
        self.feature_cov_inv = None

    def fit(self, features: np.ndarray):
        """Fit OOD detector on in-distribution features."""
        self.feature_mean = features.mean(axis=0)
        cov = np.cov(features.T)
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularization
        self.feature_cov_inv = np.linalg.inv(cov)

    def detect_ood(self, features: np.ndarray) -> np.ndarray:
        """Return True for OOD samples."""

        if self.method == 'mahalanobis':
            scores = self._mahalanobis_distance(features)
        elif self.method == 'odin':
            scores = self._odin_score(features)

        threshold_val = np.percentile(scores, self.threshold * 100)
        return scores > threshold_val

    def _mahalanobis_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance from mean."""
        diff = features - self.feature_mean
        distances = np.sqrt(np.sum(diff @ self.feature_cov_inv * diff, axis=1))
        return distances

    def _odin_score(self, logits: np.ndarray, temperature: float = 1000.0) -> np.ndarray:
        """ODIN score - based on softmax confidence."""
        # Higher ODIN score = more likely OOD
        exp_logits = np.exp(logits / temperature)
        softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        max_confidence = np.max(softmax, axis=1)
        return -max_confidence  # Negate so high = OOD
```

---

## Integration Points & Testing

### Test New Features

Create `tests/test_improvements.py`:
```python
import pytest
import torch
from src.model_registry import ModelRegistry
from src.federated.fed_avg import ByzantineRobustAggregator
from src.analysis.anomaly import OODDetector

def test_model_registry():
    """Test model registration and retrieval."""
    registry = ModelRegistry('/tmp/test_registry')
    # ... test code ...

def test_byzantine_aggregation():
    """Test robust federated aggregation."""
    aggregator = ByzantineRobustAggregator(method='median')
    # ... test code ...

def test_ood_detection():
    """Test out-of-distribution detection."""
    detector = OODDetector(method='mahalanobis')
    # ... test code ...
```

---

## Implementation Checklist

- [ ] Fix ViT patch size (1 hour)
- [ ] Add custom exceptions (30 min)
- [ ] Add type hints to remaining functions (1 hour)
- [ ] Implement model registry (2 hours)
- [ ] Complete synthetic data pipeline (2 hours)
- [ ] Add Byzantine-resistant aggregation (1.5 hours)
- [ ] Add OOD detection (1 hour)
- [ ] Create comprehensive tests (2 hours)
- [ ] Update documentation (1 hour)

**Total Estimated Time**: 12 hours

---

## Expected Improvements

After implementing all suggestions:
- **Code Quality**: A → A+
- **Security**: B- → B+ (federated learning)
- **Testing**: B- → A- (comprehensive tests)
- **Documentation**: B+ → A (feature guides)
- **Performance**: Potential 5-10% accuracy improvement (ViT fix)
- **Reliability**: Better error messages, fewer silent failures

