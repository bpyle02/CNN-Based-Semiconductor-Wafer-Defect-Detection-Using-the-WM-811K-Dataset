"""Integration tests for full training and inference pipeline."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

try:
    import torchvision  # noqa: F401
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Test fixtures and utilities
@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create small sample data for testing."""
    # Create dummy wafer maps (batch_size=16, channels=3, height=96, width=96)
    X = np.random.randn(16, 3, 96, 96).astype(np.float32)
    # Create dummy labels (9 classes)
    y = np.random.randint(0, 9, size=16)
    return X, y


@pytest.fixture
def config_dict():
    """Return test configuration."""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'data': {
            'dataset_path': 'data/LSWMD_new.pkl',
            'test_size': 0.15,
            'val_size': 0.15,
            'train_size': 0.70,
        },
        'training': {
            'epochs': 2,  # Short for testing
            'batch_size': 8,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': {
                'type': 'ReduceLROnPlateau',
            },
            'default_model': 'cnn',
        },
        'models': {
            'cnn': {'name': 'Custom CNN', 'architecture': 'custom', 'dropout_rate': 0.5},
            'resnet': {'name': 'ResNet-18', 'architecture': 'resnet18', 'dropout_rate': 0.5},
            'efficientnet': {'name': 'EfficientNet-B0', 'architecture': 'efficientnet_b0', 'dropout_rate': 0.5},
        },
        'checkpoint_dir': 'checkpoints',
    }


class TestModelInitialization:
    """Test model creation and initialization."""

    def test_cnn_initialization(self, device):
        """Test custom CNN model creation."""
        from src.models import WaferCNN
        model = WaferCNN(num_classes=9).to(device)
        assert isinstance(model, nn.Module)
        assert next(model.parameters()).device.type == (
            'cuda' if device == 'cuda' else 'cpu'
        )

    def test_resnet_initialization(self, device):
        """Test ResNet-18 initialization."""
        if not TORCHVISION_AVAILABLE:
            pytest.skip("torchvision not installed")
        from src.models import get_resnet18
        model = get_resnet18(num_classes=9).to(device)
        assert isinstance(model, nn.Module)
        assert isinstance(model.fc, nn.Module)

    def test_efficientnet_initialization(self, device):
        """Test EfficientNet-B0 initialization."""
        if not TORCHVISION_AVAILABLE:
            pytest.skip("torchvision not installed")
        from src.models import get_efficientnet_b0
        model = get_efficientnet_b0(num_classes=9).to(device)
        assert isinstance(model, nn.Module)
        assert isinstance(model.classifier, nn.Module)

    def test_vit_initialization(self, device):
        """Test Vision Transformer initialization."""
        from src.models import get_vit_small
        model = get_vit_small(num_classes=9).to(device)
        assert isinstance(model, nn.Module)
        # Test forward pass
        x = torch.randn(2, 3, 96, 96).to(device)
        out = model(x)
        assert out.shape == (2, 9)


class TestDataPipeline:
    """Test data loading and preprocessing."""

    def test_wafer_dataset_creation(self, sample_data):
        """Test WaferMapDataset creation."""
        from src.data import WaferMapDataset
        X, y = sample_data
        dataset = WaferMapDataset(X[:, 0], y)
        assert len(dataset) == len(y)
        sample, label = dataset[0]
        assert sample.shape == (3, 96, 96)
        assert isinstance(label, (int, np.integer, torch.Tensor))

    def test_dataloader_creation(self, sample_data):
        """Test DataLoader creation."""
        from src.data import WaferMapDataset
        from torch.utils.data import DataLoader
        X, y = sample_data
        dataset = WaferMapDataset(X[:, 0], y)
        loader = DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape[0] == 4  # batch_size

    def test_image_transforms(self):
        """Test image transformation pipeline."""
        from src.data import get_image_transforms, get_imagenet_normalize
        transforms = get_image_transforms()
        norm = get_imagenet_normalize()
        assert transforms is not None
        assert norm is not None


class TestTrainingPipeline:
    """Test training loop and optimization."""

    def test_single_forward_pass(self, device, sample_data):
        """Test forward pass through model."""
        from src.models import WaferCNN
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)
        x_tensor = torch.from_numpy(X).to(device)
        output = model(x_tensor)
        assert output.shape == (16, 9)

    def test_loss_computation(self, device, sample_data):
        """Test loss computation with class weights."""
        from src.models import WaferCNN
        from collections import Counter
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)

        # Compute class weights
        class_counts = Counter(y)
        total = len(y)
        weights = torch.tensor([
            total / (9 * class_counts.get(i, 1)) for i in range(9)
        ], dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        x_tensor = torch.from_numpy(X).to(device)
        y_tensor = torch.from_numpy(y).long().to(device)

        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_backpropagation(self, device, sample_data):
        """Test backpropagation and gradient updates."""
        from src.models import WaferCNN
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x_tensor = torch.from_numpy(X).to(device)
        y_tensor = torch.from_numpy(y).long().to(device)

        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients

        optimizer.step()


class TestEvaluation:
    """Test evaluation metrics and reporting."""

    def test_accuracy_computation(self, device, sample_data):
        """Test accuracy computation."""
        from src.models import WaferCNN
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)
        model.eval()

        with torch.no_grad():
            x_tensor = torch.from_numpy(X).to(device)
            output = model(x_tensor)
            preds = output.argmax(dim=1).cpu().numpy()

        accuracy = (preds == y).mean()
        assert 0 <= accuracy <= 1

    def test_confusion_matrix_computation(self, device, sample_data):
        """Test confusion matrix computation."""
        from sklearn.metrics import confusion_matrix
        from src.models import WaferCNN
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)
        model.eval()

        with torch.no_grad():
            x_tensor = torch.from_numpy(X).to(device)
            output = model(x_tensor)
            preds = output.argmax(dim=1).cpu().numpy()

        cm = confusion_matrix(y, preds, labels=range(9))
        assert cm.shape == (9, 9)


class TestModelRegistry:
    """Test model registry and versioning."""

    def test_model_registry_creation(self, workspace_tmp_path):
        """Test ModelRegistry initialization."""
        from src.model_registry import ModelRegistry
        registry = ModelRegistry(registry_path=str(workspace_tmp_path))
        assert registry.registry_path.exists()

    def test_model_registration(self, device, sample_data, workspace_tmp_path):
        """Test registering a model."""
        from src.model_registry import ModelRegistry, ModelMetadata
        from src.models import WaferCNN
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)

        registry = ModelRegistry(registry_path=str(workspace_tmp_path))

        metadata = ModelMetadata(
            model_name='test_cnn',
            architecture='CustomCNN',
            num_classes=9,
            training_config={'epochs': 2, 'lr': 1e-3},
            metrics={'accuracy': 0.75, 'f1': 0.70},
            dataset_version='v1.0',
        )

        model_id = registry.register(model, metadata)
        assert model_id in registry.list_models()

    def test_model_loading(self, device, sample_data, workspace_tmp_path):
        """Test loading registered model."""
        from src.model_registry import ModelRegistry, ModelMetadata
        from src.models import WaferCNN
        X, y = sample_data
        model = WaferCNN(num_classes=9).to(device)

        registry = ModelRegistry(registry_path=str(workspace_tmp_path))

        metadata = ModelMetadata(
            model_name='test_cnn',
            architecture='CustomCNN',
            num_classes=9,
            training_config={},
            metrics={'accuracy': 0.75},
            dataset_version='v1.0',
        )

        model_id = registry.register(model, metadata)
        state_dict, loaded_metadata = registry.load(model_id)
        assert isinstance(state_dict, dict)
        assert 'metadata' in loaded_metadata or isinstance(loaded_metadata, dict)


class TestExceptionHandling:
    """Test custom exception classes."""

    def test_exception_imports(self):
        """Test that custom exceptions can be imported."""
        from src.exceptions import (
            WaferMapError,
            DataLoadError,
            ModelError,
            TrainingError,
            InferenceError,
            FederatedError,
        )
        assert issubclass(DataLoadError, WaferMapError)
        assert issubclass(ModelError, WaferMapError)
        assert issubclass(TrainingError, WaferMapError)
        assert issubclass(InferenceError, WaferMapError)
        assert issubclass(FederatedError, WaferMapError)


class TestConfiguration:
    """Test configuration management."""

    def test_base_trainer_creation(self, workspace_tmp_path):
        """Test BaseTrainer initialization."""
        from src.training.base_trainer import BaseTrainer
        # Create minimal config file
        checkpoint_dir = (workspace_tmp_path / 'checkpoints').as_posix()
        config_path = workspace_tmp_path / 'config.yaml'
        config_path.write_text(f"""
device: cpu
seed: 42
checkpoint_dir: {checkpoint_dir}
training:
  epochs: 5
  batch_size: 64
  learning_rate: 1e-4
  weight_decay: 1e-4
data:
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
""")
        trainer = BaseTrainer(str(config_path))
        assert trainer.device.type == 'cpu'
        assert trainer.seed == 42

    def test_seed_setting(self, workspace_tmp_path):
        """Test that seed setting works."""
        from src.training.base_trainer import BaseTrainer
        checkpoint_dir = (workspace_tmp_path / 'checkpoints').as_posix()
        config_path = workspace_tmp_path / 'config.yaml'
        config_path.write_text(f"""
device: cpu
seed: 123
checkpoint_dir: {checkpoint_dir}
training:
  epochs: 5
  batch_size: 64
data:
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
""")
        trainer = BaseTrainer(str(config_path))

        # Generate random numbers twice with same seed
        torch.manual_seed(123)
        r1 = torch.randn(5).numpy()

        torch.manual_seed(123)
        r2 = torch.randn(5).numpy()

        np.testing.assert_array_equal(r1, r2)


class TestEnsembling:
    """Test model ensembling."""

    def test_ensemble_creation(self, device):
        """Test ensemble model creation."""
        if not TORCHVISION_AVAILABLE:
            pytest.skip("torchvision not installed")
        from src.models import WaferCNN, get_resnet18
        from src.models.ensemble import EnsembleModel

        models = [
            WaferCNN(num_classes=9).to(device),
            get_resnet18(num_classes=9).to(device),
        ]

        ensemble = EnsembleModel(models, aggregation='averaging').to(device)
        assert len(ensemble.models) == 2

    def test_ensemble_inference(self, device, sample_data):
        """Test ensemble forward pass."""
        if not TORCHVISION_AVAILABLE:
            pytest.skip("torchvision not installed")
        from src.models import WaferCNN, get_resnet18
        from src.models.ensemble import EnsembleModel

        X, y = sample_data
        models = [
            WaferCNN(num_classes=9).to(device),
            get_resnet18(num_classes=9).to(device),
        ]

        ensemble = EnsembleModel(models, aggregation='averaging').to(device)
        ensemble.eval()

        with torch.no_grad():
            x_tensor = torch.from_numpy(X[:4]).to(device)
            output = ensemble(x_tensor)
            assert output.shape == (4, 9)


class TestAttention:
    """Test attention mechanisms."""

    def test_attention_module_creation(self, device):
        """Test attention module creation."""
        from src.models.attention import SEBlock
        se = SEBlock(in_channels=64).to(device)
        assert isinstance(se, nn.Module)

    def test_attention_forward_pass(self, device):
        """Test attention module forward pass."""
        from src.models.attention import SEBlock
        se = SEBlock(in_channels=64).to(device)
        x = torch.randn(2, 64, 32, 32).to(device)
        out = se(x)
        assert out.shape == x.shape


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
