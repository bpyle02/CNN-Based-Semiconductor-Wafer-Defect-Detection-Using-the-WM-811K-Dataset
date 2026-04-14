"""Unit tests for src/models/cnn.py, vit.py, pretrained.py, ensemble.py."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn import WaferCNN, count_parameters
from src.models.vit import ViT, get_vit_tiny, get_vit_small
from src.models.pretrained import get_resnet18, get_efficientnet_b0, get_frozen_params
from src.models.ensemble import EnsembleModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch():
    """Small batch of dummy images."""
    return torch.randn(2, 3, 96, 96)


@pytest.fixture
def cnn():
    return WaferCNN(num_classes=9)


# ---------------------------------------------------------------------------
# WaferCNN
# ---------------------------------------------------------------------------

class TestWaferCNN:
    def test_forward_shape(self, cnn, batch):
        out = cnn(batch)
        assert out.shape == (2, 9)

    def test_single_sample(self, cnn):
        out = cnn(torch.randn(1, 3, 96, 96))
        assert out.shape == (1, 9)

    def test_custom_num_classes(self, batch):
        model = WaferCNN(num_classes=4)
        out = model(batch)
        assert out.shape == (2, 4)

    def test_parameter_count_positive(self, cnn):
        total, trainable = count_parameters(cnn)
        assert total > 0
        assert trainable > 0

    def test_all_params_trainable(self, cnn):
        """Custom CNN has no frozen layers."""
        total, trainable = count_parameters(cnn)
        assert total == trainable

    def test_output_dtype(self, cnn, batch):
        out = cnn(batch)
        assert out.dtype == torch.float32

    def test_kaiming_init_conv_weights(self, cnn):
        """Conv layer weights should have reasonable statistics after kaiming init."""
        for module in cnn.modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                # Kaiming normal with fan_out produces std ~ sqrt(2/fan_out)
                # Just verify it's not all zeros or constant
                assert w.std() > 0.01
                assert w.mean().abs() < 0.5
                break  # checking first conv is sufficient

    def test_batchnorm_init(self, cnn):
        """BatchNorm weight=1, bias=0 after init."""
        for module in cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert torch.allclose(module.weight.data, torch.ones_like(module.weight.data))
                assert torch.allclose(module.bias.data, torch.zeros_like(module.bias.data))
                break

    def test_can_disable_batchnorm(self, batch):
        model = WaferCNN(num_classes=9, use_batch_norm=False)
        out = model(batch)
        assert out.shape == (2, 9)
        assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class TestResNet18:
    def test_forward_shape(self, batch):
        model = get_resnet18(num_classes=9, pretrained=False)
        out = model(batch)
        assert out.shape == (2, 9)

    def test_frozen_params_exist(self):
        model = get_resnet18(num_classes=9, pretrained=False)
        frozen = get_frozen_params(model)
        assert len(frozen) > 0, "ResNet-18 should have frozen parameters"

    def test_layer4_unfrozen(self):
        model = get_resnet18(num_classes=9, pretrained=False)
        for name, param in model.named_parameters():
            if name.startswith('layer4'):
                assert param.requires_grad, f"{name} should be unfrozen"

    def test_early_layers_frozen(self):
        model = get_resnet18(num_classes=9, pretrained=False)
        for name, param in model.named_parameters():
            if name.startswith('layer1') or name.startswith('layer2') or name.startswith('layer3'):
                assert not param.requires_grad, f"{name} should be frozen"

    def test_fc_head_unfrozen(self):
        model = get_resnet18(num_classes=9, pretrained=False)
        for name, param in model.named_parameters():
            if name.startswith('fc'):
                assert param.requires_grad, f"{name} should be unfrozen"

    def test_no_freezing_when_boundary_disabled(self):
        model = get_resnet18(num_classes=9, pretrained=False, freeze_until=None)
        assert len(get_frozen_params(model)) == 0


# ---------------------------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------------------------

class TestEfficientNetB0:
    def test_forward_shape(self, batch):
        model = get_efficientnet_b0(num_classes=9, pretrained=False)
        out = model(batch)
        assert out.shape == (2, 9)

    def test_frozen_params_exist(self):
        model = get_efficientnet_b0(num_classes=9, pretrained=False)
        frozen = get_frozen_params(model)
        assert len(frozen) > 0, "EfficientNet-B0 should have frozen parameters"

    def test_classifier_unfrozen(self):
        model = get_efficientnet_b0(num_classes=9, pretrained=False)
        for name, param in model.named_parameters():
            if name.startswith('classifier'):
                assert param.requires_grad, f"{name} should be unfrozen"


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------

class TestViT:
    def test_vit_tiny_forward_shape(self, batch):
        model = get_vit_tiny(num_classes=9)
        out = model(batch)
        assert out.shape == (2, 9)

    def test_vit_small_forward_shape(self):
        model = get_vit_small(num_classes=9)
        out = model(torch.randn(1, 3, 96, 96))
        assert out.shape == (1, 9)

    def test_custom_vit(self):
        model = ViT(
            image_size=96,
            patch_size=16,
            in_channels=3,
            num_classes=5,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            mlp_dim=256,
        )
        out = model(torch.randn(2, 3, 96, 96))
        assert out.shape == (2, 5)

    def test_vit_has_parameters(self):
        model = get_vit_tiny(num_classes=9)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_patch_count(self):
        model = ViT(image_size=96, patch_size=8)
        expected_patches = (96 // 8) ** 2  # 144
        assert model.num_patches == expected_patches


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    @pytest.fixture
    def two_cnns(self):
        m1 = WaferCNN(num_classes=9)
        m2 = WaferCNN(num_classes=9)
        return [m1, m2]

    def test_averaging_forward(self, two_cnns, batch):
        ensemble = EnsembleModel(two_cnns, aggregation="averaging")
        ensemble.eval()
        out = ensemble(batch)
        assert out.shape == (2, 9)

    def test_voting_forward(self, two_cnns, batch):
        ensemble = EnsembleModel(two_cnns, aggregation="voting")
        ensemble.eval()
        out = ensemble(batch)
        assert out.shape == (2, 9)

    def test_weighted_averaging_forward(self, two_cnns, batch):
        ensemble = EnsembleModel(two_cnns, aggregation="weighted_averaging", weights=[0.7, 0.3])
        ensemble.eval()
        out = ensemble(batch)
        assert out.shape == (2, 9)

    def test_weighted_averaging_default_weights(self, two_cnns, batch):
        """Without explicit weights, they default to uniform."""
        ensemble = EnsembleModel(two_cnns, aggregation="weighted_averaging")
        ensemble.eval()
        out = ensemble(batch)
        assert out.shape == (2, 9)

    def test_invalid_aggregation_raises(self, two_cnns):
        with pytest.raises(AssertionError, match="Unknown aggregation"):
            EnsembleModel(two_cnns, aggregation="invalid_method")

    def test_voting_output_is_onehot_like(self, two_cnns, batch):
        """Voting returns one-hot logits (exactly one 1.0 per row)."""
        ensemble = EnsembleModel(two_cnns, aggregation="voting")
        ensemble.eval()
        out = ensemble(batch)
        # Each row should have exactly one 1.0 and rest 0.0
        for row in out:
            assert row.sum().item() == pytest.approx(1.0, abs=1e-6)
            assert row.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_num_models(self, two_cnns):
        ensemble = EnsembleModel(two_cnns, aggregation="averaging")
        assert ensemble.num_models == 2

    def test_eval_mode_propagates(self, two_cnns):
        ensemble = EnsembleModel(two_cnns, aggregation="averaging")
        ensemble.eval()
        for model in ensemble.models:
            assert not model.training

    def test_train_mode_propagates(self, two_cnns):
        ensemble = EnsembleModel(two_cnns, aggregation="averaging")
        ensemble.eval()
        ensemble.train()
        for model in ensemble.models:
            assert model.training
