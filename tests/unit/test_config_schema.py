"""Unit tests for the canonical repository configuration schema."""

import pytest
from pydantic import ValidationError

from src.config import Config, canonicalize_model_name, load_config


class TestConfigSchema:
    def test_repo_config_loads_all_major_sections(self):
        cfg = load_config("config.yaml")

        assert cfg.training.default_model == "all"
        assert cfg.training.mixed_precision is True  # AMP is the repo default
        assert cfg.training.loss.label_smoothing == 0.0
        assert cfg.models.efficientnet.architecture == "efficientnet_b0"
        assert cfg.models.cnn.input_channels == 3
        assert cfg.validation.compute_gradcam is True
        assert cfg.inference.gradcam.enabled is True
        assert cfg.paths.checkpoint_dir == cfg.checkpoint_dir == "checkpoints"
        assert cfg.device == "cuda"
        assert cfg.to_dict()["device"]["type"] == "cuda"

    def test_model_alias_is_normalized(self):
        cfg = Config.from_dict(
            {
                "training": {
                    "default_model": "effnet",
                }
            }
        )
        assert cfg.training.default_model == "efficientnet"
        assert canonicalize_model_name("effnet") == "efficientnet"

    def test_unknown_config_keys_are_rejected(self):
        with pytest.raises(ValidationError):
            Config.from_dict(
                {
                    "data": {
                        "dataset_path": "data/LSWMD_new.pkl",
                        "unexpected_field": True,
                    }
                }
            )
