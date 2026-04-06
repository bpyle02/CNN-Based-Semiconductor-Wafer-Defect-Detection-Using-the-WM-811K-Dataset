"""Unit tests for merged config overlays."""

from src.config import load_merged_config


def test_base_and_train_overlays_merge():
    cfg = load_merged_config(["configs/base.yaml", "configs/train.yaml"])
    assert cfg.training.default_model == "all"
    assert cfg.training.batch_size == 64
    assert cfg.paths.checkpoint_dir == "checkpoints"


def test_base_and_inference_overlays_merge():
    cfg = load_merged_config(["configs/base.yaml", "configs/inference.yaml"])
    assert cfg.inference.device == "cpu"
    assert cfg.inference.gradcam.enabled is True
    assert cfg.device == "cuda"
