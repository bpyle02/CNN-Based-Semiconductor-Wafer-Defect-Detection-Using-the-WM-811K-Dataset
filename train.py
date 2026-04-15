#!/usr/bin/env python3
"""CLI entry point for training wafer defect models.

Usage:
    python train.py --model cnn --epochs 5 --device cuda
    python train.py --model all --epochs 5 --device cpu

This file is intentionally thin: all heavy lifting lives in
``src/data/pipeline.py``, ``src/data/loaders.py``, ``src/training/builders.py``,
and ``src/training/pipeline.py``. External consumers that import names such as
``build_model`` or ``load_and_preprocess_data`` from ``train`` continue to
work because those names are re-exported here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, load_merged_config
from src.data.loaders import (  # noqa: F401 re-exported
    create_test_loader,
    create_train_loader,
    create_val_loader,
)
from src.data.pipeline import SEED, load_and_preprocess_data  # noqa: F401 re-exported
from src.training.builders import (  # noqa: F401 re-exported
    DEFAULT_SCHEDULER_CONFIG,
    build_model,
    build_optimizer,
    build_scheduler,
)
from src.training.pipeline import TrainingPipeline, set_seed  # noqa: F401 re-exported


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser. Kept isolated for testability."""
    parser = argparse.ArgumentParser(description="Train wafer defect detection models")
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "cnn_fpn",
            "resnet",
            "efficientnet",
            "effnet",
            "vit",
            "swin",
            "ride",
            "all",
        ],
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="Balance rare classes with synthetic augmentation"
    )
    parser.add_argument(
        "--mixup", action="store_true", help="Enable Mixup/CutMix batch augmentation"
    )
    parser.add_argument(
        "--balanced-sampling", action="store_true", help="Enable class-balanced batch sampling"
    )
    parser.add_argument("--distributed", action="store_true", help="Enable DataParallel multi-GPU")
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Run MC Dropout uncertainty estimation after training",
    )
    parser.add_argument(
        "--cost-sensitive",
        action="store_true",
        help="Use CostSensitiveCE loss with WM-811K cost matrix " "(overrides loss.type)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Load pretrained backbone (e.g. from SimCLR)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Configuration file path. Repeat to merge overlays in order.",
    )
    return parser


def _load_config(config_paths):
    """Resolve --config overlays, falling back to config.yaml if present."""
    if config_paths:
        if len(config_paths) == 1:
            config = load_config(str(config_paths[0]))
        else:
            config = load_merged_config(config_paths)
        logger.info(
            "Loaded defaults from %s",
            ", ".join(str(path) for path in config_paths),
        )
        return config
    if Path("config.yaml").exists():
        logger.info("Loaded defaults from config.yaml")
        return load_config("config.yaml")
    logger.info("No config.yaml found, using hardcoded defaults")
    return None


def _embed_run_id(results_dir: Path, run_id: str) -> None:
    """Embed run_id into the metrics.json produced by the pipeline, if present."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics_payload, dict):
            return
        metrics_payload["run_id"] = run_id
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
        logger.warning("Could not embed run_id into %s: %s", metrics_path, exc)


def main():
    """Main training entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    from src.utils.logging_setup import setup_logging
    from src.utils.seeding import seed_everything

    run_id = setup_logging(level="INFO")
    seed_everything(args.seed if args.seed is not None else SEED)
    logger.info("Starting run run_id=%s model=%s", run_id, args.model)

    config = _load_config(args.config)

    pipeline = TrainingPipeline(args, config)
    exit_code = pipeline.run()

    results_dir = getattr(pipeline, "results_dir", None)
    if results_dir is not None:
        _embed_run_id(results_dir, run_id)

    return exit_code


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.exit(main())
