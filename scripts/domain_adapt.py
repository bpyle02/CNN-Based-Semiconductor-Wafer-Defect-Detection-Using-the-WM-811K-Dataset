#!/usr/bin/env python3
"""Domain adaptation for cross-plant wafer defect detection.

Loads a pretrained model and adapts it from a source domain to a target
domain using fine-tuning, CORAL alignment, or adversarial training.

Usage:
    python scripts/domain_adapt.py --checkpoint checkpoints/best_cnn.pth --model-type cnn \
        --source-data data/source.pkl --target-data data/target.pkl --method coral
    python scripts/domain_adapt.py --checkpoint checkpoints/best_resnet.pth --model-type resnet \
        --source-data data/LSWMD_new.pkl --target-data data/LSWMD_new.pkl --method finetune
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

METHOD_MAP = {"finetune": "fine_tuning", "coral": "coral", "adversarial": "adversarial"}


def load_model(model_type: str, checkpoint_path: str, device: str) -> nn.Module:
    """Load a model from checkpoint."""
    from src.models import WaferCNN, get_resnet18, get_efficientnet_b0

    if model_type == "cnn":
        model = WaferCNN(num_classes=9)
    elif model_type == "resnet":
        model = get_resnet18(num_classes=9, pretrained=False, freeze_until=None)
    elif model_type == "efficientnet":
        model = get_efficientnet_b0(num_classes=9, pretrained=False, freeze_until=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"])
    model.to(device)
    return model


def build_loader(data_path: str, batch_size: int = 64):
    """Build a DataLoader from a wafer map pickle file."""
    from src.data.dataset import load_dataset, KNOWN_CLASSES
    from src.data.preprocessing import WaferMapDataset, get_image_transforms

    df = load_dataset(Path(data_path))
    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    labels = le.transform(df["failureClass"].values)
    maps = df["waferMap"].tolist()

    transform = get_image_transforms(augment=True)
    dataset = WaferMapDataset(maps, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Domain adaptation for wafer defect models")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--model-type", required=True, choices=["cnn", "resnet", "efficientnet"])
    parser.add_argument("--source-data", required=True, help="Path to source domain pickle")
    parser.add_argument("--target-data", required=True, help="Path to target domain pickle")
    parser.add_argument("--method", default="coral", choices=["finetune", "coral", "adversarial"],
                        help="Adaptation method (default: coral)")
    parser.add_argument("--epochs", type=int, default=5, help="Adaptation epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-path", default="checkpoints/adapted_model.pth",
                        help="Where to save the adapted model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        from src.training.domain_adaptation import DomainAdaptationTrainer
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}. Install project requirements first.")
        return 1

    logger.info(f"Loading {args.model_type} model from {args.checkpoint}")
    model = load_model(args.model_type, args.checkpoint, args.device)

    logger.info(f"Loading source data from {args.source_data}")
    source_loader = build_loader(args.source_data, args.batch_size)
    logger.info(f"Loading target data from {args.target_data}")
    target_loader = build_loader(args.target_data, args.batch_size)
    logger.info(f"Source batches: {len(source_loader)}, Target batches: {len(target_loader)}")

    internal_method = METHOD_MAP[args.method]
    logger.info(f"Initializing DomainAdaptationTrainer (method={internal_method})")
    trainer = DomainAdaptationTrainer(model=model, method=internal_method, device=args.device)
    trainer.prepare_for_adaptation(freeze_backbone=True)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if trainer.discriminator is not None:
        trainable_params += list(trainer.discriminator.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)

    logger.info(f"Running {args.method} adaptation for {args.epochs} epochs...")
    history = trainer.adapt(
        source_loader=source_loader,
        target_loader=target_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
    )

    # Save adapted model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    logger.info(f"\n{'='*50}")
    logger.info(f"Domain Adaptation Complete ({args.method})")
    logger.info(f"{'='*50}")
    if history:
        final_loss = history.get("loss", [None])[-1]
        logger.info(f"Final loss:          {final_loss:.6f}" if final_loss else "Final loss: N/A")
        if "coral_loss" in history:
            logger.info(f"Final CORAL loss:    {history['coral_loss'][-1]:.6f}")
        if "disc_loss" in history:
            logger.info(f"Final disc loss:     {history['disc_loss'][-1]:.6f}")
    logger.info(f"Model saved to:      {output_path}")
    logger.info(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
