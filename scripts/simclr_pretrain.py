#!/usr/bin/env python3
"""Self-supervised pretraining with SimCLR on unlabeled wafer maps.

Creates a SimCLR encoder around a WaferCNN backbone, trains using
contrastive learning (NT-Xent loss), and saves the pretrained backbone
weights for downstream fine-tuning.

Usage:
    python scripts/simclr_pretrain.py --epochs 10 --device cuda
    python scripts/simclr_pretrain.py --output-path checkpoints/simclr_backbone.pth --batch-size 128
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def build_paired_loader(data_path: str = None, batch_size: int = 64):
    """Build a DataLoader that yields duplicate batches for contrastive views.

    SimCLR requires two augmented views per sample. We rely on random
    augmentation in the dataset transform so that each call to __getitem__
    produces a different augmented view. The trainer handles splitting.
    """
    from src.data.dataset import load_dataset, KNOWN_CLASSES
    from src.data.preprocessing import WaferMapDataset, get_image_transforms

    df = load_dataset(Path(data_path) if data_path else None)
    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    labels = le.transform(df["failureClass"].values)
    maps = df["waferMap"].tolist()

    # Use augmentation so each access produces a different view
    transform = get_image_transforms(augment=True)
    dataset = WaferMapDataset(maps, labels, transform=transform)

    # Double the batch: first half = view_i, second half = view_j
    # SimCLRPretrainer.train_epoch splits on batch_size//2
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=True, drop_last=True)
    return loader


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SimCLR self-supervised pretraining")
    parser.add_argument("--data-path", default=None, help="Path to LSWMD_new.pkl")
    parser.add_argument("--epochs", type=int, default=10, help="Pretraining epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-view batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent temperature")
    parser.add_argument("--output-path", default="checkpoints/simclr_backbone.pth",
                        help="Where to save backbone weights")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        from src.training.simclr import SimCLREncoder, SimCLRPretrainer
        from src.models import WaferCNN
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}. Install project requirements first.")
        return 1

    logger.info("Building dataloader for contrastive pretraining...")
    train_loader = build_paired_loader(args.data_path, args.batch_size)
    logger.info(f"Dataloader: {len(train_loader)} batches (batch_size={args.batch_size * 2})")

    # Build backbone (WaferCNN without classification head output)
    backbone = WaferCNN(num_classes=9)
    feature_dim = backbone.feature_channels[-1]  # 256 by default

    logger.info(f"Building SimCLR encoder (feature_dim={feature_dim})...")
    encoder = SimCLREncoder(backbone, feature_dim=feature_dim, projection_dim=128)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)

    pretrainer = SimCLRPretrainer(
        encoder=encoder,
        optimizer=optimizer,
        device=args.device,
        temperature=args.temperature,
    )

    logger.info(f"Starting SimCLR pretraining for {args.epochs} epochs...")
    losses = pretrainer.pretrain(train_loader, epochs=args.epochs)

    # Save backbone weights
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pretrained_backbone = pretrainer.get_backbone()
    torch.save(pretrained_backbone.state_dict(), output_path)

    logger.info(f"\n{'='*50}")
    logger.info(f"SimCLR Pretraining Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Epochs:              {args.epochs}")
    logger.info(f"Final loss:          {losses[-1]:.6f}")
    logger.info(f"Best loss:           {min(losses):.6f}")
    logger.info(f"Backbone saved to:   {output_path}")
    logger.info(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
