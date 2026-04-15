#!/usr/bin/env python3
"""
Model compression toolkit: quantization, pruning, and knowledge distillation.

Reduces model size and inference latency while maintaining accuracy.

Techniques:
- Quantization: Convert FP32 to INT8 (4x size reduction)
- Pruning: Remove small-magnitude weights (typically 30-50% sparsity)
- Knowledge Distillation: Train compact student from powerful teacher

Usage:
    python compress_model.py --model cnn --method quantize --checkpoint checkpoints/best_cnn.pth
    python compress_model.py --model resnet --method prune --sparsity 0.3
    python compress_model.py --student cnn --teacher resnet --method distill --teacher-checkpoint checkpoints/best_resnet.pth

References:
    [42] Hinton et al. (2015). "Distilling the Knowledge in a Neural Network". arXiv:1503.02531
    [43] Han et al. (2016). "Deep Compression". arXiv:1510.00149
    [44] Jacob et al. (2018). "Quantization for Efficient Inference". arXiv:1712.05877
    [59] (2021). "Light-Weight CNN for Wafer Map Defect Detection"
    [106] Touvron et al. (2021). "DeiT III: Efficient Training". arXiv:2204.07118
    [141] (2021). "Deploying ML Models in Semiconductor Manufacturing"
    [142] (2020). "TensorRT Optimization for Production Inference"
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import evaluate_model
from src.config import load_config
from src.data import (
    KNOWN_CLASSES,
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    load_dataset,
    preprocess_wafer_maps,
    seed_worker,
)
from src.models import WaferCNN, get_efficientnet_b0, get_resnet18
from src.training.trainer import train_model


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: str = "cuda") -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"Warning: Checkpoint not found at {checkpoint_path}")
        return model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle both raw state_dicts and wrapped dictionaries
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {checkpoint_path}")
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def quantize_model(model: nn.Module, dtype: str = "int8") -> nn.Module:
    """Quantize model to lower precision."""
    logger.info(f"\nQuantizing model to {dtype}...")

    if dtype == "int8":
        # Use PyTorch's quantization
        model_q = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
    elif dtype == "float16":
        # Convert to FP16
        model_q = model.half()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    total, _ = count_parameters(model)
    total_q, _ = count_parameters(model_q)

    logger.info(f"Model size reduced significantly (dtype={dtype})")
    return model_q


def prune_model(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
    """Prune model by removing small-magnitude weights."""
    logger.info(f"\nPruning model to {sparsity * 100:.1f}% sparsity...")

    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_params += module.weight.numel()

            # Compute threshold
            threshold = np.percentile(np.abs(module.weight.data.cpu().numpy()), sparsity * 100)

            # Prune weights below threshold
            mask = torch.abs(module.weight.data) > threshold
            pruned_params += (~mask).sum().item()
            module.weight.data *= mask.float()

    logger.info(
        f"Pruned {pruned_params:,} / {total_params:,} parameters ({pruned_params / total_params * 100:.2f}%)"
    )

    return model


class DistillationTrainer:
    """Handles Knowledge Distillation training."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        self.teacher.eval()

    def train(self, epochs: int = 5, lr: float = 1e-3):
        logger.info(
            f"\nStarting Distillation (T={self.temperature}, α={self.alpha}, {epochs} epochs)..."
        )

        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction="batchmean")

        history = []

        for epoch in range(epochs):
            self.student.train()
            train_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

                student_logits = self.student(inputs)

                loss_ce = criterion_ce(student_logits, targets)
                loss_kd = criterion_kd(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                ) * (self.temperature**2)

                loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.student.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.student(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_acc = correct / total
            logger.info(
                f"  Epoch {epoch+1}/{epochs}: Loss={train_loss/len(self.train_loader):.4f}, Val Acc={val_acc:.4f}"
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss / len(self.train_loader),
                    "val_acc": val_acc,
                }
            )

        return self.student, history


def get_data_loaders(dataset_path: str, batch_size: int = 64, model_type: str = "cnn"):
    """Prepare data loaders for distillation."""
    df = load_dataset(Path(dataset_path))
    labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    labels = le.fit_transform(df_clean["failureClass"])
    maps = df_clean["waferMap"].values

    X_train, X_val, y_train, y_val = train_test_split(
        maps, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Preprocess
    X_train = np.array(preprocess_wafer_maps(X_train.tolist()))
    X_val = np.array(preprocess_wafer_maps(X_val.tolist()))

    if model_type == "cnn":
        transform = get_image_transforms()
    else:
        transform = torch.nn.Sequential(get_image_transforms(), get_imagenet_normalize())

    train_ds = WaferMapDataset(X_train, y_train, transform=transform)
    val_ds = WaferMapDataset(
        X_val, y_val, transform=None if model_type == "cnn" else get_imagenet_normalize()
    )

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g
    )

    return train_loader, val_loader


def main() -> int:
    """Main compression entry point."""
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(description="Model compression toolkit")
    parser.add_argument(
        "--method",
        choices=["quantize", "prune", "distill"],
        default="quantize",
        help="Compression method",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "effnet"],
        default="cnn",
        help="Model to compress (for quantize/prune)",
    )
    parser.add_argument("--student", default="cnn", help="Student model architecture (for distill)")
    parser.add_argument(
        "--teacher", default="resnet", help="Teacher model architecture (for distill)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (or teacher checkpoint if distilling)",
    )
    parser.add_argument(
        "--dtype", choices=["int8", "float16"], default="int8", help="Quantization dtype"
    )
    parser.add_argument("--sparsity", type=float, default=0.3, help="Pruning sparsity (0-1)")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="Distillation alpha")
    parser.add_argument("--epochs", type=int, default=5, help="Distillation epochs")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=config.device)
    parser.add_argument("--data-path", type=str, default=config.data.dataset_path)
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")

    args = parser.parse_args()

    device = args.device
    logger.info(f"Device: {device}")

    if args.method == "quantize":
        logger.info(f"\n{'='*70}")
        logger.info(f"QUANTIZATION: {args.model}")
        logger.info(f"{'='*70}")

        model = (
            WaferCNN(num_classes=9)
            if args.model == "cnn"
            else (
                get_resnet18(num_classes=9)
                if args.model == "resnet"
                else get_efficientnet_b0(num_classes=9)
            )
        )
        if args.checkpoint:
            model = load_checkpoint(args.checkpoint, model, device)

        model_q = quantize_model(model, dtype=args.dtype)
        if args.output:
            torch.save(model_q.state_dict(), args.output)
            logger.info(f"Saved quantized model to {args.output}")

    elif args.method == "prune":
        logger.info(f"\n{'='*70}")
        logger.info(f"PRUNING: {args.model}")
        logger.info(f"{'='*70}")

        model = (
            WaferCNN(num_classes=9)
            if args.model == "cnn"
            else (
                get_resnet18(num_classes=9)
                if args.model == "resnet"
                else get_efficientnet_b0(num_classes=9)
            )
        )
        if args.checkpoint:
            model = load_checkpoint(args.checkpoint, model, device)

        model_p = prune_model(model, sparsity=args.sparsity)
        if args.output:
            torch.save(model_p.state_dict(), args.output)
            logger.info(f"Saved pruned model to {args.output}")

    elif args.method == "distill":
        logger.info(f"\n{'='*70}")
        logger.info(f"KNOWLEDGE DISTILLATION: {args.student} <- {args.teacher}")
        logger.info(f"{'='*70}")

        student = (
            WaferCNN(num_classes=9)
            if args.student == "cnn"
            else (
                get_resnet18(num_classes=9)
                if args.student == "resnet"
                else get_efficientnet_b0(num_classes=9)
            )
        )
        teacher = (
            WaferCNN(num_classes=9)
            if args.teacher == "cnn"
            else (
                get_resnet18(num_classes=9)
                if args.teacher == "resnet"
                else get_efficientnet_b0(num_classes=9)
            )
        )

        if not args.checkpoint:
            logger.error("Distillation requires a teacher checkpoint. Use --checkpoint")
            return 1

        teacher = load_checkpoint(args.checkpoint, teacher, device)
        train_loader, val_loader = get_data_loaders(args.data_path, model_type=args.student)

        distiller = DistillationTrainer(
            teacher,
            student,
            train_loader,
            val_loader,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha,
        )
        student, _ = distiller.train(epochs=args.epochs)

        if args.output:
            torch.save(student.state_dict(), args.output)
            logger.info(f"Saved distilled student model to {args.output}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.exit(main())
