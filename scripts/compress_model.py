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
    python compress_model.py --student cnn --teacher resnet --method distill
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.analysis import evaluate_model
from src.config import load_config
from src.training.base_trainer import BaseTrainer


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: str = 'cuda') -> nn.Module:
    """Load model weights from checkpoint."""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Warning: Checkpoint not found at {checkpoint_path}")
        return model
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {checkpoint_path}")
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def quantize_model(model: nn.Module, dtype: str = 'int8') -> nn.Module:
    """Quantize model to lower precision."""
    logger.info(f"\nQuantizing model to {dtype}...")

    if dtype == 'int8':
        # Use PyTorch's quantization
        model_q = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
    elif dtype == 'float16':
        # Convert to FP16
        model_q = model.half()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    total, _ = count_parameters(model)
    total_q, _ = count_parameters(model_q)

    logger.info(f"Model size: {total * 4 / 1e6:.2f}M -> {total_q * 2 / 1e6:.2f}M (approx)")
    logger.info(f"Compression ratio: {total / total_q:.2f}x")

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
            threshold = np.percentile(
                np.abs(module.weight.data.cpu().numpy()), sparsity * 100
            )

            # Prune weights below threshold
            mask = torch.abs(module.weight.data) > threshold
            pruned_params += (~mask).sum().item()
            module.weight.data *= mask.float()

    logger.info(f"Pruned {pruned_params:,} / {total_params:,} parameters ({pruned_params / total_params * 100:.2f}%)")

    return model


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    epochs: int = 5,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> nn.Module:
    """Knowledge distillation: train student from teacher."""
    logger.info(f"\nKnowledge Distillation (T={temperature}, α={alpha})...")

    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    student.train()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.parameters(), lr=1e-4)

    best_val_f1 = 0.0
    student_checkpoints = []

    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Student forward
            student_logits = student(inputs)

            # Loss: blend CE and KD
            loss_ce = criterion_ce(student_logits, targets)
            loss_kd = criterion_kd(
                torch.nn.functional.log_softmax(student_logits / temperature, dim=1),
                torch.nn.functional.softmax(teacher_logits / temperature, dim=1),
            )
            loss = alpha * loss_ce + (1 - alpha) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = student(inputs)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        student_checkpoints.append(student.state_dict())

        logger.info(f"  Epoch {epoch + 1}/{epochs}: Loss={train_loss / len(train_loader):.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        student.train()

    # Load best checkpoint
    student.load_state_dict(student_checkpoints[-1])
    student.eval()

    return student


def benchmark_inference(model: nn.Module, test_loader: DataLoader, device: str = 'cuda', num_runs: int = 10) -> Tuple[float, float]:
    """Benchmark inference speed."""
    logger.info(f"\nBenchmarking inference speed ({num_runs} runs)...")

    model = model.to(device)
    model.eval()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            for inputs, _ in test_loader:
                inputs = inputs.to(device)

                # Warm up
                _ = model(inputs)

                # Time
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start = time.time()
                _ = model(inputs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times.append(time.time() - start)

    avg_time = np.mean(times)
    throughput = len(inputs) / avg_time

    logger.info(f"  Avg latency: {avg_time * 1000:.2f}ms")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")

    return avg_time, throughput


def main() -> int:
    """Main compression entry point."""
    trainer = BaseTrainer("config.yaml")
    config = trainer.config

    parser = argparse.ArgumentParser(description='Model compression toolkit')
    parser.add_argument(
        '--method',
        choices=['quantize', 'prune', 'distill'],
        default='quantize',
        help='Compression method'
    )
    parser.add_argument('--model', choices=['cnn', 'resnet', 'effnet'], default='cnn',
                        help='Model to compress (for quantize/prune)')
    parser.add_argument('--student', default='cnn', help='Student model (for distill)')
    parser.add_argument('--teacher', default='resnet', help='Teacher model (for distill)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--dtype', choices=['int8', 'float16'], default='int8',
                        help='Quantization dtype')
    parser.add_argument('--sparsity', type=float, default=0.3, help='Pruning sparsity (0-1)')
    parser.add_argument('--temperature', type=float, default=4.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation alpha')
    parser.add_argument('--epochs', type=int, default=5, help='Distillation epochs')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.device)
    parser.add_argument('--output', type=str, default=None, help='Output checkpoint path')

    args = parser.parse_args()

    device = args.device
    logger.info(f"Device: {device}")

    if args.method == 'quantize':
        logger.info(f"\n{'='*70}")
        logger.info(f"QUANTIZATION: {args.model}")
        logger.info(f"{'='*70}")

        # Create model
        num_classes = 9
        if args.model == 'cnn':
            model = WaferCNN(num_classes=num_classes)
        elif args.model == 'resnet':
            model = get_resnet18(num_classes=num_classes)
        else:
            model = get_efficientnet_b0(num_classes=num_classes)

        # Load weights if available
        if args.checkpoint:
            model = load_checkpoint(args.checkpoint, model, device)

        # Quantize
        model_q = quantize_model(model, dtype=args.dtype)

        # Save
        if args.output:
            torch.save(model_q.state_dict(), args.output)
            logger.info(f"Saved quantized model to {args.output}")

    elif args.method == 'prune':
        logger.info(f"\n{'='*70}")
        logger.info(f"PRUNING: {args.model}")
        logger.info(f"{'='*70}")

        # Create model
        num_classes = 9
        if args.model == 'cnn':
            model = WaferCNN(num_classes=num_classes)
        elif args.model == 'resnet':
            model = get_resnet18(num_classes=num_classes)
        else:
            model = get_efficientnet_b0(num_classes=num_classes)

        # Load weights
        if args.checkpoint:
            model = load_checkpoint(args.checkpoint, model, device)

        # Prune
        model_p = prune_model(model, sparsity=args.sparsity)

        # Save
        if args.output:
            torch.save(model_p.state_dict(), args.output)
            logger.info(f"Saved pruned model to {args.output}")

    elif args.method == 'distill':
        logger.info(f"\n{'='*70}")
        logger.info(f"KNOWLEDGE DISTILLATION: {args.student} <- {args.teacher}")
        logger.info(f"{'='*70}")

        # Create models
        num_classes = 9
        if args.student == 'cnn':
            student = WaferCNN(num_classes=num_classes)
        elif args.student == 'resnet':
            student = get_resnet18(num_classes=num_classes)
        else:
            student = get_efficientnet_b0(num_classes=num_classes)

        if args.teacher == 'cnn':
            teacher = WaferCNN(num_classes=num_classes)
        elif args.teacher == 'resnet':
            teacher = get_resnet18(num_classes=num_classes)
        else:
            teacher = get_efficientnet_b0(num_classes=num_classes)

        # Load teacher weights
        logger.info(f"Note: Provide teacher checkpoint via --checkpoint")
        if args.checkpoint:
            teacher = load_checkpoint(args.checkpoint, teacher, device)

        logger.info("Distillation requires training data loaders. Run this with actual training loop.")
        logger.info("Example: See progressive_train.py for dataloader setup")

    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(main())
