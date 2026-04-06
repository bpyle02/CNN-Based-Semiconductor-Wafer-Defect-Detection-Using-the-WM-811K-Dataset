#!/usr/bin/env python3
"""
Refactored training script using BaseTrainer and ModelRegistry.

This is an alternative to the primary CLI entry point (root train.py).
Use root train.py for standard training; use this script when you need
BaseTrainer integration or ModelRegistry-based model management.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset, preprocess_wafer_maps, get_image_transforms, get_imagenet_normalize, WaferMapDataset, seed_worker
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.training import train_model
from src.analysis import evaluate_model, count_params, count_trainable
from src.config import Config, load_config
from src.training.base_trainer import BaseTrainer
from src.model_registry import ModelRegistry, ModelMetadata
from src.exceptions import DataLoadError, ModelError


KNOWN_CLASSES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]


class WaferTrainer(BaseTrainer):
    """Main training orchestrator."""

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        self.registry = ModelRegistry()

    def load_and_preprocess(
        self,
        dataset_path: Path,
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 64,
    ) -> Tuple[Dict[str, Any], Dict[str, DataLoader], LabelEncoder]:
        """Load dataset, preprocess, and create train/val/test splits."""
        logger.info("\n" + "="*70)
        logger.info("LOADING AND PREPROCESSING DATA")
        logger.info("="*70)

        try:
            df = load_dataset(dataset_path)
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset from {dataset_path}: {e}")

        labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
        df_clean = df[labeled_mask].reset_index(drop=True)
        
        le = LabelEncoder()
        df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

        wafer_maps = df_clean['waferMap'].values
        labels = df_clean['label_encoded'].values

        temp_size = test_size + val_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            np.arange(len(labels)), labels, test_size=temp_size,
            stratify=labels, random_state=self.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size / temp_size,
            stratify=y_temp, random_state=self.seed
        )

        logger.info(f"\nSplit sizes: Train={len(y_train):,}, Val={len(y_val):,}, Test={len(y_test):,}")

        train_maps = preprocess_wafer_maps([wafer_maps[i] for i in X_train])
        val_maps = preprocess_wafer_maps([wafer_maps[i] for i in X_val])
        test_maps = preprocess_wafer_maps([wafer_maps[i] for i in X_test])

        class_counts_train = Counter(y_train)
        loss_weights = torch.tensor(
            [len(y_train) / (len(KNOWN_CLASSES) * class_counts_train[c]) for c in range(len(KNOWN_CLASSES))],
            dtype=torch.float32
        ).to(self.device)

        train_aug = get_image_transforms()
        imagenet_norm = get_imagenet_normalize()

        # Compose augmentation + ImageNet norm for pretrained models
        pretrained_train_transform = transforms.Compose([
            *train_aug.transforms,
            imagenet_norm,
        ])
        pretrained_val_transform = transforms.Compose([imagenet_norm])

        # CNN: augmentation only (no ImageNet normalization)
        g = torch.Generator().manual_seed(42)
        cnn_train_loader = DataLoader(WaferMapDataset(train_maps, y_train, transform=train_aug), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        cnn_val_loader = DataLoader(WaferMapDataset(val_maps, y_val, transform=None), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        cnn_test_loader = DataLoader(WaferMapDataset(test_maps, y_test, transform=None), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        # Pretrained (ResNet, EfficientNet): augmentation + ImageNet normalization
        g_pre = torch.Generator().manual_seed(42)
        pretrained_train_loader = DataLoader(WaferMapDataset(train_maps, y_train, transform=pretrained_train_transform), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g_pre)
        pretrained_val_loader = DataLoader(WaferMapDataset(val_maps, y_val, transform=pretrained_val_transform), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g_pre)
        pretrained_test_loader = DataLoader(WaferMapDataset(test_maps, y_test, transform=pretrained_val_transform), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g_pre)

        loaders = {
            'cnn': (cnn_train_loader, cnn_val_loader, cnn_test_loader),
            'resnet': (pretrained_train_loader, pretrained_val_loader, pretrained_test_loader),
            'effnet': (pretrained_train_loader, pretrained_val_loader, pretrained_test_loader),
        }

        data = {
            'class_names': le.classes_.tolist(),
            'loss_weights': loss_weights,
        }

        return data, loaders, le

    def run(self, args):
        """Execute training loop."""
        # Override config
        if args.model != 'all':
            self.config.training.default_model = args.model
        self.config.training.epochs = args.epochs
        self.config.training.batch_size = args.batch_size

        # Data path
        data_path = args.data_path or Path(self.config.data.dataset_path)
        if not data_path.is_absolute():
            data_path = Path(__file__).parent.parent / data_path

        data, loaders, le = self.load_and_preprocess(
            data_path,
            test_size=self.config.data.test_size,
            val_size=self.config.data.val_size,
            batch_size=self.config.training.batch_size
        )
        
        class_names = data['class_names']
        criterion = nn.CrossEntropyLoss(weight=data['loss_weights'])

        models_to_train = ['cnn', 'resnet', 'effnet'] if self.config.training.default_model == 'all' else [self.config.training.default_model]
        
        for model_type in models_to_train:
            logger.info(f"\nTraining {model_type.upper()}...")
            
            if model_type == 'cnn':
                model = WaferCNN(num_classes=len(class_names)).to(self.device)
                lr = args.lr or self.config.training.learning_rate.get('cnn', 1e-3)
            elif model_type == 'resnet':
                model = get_resnet18(num_classes=len(class_names)).to(self.device)
                lr = args.lr or self.config.training.learning_rate.get('resnet', 1e-4)
            else:
                model = get_efficientnet_b0(num_classes=len(class_names)).to(self.device)
                lr = args.lr or self.config.training.learning_rate.get('effnet', 1e-4)

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            train_loader, val_loader, test_loader = loaders[model_type]
            
            model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                epochs=self.config.training.epochs, device=str(self.device)
            )

            preds, labels, metrics = evaluate_model(model, test_loader, class_names, model_type, str(self.device))
            
            # Register model
            metadata = ModelMetadata(
                model_name=model_type,
                architecture=model.__class__.__name__,
                num_classes=len(class_names),
                training_config=self.config.__dict__, # Simplification
                metrics=metrics,
                dataset_version="wm811k_v1"
            )
            model_id = self.registry.register(model, metadata)
            logger.info(f"Model registered as: {model_id}")


def main():
    parser = argparse.ArgumentParser(description='Train wafer defect detection models')
    parser.add_argument('--model', choices=['cnn', 'resnet', 'effnet', 'all'], default='cnn')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--data-path', type=Path, default=None)
    parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    trainer = WaferTrainer(args.config)
    trainer.run(args)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
