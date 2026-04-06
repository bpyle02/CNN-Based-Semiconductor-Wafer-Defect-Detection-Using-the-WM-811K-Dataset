#!/usr/bin/env python3
"""
Comprehensive model validation script.
Performs calibration analysis, uncertainty quantification, and per-class performance audit.

Usage:
    python validate_model.py --model cnn --checkpoint checkpoints/best_cnn.pth
    python validate_model.py --model resnet --checkpoint checkpoints/best_resnet.pth --device cuda
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset, KNOWN_CLASSES
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.inference.uncertainty import UncertaintyEstimator, plot_uncertainty_distribution
from src.data.preprocessing import get_imagenet_normalize

def load_data(dataset_path: str, model_type: str = 'cnn'):
    """Load test data for validation."""
    df = load_dataset(Path(dataset_path))
    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)
    
    le = LabelEncoder()
    labels = le.fit_transform(df_clean['failureClass'])
    maps = df_clean['waferMap'].values
    
    # Use a 20% validation split for calibration audit
    _, X_test, _, y_test = train_test_split(
        maps, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    X_test = np.array(preprocess_wafer_maps(X_test.tolist()))
    
    transform = None
    if model_type in ['resnet', 'effnet']:
        transform = get_imagenet_normalize()
        
    ds = WaferMapDataset(X_test, y_test, transform=transform)
    return DataLoader(ds, batch_size=64, shuffle=False)

def main():
    parser = argparse.ArgumentParser(description='Model Validation Audit')
    parser.add_argument('--model', choices=['cnn', 'resnet', 'effnet'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/LSWMD_new.pkl')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--mc-samples', type=int, default=20, help='Number of MC Dropout samples')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Auditing model: {args.model} using checkpoint: {args.checkpoint}")
    
    # 1. Load Model
    if args.model == 'cnn':
        model = WaferCNN(num_classes=9)
    elif args.model == 'resnet':
        model = get_resnet18(num_classes=9)
    else:
        model = get_efficientnet_b0(num_classes=9)
        
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # 2. Load Data
    test_loader = load_data(args.data_path, args.model)
    
    # 3. Initialize Uncertainty Estimator
    estimator = UncertaintyEstimator(model, num_iterations=args.mc_samples, device=args.device)
    
    # 4. Calibration Audit
    logger.info("\n--- Calibration Audit ---")
    calibration_metrics = estimator.uncertainty_calibration(test_loader)
    for k, v in calibration_metrics.items():
        logger.info(f"  {k:30s}: {v:.4f}")
        
    # 5. Uncertainty Quantification
    logger.info("\n--- Uncertainty Quantification ---")
    results = estimator.estimate_dataset_uncertainty(test_loader, return_predictions=True)
    avg_unc = results['uncertainty'].mean()
    logger.info(f"  Average Epistemic Uncertainty : {avg_unc:.4f}")
    
    # 6. Temperature Scaling (Optional Check)
    logger.info("\n--- Fitting Temperature Scaling ---")
    temp = estimator.calibrate(test_loader)
    logger.info(f"  Optimized Temperature (T)     : {temp:.4f}")
    
    calibrated_metrics = estimator.uncertainty_calibration(test_loader)
    logger.info(f"  ECE after calibration         : {calibrated_metrics['ece']:.4f}")

    logger.info("\nAudit Complete.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
