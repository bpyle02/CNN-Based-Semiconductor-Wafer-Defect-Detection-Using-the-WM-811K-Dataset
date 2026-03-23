"""
Example: Monte Carlo Dropout Uncertainty Quantification for Wafer Defect Detection.

Demonstrates:
1. MC Dropout wrapper initialization
2. Uncertainty estimation on test set
3. Confidence intervals for predictions
4. Calibration analysis
5. Active learning (uncertain sample selection)
6. Visualization of uncertainty distributions

Reference: Gal & Ghahramani "Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning" (ICML 2016)

Usage:
    python uncertainty_example.py --model resnet --checkpoint checkpoints/best_resnet.pth
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.data import (
    load_dataset,
    preprocess_wafer_maps,
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
)
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.inference.uncertainty import (
    MCDropoutModel,
    UncertaintyEstimator,
    plot_uncertainty_distribution,
)


def load_model(model_type: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load pretrained model from checkpoint."""
    if model_type == 'cnn':
        model = WaferCNN(num_classes=9)
    elif model_type == 'resnet':
        model = get_resnet18(num_classes=9)
    elif model_type == 'efficientnet':
        model = get_efficientnet_b0(num_classes=9)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


def main(args):
    """Main uncertainty quantification pipeline."""
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"\nMonte Carlo Dropout Uncertainty Quantification")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"MC Iterations: {args.num_iterations}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    df = load_dataset()
    df = df[df['failureClass'] != 'unknown'].reset_index(drop=True)
    raw_maps = df['waferMap'].values
    labels = df['failureClass'].values

    # Preprocess
    print("Preprocessing wafer maps...")
    processed_maps = preprocess_wafer_maps(raw_maps)

    # Create test dataset (stratified split)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    _, test_maps, _, test_labels = train_test_split(
        processed_maps, labels_encoded,
        test_size=0.15, random_state=42, stratify=labels_encoded
    )

    # Create dataset with proper transforms
    class_names = le.classes_.tolist()
    transform = (
        get_image_transforms(augment=False)
        if args.model == 'cnn'
        else get_imagenet_normalize()
    )
    test_dataset = WaferMapDataset(test_maps, test_labels, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Load model
    print(f"Loading {args.model} model...")
    model = load_model(args.model, args.checkpoint, device)
    model.eval()

    # Initialize MC Dropout wrapper
    print(f"Initializing MC Dropout wrapper (T={args.num_iterations})...")
    mc_model = MCDropoutModel(model, num_iterations=args.num_iterations, device=device)

    # Initialize uncertainty estimator
    estimator = UncertaintyEstimator(model, num_iterations=args.num_iterations, device=device)

    print("\n" + "="*60)
    print("UNCERTAINTY ESTIMATION ON TEST SET")
    print("="*60)

    # Estimate uncertainty for all test samples
    print("\nEstimating uncertainty for test set...")
    results = estimator.estimate_dataset_uncertainty(test_loader, return_predictions=True)

    uncertainties = results['uncertainty']
    mean_probs = results['mean_probs']
    predictions = results['predictions']
    entropy = results['entropy']
    true_labels = results['true_labels']

    # Basic statistics
    print(f"\nUncertainty Statistics:")
    print(f"  Mean:    {uncertainties.mean():.4f}")
    print(f"  Std:     {uncertainties.std():.4f}")
    print(f"  Min:     {uncertainties.min():.4f}")
    print(f"  Max:     {uncertainties.max():.4f}")
    print(f"  Median:  {np.median(uncertainties):.4f}")

    print(f"\nPredictive Entropy Statistics:")
    print(f"  Mean:    {entropy.mean():.4f}")
    print(f"  Std:     {entropy.std():.4f}")
    print(f"  Min:     {entropy.min():.4f}")
    print(f"  Max:     {entropy.max():.4f}")

    # Accuracy
    correct = (predictions == true_labels).astype(np.float32)
    accuracy = correct.mean()
    print(f"\nAccuracy: {accuracy:.4f}")

    # Per-class accuracy and uncertainty
    print(f"\nPer-Class Analysis:")
    print(f"{'Class':<20} {'Samples':<10} {'Accuracy':<12} {'Mean UNC':<12}")
    print(f"{'-'*54}")
    for i, class_name in enumerate(class_names):
        mask = (predictions == i)
        if mask.sum() > 0:
            class_acc = correct[mask].mean()
            class_unc = uncertainties[mask].mean()
            print(f"{class_name:<20} {mask.sum():<10} {class_acc:<12.4f} {class_unc:<12.4f}")

    # Calibration analysis
    print("\n" + "="*60)
    print("UNCERTAINTY CALIBRATION")
    print("="*60)
    print("\nComputing calibration metrics...")
    cal_metrics = estimator.uncertainty_calibration(test_loader)

    print(f"\nCalibration Metrics:")
    print(f"  Brier Score: {cal_metrics['brier_score']:.4f}")
    print(f"  ECE (Expected Calibration Error): {cal_metrics['ece']:.4f}")
    print(f"  Uncertainty-Accuracy Correlation: {cal_metrics['uncertainty_accuracy_correlation']:.4f}")
    print(f"  Mean Uncertainty: {cal_metrics['mean_uncertainty']:.4f}")
    print(f"  Std Uncertainty: {cal_metrics['std_uncertainty']:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if cal_metrics['brier_score'] < 0.3:
        print(f"  ✓ Brier score is good (< 0.3)")
    else:
        print(f"  ✗ Brier score is high (> 0.3)")

    if abs(cal_metrics['uncertainty_accuracy_correlation']) > 0.3:
        corr_sign = "negative" if cal_metrics['uncertainty_accuracy_correlation'] < 0 else "positive"
        print(f"  ✓ Strong {corr_sign} correlation (well-calibrated if negative)")
    else:
        print(f"  ✗ Weak correlation (poorly calibrated)")

    # Confidence intervals
    print("\n" + "="*60)
    print("CONFIDENCE INTERVALS")
    print("="*60)

    print("\nComputing 95% confidence intervals for first 5 test samples...")
    first_batch = next(iter(test_loader))
    if isinstance(first_batch, (tuple, list)):
        first_batch = first_batch[0]
    first_batch = first_batch[:min(5, first_batch.shape[0])]
    first_batch = first_batch.to(device)

    median, lower, upper = mc_model.confidence_intervals(first_batch)

    print(f"\n{'Sample':<10} {'Class':<20} {'Median':<12} {'95% CI':<30}")
    print(f"{'-'*72}")
    for i in range(median.shape[0]):
        pred_class = median[i].argmax()
        pred_prob = median[i, pred_class]
        ci_lower = lower[i, pred_class]
        ci_upper = upper[i, pred_class]
        class_name = class_names[pred_class]
        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        print(f"Sample {i:<3} {class_name:<20} {pred_prob:<12.4f} {ci_str:<30}")

    # Active learning: uncertain sample selection
    print("\n" + "="*60)
    print("ACTIVE LEARNING: UNCERTAIN SAMPLE SELECTION")
    print("="*60)

    print("\nSelecting top-10 most uncertain samples (entropy metric)...")
    uncertain_samples = estimator.get_uncertain_samples(
        test_loader, k=10, metric='entropy'
    )

    print(f"\n{'Rank':<6} {'Entropy':<12} {'Max Prob':<12} {'Top-2 Margin':<12}")
    print(f"{'-'*42}")
    for i, idx in enumerate(uncertain_samples['indices']):
        entropy_score = uncertain_samples['uncertainties'][i]
        max_prob = uncertain_samples['predictions'][i].max()
        margin = uncertain_samples['top2_margin'][i]
        print(f"{i+1:<6} {entropy_score:<12.4f} {max_prob:<12.4f} {margin:<12.4f}")

    # Visualization
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    print("\nGenerating uncertainty distribution plots...")
    plot_uncertainty_distribution(
        uncertainties,
        predictions=mean_probs,
        true_labels=true_labels,
        class_names=class_names,
    )

    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Monte Carlo Dropout Uncertainty Quantification Example"
    )
    parser.add_argument(
        '--model', type=str, choices=['cnn', 'resnet', 'efficientnet'],
        default='cnn', help='Model type'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoints/best_cnn.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num-iterations', type=int, default=50,
        help='Number of MC Dropout iterations'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Compute device (cuda or cpu)'
    )

    args = parser.parse_args()
    main(args)
