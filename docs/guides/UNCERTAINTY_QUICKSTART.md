# Monte Carlo Dropout Uncertainty Quantification — Quick Start

## 5-Minute Setup

### 1. Load Model and Enable MC Dropout

```python
import torch
from src.models import get_resnet18
from src.inference.uncertainty import MCDropoutModel, UncertaintyEstimator

# Load pretrained model
model = get_resnet18()
model.load_state_dict(torch.load('checkpoints/best_resnet.pth'))

# Initialize MC Dropout wrapper
mc_model = MCDropoutModel(model, num_iterations=50, device='cuda')
estimator = UncertaintyEstimator(model, num_iterations=50, device='cuda')
```

### 2. Get Predictions with Uncertainty

```python
# Single sample
x = torch.randn(1, 3, 96, 96).cuda()
probs, uncertainty = mc_model.predict_with_uncertainty(x)

print(f"Predicted class: {probs.argmax()}")
print(f"Confidence: {probs.max():.4f}")
print(f"Uncertainty: {uncertainty[0]:.4f}")
```

### 3. Analyze Full Dataset

```python
# Estimate uncertainty for all test samples
results = estimator.estimate_dataset_uncertainty(test_loader)

print(f"Mean uncertainty: {results['uncertainty'].mean():.4f}")
print(f"Mean entropy: {results['entropy'].mean():.4f}")
print(f"Accuracy: {(results['predictions'] == results['true_labels']).mean():.4f}")
```

### 4. Check Calibration

```python
# Is model well-calibrated?
metrics = estimator.uncertainty_calibration(test_loader)

print(f"Brier Score: {metrics['brier_score']:.4f}  (< 0.3 is good)")
print(f"ECE: {metrics['ece']:.4f}  (< 0.1 is well-calibrated)")
print(f"Corr(Uncertainty, Correctness): {metrics['uncertainty_accuracy_correlation']:.4f}")
```

### 5. Active Learning: Select Uncertain Samples

```python
# Top 100 most uncertain samples to label
uncertain = estimator.get_uncertain_samples(
    unlabeled_loader, k=100, metric='entropy'
)

# These indices should be labeled next
indices_to_label = uncertain['indices']
entropies = uncertain['uncertainties']
```

### 6. Visualize Uncertainty

```python
from src.inference.uncertainty import plot_uncertainty_distribution

plot_uncertainty_distribution(
    uncertainties=results['uncertainty'],
    predictions=results['mean_probs'],
    true_labels=results['true_labels'],
    class_names=class_names
)
```

## Common Patterns

### Pattern 1: Confident vs. Uncertain Predictions

```python
probs, entropy = mc_model.predict_proba_with_uncertainty(x)

# Get confident predictions
confident_mask = entropy < entropy.quantile(0.25)
uncertain_mask = entropy > entropy.quantile(0.75)

print(f"Confident samples: {confident_mask.sum()}")
print(f"Uncertain samples: {uncertain_mask.sum()}")
```

### Pattern 2: Confidence Intervals for Predictions

```python
median, lower, upper = mc_model.confidence_intervals(x)

# Example: Show first sample's 95% credible interval for each class
for c in range(num_classes):
    print(f"Class {c}: {median[0, c]:.3f} [{lower[0, c]:.3f}, {upper[0, c]:.3f}]")
```

### Pattern 3: Find Mislabeled or Out-of-Distribution Samples

```python
results = estimator.estimate_dataset_uncertainty(test_loader)

# Uncertain predictions that are wrong (likely mislabeled or OOD)
wrong = results['predictions'] != results['true_labels']
uncertain = results['entropy'] > results['entropy'].mean()
suspicious = wrong & uncertain

print(f"Potentially mislabeled: {suspicious.sum()} samples")
```

### Pattern 4: Per-Class Uncertainty Analysis

```python
for c, class_name in enumerate(class_names):
    mask = results['predictions'] == c
    if mask.sum() > 0:
        class_unc = results['uncertainty'][mask]
        print(f"{class_name}: mean_unc={class_unc.mean():.4f}, std={class_unc.std():.4f}")
```

## Integration with Training

### Add to train.py

```python
from src.inference.uncertainty import UncertaintyEstimator

# After training loop
model.load_state_dict(torch.load(best_checkpoint))

# Compute uncertainty metrics on test set
estimator = UncertaintyEstimator(model, num_iterations=50, device=device)
results = estimator.estimate_dataset_uncertainty(test_loader)
metrics = estimator.uncertainty_calibration(test_loader)

# Log results
print(f"\nUncertainty Analysis:")
print(f"  Mean Uncertainty: {results['uncertainty'].mean():.4f}")
print(f"  Brier Score: {metrics['brier_score']:.4f}")
print(f"  ECE: {metrics['ece']:.4f}")
```

## API Cheat Sheet

| Task | Code |
|---|---|
| **Single prediction** | `probs, unc = mc_model.predict_with_uncertainty(x)` |
| **Per-class uncertainty** | `mean_p, std_p, entropy = mc_model.predict_proba_with_uncertainty(x)` |
| **Confidence intervals** | `median, lower, upper = mc_model.confidence_intervals(x)` |
| **Dataset analysis** | `results = estimator.estimate_dataset_uncertainty(loader)` |
| **Uncertain samples** | `uncertain = estimator.get_uncertain_samples(loader, k=100)` |
| **Calibration check** | `metrics = estimator.uncertainty_calibration(loader)` |
| **Plot** | `plot_uncertainty_distribution(uncertainties, predictions, true_labels, class_names)` |

## Interpretation

### Uncertainty Levels
- **< 0.05**: Very confident, trust prediction
- **0.05-0.15**: Confident
- **0.15-0.25**: Somewhat uncertain
- **> 0.25**: Very uncertain, flag for review

### Calibration Signals
| Metric | Good | Bad |
|---|---|---|
| Brier Score | < 0.3 | > 0.5 |
| ECE | < 0.1 | > 0.2 |
| Corr(Unc, Correct) | -0.3 to -0.7 | > -0.1 |

## Troubleshooting

**Q: Uncertainty is always very high (close to log(num_classes))**
- A: Dropout might be disabled. Check that model has Dropout layers with p > 0.

**Q: Uncertainty doesn't vary across samples**
- A: Try increasing T (num_iterations). Start with T=100 instead of 50.

**Q: Runtime is slow**
- A: MC Dropout adds ~50x overhead. Options:
  - Reduce T (20 instead of 50, if acceptable)
  - Use smaller batch size
  - Use simpler model (custom CNN vs. EfficientNet)
  - Use GPU

**Q: Uncertainty is anticorrelated with correctness (high on correct, low on wrong)**
- A: Model is poorly calibrated. Possible fixes:
  - Retrain with higher dropout (0.5-0.7)
  - Use ensemble methods (average multiple models)
  - Apply post-hoc temperature scaling

## Next Steps

1. Read full guide: `docs/UNCERTAINTY_QUANTIFICATION.md`
2. Run example: `python uncertainty_example.py --model resnet`
3. Run tests: `python -m pytest tests/test_uncertainty.py -v`
4. Integrate into your training pipeline

## Citation

If using MC Dropout for uncertainty, cite:

```bibtex
@inproceedings{gal2016dropout,
  title={Dropout as a Bayesian approximation: Representing model uncertainty in deep learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={International conference on machine learning},
  pages={1050--1059},
  year={2016},
  organization={PMLR}
}
```
