# Monte Carlo Dropout Uncertainty Quantification

## Overview

This implementation provides **Bayesian uncertainty estimation** for deep learning models using Monte Carlo (MC) Dropout, based on the theoretical framework of Gal & Ghahramani (ICML 2016): "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning."

### Key Insight

By enabling dropout during inference and running multiple forward passes, we approximate the posterior distribution over model predictions. This is equivalent to sampling from a Bayesian neural network, enabling principled uncertainty quantification.

## Quick Start (5 Minutes)

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
results = estimator.estimate_dataset_uncertainty(test_loader)

print(f"Mean uncertainty: {results['uncertainty'].mean():.4f}")
print(f"Mean entropy: {results['entropy'].mean():.4f}")
print(f"Accuracy: {(results['predictions'] == results['true_labels']).mean():.4f}")
```

### 4. Check Calibration

```python
metrics = estimator.uncertainty_calibration(test_loader)

print(f"Brier Score: {metrics['brier_score']:.4f}  (< 0.3 is good)")
print(f"ECE: {metrics['ece']:.4f}  (< 0.1 is well-calibrated)")
print(f"Corr(Uncertainty, Correctness): {metrics['uncertainty_accuracy_correlation']:.4f}")
```

### 5. Active Learning: Select Uncertain Samples

```python
uncertain = estimator.get_uncertain_samples(
    unlabeled_loader, k=100, metric='entropy'
)
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

### Common Patterns

**Confident vs. Uncertain Predictions:**
```python
probs, entropy = mc_model.predict_proba_with_uncertainty(x)
confident_mask = entropy < entropy.quantile(0.25)
uncertain_mask = entropy > entropy.quantile(0.75)
```

**Confidence Intervals:**
```python
median, lower, upper = mc_model.confidence_intervals(x)
for c in range(num_classes):
    print(f"Class {c}: {median[0, c]:.3f} [{lower[0, c]:.3f}, {upper[0, c]:.3f}]")
```

**Find Mislabeled or OOD Samples:**
```python
results = estimator.estimate_dataset_uncertainty(test_loader)
wrong = results['predictions'] != results['true_labels']
uncertain = results['entropy'] > results['entropy'].mean()
suspicious = wrong & uncertain
print(f"Potentially mislabeled: {suspicious.sum()} samples")
```

**Per-Class Uncertainty:**
```python
for c, class_name in enumerate(class_names):
    mask = results['predictions'] == c
    if mask.sum() > 0:
        class_unc = results['uncertainty'][mask]
        print(f"{class_name}: mean_unc={class_unc.mean():.4f}, std={class_unc.std():.4f}")
```

### API Cheat Sheet

| Task | Code |
|---|---|
| **Single prediction** | `probs, unc = mc_model.predict_with_uncertainty(x)` |
| **Per-class uncertainty** | `mean_p, std_p, entropy = mc_model.predict_proba_with_uncertainty(x)` |
| **Confidence intervals** | `median, lower, upper = mc_model.confidence_intervals(x)` |
| **Dataset analysis** | `results = estimator.estimate_dataset_uncertainty(loader)` |
| **Uncertain samples** | `uncertain = estimator.get_uncertain_samples(loader, k=100)` |
| **Calibration check** | `metrics = estimator.uncertainty_calibration(loader)` |
| **Plot** | `plot_uncertainty_distribution(uncertainties, predictions, true_labels, class_names)` |

### Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Uncertainty always very high | Check model has Dropout layers with p > 0 |
| Uncertainty doesn't vary | Increase T (num_iterations) to 100+ |
| Runtime is slow | Reduce T, use smaller batch, or use GPU |
| Anticorrelated with correctness | Retrain with higher dropout (0.5-0.7) or use ensemble |

---

## Mathematical Foundation

### Bayesian Approximation via Dropout

For a neural network with dropout, if we run inference T times with dropout enabled:

```
y_1, y_2, ..., y_T ≈ samples from posterior p(y|x, D)
```

where:
- `y_t` = model output at iteration t
- `x` = input
- `D` = training data

**Epistemic Uncertainty** (model uncertainty, reducible with more data):
```
σ²_epistemic ≈ Var[E[p(y|x)]] = E[p(y|x)²] - (E[p(y|x)])²
```

**Aleatoric Uncertainty** (data noise, irreducible):
```
σ²_aleatoric ≈ E[Var[p(y|x)]] = expected entropy of predictions
```

### Predictive Entropy

For multi-class classification, **predictive entropy** measures overall uncertainty:

```
H(y|x) = -Σ_c p(c|x) * log p(c|x)
```

- Low entropy → model is confident
- High entropy → model is uncertain (good for active learning)

## API Reference

### MCDropoutModel

Core wrapper for Monte Carlo Dropout inference.

```python
from src.inference.uncertainty import MCDropoutModel

# Initialize with pretrained model
mc_model = MCDropoutModel(
    model=trained_model,
    num_iterations=50,  # T samples from posterior
    device='cuda'
)

# Get predictions with uncertainty
mean_probs, epistemic_unc = mc_model.predict_with_uncertainty(x)
# mean_probs: shape (B, C), mean probability per class
# epistemic_unc: shape (B,), epistemic uncertainty per sample

# Get per-class probabilities and entropy
mean_probs, std_probs, entropy = mc_model.predict_proba_with_uncertainty(x)
# std_probs: shape (B, C), standard deviation of probabilities
# entropy: shape (B,), predictive entropy per sample

# Compute confidence intervals
median, lower, upper = mc_model.confidence_intervals(
    x, percentiles=(2.5, 97.5)
)
# Returns 95% credible intervals for each class probability
```

**Parameters:**
- `model`: PyTorch model (any architecture with Dropout layers)
- `num_iterations`: Number of MC passes (default 50; more = better approximation)
- `device`: 'cpu' or 'cuda'

**Returns:**
- Probability distributions across T forward passes
- Epistemic uncertainty (std of predictions)
- Predictive entropy
- Confidence intervals (percentile-based)

### UncertaintyEstimator

High-level API for dataset-level uncertainty analysis.

```python
from src.inference.uncertainty import UncertaintyEstimator

estimator = UncertaintyEstimator(model, num_iterations=50, device='cuda')

# Estimate uncertainty for all samples
results = estimator.estimate_dataset_uncertainty(test_loader)
# Returns: uncertainty, mean_probs, entropy, predictions, true_labels

# Get top-K most uncertain samples (active learning)
uncertain = estimator.get_uncertain_samples(
    test_loader, k=100, metric='entropy'  # or 'variance', 'margin'
)
# uncertain['indices']: indices of uncertain samples
# uncertain['uncertainties']: entropy scores for each

# Compute calibration metrics
metrics = estimator.uncertainty_calibration(test_loader)
# Returns: brier_score, ece, uncertainty_accuracy_correlation
```

**Uncertainty Metrics:**
- `'entropy'`: Predictive entropy H(y|x) — standard Bayesian measure
- `'variance'`: Mean class probability variance — model disagreement
- `'margin'`: 1 - (p_1 - p_2) where p_1, p_2 are top-2 class probabilities

### Visualization

```python
from src.inference.uncertainty import plot_uncertainty_distribution

plot_uncertainty_distribution(
    uncertainties=unc_array,
    predictions=mean_probs,
    true_labels=true_labels,
    class_names=class_names
)
```

Generates 4 subplots:
1. **Uncertainty histogram** — distribution of epistemic uncertainty
2. **Calibration curve** — uncertainty vs. prediction confidence
3. **Per-class uncertainty** — boxplots by predicted class
4. **Correctness correlation** — uncertainty for correct vs. incorrect predictions

## Usage Examples

### Example 1: Basic Uncertainty Estimation

```python
import torch
from src.models import get_resnet18
from src.inference.uncertainty import MCDropoutModel

# Load pretrained model
model = get_resnet18()
model.load_state_dict(torch.load('checkpoint.pth'))

# Wrap with MC Dropout
mc_model = MCDropoutModel(model, num_iterations=50, device='cuda')

# Get single prediction with uncertainty
x = torch.randn(1, 3, 96, 96).cuda()
probs, unc = mc_model.predict_with_uncertainty(x)

print(f"Predicted class: {probs.argmax()}")
print(f"Uncertainty: {unc[0]:.4f}")
print(f"Confidence: {probs.max():.4f}")
```

### Example 2: Active Learning

```python
from src.inference.uncertainty import UncertaintyEstimator

estimator = UncertaintyEstimator(model, num_iterations=50)

# Select uncertain samples for labeling
uncertain = estimator.get_uncertain_samples(
    unlabeled_loader, k=100, metric='entropy'
)

# Get indices of samples to label
to_label = uncertain['indices']
uncertainties = uncertain['uncertainties']

# Sort by uncertainty (descending)
sorted_idx = np.argsort(-uncertainties)
print(f"Top 10 uncertain samples to label:")
for i in range(10):
    idx = to_label[sorted_idx[i]]
    ent = uncertainties[sorted_idx[i]]
    print(f"  Sample {idx}: entropy={ent:.4f}")
```

### Example 3: Calibration Analysis

```python
# Check model calibration
metrics = estimator.uncertainty_calibration(test_loader)

print("Calibration Report:")
print(f"  Brier Score: {metrics['brier_score']:.4f}")
print(f"    (< 0.3 is good, measures probability accuracy)")
print(f"  ECE: {metrics['ece']:.4f}")
print(f"    (< 0.1 is well-calibrated)")
print(f"  Corr(Uncertainty, Correctness): {metrics['uncertainty_accuracy_correlation']:.4f}")
print(f"    (negative = well-calibrated, model knows when it's wrong)")
```

### Example 4: Confidence Intervals

```python
# Get prediction credible intervals
x = torch.randn(5, 3, 96, 96).cuda()
median, lower, upper = mc_model.confidence_intervals(x, percentiles=(2.5, 97.5))

for i in range(5):
    pred_class = median[i].argmax()
    prob = median[i, pred_class]
    ci = f"[{lower[i, pred_class]:.3f}, {upper[i, pred_class]:.3f}]"
    print(f"Sample {i}: Class {pred_class}, P={prob:.3f} 95% CI {ci}")
```

## Integration with Training Pipeline

### Enable MC Dropout for Trained Models

Models already have dropout layers from training. MC Dropout just enables them at inference:

```python
from src.inference.uncertainty import enable_dropout, disable_dropout

# Enable dropout at inference time
model.eval()  # Still in eval mode (no batch norm updates)
enable_dropout(model)

# Run MC Dropout inference
mc_model = MCDropoutModel(model, num_iterations=50)

# Standard inference (no uncertainty)
disable_dropout(model)
```

### Integration with train.py

```python
from src.inference.uncertainty import UncertaintyEstimator

# After training
model.load_state_dict(torch.load('best_model.pth'))

# Add uncertainty analysis
estimator = UncertaintyEstimator(model, num_iterations=50, device=device)
results = estimator.estimate_dataset_uncertainty(test_loader)

# Report uncertainty metrics
print(f"Mean Uncertainty: {results['uncertainty'].mean():.4f}")
print(f"Mean Entropy: {results['entropy'].mean():.4f}")

# Calibration check
cal_metrics = estimator.uncertainty_calibration(test_loader)
print(f"Brier Score: {cal_metrics['brier_score']:.4f}")
```

## Interpretation Guide

### Uncertainty Levels

| Uncertainty | Interpretation | Action |
|---|---|---|
| **Low** (<0.05) | Model very confident | Trust prediction |
| **Moderate** (0.05-0.15) | Model confident with some doubt | Good for classification |
| **High** (>0.15) | Model uncertain | Flag for human review / active learning |

### Calibration Metrics

**Brier Score** (mean squared error of probabilities):
- **Good**: < 0.3
- **Acceptable**: 0.3-0.5
- **Poor**: > 0.5

**Expected Calibration Error (ECE)**:
- Measures gap between predicted confidence and actual accuracy
- **Good**: < 0.1
- **Acceptable**: 0.1-0.2
- **Poor**: > 0.2

**Uncertainty-Accuracy Correlation**:
- Measures how well uncertainty predicts correctness
- **Well-calibrated**: -0.3 to -0.7 (negative, model knows when uncertain)
- **Poorly calibrated**: > -0.3 (no correlation or positive)

### When Uncertainty is High/Low

**High uncertainty on correct predictions:**
- ❌ Bad sign (model doesn't know what it knows)
- May indicate data distribution shift or edge cases

**High uncertainty on incorrect predictions:**
- ✅ Good sign (model knows when it fails)
- Indicates well-calibrated uncertainty

**Low uncertainty on all predictions:**
- ❌ Red flag (dropout might be disabled or T too small)
- Check that dropout is enabled and T ≥ 50

## Hyperparameter Tuning

### Number of MC Iterations (T)

Effect of increasing T:
- **T=10**: Fast, rough uncertainty estimates, high variance
- **T=50**: Good balance (default)
- **T=100**: Better estimates, 2x slower
- **T=200**: Excellent estimates for high-stakes decisions, 4x slower

**Recommendation**: Start with T=50, increase if uncertainty varies significantly or needed for critical applications.

```python
# Compare uncertainty estimates for different T values
for T in [10, 50, 100]:
    mc_model = MCDropoutModel(model, num_iterations=T)
    _, unc = mc_model.predict_with_uncertainty(x)
    print(f"T={T}: uncertainty={unc.mean():.4f}")
```

### Dropout Probability

Original model's dropout rates are preserved. To increase uncertainty:
- **Retrain** with higher dropout (0.5-0.7 in fully connected layers)
- More stochasticity → more varied predictions → higher uncertainty

## Limitations & Considerations

### What MC Dropout Captures

✅ Epistemic uncertainty (model uncertainty)
- Well-approximated by dropout variance
- Reducible with more data
- Good for active learning

⚠️ Aleatoric uncertainty (data noise)
- Partially captured by entropy
- Not fully separated from epistemic
- Consider separate aleatoric heads if critical

### When to Use / Avoid

**Use MC Dropout when:**
- You have dropout in your model
- You need uncertainty estimates quickly
- You want interpretable Bayesian approximation
- You need active learning sample selection

**Avoid or augment when:**
- Model has little to no dropout
- You need precise aleatoric/epistemic separation
- You need calibrated confidence scores for high-stakes decisions (also use post-hoc calibration)
- You need uncertainty in feature representations (use intermediate layers)

## References

1. **Gal & Ghahramani (2016)**: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" — ICML 2016
   - [Paper](https://arxiv.org/abs/1506.02142)
   - Core theoretical foundation

2. **Kendall & Gal (2017)**: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
   - Distinguishes aleatoric vs. epistemic uncertainty
   - Recommended for understanding uncertainty types

3. **Lakshminarayanan et al. (2017)**: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
   - Alternative approach (ensemble methods)
   - Good for comparison

4. **Guo et al. (2017)**: "On Calibration of Modern Neural Networks"
   - Temperature scaling and post-hoc calibration
   - Recommended for high-stakes applications

## Testing

Run unit tests:

```bash
python -m pytest tests/test_uncertainty.py -v
```

All 16 tests pass, covering:
- MC Dropout forward passes
- Uncertainty estimation
- Confidence intervals
- Calibration metrics
- Active learning sample selection
- Helper functions

## Performance

Typical runtime for T=50 iterations:

| Batch Size | Model | Runtime |
|---|---|---|
| 1 | ResNet-18 | ~50ms |
| 32 | ResNet-18 | ~500ms |
| 64 | ResNet-18 | ~1s |
| 64 | EfficientNet-B0 | ~1.5s |
| 64 | Custom CNN | ~200ms |

MC Dropout adds ~50x overhead compared to single forward pass.

## Future Extensions

Potential enhancements (not yet implemented):

1. **Separate Aleatoric Head**: Learn data noise explicitly
2. **Temperature Scaling**: Post-hoc calibration
3. **Ensemble + MC Dropout**: Combine multiple models with MC Dropout
4. **Variational Inference**: Replace dropout with proper Bayesian layers
5. **Uncertainty in Representations**: Extract uncertainty from hidden layers
