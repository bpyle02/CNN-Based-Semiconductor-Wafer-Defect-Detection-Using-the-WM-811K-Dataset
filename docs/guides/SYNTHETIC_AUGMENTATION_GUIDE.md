# Synthetic Data Augmentation Guide

## Overview

The synthetic augmentation module provides two complementary approaches for generating synthetic wafer maps to address class imbalance:

1. **Rule-Based Generator** (Fast, CPU-friendly, deterministic patterns)
2. **GAN-Based Generator** (Slower, GPU-recommended, realistic patterns)

## Quick Start

### Rule-Based Augmentation (Recommended for CPU)

```python
from src.augmentation.synthetic import WaferMapGenerator
import numpy as np

# Initialize generator
gen = WaferMapGenerator(image_size=96)

# Generate samples for specific classes
center_wafer = gen.generate_sample('Center', intensity=0.7)
donut_wafer = gen.generate_sample('Donut', intensity=0.8)
none_wafer = gen.generate_sample('none')

print(center_wafer.shape)  # (96, 96)
```

### GAN-Based Augmentation (Better quality, requires training)

```python
from src.augmentation.synthetic import SyntheticDataAugmenter
import torch
import numpy as np

# Initialize augmenter with GAN
augmenter = SyntheticDataAugmenter(
    generator_type='gan',
    image_size=96,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Train GAN on your wafer data
wafer_maps = np.random.rand(100, 96, 96)  # Your preprocessed wafers
history = augmenter.train_generator(
    wafer_maps,
    epochs=20,
    batch_size=32,
    verbose=True
)

# Generate synthetic samples
synthetic_wafers = augmenter.generate_samples(num_samples=50)
print(synthetic_wafers.shape)  # (50, 96, 96)
```

## API Reference

### WaferMapGenerator

Deterministic, rule-based generator using geometric patterns.

#### Methods

##### `generate_sample(defect_class, intensity=None, noise_level=0.1) -> np.ndarray`

Generate a synthetic wafer map for a specific defect class.

**Parameters:**
- `defect_class` (str): One of `['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']`
- `intensity` (float, optional): Defect intensity [0, 1]. If None, random.
- `noise_level` (float): Gaussian noise std dev [0, 1]

**Returns:** Synthetic wafer map (96, 96) normalized to [0, 1]

**Example:**
```python
gen = WaferMapGenerator()
for _ in range(10):
    sample = gen.generate_sample('Center', intensity=0.6, noise_level=0.05)
```

### SimpleWaferGAN

Neural network-based GAN for realistic wafer generation.

**Architecture:**
- **Generator**: Latent vector (100D) → 96×96 wafer via transposed convolutions
- **Discriminator**: 96×96 image → real/fake classification
- **Loss**: Binary cross-entropy

**Suitable for:**
- Learning actual defect patterns from real data
- Generating diverse, realistic wafers
- Transfer to other defect detection tasks

### SyntheticDataAugmenter

High-level API for augmentation with both generators.

#### Methods

##### `__init__(generator_type='rule-based', latent_dim=100, image_size=96, device=None)`

Initialize augmenter.

**Parameters:**
- `generator_type` (str): 'gan' or 'rule-based'
- `latent_dim` (int): GAN latent dimension (default 100)
- `image_size` (int): Output size (default 96)
- `device` (torch.device): CPU or CUDA. Auto-selects if None.

##### `train_generator(wafer_maps, epochs=10, batch_size=32, learning_rate=0.0002, beta1=0.5, verbose=True) -> dict`

Train GAN on unlabeled or labeled wafers.

**Parameters:**
- `wafer_maps` (np.ndarray): Shape (N, H, W), normalized [0, 1]
- `epochs` (int): Training epochs
- `batch_size` (int): Batch size
- `learning_rate` (float): Adam learning rate
- `beta1` (float): Adam beta1 parameter
- `verbose` (bool): Show progress

**Returns:** Training history with 'gen_loss' and 'disc_loss'

**Raises:** RuntimeError if generator_type != 'gan'

**Note:** Only for GAN. Use for rule-based is no-op.

**Example:**
```python
augmenter = SyntheticDataAugmenter('gan', device=torch.device('cuda'))
history = augmenter.train_generator(
    wafer_maps,
    epochs=20,
    batch_size=32,
    learning_rate=0.0002
)
```

##### `generate_samples(num_samples, class_label=None) -> np.ndarray`

Generate synthetic wafer maps.

**Parameters:**
- `num_samples` (int): Number of samples to generate
- `class_label` (str, optional): Class for rule-based. Ignored for GAN.

**Returns:** Synthetic maps (num_samples, H, W)

**Example:**
```python
# Rule-based
samples = augmenter.generate_samples(100, class_label='Center')

# GAN (class_label ignored)
samples = augmenter.generate_samples(100)
```

##### `augment_dataset(original_maps, original_labels, target_samples_per_class, class_names=None) -> tuple`

Balance dataset by generating synthetic samples for underrepresented classes.

**Parameters:**
- `original_maps` (np.ndarray): Shape (N, H, W)
- `original_labels` (np.ndarray): Shape (N,), integer class indices
- `target_samples_per_class` (int): Target count per class
- `class_names` (list, optional): Class names for logging

**Returns:** Tuple of (augmented_maps, augmented_labels)

**Example:**
```python
augmented_maps, augmented_labels = augmenter.augment_dataset(
    original_maps,
    original_labels,
    target_samples_per_class=5000,
    class_names=['Center', 'Donut', ..., 'none']
)
```

##### `visualize_generated_samples(class_names=None, num_samples_per_class=3, figsize=(15, 12)) -> None`

Generate and visualize samples in a grid.

**Parameters:**
- `class_names` (list, optional): Class names
- `num_samples_per_class` (int): Samples per class (default 3)
- `figsize` (tuple): Figure size

**Output:** Saves `gan_generated_samples.png` or `rule_based_generated_samples.png`

**Example:**
```python
augmenter.visualize_generated_samples(
    class_names=['Center', 'Donut', 'Edge-Loc', ...],
    num_samples_per_class=5
)
```

##### `save_gan(path) -> None` and `load_gan(path) -> None`

Persist and restore trained GANs.

**Example:**
```python
# Save after training
augmenter.save_gan('my_gan.pth')

# Load in new session
augmenter.load_gan('my_gan.pth')
```

### balance_dataset_with_synthetic()

Helper function for one-shot dataset balancing.

```python
def balance_dataset_with_synthetic(
    wafer_maps: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    generator_type: str = 'rule-based',
    strategy: str = 'oversample_to_max',
    oversample_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
```

**Parameters:**
- `wafer_maps` (np.ndarray): Original maps (N, H, W)
- `labels` (np.ndarray): Original labels (N,)
- `class_names` (list, optional): For logging
- `generator_type` (str): 'gan' or 'rule-based'
- `strategy` (str):
  - 'oversample_to_max': Match majority class (default)
  - 'oversample_to_mean': Match mean class count
  - 'oversample_to_custom': Custom ratio (set via oversample_ratio)
- `oversample_ratio` (float): Target ratio relative to max [0, 1]

**Returns:** Tuple of (balanced_maps, balanced_labels)

**Example:**
```python
from src.augmentation.synthetic import balance_dataset_with_synthetic

augmented_maps, augmented_labels = balance_dataset_with_synthetic(
    wafer_maps,
    labels,
    class_names=['Center', 'Donut', ..., 'none'],
    generator_type='rule-based',
    strategy='oversample_to_max'
)
```

## Usage Patterns

### Pattern 1: Quick Balancing (Rule-Based)

For fast CPU-based balancing without training:

```python
from src.augmentation.synthetic import balance_dataset_with_synthetic

augmented_maps, augmented_labels = balance_dataset_with_synthetic(
    original_maps,
    original_labels,
    generator_type='rule-based'
)
```

**Time:** ~1-2 minutes for 100K samples (CPU)

### Pattern 2: Quality Augmentation (GAN)

For best quality realistic samples:

```python
from src.augmentation.synthetic import SyntheticDataAugmenter
import torch

augmenter = SyntheticDataAugmenter('gan', device=torch.device('cuda'))
history = augmenter.train_generator(original_maps, epochs=20)
augmented_maps, augmented_labels = augmenter.augment_dataset(
    original_maps,
    original_labels,
    target_samples_per_class=5000
)
```

**Time:** ~2-4 hours for 20 epochs on 100K samples (single GPU)

### Pattern 3: Integration with Training Pipeline

In your training script:

```python
from src.augmentation.synthetic import balance_dataset_with_synthetic
from src.data.preprocessing import preprocess_wafer_maps
from src.models import WaferCNN
from src.training.trainer import train_model

# Load and preprocess
df = load_dataset()
raw_maps = extract_wafer_maps(df)
preprocessed_maps = preprocess_wafer_maps(raw_maps)
labels = extract_labels(df)

# Augment BEFORE splitting
if use_augmentation:
    preprocessed_maps, labels = balance_dataset_with_synthetic(
        preprocessed_maps, labels,
        generator_type='rule-based'
    )

# Split and train normally
train_dataset = WaferMapDataset(preprocessed_maps, labels)
train_loader = DataLoader(train_dataset, batch_size=64)

model = WaferCNN(num_classes=9)
trained_model, history = train_model(model, train_loader, ...)
```

### Pattern 4: Custom Class Balancing

Target specific minority classes:

```python
from src.augmentation.synthetic import WaferMapGenerator

gen = WaferMapGenerator()
custom_samples = []
custom_labels = []

# Generate 1000 samples each for rare classes
rare_classes = ['Edge-Ring', 'Scratch']
for class_name in rare_classes:
    for _ in range(1000):
        sample = gen.generate_sample(class_name)
        custom_samples.append(sample)
        custom_labels.append(class_name)

# Combine with original
all_maps = np.vstack([original_maps, np.array(custom_samples)])
all_labels = np.hstack([original_labels, np.array(custom_labels)])
```

## Performance Characteristics

### Rule-Based Generator

| Metric | Value |
|--------|-------|
| Speed (1000 samples) | ~2 seconds (CPU) |
| Memory | <100 MB |
| Quality | Deterministic, geometrically correct |
| Realism | Good for simple defects, less for complex |
| Training | None (instant) |

**Best for:**
- Rapid prototyping
- CPU-constrained environments
- Balanced classes after synthesis

### GAN-Based Generator

| Metric | Value |
|--------|-------|
| Speed (1000 samples) | ~30-60 sec (GPU), ~5 min (CPU) |
| Memory | ~2-4 GB (training), ~200 MB (inference) |
| Quality | Learned from data, highly realistic |
| Realism | Excellent, learns actual patterns |
| Training | 2-4 hours (100K samples, 1 GPU) |

**Best for:**
- Production systems
- High-quality synthetic data
- Transfer learning scenarios
- Research publications

## Troubleshooting

### Issue: GAN training diverges (D_loss → 0)

**Solution:** Reduce learning rate or add more data.

```python
augmenter.train_generator(
    maps,
    learning_rate=0.0001,  # Reduce 10x
    epochs=30
)
```

### Issue: Generated samples are blurry/low quality

**Solution:** Increase training epochs or use pre-trained checkpoint.

```python
# Train longer
history = augmenter.train_generator(maps, epochs=50)

# Or load from checkpoint
augmenter.load_gan('good_checkpoint.pth')
generated = augmenter.generate_samples(100)
```

### Issue: Out of memory with GAN training

**Solution:** Reduce batch size or use gradient accumulation.

```python
augmenter.train_generator(
    maps,
    batch_size=16,  # Reduce from 32
    epochs=10
)
```

### Issue: Augmentation creates unrealistic patterns

**Solution:** Adjust noise level in rule-based generator.

```python
gen = WaferMapGenerator()
sample = gen.generate_sample('Center', intensity=0.7, noise_level=0.05)
```

## Integration Checklist

- [ ] Import augmentation module: `from src.augmentation.synthetic import ...`
- [ ] Choose generator type: GAN (quality) or rule-based (speed)
- [ ] Preprocess wafer maps to (96, 96) normalized [0, 1]
- [ ] Call `balance_dataset_with_synthetic()` or train GAN manually
- [ ] Verify augmented class distribution
- [ ] Visualize generated samples before training
- [ ] Train model on augmented dataset
- [ ] Compare metrics with/without augmentation
- [ ] Save trained GAN if planning reuse

## Example Scripts

Three example scripts provided:

1. **`synthetic_augment.py`** - Standalone demonstration
   ```bash
   python synthetic_augment.py --generator rule-based --visualize-only
   python synthetic_augment.py --generator gan --gan-epochs 10
   ```

2. **`train_with_synthetic_augmentation.py`** - Full training pipeline
   ```bash
   python train_with_synthetic_augmentation.py --model resnet --augmentation rule-based
   python train_with_synthetic_augmentation.py --model cnn --augmentation gan
   ```

3. **Jupyter Notebook** - Interactive exploration (planned)

## References

- **GAN Architecture:** Standard DCGAN design with transposed convolutions
- **Loss Function:** Binary cross-entropy (BCE)
- **Optimization:** Adam optimizer with β₁=0.5, β₂=0.999
- **Latent Dimension:** 100D normal distribution

## FAQ

**Q: Should I use rule-based or GAN?**
A: Start with rule-based (fast). If results aren't satisfactory, try GAN (better quality).

**Q: Can I use both generators together?**
A: Yes! Train GAN on real data, then mix generated outputs with rule-based samples.

**Q: What happens if I augment too much?**
A: Risk of overfitting to synthetic patterns. Start conservatively (20-50% synthetic).

**Q: Does augmentation guarantee better model performance?**
A: Usually yes for imbalanced datasets. Always validate on held-out test set.

**Q: Can I save/load trained GANs?**
A: Yes! Use `save_gan()` and `load_gan()` for persistence.

## Citation

If using this module in research, cite as:

```bibtex
@software{wafer_augmentation_2024,
  title={Synthetic Data Augmentation for Wafer Defect Detection},
  author={Rettura, B. and Paul, A. and Pyle, B. and Rajan, A.},
  year={2024},
  url={https://github.com/your-repo/wafer-defect-detection}
}
```
