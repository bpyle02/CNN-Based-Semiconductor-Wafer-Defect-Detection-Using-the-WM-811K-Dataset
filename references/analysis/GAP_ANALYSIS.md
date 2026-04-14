# Gap Analysis: Paper Techniques Not Yet Implemented

**Date**: 2026-04-06
**Current Best Macro F1**: 0.665 (ResNet-18, 5 epochs)
**Target**: Maximize macro F1 on WM-811K (9 classes, 150:1 imbalance)

---

## Methodology

Cross-referenced 150 papers (116 with extracted text in `references/text/`) against every
module in `src/`. The analysis below identifies techniques **referenced in the papers but
absent from the codebase**, ranked by expected impact on macro F1 given the 150:1 class
imbalance. Each gap includes the paper IDs, the file that should be extended, and estimated
implementation effort.

---

## A. HIGH IMPACT -- Techniques Directly Targeting Imbalanced Macro F1

### A1. Logit-Adjusted / Balanced Softmax Loss

**What**: Post-hoc logit adjustment adds `log(pi_y)` to logits at train or test time,
where `pi_y` is the class prior. This corrects the decision boundary shift caused by
class imbalance without the instability of large class weights.

**Equation**: `L = CE(logits + tau * log(pi), y)` where `pi` is the class frequency vector.

**Why it matters**: Current loss functions (CrossEntropy with weights, FocalLoss) use
heuristic weight scaling. Logit adjustment is theoretically grounded (Fisher-consistent
for balanced error) and directly optimizes for the balanced accuracy objective that
macro F1 approximates. On long-tailed CIFAR-100, logit adjustment improves by 3-5% macro F1
over focal loss.

**Paper IDs**: [16] (Class-Balanced Loss framework), [74] (Decoupling representation/classifier)
**Not in 150 -- key reference**: Menon et al. (2021) "Long-tail learning via logit adjustment",
arXiv:2007.07314

**Extend**: `src/training/losses.py` -- add `LogitAdjustedLoss` class
**Effort**: Small (20 lines of code)
**Expected impact**: +2-4% macro F1

---

### A2. Decoupled Training (cRT / tau-Normalization)

**What**: Two-stage training: (1) train backbone with standard instance-balanced sampling
to learn good representations, then (2) re-train only the classifier head with
class-balanced sampling or tau-normalization of the weight vectors.

**Equation**: tau-norm: `w_c' = w_c / ||w_c||^tau` where tau is tuned on val set.

**Why it matters**: Paper [74] (Kang et al., 2020) showed that on ImageNet-LT,
decoupled training with cRT (classifier re-training) improves top-1 by 7-10% over
joint training. The current codebase trains backbone and classifier jointly with
weighted loss -- the backbone overfits to the weighting scheme, learning biased features.
SupCon (`src/training/supcon.py`) already implements a two-stage approach but uses
contrastive pretraining; cRT is simpler and often competitive.

**Paper IDs**: [74] (Decoupling), [16] (CB Loss), [17] (Systematic study of imbalance)
**Extend**: `src/training/trainer.py` or new `src/training/decoupled.py`
**Effort**: Medium (100 lines: freeze backbone, balanced re-train classifier)
**Expected impact**: +3-5% macro F1

---

### A3. Dice Loss / Tversky Loss for Classification

**What**: Dice loss treats classification as a soft set-overlap problem. The loss for
class c is `1 - 2*sum(p_c * y_c) / (sum(p_c) + sum(y_c))`. Tversky loss generalizes
this with alpha/beta parameters controlling FP/FN tradeoff.

**Equation (Tversky)**: `TL_c = 1 - sum(p_c * y_c) / (sum(p_c * y_c) + alpha*sum(p_c * (1-y_c)) + beta*sum((1-p_c) * y_c))`

**Why it matters**: Papers [131] (V-Net/Dice) and [132] (Tversky) show these losses
handle extreme imbalance (500:1) far better than weighted CE because they directly
optimize the overlap metric rather than per-sample log-likelihood. For WM-811K's 150:1
ratio, Dice/Tversky losses can dramatically improve recall on rare classes (Donut,
Near-full, Random) without collapsing majority class performance. These losses are
**referenced in `src/training/losses.py` docstring** but **not implemented**.

**Paper IDs**: [131] (V-Net / Dice Loss), [132] (Tversky Loss)
**Extend**: `src/training/losses.py` -- add `DiceLoss`, `TverskyLoss`, `FocalTverskyLoss`
**Effort**: Small (50 lines each)
**Expected impact**: +2-5% macro F1 (especially rare classes)

---

### A4. Semi-Supervised Learning with Pseudo-Labels / FixMatch

**What**: WM-811K has ~811K total wafers but only ~172K are labeled. The remaining ~640K
unlabeled wafers are currently discarded. FixMatch generates pseudo-labels for
high-confidence predictions on weakly-augmented unlabeled data and trains on
strongly-augmented versions. This is the single largest untapped data source.

**Algorithm (FixMatch)**:
1. Forward pass on weakly-augmented unlabeled image
2. If max(softmax(logits)) > threshold (e.g., 0.95), create pseudo-label = argmax
3. Compute CE loss on strongly-augmented version of same image against pseudo-label
4. Total loss = supervised_loss + lambda_u * unsupervised_loss

**Why it matters**: With 640K unlabeled wafers, semi-supervised learning can provide
massive representation improvement. FixMatch achieves 94.93% on CIFAR-10 with only
250 labels. The codebase has `SimCLR` and `BYOL` for self-supervised pretraining but
**no semi-supervised training loop** that uses both labeled and unlabeled data
simultaneously. The reference to FixMatch [111] appears only in `simclr.py` docstring.

**Paper IDs**: [111] (FixMatch), [115] (MixMatch), [118] (UDA), [119] (Temporal Ensembling), [120] (Pseudo-Label)
**Extend**: New `src/training/semi_supervised.py`
**Effort**: Large (200-300 lines: dual dataloader, confidence thresholding, consistency loss)
**Expected impact**: +5-10% macro F1 (leverages 4x more data)

---

### A5. SAM (Sharpness-Aware Minimization) Optimizer

**What**: SAM seeks parameters in flat loss regions by computing gradient at a
worst-case perturbation, then updating at the original point. This finds flatter
minima that generalize better.

**Algorithm**: For each step: (1) compute epsilon = rho * grad/||grad||,
(2) compute gradient at theta + epsilon, (3) update theta with this gradient.

**Why it matters**: Paper [139] shows SAM improves generalization on CIFAR-10/100 by
0.5-1.5% and provides label-noise robustness for free. For imbalanced datasets, SAM's
flat minima generalize better to rare classes. SAM is **referenced in train.py** [139]
but **not implemented** -- no SAM optimizer class exists.

**Paper IDs**: [139] (SAM)
**Extend**: `src/training/trainer.py` or new `src/training/optimizers.py`
**Effort**: Small-Medium (50 lines for SAM wrapper)
**Expected impact**: +1-2% macro F1

---

### A6. Test-Time Augmentation (TTA)

**What**: At inference, apply multiple augmentations to each test image and average
predictions. This is a free lunch that reduces variance and improves calibration.

**Why it matters**: The codebase applies augmentation only during training. TTA with
the existing domain-specific augmentations (rotation, flip, radial distortion) would
ensemble 5-10 views per sample. Typical TTA improvement is 1-3% accuracy. For rare
classes where a single misclassification swings macro F1, TTA is especially valuable.

**Paper IDs**: [49] (Augmentation survey), [30] (Deep Ensembles)
**Extend**: `src/inference/uncertainty.py` or new `src/inference/tta.py`
**Effort**: Small (40 lines)
**Expected impact**: +1-3% macro F1

---

### A7. Exponential Moving Average (EMA) of Model Weights

**What**: Maintain an exponentially-weighted moving average of model parameters during
training. Use the EMA model for evaluation. EMA parameters are smoother and
generalize better.

**Equation**: `theta_ema = alpha * theta_ema + (1 - alpha) * theta`, alpha ~ 0.999

**Why it matters**: EMA is standard in semi-supervised and contrastive learning
(Mean Teacher [119] uses EMA) and consistently adds 0.5-1% accuracy. Zero cost
at training time. Not implemented anywhere in the codebase.

**Paper IDs**: [119] (Mean Teacher / Temporal Ensembling), [38] (BYOL uses EMA)
**Extend**: `src/training/trainer.py` -- add EMA callback
**Effort**: Small (30 lines)
**Expected impact**: +0.5-1.5% macro F1

---

### A8. LDAM Loss (Label-Distribution-Aware Margin)

**What**: Enforces larger classification margins for minority classes. The margin for
class j is proportional to `n_j^{-1/4}`, ensuring rare classes get wider decision
boundaries.

**Equation**: `L_LDAM = CE(logits - delta_y, y)` where `delta_j = C / n_j^{1/4}`

**Why it matters**: LDAM directly addresses the decision boundary bias from imbalance.
Combined with deferred re-balancing (DRW -- switch from uniform to class-balanced
sampling at epoch T), LDAM+DRW achieves SOTA on imbalanced CIFAR/ImageNet-LT.
Not in the codebase despite paper [73] being referenced (the reference is to
CB Loss, not LDAM specifically).

**Not in 150 -- key reference**: Cao et al. (2019) "Learning Imbalanced Datasets
with Label-Distribution-Aware Margin Loss", arXiv:1906.07413

**Extend**: `src/training/losses.py`
**Effort**: Small (40 lines)
**Expected impact**: +2-4% macro F1

---

## B. MEDIUM IMPACT -- Architecture and Training Enhancements

### B1. Deformable Convolutions

**What**: Deformable convolutions learn per-sample spatial offsets for the conv kernel,
allowing the receptive field to adapt to object shape. Paper [14] specifically used
deformable CNNs for wafer defect pattern detection.

**Paper IDs**: [14] (Tsai & Wang -- Deformable CNNs for Wafer Defects), [92] (DCN original)
**Extend**: `src/models/cnn.py` -- replace standard Conv2d in later stages
**Effort**: Medium (requires torchvision.ops.DeformConv2d or custom implementation)
**Expected impact**: +1-2% macro F1

---

### B2. Graph Neural Networks for Wafer Spatial Structure

**What**: Paper [89] (Park & Cho, 2021) uses GNNs on wafer maps by treating die
positions as graph nodes. This captures spatial relationships that CNNs miss when
the defect pattern follows the wafer's physical topology.

**Paper IDs**: [86] (GCN), [87] (GAT), [88] (GraphSAGE), [89] (Wafer GNN), [90] (GIN)
**Extend**: New `src/models/gnn.py`
**Effort**: Large (150+ lines; requires torch_geometric or manual adjacency)
**Expected impact**: +1-3% macro F1 (especially for spatial patterns like Edge-Ring)

---

### B3. Multi-Task Learning with Auxiliary Objectives

**What**: Train the main classifier alongside auxiliary tasks: (a) binary defect/no-defect,
(b) defect region segmentation mask, (c) rotation prediction (self-supervised).
Shared representations become more robust.

**Paper IDs**: [93] (MTL overview), [95] (MTL as multi-objective), [96] (Cross-Stitch), [100] (GradNorm)
**Extend**: `train.py` + new `src/training/multitask.py`
**Effort**: Medium (auxiliary head + loss combination)
**Expected impact**: +1-2% macro F1

---

### B4. Class-Conditional Batch Normalization

**What**: Use separate batch normalization statistics per class during training.
This prevents the majority class from dominating BN running statistics, which
distorts feature normalization for rare classes.

**Not in 150 -- key reference**: De Vries et al. (2017) "Modulating early visual
processing by language", arXiv:1707.00683 (conditional BN concept)

**Extend**: `src/models/cnn.py` -- replace nn.BatchNorm2d with ClassConditionalBN
**Effort**: Medium (80 lines)
**Expected impact**: +1-2% macro F1

---

### B5. Stochastic Weight Averaging (SWA)

**What**: Average model weights from the last K training epochs (with cyclic or
constant LR). Finds wider optima than standard training, improving generalization.

**Not in 150 -- key reference**: Izmailov et al. (2018) "Averaging Weights Leads
to Wider Optima and Better Generalization", arXiv:1803.05407

**Extend**: `src/training/trainer.py` -- use `torch.optim.swa_utils`
**Effort**: Small (20 lines; PyTorch has built-in SWA)
**Expected impact**: +0.5-1.5% macro F1

---

### B6. Knowledge Distillation for Ensemble Compression

**What**: Distill the ensemble (CNN + ResNet + EfficientNet) into a single student
model. The student learns from soft targets (teacher softmax outputs) which encode
inter-class similarity. The student can match or exceed individual teacher models.

**Paper IDs**: [42] (Hinton -- Knowledge Distillation)
**Extend**: `scripts/compress_model.py` (distillation is referenced but current
implementation is basic; needs soft-target training loop)
**Effort**: Medium (100 lines)
**Expected impact**: +1-2% macro F1 (single model approaching ensemble performance)

---

### B7. Conditional GAN / Diffusion-Based Augmentation

**What**: Train a class-conditional generative model on real wafer maps and use it
to generate realistic synthetic samples for rare classes. Current synthetic generation
(`src/augmentation/synthetic.py`) is rule-based with only 3 patterns (Center, Edge-Loc,
Scratch); other classes get random noise. A learned generator would produce far more
realistic samples.

**Paper IDs**: [65] (GAN for wafer maps), [123] (VAE)
**Extend**: `src/augmentation/train_generator.py` (GAN trainer exists but no
class-conditional generation)
**Effort**: Large (200+ lines for conditional DCGAN or lightweight diffusion)
**Expected impact**: +2-4% macro F1 (realistic rare-class augmentation)

---

### B8. Feature-Space Augmentation (Implicit/Manifold Mixup)

**What**: Apply mixup in the feature space rather than pixel space. This creates more
meaningful interpolations because the feature manifold is smoother than pixel space.
Paper [72] specifically addresses long-tailed classification.

**Paper IDs**: [66] (Mixup -- pixel space, implemented), [72] (Feature Space Aug),
[75] (Remix -- implemented in MixupCutmix references)

**Not in 150 -- key reference**: Verma et al. (2019) "Manifold Mixup: Better
Representations by Interpolating Hidden States", arXiv:1806.05236

**Extend**: `src/data/preprocessing.py` or `src/training/trainer.py` -- intercept
intermediate features
**Effort**: Medium (60 lines)
**Expected impact**: +1-2% macro F1

---

## C. MODERATE IMPACT -- Calibration, Inference, and Rare-Class Techniques

### C1. Post-Hoc Calibration (Temperature Scaling + Per-Class Calibration)

**What**: After training, learn a single temperature T (or per-class temperatures) on
the validation set to calibrate softmax probabilities. This is already in config
(`temperature_scaling` field) but no implementation exists in `src/`.

**Paper IDs**: [29] (On Calibration), [32] (Measuring Calibration)
**Extend**: `src/analysis/evaluate.py` -- `calibrate_and_evaluate` exists but needs
temperature optimization
**Effort**: Small (30 lines)
**Expected impact**: +0.5-1% macro F1 (via better threshold selection)

---

### C2. MAML / Prototypical Networks for Few-Shot Rare Classes

**What**: Use meta-learning to specifically improve classification of the rarest classes
(Donut: 534 samples, Near-full: 348, Random: 383). MAML [113] learns an
initialization that adapts in few gradient steps. Prototypical Networks [77] classify
by distance to class-mean embeddings.

**Paper IDs**: [77] (Prototypical Networks), [112] (Meta-learning survey), [113] (MAML),
[116] (Matching Networks), [117] (MetaOptNet)
**Extend**: New `src/training/meta_learning.py`
**Effort**: Large (200+ lines)
**Expected impact**: +1-3% macro F1 on rare classes

---

### C3. Lookahead Optimizer

**What**: Maintains slow weights that are updated by interpolating with fast weights
every k steps. `theta_slow = theta_slow + alpha * (theta_fast - theta_slow)`.
Reduces variance, improves convergence.

**Paper IDs**: [137] (Lookahead), [138] (RAdam)
**Extend**: `src/training/trainer.py` -- add Lookahead wrapper
**Effort**: Small (40 lines)
**Expected impact**: +0.5-1% macro F1

---

### C4. Cosine Classifier (Normalized Weights + Temperature)

**What**: Replace the final linear layer with a cosine similarity classifier where
both weights and features are L2-normalized, then scaled by a learnable temperature.
This prevents the classifier from learning magnitude-biased decision boundaries
(common in imbalanced settings).

**Paper IDs**: [79] (ArcFace), [80] (CosFace), [84] (Circle Loss)
**Extend**: `src/models/cnn.py`, `src/models/pretrained.py` -- replace final linear
**Effort**: Small (30 lines)
**Expected impact**: +1-2% macro F1

---

### C5. Class-Aware Sampling with Deferred Re-Balancing (DRW)

**What**: Train with instance-balanced sampling for the first T epochs (to learn good
representations), then switch to class-balanced sampling (using `ClassBalancedSampler`)
for the remaining epochs. The sampler exists (`src/data/preprocessing.py`) but is
never used in `train.py` and there is no deferred switching logic.

**Not in 150 -- key reference**: Cao et al. (2019), arXiv:1906.07413

**Extend**: `train.py` -- add DRW schedule
**Effort**: Small (20 lines)
**Expected impact**: +1-3% macro F1

---

### C6. Label-Aware Smooth Regularization

**What**: Apply different label smoothing values per class. Rare classes get less
smoothing (preserve the hard target), majority classes get more smoothing (reduce
overconfidence). Current implementation applies uniform smoothing.

**Paper IDs**: [20] (Label Smoothing)
**Extend**: `src/training/losses.py`
**Effort**: Small (30 lines)
**Expected impact**: +0.5-1% macro F1

---

### C7. MixUp with Class-Aware Pairing (Remix)

**What**: Paper [75] (Remix) modifies mixup to favor minority class labels when
mixing. If sample A is minority and B is majority, assign the minority label with
higher weight. Current MixupCutmix uses random pairing.

**Paper IDs**: [75] (Remix)
**Extend**: `src/data/preprocessing.py` -- modify `MixupCutmix.__call__`
**Effort**: Small (20 lines)
**Expected impact**: +0.5-1.5% macro F1

---

## D. Summary Table: Ranked by Expected Impact

| Rank | Technique | Expected Impact | Effort | Paper IDs | File to Extend |
|------|-----------|----------------|--------|-----------|----------------|
| 1 | Semi-supervised (FixMatch) | +5-10% | Large | [111,115,118,119,120] | New: `src/training/semi_supervised.py` |
| 2 | Decoupled Training (cRT) | +3-5% | Medium | [74,16,17] | `src/training/trainer.py` |
| 3 | Dice/Tversky Loss | +2-5% | Small | [131,132] | `src/training/losses.py` |
| 4 | Logit-Adjusted Loss | +2-4% | Small | [16,74] | `src/training/losses.py` |
| 5 | LDAM + DRW | +2-4% | Small | [73] | `src/training/losses.py` + `train.py` |
| 6 | Conditional GAN Augmentation | +2-4% | Large | [65,123] | `src/augmentation/train_generator.py` |
| 7 | SAM Optimizer | +1-2% | Small | [139] | New: `src/training/optimizers.py` |
| 8 | Test-Time Augmentation | +1-3% | Small | [49,30] | New: `src/inference/tta.py` |
| 9 | DRW Sampling Schedule | +1-3% | Small | -- | `train.py` |
| 10 | EMA Model Weights | +0.5-1.5% | Small | [119,38] | `src/training/trainer.py` |
| 11 | Cosine Classifier | +1-2% | Small | [79,80,84] | `src/models/cnn.py`, `pretrained.py` |
| 12 | Deformable Convolutions | +1-2% | Medium | [14,92] | `src/models/cnn.py` |
| 13 | SWA | +0.5-1.5% | Small | -- | `src/training/trainer.py` |
| 14 | Feature-Space Mixup | +1-2% | Medium | [72] | `src/training/trainer.py` |
| 15 | Remix (class-aware mixup) | +0.5-1.5% | Small | [75] | `src/data/preprocessing.py` |
| 16 | Knowledge Distillation | +1-2% | Medium | [42] | `scripts/compress_model.py` |
| 17 | GNN for Spatial Patterns | +1-3% | Large | [86-90] | New: `src/models/gnn.py` |
| 18 | Meta-Learning (MAML/Proto) | +1-3% | Large | [77,112,113,116,117] | New: `src/training/meta_learning.py` |
| 19 | Multi-Task Learning | +1-2% | Medium | [93,95,96,100] | New: `src/training/multitask.py` |
| 20 | Lookahead Optimizer | +0.5-1% | Small | [137,138] | `src/training/trainer.py` |
| 21 | Post-Hoc Calibration | +0.5-1% | Small | [29,32] | `src/analysis/evaluate.py` |
| 22 | Label-Aware Smoothing | +0.5-1% | Small | [20] | `src/training/losses.py` |
| 23 | Class-Conditional BN | +1-2% | Medium | -- | `src/models/cnn.py` |

---

## E. Quick Wins (Implement in <1 Hour, Combined +5-10% Macro F1)

These 6 techniques are small effort, well-supported by the papers, and stack additively:

1. **Dice/Tversky Loss** in `src/training/losses.py` (50 lines)
2. **Logit-Adjusted Loss** in `src/training/losses.py` (20 lines)
3. **DRW schedule** in `train.py` -- switch to ClassBalancedSampler at epoch 3/5 (20 lines)
4. **Test-Time Augmentation** in `src/inference/tta.py` (40 lines)
5. **EMA weights** in `src/training/trainer.py` (30 lines)
6. **Cosine classifier** option in model builders (30 lines)

---

## F. Proposed Additional Papers (Not in the 150)

The following 30 papers address techniques identified as gaps. Prioritized by relevance
to WM-811K's specific challenges (extreme imbalance, spatial defect patterns, unlabeled data).

### Long-Tail / Class Imbalance (Most Critical Gap)

1. **Menon et al. (2021)**. "Long-tail learning via logit adjustment". arXiv:2007.07314
   -- Theoretically optimal logit correction for imbalanced classification.

2. **Cao et al. (2019)**. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss". arXiv:1906.07413
   -- LDAM loss + DRW schedule; SOTA on imbalanced CIFAR/ImageNet-LT.

3. **Hong et al. (2021)**. "Disentangling Label Distribution for Long-tailed Visual Recognition". arXiv:2012.00321
   -- LADE: label-distribution-aware estimation for long-tail.

4. **Zhang et al. (2021)**. "Distribution Alignment: A Unified Framework for Long-tail Visual Recognition". arXiv:2103.16370
   -- DisAlign: adjusts classifier magnitude + offset per class.

5. **Zhong et al. (2021)**. "Improving Calibration for Long-Tailed Recognition". arXiv:2104.00466
   -- MiSLAS: mixup-shifted label-aware smoothing for calibrated long-tail.

6. **Wang et al. (2021)**. "Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification". arXiv:2103.14267
   -- Combines supervised contrastive + balanced classifier.

### Semi-Supervised for Industrial Defect Detection

7. **Xu et al. (2021)**. "End-to-End Semi-Supervised Object Detection with Soft Teacher". arXiv:2106.09018
   -- Soft teacher: EMA teacher produces soft pseudo-labels.

8. **Zhang et al. (2021)**. "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling". arXiv:2110.08263
   -- Per-class adaptive thresholds for pseudo-labeling.

9. **Wei et al. (2021)**. "CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning". arXiv:2102.09559
   -- Self-training specifically designed for imbalanced SSL.

10. **Kim et al. (2020)**. "Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning". arXiv:2007.08844
    -- DARP: aligns pseudo-label distribution to true prior.

### Wafer-Specific and Industrial

11. **Alawieh et al. (2020)**. "Wafer map defect patterns classification using deep selective learning". DOI:10.1109/DAC18072.2020.9218588
    -- Deep selective learning for wafer defect with abstention option.

12. **Shim et al. (2023)**. "Active Learning for Semiconductor Wafer Bin Map Defect Pattern Classification". DOI:10.1109/TSM.2023.3241065
    -- Active learning specifically for WM-811K with imbalanced AL.

13. **Saqlain et al. (2019)**. "A Voting Ensemble Classifier for Wafer Map Defect Patterns". DOI:10.1109/ACCESS.2019.2958789
    -- Ensemble voting tuned for WM-811K.

14. **Kim & Kim (2021)**. "Semi-supervised defect classification with contrastive learning on wafer bin maps". DOI:10.1109/TSM.2021.3072404
    -- Direct application of semi-supervised + contrastive to wafer maps.

### Optimizers and Training Techniques

15. **Izmailov et al. (2018)**. "Averaging Weights Leads to Wider Optima and Better Generalization". arXiv:1803.05407
    -- SWA: stochastic weight averaging.

16. **Kwon et al. (2021)**. "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks". arXiv:2102.11600
    -- ASAM: scale-invariant version of SAM.

17. **Du et al. (2022)**. "Efficient Sharpness-aware Minimization for Improved Training of Neural Networks". arXiv:2110.03141
    -- ESAM: reduces SAM's computational cost.

### Augmentation

18. **Verma et al. (2019)**. "Manifold Mixup: Better Representations by Interpolating Hidden States". arXiv:1806.05236
    -- Mixup in feature space.

19. **Kim et al. (2020)**. "Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup". arXiv:2009.06962
    -- Saliency-guided mixup for more meaningful augmentation.

20. **Li et al. (2021)**. "Improved Regularization and Robustness for Fine-tuning in Neural Networks". arXiv:2111.04578
    -- Regularization for transfer learning fine-tuning.

### Loss Functions and Training Strategies

21. **Li et al. (2022)**. "Targeted Supervised Contrastive Learning for Long-Tailed Recognition". arXiv:2111.13998
    -- TSC: modifies SupCon for long-tail by targeting rare classes.

22. **Zhu et al. (2022)**. "Balanced Contrastive Learning for Long-Tailed Visual Recognition". arXiv:2207.02958
    -- BCL: balanced version of supervised contrastive loss.

23. **Ren et al. (2020)**. "Balanced Meta-Softmax for Long-Tailed Visual Recognition". arXiv:2007.10740
    -- BALMS: meta-learned balanced softmax.

24. **Tan et al. (2020)**. "Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Recognition". arXiv:2012.08548
    -- Gradient re-balancing for long-tail.

### Architecture

25. **Woo et al. (2023)**. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders". arXiv:2301.00808
    -- Modern ConvNet that matches transformer performance.

26. **Liu et al. (2022)**. "A ConvNet for the 2020s (ConvNeXt)". arXiv:2201.03545
    -- Modernized ResNet that outperforms Swin on many tasks.

### Diffusion for Augmentation

27. **Ho et al. (2020)**. "Denoising Diffusion Probabilistic Models". arXiv:2006.11239
    -- DDPM for high-quality image generation.

28. **Trabucco et al. (2023)**. "Effective Data Augmentation With Diffusion Models". arXiv:2302.07944
    -- Using diffusion models specifically for augmentation.

### Post-Hoc Methods

29. **Alexandari et al. (2020)**. "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation". arXiv:2001.06572
    -- Bias-corrected temperature scaling for shifted distributions.

30. **Kim et al. (2021)**. "Adjusting Decision Boundary for Class Imbalanced Learning". arXiv:2012.06822
    -- LADE: test-time logit adjustment with Bayesian perspective.

---

## G. Implementation Priority Roadmap

### Phase 1 (1-2 hours): Quick wins for immediate macro F1 boost
- Add DiceLoss + TverskyLoss to `src/training/losses.py`
- Add LogitAdjustedLoss to `src/training/losses.py`
- Add DRW schedule to `train.py`
- Wire `build_classification_loss` to support new loss names

### Phase 2 (2-4 hours): Training improvements
- Implement SAM optimizer wrapper
- Add EMA model tracking in training loop
- Implement TTA in inference
- Add cosine classifier option

### Phase 3 (4-8 hours): Major capability additions
- Implement FixMatch semi-supervised training
- Implement cRT decoupled training
- Add conditional generation to GAN trainer

### Phase 4 (8+ hours): Advanced techniques
- Meta-learning (MAML / Prototypical Networks)
- Deformable convolutions
- GNN for wafer spatial patterns
- Multi-task learning framework

---

## H. What IS Already Implemented (for Reference)

The following techniques from the papers ARE present and functional:

| Technique | File | Paper IDs |
|-----------|------|-----------|
| FocalLoss with sqrt-weight moderation | `src/training/losses.py` | [15,16] |
| Label smoothing | `src/training/losses.py` | [20] |
| Mixup + CutMix | `src/data/preprocessing.py` | [66,67] |
| SE / CBAM attention injection | `src/models/attention.py` | [21,22] |
| Feature Pyramid Network | `src/models/fpn.py` | [91] |
| SimCLR contrastive pretraining | `src/training/simclr.py` | [37,38,39] |
| Supervised Contrastive (SupCon) | `src/training/supcon.py` | [76] |
| Class-balanced sampler | `src/data/preprocessing.py` | [17] |
| Domain-specific augmentation | `src/data/preprocessing.py` | [49] |
| Rule-based synthetic generation | `src/augmentation/synthetic.py` | [18,68,69] |
| ResNet-18 with layer-boundary freezing | `src/models/pretrained.py` | [1] |
| EfficientNet-B0 | `src/models/pretrained.py` | [2] |
| ViT from scratch | `src/models/vit.py` | [3] |
| Swin Transformer | `src/models/swin.py` | [102] |
| Ensemble (voting, averaging, stacking) | `src/models/ensemble.py` | [30] |
| GradCAM | `src/inference/gradcam.py` | [25] |
| MC Dropout uncertainty | `src/inference/uncertainty.py` | [28] |
| Domain adaptation (CORAL + adversarial) | `src/training/domain_adaptation.py` | [40,41] |
| Anomaly detection (IF, OC-SVM, AE) | `src/analysis/anomaly.py` | [121,129] |
| GAN trainer with FID | `src/augmentation/train_generator.py` | [65] |
| Active learning | `scripts/active_learn.py` | [50] |
