# Data Card: WM-811K

Format based on the [Google Data Card](https://modelcards.withgoogle.com/about)
template. This card describes the WM-811K dataset as used by this repository.

---

## Dataset Overview

- **Name**: WM-811K (Wafer Map 811K).
- **Primary citation**: M.-J. Wu, J.-S. R. Jang, and J.-L. Chen, "Wafer Map
  Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets,"
  *IEEE Transactions on Semiconductor Manufacturing*, vol. 28, no. 1, pp. 1-12,
  Feb. 2015.
- **Total wafer maps**: 811,457.
- **Labeled subset**: 172,950 wafers across 9 classes (the remainder are
  unlabeled / "unknown" and are excluded from supervised training in this
  repository).
- **Classes (9)**: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`,
  `Near-full`, `Random`, `Scratch`, `none`.
- **Modality**: 2-D integer wafer bin maps (values in `{0, 1, 2}` for
  `background / pass / fail`) of heterogeneous native resolution.
- **Format in this repo**: `data/LSWMD_new.pkl` (pandas pickle, shape
  `(811457, 7)`), schema: `waferMap`, `failureType`, `trianTestLabel`,
  plus lot/die metadata.

---

## Sensitive Data

- **None.** Wafer maps contain no PII, personal data, or identifiable
  operator information.
- Lot IDs are pseudonymous integers; no original manufacturer or
  individual is identifiable from the data as distributed.

---

## Dataset Version & License

- **Dataset version used**: The Kaggle redistribution at
  <https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map>, preprocessed
  into `LSWMD_new.pkl`.
- **Original rights holder**: The dataset accompanies Wu et al. 2015
  (IEEE TSM). The 2015 paper and associated MIR Lab release define the
  terms of academic use.
- **Redistribution**: The raw pickle is not versioned in this repository
  (2.2 GB). Users must obtain it from Kaggle or a permitted mirror.
- **License of this card and any derived metrics**: Same as the host
  repository (coursework / All Rights Reserved on code; facts about the
  dataset are not themselves copyrightable).
- **Allowed uses**: Academic research, benchmarking, and coursework. Any
  commercial or production deployment must revisit the upstream rights.

---

## Data Collection Process

- **Source**: Pulled from real production semiconductor fabs, circa 2013-2015,
  as reported by Wu et al. 2015.
- **Collection mechanism**: Inline electrical test results were rasterized
  into per-wafer die-bin maps.
- **Labeling mechanism**: Human fab engineers inspected each wafer map and
  assigned one of the 9 pattern labels. The bulk of the dataset (~639,000
  wafers) was left unlabeled.
- **Time span**: 2013-2015.
- **Process technology**: Commercial foundry processes of that era;
  specific node(s) are not published.
- **Inter-annotator agreement**: Not reported by the source paper. Labels
  must be treated as single-annotator expert judgments.

---

## Class Distribution

From the labeled subset (n = 172,950):

| Defect Class | Count     | Percent  |
|--------------|----------:|---------:|
| none         | 147,431   | 85.24%   |
| Edge-Ring    |   9,680   |  5.60%   |
| Edge-Loc     |   5,189   |  3.00%   |
| Center       |   4,294   |  2.48%   |
| Loc          |   3,593   |  2.08%   |
| Scratch      |   1,193   |  0.69%   |
| Random       |     866   |  0.50%   |
| Donut        |     555   |  0.32%   |
| Near-full    |     149   |  0.09%   |
| **Total**    | **172,950** | **100.00%** |

Imbalance ratio (`none` : `Near-full`) is ~990:1. Any evaluation on this
dataset must report Macro F1 or another imbalance-aware aggregate;
accuracy alone is uninformative.

---

## Known Biases

- **Severe class imbalance (85% "none")**. Naive training converges to an
  "always predict none" classifier unless inverse-frequency weighting,
  DRW, focal loss, or a balanced sampler is used.
- **Rare-class instability**. Classes with <1% prevalence (Donut, Random,
  Near-full) produce high-variance per-class metrics; single-seed
  experiments can shift F1 by 0.05-0.10 between runs.
- **Labeler drift**. With single-expert labels over a 2-3 year collection
  window, gradual drift in the boundary between adjacent classes
  (e.g., Loc vs Edge-Loc vs Scratch) is plausible but unmeasured.
- **Era-specific pattern prevalence**. The frequency of each failure mode
  reflects the process tech, defect-density, and tooling of early-2010s
  fabs. A modern advanced-node fab will have a different prior over
  failure classes (e.g., more patterning-related, fewer particle-type
  defects).
- **Resolution heterogeneity**. Native wafer-map resolution varies widely
  across lots; resizing to 96x96 loses fine-structure information on
  large wafers and over-smooths on small ones.
- **Unlabeled majority**. 78% of the dataset is unlabeled. Training on
  only the labeled subset means the supervised estimator sees a
  non-random sample of production wafers — engineers may have preferentially
  labeled interesting / ambiguous cases.

---

## Preprocessing

- **Resize**: All wafer maps resized to **96 x 96** via
  `skimage.transform.resize` (bilinear, anti-aliasing on).
- **Value normalization**: Integer values `{0, 1, 2}` divided by `2.0` to
  land in `[0, 0.5, 1.0]`.
- **Channel replication**: The single resulting channel is replicated to
  3 channels so that ImageNet-pretrained backbones
  (ResNet, EfficientNet, ViT, Swin) ingest the expected shape.
- **No mean/std standardization**: Inputs are passed as-is to the model;
  models that need ImageNet statistics apply them inside their first layer
  or first normalization module.
- **Label encoding**: The 9 classes are mapped to integer IDs (0-8) in a
  deterministic order defined in `src/data/dataset.py`.

---

## Splits

- **Strategy**: Stratified 70 / 15 / 15 (train / val / test).
- **Seed**: `42` (global, fixed; also controls shuffling and weight init).
- **Rationale**: Stratification preserves the 85% "none" prevalence in
  every split so reported metrics generalize to the overall label
  distribution. Fixed seed enables exact reproducibility.
- **Approximate sizes**:
  - Train: ~121,000 wafers.
  - Val: ~25,950 wafers.
  - Test: ~25,950 wafers (Near-full test support = 22).

---

## Intended Use

- Benchmarking wafer-map defect-pattern classifiers.
- Long-tail / imbalanced-classification research.
- Coursework and didactic examples of ML on real industrial data.
- Baseline for anomaly / OOD detection on structured binary-valued images.

## Out-of-Scope Use

- Production fab deployment without domain adaptation and drift monitoring.
- Training a model intended for a non-WM-811K wafer inspection task
  (different sensor, different technology node) without re-labeling.
- Any privacy / demographic inference (no such information exists in
  the data).
- Claims of generality across fab technologies or vendors — this dataset
  is from one documented study and should not be interpreted as a global
  wafer-defect prior.
