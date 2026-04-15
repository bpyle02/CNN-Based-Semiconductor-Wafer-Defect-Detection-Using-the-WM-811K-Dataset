# Model Card: WM-811K Wafer Defect Classifier

Format based on the [Google Model Cards](https://modelcards.withgoogle.com) template
(Mitchell et al., 2019). This card covers the family of classifiers trained in this
repository for 9-class wafer map defect classification on WM-811K.

---

## Model Details

- **Name**: WM-811K Wafer Defect Classifier (AI 570 Team 4)
- **Model family**: Custom CNN, ResNet-18 (transfer), EfficientNet-B0 (transfer),
  ViT-S/16, Swin-T, RIDE (multi-expert long-tail), and an ensemble/distilled student
  built on top of the six.
- **Architecture**: Image classifier (3 channel 96x96 input, 9 logits output).
  Backbones initialized from ImageNet weights for ResNet / EfficientNet / ViT / Swin;
  custom CNN trained from scratch.
- **Input**: RGB-replicated wafer map, 96x96, values in [0, 1] (original values 0, 1, 2
  divided by 2.0).
- **Output**: Softmax probability over
  `{Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none}`.
- **Version**: 0.1.0 (coursework snapshot, 2026-04-14 run).
- **Training date**: 2026-04-14 (Colab T4 runs, 10 epochs each). See
  `results/metrics.json` for per-run provenance timestamps.
- **License**: Source under repository license (All Rights Reserved for the
  coursework). Weights are not distributed. WM-811K itself is governed by
  Wu et al. 2015 (IEEE TSM).
- **Contact**: Open an issue at
  <https://github.com/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/issues>
- **Citation**: See `CITATION.cff` in the repository root.

---

## Intended Use

### Primary intended use
Classify wafer map patterns from WM-811K-style inline inspection data into
the nine canonical defect classes for **offline analysis, research, and
coursework**. Typical downstream uses:

- Benchmarking new long-tail / imbalance-aware techniques on a well-known dataset.
- Exploratory analysis of which defect patterns dominate a given lot.
- Grad-CAM and MC-dropout uncertainty studies on wafer-map CNNs.

### Primary intended users
ML researchers, students, and process/yield engineers who want a baseline
for internal experimentation.

### Out-of-scope uses
- **Real-time in-fab deployment** without a full calibration verification and
  drift-monitoring layer. The distribution seen in WM-811K (2013-2015 fab
  technology) is not guaranteed to match a modern process node.
- Automated lot disposition or wafer scrap decisions with no human in the loop.
- Any safety-critical or contractual quality signal — this model is coursework-grade.
- Generalization to other inline inspection datasets (e.g., AOI images of die
  surfaces, SEM imagery). The model was trained only on WM-811K bin maps.
- Use on uncalibrated logits. The raw softmax is over/under-confident; apply the
  fitted temperature scaler (`results/metrics.json -> calibrated_metrics`) before
  exposing probabilities to any decision system.

---

## Factors

### Relevant factors
- **Defect class**: The model is evaluated per-class across the 9 WM-811K labels.
  Rare classes (Near-full, Donut, Random) dominate variance.
- **Wafer map resolution**: WM-811K wafer maps have heterogeneous native
  resolution. All inputs are resized to 96x96; performance on very small
  (<32x32 native) or very large (>200x200 native) wafers may degrade.
- **Pattern rarity / long-tail tier**: We group by `many-shot` (>1000 samples),
  `medium-shot` (100-1000), and `few-shot` (<100, i.e. Near-full) tiers.
- **Labeling era**: Labels were applied by fab engineers during 2013-2015;
  pattern prevalence reflects the process tech of that era.

### Evaluation factors used in this card
- Per-class F1.
- Macro F1 as primary aggregate (imbalance-aware).
- Expected Calibration Error (ECE) pre- and post-temperature scaling.
- Pattern-rarity breakdown (many/medium/few-shot).

---

## Metrics

- **Primary**: **Macro F1** — chosen because the dataset is ~85% "none" and
  accuracy is trivially inflated. Macro F1 weights each class equally.
- **Secondary**:
  - Per-class F1 (diagnostic — flags collapse modes).
  - Accuracy (for comparison with literature that reports it).
  - Weighted F1 (population-weighted, close to accuracy on this dataset).
  - **Expected Calibration Error (ECE)** pre/post temperature scaling — required
    because any downstream "confidence" signal must be honest.
  - Negative log-likelihood and Brier score for full-distribution scoring.
- **Decision thresholds**: Argmax over the 9 logits. No per-class threshold tuning.
- **Variability**: Numbers are from a single 10-epoch Colab T4 run per model.
  No variance is reported (single seed = 42); rare-class metrics — especially for
  Near-full (n=22 in test) — have high variance and should not be compared
  across runs without bootstrapping.

---

## Evaluation Data

- **Dataset**: WM-811K (Wu et al. 2015, IEEE TSM).
- **Split**: Stratified 70/15/15 (train/val/test), seed = 42, over the labeled
  subset (~172,950 wafers). Test split is the 15% slice.
- **Preprocessing**: Resize to 96x96 via `skimage.transform.resize`, divide by
  2.0 to map {0,1,2} -> [0, 0.5, 1.0], replicate single channel to 3 channels
  for ImageNet backbones.
- **Motivation**: Standard held-out split. Stratification preserves the 85%
  "none" prevalence in every split so that evaluation matches training
  distribution.

---

## Training Data

- **Dataset**: WM-811K labeled subset.
- **Split**: 70% stratified train, seed = 42 (~121,000 labeled wafers).
- **Preprocessing**: Same as evaluation data. Optional synthetic augmentation
  for rare classes and Mixup/CutMix when `--synthetic` / `--mixup` flags are
  set; default training run uses neither.
- **Class weights**: Inverse-frequency weights in cross-entropy
  (`total / (9 * count[c])`) for the CNN and ResNet baselines.
- See `docs/DATA_CARD.md` for provenance, biases, and license details.

---

## Quantitative Analyses

Numbers below are from `results/metrics.json` (2026-04-14 Colab T4 runs,
10 epochs, seed=42, 15% stratified test split).

### Headline metrics

| Model           | Accuracy | Macro F1 | Weighted F1 | ECE    | ECE (T-scaled) | Temperature |
|-----------------|---------:|---------:|------------:|-------:|---------------:|------------:|
| Custom CNN      | 0.9611   | **0.7988** | 0.9632    | 0.0068 | 0.0106          | 1.169       |
| ResNet-18       | 0.7770   | 0.6587   | 0.8352      | 0.0312 | 0.0306          | 1.144       |
| EfficientNet-B0 | 0.3742   | 0.0999   | 0.4966      | 0.4543 | 0.2749          | 2.433       |

The EfficientNet-B0 row reflects a documented training failure (inverse-frequency
CE collapse; see `results/metrics.json` note for that run). The ResNet-18 row also
partially collapsed on Edge-Loc (precision 0.137, recall 0.911) — Macro F1 is
understated for that single-class reason.

### Per-class F1 (Custom CNN, best reported)

| Class      | Precision | Recall | F1     | Support |
|------------|----------:|-------:|-------:|--------:|
| Center     | 0.861     | 0.894  | 0.877  | 644     |
| Donut      | 0.581     | 0.867  | 0.696  | 83      |
| Edge-Loc   | 0.713     | 0.819  | 0.762  | 779     |
| Edge-Ring  | 0.987     | 0.934  | 0.960  | 1452    |
| Loc        | 0.621     | 0.707  | 0.661  | 539     |
| Near-full  | 0.759     | 1.000  | 0.863  | 22      |
| Random     | 0.665     | 0.854  | 0.747  | 130     |
| Scratch    | 0.525     | 0.810  | 0.637  | 179     |
| none       | 0.993     | 0.978  | 0.985  | 22115   |

### Calibration

- Custom CNN is near-perfectly calibrated out of the box (ECE = 0.0068);
  applying temperature T = 1.169 does not improve ECE and slightly worsens
  it (0.0106). Recommendation: ship with T = 1.0.
- ResNet-18 temperature scaling is a wash (0.0312 -> 0.0306).
- EfficientNet-B0 temperature T = 2.43 reduces ECE from 0.454 to 0.275 but
  the underlying model is unusable; do not deploy.

### Long-tail tier performance (Custom CNN)

- Many-shot (none, Edge-Ring): F1 0.97 average.
- Medium-shot (Center, Edge-Loc, Loc, Scratch, Random): F1 0.74 average.
- Few-shot (Donut, Near-full): F1 0.78 average — but with n=83 and n=22 support,
  variance is high.

---

## Ethical Considerations

- **Labeling provenance**: Labels in WM-811K were applied by human fab engineers
  (Wu et al. 2015). Inter-annotator agreement is not published, and individual
  labeler bias on ambiguous patterns (e.g., Loc vs Edge-Loc vs Scratch) is
  carried through into every supervised model trained on this data.
- **Era bias**: Data was collected 2013-2015 on then-current process
  technology. Modern fabs (sub-7 nm, 3D-NAND, advanced packaging) produce
  different defect signatures. A model trained on WM-811K is a historical
  baseline, not a current-production inspector.
- **Class-imbalance bias**: 85% "none" inclines the model toward "no defect"
  predictions. In a yield-critical setting this is the expensive direction
  of error (false negatives skip a wafer that should have been flagged).
- **No PII**: Wafer maps contain no personally identifiable information.
  No individuals or operators are identifiable from the data.
- **Dual-use**: Defect classification is a process-optimization tool and
  has no weaponization pathway; however, fab yield data is commercially
  sensitive and this dataset was released under academic terms — do not
  conflate with current-production fab data.

---

## Caveats and Recommendations

- **Do not use raw logits as confidence.** Fit and apply the temperature
  scaler from `results/metrics.json -> calibrated_metrics.temperature` before
  surfacing probabilities to a downstream decision system. For the CNN the
  raw softmax is acceptable (ECE < 0.01); for ResNet/EfficientNet it is not.
- **Near-full has n=22 in the test split** (149 total across the full dataset).
  Reported F1 = 0.86 is high-variance; do not claim few-shot performance
  from a single run. Use bootstrap CIs or k-fold before making any
  few-shot claim.
- **Edge-Loc / Loc confusion** is the dominant residual error mode.
  Inspect confusion matrices (`results/*_confusion_matrix.png`) before
  deploying — a 10% Edge-Loc -> Loc flip may or may not be acceptable
  depending on downstream use.
- **Retrain before domain transfer.** Do not apply this model to non-WM-811K
  wafer maps without at least fine-tuning on a target-domain calibration set.
- **Checkpoint integrity.** Every published checkpoint must be verified via
  its `.sha256` sidecar before use. The current snapshot hashes are in
  `results/metrics.json` under `checkpoint_sha256_prefix`.
- **Reproducibility gap.** The `README.md` "expected results" table predates
  the 2026-04-14 run and shows conservative ranges (Macro F1 0.42-0.48 for
  the CNN); the committed `metrics.json` shows 0.7988. The difference is
  epoch count (5 vs 10) and minor pipeline fixes. Update the README when
  landing a new canonical run.
- **Single-seed evaluation.** All reported numbers use seed = 42. A
  multi-seed replication study is an open task.
