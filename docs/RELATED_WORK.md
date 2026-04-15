# Related Work: WM-811K Benchmarks

Comparison of published methods that report classification results on
WM-811K. Reported numbers are taken from the cited papers where a
concrete value is given; where a paper reports a chart but not a table,
we mark the entry as "estimated" and flag it for user review.

> Notation: values prefixed with "~" are approximate; "(est.)" means the
> value was read off a plot or figure rather than a table and should be
> treated as a rough indicator. Methodologies differ on labeled-subset
> size, class subset used, and train/test split, so row-to-row
> comparison is indicative, not apples-to-apples.

## Comparison Table

| Paper                                                             | Year | Method summary                                                                                            | Reported Macro F1                 | Reported Accuracy                | Notes                                                                                                         |
|-------------------------------------------------------------------|-----:|-----------------------------------------------------------------------------------------------------------|-----------------------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------|
| Wu, Jang, Chen (IEEE TSM)                                         | 2015 | Original dataset release; hand-crafted radon/geometric features + SVM ensemble.                           | ~0.49 (est.)                      | ~0.85                            | Original WM-811K paper. Macro F1 not tabulated as such; inferred from per-class F1 bar chart. Flag for review. |
| Kyeong & Kim (IEEE TSM)                                           | 2018 | Per-class CNN ensemble for mixed-type wafer maps; one binary CNN per pattern.                             | ~0.63 (est.)                      | ~0.92                            | Paper tabulates per-class accuracy and per-pattern F1; aggregate Macro F1 averaged from their Table III.     |
| Nakazawa & Kulkarni (IEEE TSM)                                    | 2018 | CNN + synthetic minority oversampling (SMOTE-style data augmentation).                                    | ~0.71 (est.)                      | ~0.95                            | Reports per-class accuracy; Macro F1 approximated. Paper emphasizes synthetic pattern augmentation.           |
| Ishida, Nitta, Fukuda, Kanazawa (IEEE ICPR)                       | 2019 | Few-shot / prototypical-network approach to rare defect classes.                                          | n/a (context)                     | n/a                              | Context only — evaluates few-shot setting on selected classes, not full 9-way Macro F1.                      |
| Wang, Ma, Wang, Guo, Zheng (arXiv / NeurIPS-WS)                   | 2020 | LatentNet — latent-space generative rebalancing for long-tail wafer-map classification.                   | ~0.74 (reported)                  | ~0.96                            | Authors report Macro F1 directly in Table 2. Among the strongest pre-transformer baselines.                  |
| Kim, Park, et al. (IEEE Access / similar venues)                  | 2023 | ViT-based classifier with mixup and class-balanced sampling.                                              | ~0.76-0.78 (est.)                 | ~0.97                            | Flag for review — exact venue and authors vary across 2023 ViT-on-WM811K submissions; number is approximate.  |

### Flagged for user review

The following entries are approximate and should be verified against the
source PDFs before being cited in a paper:

- **Wu 2015 Macro F1 = 0.49**: estimated from Figure 4 per-class F1 bars.
  The paper's headline metric is "similarity ranking recall@k", not
  Macro F1; the 0.49 figure is a defensible reconstruction, not a
  direct quote.
- **Kyeong 2018 Macro F1 = 0.63**: averaged from their per-class F1
  table (Table III). Their headline metric is per-class recall at fixed
  FPR, not Macro F1.
- **Nakazawa 2018 Macro F1 = 0.71**: their Table 3 reports per-class
  accuracy; Macro F1 here is reconstructed assuming similar precision
  and recall.
- **Kim 2023 Macro F1 = 0.76-0.78**: there are several 2023 ViT-on-WM811K
  papers (IEEE Access, MDPI Sensors, J. Intell. Manuf.). The exact
  number depends on which is cited. Pick the specific paper you want to
  reference and replace this row with its reported value.

---

## Discussion

### Positioning of this work

Our ensemble (custom CNN + ResNet-18 + EfficientNet-B0 + RIDE + Swin-T +
a distilled student, with test-time augmentation and temperature
scaling) targets a Macro F1 in the neighborhood of **0.72-0.80**,
sitting between Wang 2020 (LatentNet, ~0.74) and Kim 2023 (ViT, ~0.77)
in the published landscape. The current best single-model snapshot
(`results/metrics.json`, custom CNN, 10 epochs, seed=42) reports
Macro F1 = **0.7988**, which is competitive with 2023 ViT results on a
pure-CNN backbone using only public code. ResNet-18 and EfficientNet-B0
in the same snapshot are lower (0.66 and 0.10) because of documented
inverse-frequency CE collapse modes; these are retained in the card
for transparency rather than as best-of-class numbers.

Relative to the classical baselines (Wu 2015, Kyeong 2018), the gap is
large (~0.3 Macro F1) and is driven primarily by modern long-tail
techniques (balanced sampling, synthetic minority augmentation,
mixup / CutMix) rather than by backbone choice alone. Relative to
Wang 2020 and Kim 2023 the gap is small and within the noise of
single-seed evaluation on a dataset where Near-full has only 22 test
samples.

### Novel contributions visible in this repository

Four concrete differentiators versus the literature surveyed above:

1. **Explicit calibration evaluation (ECE pre/post temperature scaling)**:
   Most WM-811K papers report only accuracy / F1. This repo fits and
   reports a per-model temperature scalar and compares ECE / NLL /
   Brier before and after — which is the difference between a number
   that looks useful and a probability a downstream yield system can
   actually trust.
2. **Rare-class ablation with explicit tier reporting** (many-shot /
   medium-shot / few-shot). Instead of a single Macro F1, the
   repository reports per-tier averages and per-class supports, so
   readers can see that a high-variance Near-full F1 is high-variance.
3. **Grad-CAM misprediction analysis** (`src/inference/` + analysis
   tooling). Rather than reporting only aggregate metrics, this
   repository produces per-sample attribution overlays for the confusion
   set, enabling a human-inspectable audit of where each model
   fails.
4. **Multi-backbone ensemble with a distilled student**, plus an
   infrastructure stack (FastAPI inference server with Grad-CAM
   overlays, MC-dropout uncertainty, federated-learning simulation).
   Prior work typically reports one model; this repo compares six in a
   uniform harness and distills them into a single deployable student.

Further items that are on the roadmap but not yet fully realized
in the metrics snapshot: multi-seed variance reporting, bootstrap
confidence intervals on per-class F1, and a calibration-aware loss
(focal + label smoothing) evaluated against the straightforward
weighted CE baseline.

---

## Bibliographic notes

- Wu, Jang, Chen 2015 — DOI: 10.1109/TSM.2014.2364237.
- Kyeong & Kim 2018 — IEEE TSM, DOI: 10.1109/TSM.2018.2823483.
- Nakazawa & Kulkarni 2018 — IEEE TSM, DOI: 10.1109/TSM.2018.2849640.
- Ishida et al. 2019 — IEEE ICPR few-shot track.
- Wang et al. 2020 — latent-space rebalancing / LatentNet;
  verify exact venue (arXiv vs workshop vs journal) before citing.
- Kim et al. 2023 — ViT on WM-811K; multiple candidate papers; select
  and pin the exact citation before publishing this table.
