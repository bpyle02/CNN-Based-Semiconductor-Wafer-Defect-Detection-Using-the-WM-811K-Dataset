# Post-mortem: 6-hour Colab run, 2026-04-14

> **Headline:** the CNN baseline succeeded (acc 0.961, macro F1 0.799,
> ECE 0.007). ResNet-18 partially collapsed; EfficientNet-B0 fully
> collapsed. The aggregated `metrics.json` captured only the EfficientNet
> entry — both the CNN success and the ResNet/EfficientNet failures
> were invisible until we recovered them from the notebook stdout.
> Both the file-race bug and the config default that caused the
> pretrained-model collapses are fixed in commit `83ba643`.

## What was run

- Subprocess loop in `docs/colab_quickstart.ipynb` Cell 6: `cnn`, then
  `resnet`, then `efficientnet`, each as its own `train.py` invocation.
- 10 epochs each, batch 64, seed 42, Colab Pro T4.
- Total wall-clock: ~6 h (CNN 1.8 h, ResNet 1.8 h, EfficientNet 1.7 h).
- Saved artifacts: `results/20260414_112925-20260414T113106Z-3-001/`
  - All three `best_<model>.pth` checkpoints made it to disk.
  - `results/metrics.json` contained ONLY EfficientNet metrics (the bug).

## Three results

### Custom CNN — strong baseline ✅

| metric | value |
|---|---|
| accuracy | **0.9611** |
| macro F1 | **0.7988** |
| weighted F1 | 0.9632 |
| ECE | 0.0068 |
| NLL | 0.123 |
| Brier | 0.058 |
| time | 6543 s (~1.8 h) |
| best epoch | 10/10 (still improving) |

Per-class (notable):

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Near-full | 0.76 | **1.00** | 0.86 | 22 |
| Scratch | 0.53 | 0.81 | 0.64 | 179 |
| Donut | 0.58 | 0.87 | 0.70 | 83 |
| Random | 0.66 | 0.85 | 0.75 | 130 |
| none | 0.99 | 0.98 | 0.99 | 22115 |

The CNN handled rare classes well *without* DRW or balanced sampling —
this is itself an interesting baseline for the upcoming rare-class study.

### ResNet-18 — partial collapse ⚠️

| metric | value |
|---|---|
| accuracy | 0.7770 |
| macro F1 | 0.6587 |
| weighted F1 | 0.8352 |
| ECE | 0.0312 |
| time | 6410 s (~1.8 h) |
| best epoch | 9/10 |

Localized failure: Edge-Loc precision 0.137 / recall 0.911 means the
model over-predicts Edge-Loc on many true `none` samples. That single-
class confusion accounts for most of the macro F1 deficit. Validation
loss stayed in 0.6–1.1 range (no divergence).

### EfficientNet-B0 — full collapse ❌

| metric | value |
|---|---|
| accuracy | 0.3742 |
| macro F1 | 0.0999 |
| weighted F1 | 0.4966 |
| ECE | 0.4543 |
| best epoch | 4/10 |

Per-class catastrophe: Near-full recall 0.86 with precision **0.003** —
the model is predicting "Near-full" on nearly every input. Validation
loss bounced 20–123 across epochs; training loss declined smoothly to
0.65 / training accuracy 84.6%. 100× train/val loss separation = pipeline
pathology.

## Why CNN worked but pretrained models collapsed

Same loss config, same data, same seed, same hardware. The difference is
**learning rate**:
- Custom CNN: `lr=1e-3` (default for from-scratch CNN)
- ResNet / EfficientNet: `lr=1e-4` (default fine-tuning rate)

Class weights at run time (logged):
```
Center=4.47, Donut=34.58, Edge-Loc=3.70, Edge-Ring=1.99,
Loc=5.35, Near-full=129.34, Random=22.20, Scratch=16.11, none=0.13
```

A 1000× ratio between Near-full (w=129) and `none` (w=0.13). Under
inverse-frequency CE without DRW smoothing:

- A wrong Near-full prediction → ~129 contribution to batch mean loss
- A wrong `none` prediction → ~0.0001 contribution

The CNN's 10× higher learning rate moved its parameters fast enough in
early epochs to find a basin where common-class predictions were already
roughly correct before rare-class gradients dominated. Pretrained models
take small careful steps from already-useful ImageNet features — they
got dragged off the good initial manifold by the heavy rare-class
gradients. ResNet partially recovered (Edge-Loc collapse only),
EfficientNet did not.

This is the textbook failure mode DRW (Cao et al. 2019, arXiv:1906.07413)
was designed to prevent: leave loss unweighted for first N epochs while
the model learns useful representations, then enable weights for the
remaining epochs.

## Two bugs, two fixes (`83ba643`)

### Bug 1: `metrics.json` clobbered

`train.py:937` (pre-fix) did `json.dump(self.results, f)` — but
`self.results` is empty at the start of each subprocess. Three
sequential `train.py` calls = three sequential overwrites. Only the last
survives.

**Fix:** load existing metrics.json, merge new results, write back.
Subprocess loops accumulate correctly. Backward-compatible.

### Bug 2: `weighted: true` default with no DRW

`config.yaml:179` shipped `training.loss.weighted: true` and `drw_epoch: 0`.
That's the loaded gun this run discharged.

**Fix:** flipped default to `weighted: false`. Comment in config.yaml
explains the trade-off and points to the rare-class study (in progress)
that benchmarks weighted-CE+DRW vs focal vs balanced-sampling vs
synthetic-augmentation vs unweighted-baseline on 3 seeds each.

## Recovery

The CNN and ResNet metrics that never made it to `metrics.json` were
recovered from notebook stdout (user copy-pasted them verbatim from
Colab cell history). All three model entries are now in
`results/metrics.json` with `source` annotations and notes flagging the
two collapse cases.

## Where I disagree with the parallel review

A second agent reviewed the (single-model) `metrics.json` and diagnosed
the EfficientNet collapse as "class imbalance not addressed: add weighting".

That reading is backwards. `train.py:528` shows the inverse-frequency
`loss_weights` tensor is passed to CE whenever `loss_cfg.weighted` is
true, regardless of `drw_epoch` or `adaptive_rebalance`. The run *was*
weighting heavily. Adding more weighting deepens the collapse.

The peer-review agent's other points are valid follow-ups:
- LR too high for fine-tuning (1e-4 → 1e-5 with warmup + cosine decay)
- EMA disabled; would have smoothed noisy validation metrics
- Best epoch 4/10 → loss-landscape collapse, not missing regularizer

These belong in the rare-class study or its follow-ups.

## Performance fixes shipped in the same batch

Three independent throughput improvements, motivated by an earlier
data-pipeline audit:

1. `mixed_precision: true` (was false) — 15-25% AMP speedup on CUDA
2. `persistent_workers=True, prefetch_factor=2` on DataLoader — 10-20%
3. `cudnn.benchmark=True` by default in `set_seed()` — 8-15% on Ampere

Combined: ~30-50% faster training on Colab Pro. Next 10-epoch CNN run
should finish in ~1 h instead of 1.8 h.

## Recommendations

1. **Adopt the CNN result as the team's headline baseline.** It's
   genuinely strong: 0.961 acc, 0.799 macro F1, calibrated (ECE 0.007),
   with strong rare-class recall.
2. **Re-train ResNet-18 and EfficientNet-B0** under the new defaults.
   Expected to land near the CNN with proper warmup + lower fine-tune LR.
3. **Run the rare-class study** to compare rebalancing interventions
   head-to-head. The CNN's strong rare-class recall *without*
   intervention is an interesting baseline.
4. **The two collapsed checkpoints stay in the repo as teaching
   artifacts.** They cleanly demonstrate the failure mode of unsmoothed
   inverse-frequency CE on extreme imbalance — useful for the report.
