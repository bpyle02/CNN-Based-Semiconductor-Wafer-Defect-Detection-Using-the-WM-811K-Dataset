# results/

Conventional output location for training and evaluation artifacts.

## What goes here

- `metrics.json` — latest canonical metrics snapshot (tracked in git).
- Confusion matrices, per-class bar plots, calibration curves generated
  by `src/analysis/visualize.py`.
- Grad-CAM overlay grids from `src/inference/visualize.py`.

## What does NOT go here (gitignored)

- Model checkpoints (`*.pth`, `*.pt`) — too big; upload to W&B / MLflow
  or your own Drive.
- Full run directories written by `train.py --save-dir …` — those live in
  `wafer_runs/` (gitignored).
- Any CSV / NPZ larger than ~1 MB — attach to a GitHub Release or put on
  Drive and link from the PR.

## Refreshing `metrics.json`

Any time you land a meaningful model change, run the full pipeline and
commit the updated `metrics.json` in the same PR so CI and the README
reflect reality.
