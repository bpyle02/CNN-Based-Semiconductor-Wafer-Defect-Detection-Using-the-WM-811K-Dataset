# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use
[SemVer](https://semver.org/) once the project reaches `1.0.0`.

## [Unreleased]

### Added
- Baseline metrics snapshot at `results/metrics.baseline.json` with a
  regression-guard script (`scripts/check_metrics.py`) wired into CI.
- `docs/REPRODUCE_METRICS.md` — end-to-end recipe to bit-for-bit reproduce
  the committed baseline from a clean clone.
- `.github/PULL_REQUEST_TEMPLATE.md` — reviewer discipline checklist
  (linked issue, `make check-all`, seed 42, metrics check, docs).

### Fixed
- `Makefile` `smoke` target no longer passes the non-existent `--smoke-test`
  flag to `train.py`.
- `configs/colab_fast.yaml` no longer declares `training.checkpointing.save_frequency`
  (Pydantic strict-extra would reject the overlay).
- `scripts/smoke_test.sh` no longer depends on `data/LSWMD_new.pkl` — the
  step needed for a dataset-less CI environment has been removed; the
  full-dataset smoke is still available via `make smoke`.
- Notebook Cell 6 `USE_DRIVE_CKPT` comment points at the correct Drive-mount
  cells (3 or 8) instead of Cell 2 (GPU check).

## [0.1.0] — 2026-04-14

First tagged baseline for the AI 570 Team 4 coursework submission. Tag and
GitHub Release are pending (see `TODO_BRANDON.md`).

### Headline result

Custom CNN on WM-811K test split, seed 42, 10 epochs, Colab T4:

| metric | value |
|---|---|
| accuracy | 0.9611 |
| macro F1 | 0.7988 |
| weighted F1 | 0.9632 |
| ECE | 0.0068 |

> The numbers committed in `results/metrics.json` / `metrics.baseline.json`
> come from a shorter 1-epoch run and are weaker. See
> `docs/REPRODUCE_METRICS.md` for how to rerun the 10-epoch Colab baseline
> that produced the headline numbers above.

### Added
- PyTorch training pipeline: custom CNN, ResNet-18, EfficientNet-B0,
  ViT, Swin, with transfer-learning options.
- Advanced training features: early stopping, EMA, DRW, MixUp, CutMix,
  SimCLR pretraining, adaptive rebalancing, synthetic augmentation.
- FastAPI inference server (`src/inference/server.py`) with Grad-CAM
  endpoint and a TorchServe-compatible `ModelServer` class.
- Colab quickstart notebook (`docs/colab_quickstart.ipynb`) with env
  snapshot, 1-epoch preflight, tee-style training cell, and tight Drive
  checkpoint integration.
- CI workflows: `ci.yml` (pytest across 3.10–3.13), `model_validation.yml`
  (manual GPU run trigger).
- Developer tooling: Makefile, `scripts/smoke_test.sh`, pre-commit
  (black, isort, flake8, mypy, nbstripout), `scripts/doctor.py`,
  `scripts/generate_lock.py`.
- Packaging: `pyproject.toml` + `requirements.txt` + `requirements-lock.txt`,
  multi-stage Dockerfile (development / production / jupyter).
- Team docs: `README.md` (classmate-replication guide), `CONTRIBUTING.md`,
  `CITATION.cff`, `.github/CODEOWNERS`, `LICENSE` (All-Rights-Reserved
  placeholder pending team vote).
- 188 unit + integration tests covering models, training, data, inference,
  federated, OOD, uncertainty quantification.

### Notes
- Dataset (`LSWMD_new.pkl`) is not in the repo. See `README.md §2` for how
  to acquire it from Kaggle / IEEE DataPort / Google Drive.
- A permanent open-source license has not yet been chosen. Until the team
  votes, the repo is viewable on GitHub but not relicensable.
