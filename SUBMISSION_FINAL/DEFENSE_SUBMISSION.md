# PhD Defense Submission Guide

## Purpose

This is the operational submission note for the committee-facing package as it exists on `2026-03-22`.

Use this file together with:

- `docs/DEFENSE_PACKET.md`
- `docs/FINAL_STATUS_REPORT.md`
- `docs/presentation.pdf`
- `docs/wafer_defect_detection_report.pdf`

## Verified State

- Repository validation date: `2026-03-22`
- Current automated verification: `pytest -q` -> `45 passed, 6 skipped`
- The inference API supports Grad-CAM overlays in prediction responses.
- The repository includes a defense smoke-demo script and wrapper.

This repository is strongest as a defended software artifact, not as a stand-alone benchmark claim. Quantitative result tables should come from separately archived experiment outputs and checkpoints.

## Authoritative Submission Files

Committee-facing materials should prioritize these files:

1. `docs/DEFENSE_PACKET.md`
2. `docs/presentation.pdf`
3. `docs/presentation.tex`
4. `docs/wafer_defect_detection_report.pdf`
5. `docs/wafer_defect_detection_report.tex`
6. `docs/FINAL_STATUS_REPORT.md`
7. `docs/wafer_defect_detection_run.ipynb`

Operational files for replay:

1. `README.md`
2. `requirements.txt`
3. `pytest.ini`
4. `scripts/run_defense_demo.ps1`
5. `scripts/defense_smoke_demo.py`
6. `scripts/finalize_submission.py`

## Final Submission Workflow

### 1. Verify the repository

```bash
pytest -q
```

Expected local result in the current workspace:

```text
45 passed, 6 skipped
```

### 2. Package the committee bundle

```bash
python scripts/finalize_submission.py
```

That script creates:

- `SUBMISSION_FINAL/`
- `SUBMISSION_FINAL.zip`

### 3. Run the live demo path

```bash
./scripts/run_defense_demo.ps1
```

What it demonstrates:

- model loading
- health check
- prediction request handling
- Grad-CAM overlay generation

## Recommended Presentation Sequence

1. Open `docs/presentation.pdf`
2. State the repo's verified scope as a software artifact
3. Show `pytest -q`
4. Run `./scripts/run_defense_demo.ps1`
5. Show the generated Grad-CAM artifact if the environment has FastAPI installed
6. Close with limitations rather than overstating deployment maturity

## Known Blockers And Caveats

- The repository does not ship a committee-approved trained checkpoint.
- The smoke demo requires `fastapi` in the same environment that resolves `pytest`.
- The checked-in PDFs are present, but rebuilding them still depends on a working local LaTeX installation.
- Historical course-era documents remain in `docs/`; the files listed above are the authoritative defense materials.

## Committee-Safe Claims

Claims supported directly by the repository:

- The codebase is modular and testable.
- Core inference, uncertainty, anomaly, and federated-learning utilities are implemented.
- The inference API now supports explainability through Grad-CAM.
- The current local test suite passes in the checked workspace.

Claims that should not be made without separate evidence:

- final benchmark numbers
- production-readiness certification
- security hardening or deployment guarantees
- reproducible quantitative performance without pinned datasets and checkpoints
