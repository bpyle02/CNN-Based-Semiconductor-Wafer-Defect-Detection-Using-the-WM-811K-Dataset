# Environment Doctor

`scripts/doctor.py` is a lightweight diagnostic tool for validating the local Python environment used by this repository.

## What it checks

- Python executable and version
- Supported Python range
- Active conda environment and recommended `base` workflow
- `python`, `pytest`, and `pip` path alignment
- Core package resolution and install locations
- `config.yaml` presence and dataset path health
- Whether packages are resolving from inside the active conda prefix or leaking in from user-site installs

## Usage

```bash
conda run -n base python scripts/doctor.py
conda run -n base python scripts/doctor.py --json
```

## Output modes

- Human-readable mode prints warnings and package locations directly to the terminal.
- `--json` emits a machine-readable summary that is suitable for CI, scripts, or release checks.
- The doctor warns when the active conda environment is not `base`, because the repository workflow is standardized on `conda run -n base ...`.
- The doctor also notes when core packages resolve outside the active conda prefix, which usually means the environment is not fully unified.
