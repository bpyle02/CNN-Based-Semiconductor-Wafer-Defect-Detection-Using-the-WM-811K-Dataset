# Environment Doctor

`scripts/doctor.py` is a lightweight diagnostic tool for validating the local Python environment used by this repository.

## What it checks

- Python executable and version
- Supported Python range
- `python`, `pytest`, and `pip` path alignment
- Core package resolution and install locations
- `config.yaml` presence and dataset path health

## Usage

```bash
python scripts/doctor.py
python scripts/doctor.py --json
```

## Output modes

- Human-readable mode prints warnings and package locations directly to the terminal.
- `--json` emits a machine-readable summary that is suitable for CI, scripts, or release checks.
