"""``wafer-cli calibrate`` — temperature scaling / calibration metrics.

No dedicated legacy script exists for just "calibrate"; the closest
existing CLIs are:

* ``scripts/validate_model.py`` — full MC-Dropout + calibration audit
  (reused by ``wafer-cli eval``), and
* ``scripts/pr_curves_ece.py`` — per-class PR + ECE
  (reused by ``wafer-cli pr-ece``).

For CLI parity we forward to ``scripts/validate_model.py`` since its
``--- Fitting Temperature Scaling ---`` block is exactly the
calibration workflow a user would expect from ``calibrate``. When a
standalone calibration CLI is introduced later, swap the import here.
"""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def calibrate(ctx: typer.Context) -> None:
    """Fit temperature scaling + report ECE. Flags forwarded to scripts/validate_model.py."""
    from scripts.validate_model import main as validate_main

    rc = forward_to_main(validate_main, "calibrate", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
