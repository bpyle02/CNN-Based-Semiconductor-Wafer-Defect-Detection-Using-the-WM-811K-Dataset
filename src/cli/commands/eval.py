"""``wafer-cli eval`` — forwards to ``scripts/validate_model.py``.

The legacy script runs calibration + MC-Dropout uncertainty audit on a
trained checkpoint. The top-level module name is ``eval`` (an exported
Python builtin shadowed only inside this module's namespace).
"""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def eval_cmd(ctx: typer.Context) -> None:
    """Evaluate a trained model. Flags forwarded to scripts/validate_model.py."""
    from scripts.validate_model import main as validate_main

    rc = forward_to_main(validate_main, "eval", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
