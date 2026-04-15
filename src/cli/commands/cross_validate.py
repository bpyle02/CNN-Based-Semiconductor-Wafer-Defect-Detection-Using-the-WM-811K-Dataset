"""``wafer-cli cross-validate`` — forwards to ``scripts/cross_validate.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def cross_validate(ctx: typer.Context) -> None:
    """K-fold cross-validation. Flags forwarded to scripts/cross_validate.py."""
    from scripts.cross_validate import main as cv_main

    rc = forward_to_main(cv_main, "cross-validate", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
