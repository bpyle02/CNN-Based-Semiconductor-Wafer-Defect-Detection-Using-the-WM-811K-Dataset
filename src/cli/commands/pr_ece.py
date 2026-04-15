"""``wafer-cli pr-ece`` — forwards to ``scripts/pr_curves_ece.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def pr_ece(ctx: typer.Context) -> None:
    """Per-class PR + ECE. Flags forwarded to scripts/pr_curves_ece.py."""
    from scripts.pr_curves_ece import main as pr_ece_main

    rc = forward_to_main(pr_ece_main, "pr-ece", ctx.args, accepts_argv=True)
    raise typer.Exit(code=rc)
