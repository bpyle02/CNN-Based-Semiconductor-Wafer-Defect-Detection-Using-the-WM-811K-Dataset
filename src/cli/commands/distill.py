"""``wafer-cli distill`` — forwards to ``scripts/distill.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def distill(ctx: typer.Context) -> None:
    """Knowledge distillation. Flags forwarded to scripts/distill.py."""
    from scripts.distill import main as distill_main

    rc = forward_to_main(distill_main, "distill", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
