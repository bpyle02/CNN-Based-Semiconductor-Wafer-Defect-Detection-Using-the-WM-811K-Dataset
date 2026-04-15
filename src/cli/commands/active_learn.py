"""``wafer-cli active-learn`` — forwards to ``scripts/active_learn.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def active_learn(ctx: typer.Context) -> None:
    """Active learning study. Flags forwarded to scripts/active_learn.py."""
    from scripts.active_learn import main as active_main

    rc = forward_to_main(active_main, "active-learn", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
