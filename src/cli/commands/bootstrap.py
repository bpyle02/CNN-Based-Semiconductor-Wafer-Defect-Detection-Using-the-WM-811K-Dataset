"""``wafer-cli bootstrap`` — forwards to ``scripts/bootstrap_ci.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def bootstrap(ctx: typer.Context) -> None:
    """Bootstrap CIs. Flags forwarded to scripts/bootstrap_ci.py."""
    from scripts.bootstrap_ci import main as bootstrap_main

    rc = forward_to_main(bootstrap_main, "bootstrap", ctx.args, accepts_argv=True)
    raise typer.Exit(code=rc)
