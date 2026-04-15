"""``wafer-cli ood`` — forwards to ``scripts/ood_analysis.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def ood(ctx: typer.Context) -> None:
    """OOD / anomaly analysis. Flags forwarded to scripts/ood_analysis.py."""
    from scripts.ood_analysis import main as ood_main

    rc = forward_to_main(ood_main, "ood", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
