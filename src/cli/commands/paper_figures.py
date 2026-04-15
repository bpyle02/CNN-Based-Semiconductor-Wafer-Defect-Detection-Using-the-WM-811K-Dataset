"""``wafer-cli paper-figures`` — forwards to ``scripts/paper_figures.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def paper_figures(ctx: typer.Context) -> None:
    """Regenerate paper figures. Flags forwarded to scripts/paper_figures.py."""
    from scripts.paper_figures import main as paper_main

    rc = forward_to_main(paper_main, "paper-figures", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
