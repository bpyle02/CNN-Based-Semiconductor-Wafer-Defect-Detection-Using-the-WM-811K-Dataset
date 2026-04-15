"""``wafer-cli federated`` — forwards to ``scripts/federated_demo.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def federated(ctx: typer.Context) -> None:
    """FedAvg demo. Flags forwarded to scripts/federated_demo.py."""
    from scripts.federated_demo import main as federated_main

    rc = forward_to_main(federated_main, "federated", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
