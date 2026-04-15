"""``wafer-cli quantize`` — forwards to ``scripts/quantize.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def quantize(ctx: typer.Context) -> None:
    """Post-training quantization. Flags forwarded to scripts/quantize.py."""
    from scripts.quantize import main as quantize_main

    rc = forward_to_main(quantize_main, "quantize", ctx.args, accepts_argv=True)
    raise typer.Exit(code=rc)
