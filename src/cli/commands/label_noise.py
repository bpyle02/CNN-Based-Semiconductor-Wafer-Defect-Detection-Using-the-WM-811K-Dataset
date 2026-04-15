"""``wafer-cli label-noise`` — forwards to ``scripts/label_noise.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def label_noise(ctx: typer.Context) -> None:
    """Label-noise robustness. Flags forwarded to scripts/label_noise.py."""
    from scripts.label_noise import main as noise_main

    rc = forward_to_main(noise_main, "label-noise", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
