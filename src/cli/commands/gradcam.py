"""``wafer-cli gradcam`` — forwards to ``scripts/gradcam_errors.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def gradcam(ctx: typer.Context) -> None:
    """Grad-CAM on misclassifications. Flags forwarded to scripts/gradcam_errors.py."""
    from scripts.gradcam_errors import main as gradcam_main

    rc = forward_to_main(gradcam_main, "gradcam", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
