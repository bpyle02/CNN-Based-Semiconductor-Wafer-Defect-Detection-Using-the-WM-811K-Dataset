"""``wafer-cli export-onnx`` — forwards to ``scripts/export_onnx.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def export_onnx(ctx: typer.Context) -> None:
    """Export a checkpoint to ONNX. Flags forwarded to scripts/export_onnx.py."""
    from scripts.export_onnx import main as export_main

    rc = forward_to_main(export_main, "export-onnx", ctx.args, accepts_argv=True)
    raise typer.Exit(code=rc)
