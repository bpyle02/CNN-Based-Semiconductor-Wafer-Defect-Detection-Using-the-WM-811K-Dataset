"""``wafer-cli train`` — forwards to the repo-root ``train.py`` entry point."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def train(ctx: typer.Context) -> None:
    """Train a model. All flags are forwarded to ``train.py``'s argparse.

    Run ``wafer-cli train --help`` for the (Typer) wrapper help, or pass
    any underlying ``train.py`` flags directly
    (``--model``, ``--epochs``, ``--config``, etc.).
    """
    # Lazy import: keep ``from src.cli.main import app`` cheap.
    from train import main as train_main  # type: ignore[import-not-found]

    rc = forward_to_main(train_main, "train", ctx.args, accepts_argv=False)
    raise typer.Exit(code=rc)
