"""``wafer-cli benchmark`` — forwards to ``scripts/benchmark.py``."""

from __future__ import annotations

import typer

from src.cli._common import forward_to_main


def benchmark(ctx: typer.Context) -> None:
    """Benchmark a trained model. Flags forwarded to scripts/benchmark.py."""
    from scripts.benchmark import main as benchmark_main

    rc = forward_to_main(benchmark_main, "benchmark", ctx.args, accepts_argv=True)
    raise typer.Exit(code=rc)
