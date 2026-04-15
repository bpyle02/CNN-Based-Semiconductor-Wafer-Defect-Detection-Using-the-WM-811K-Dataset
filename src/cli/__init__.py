"""Unified Typer-based CLI for the wafer-defect-detection project.

Exposes every existing script under a single ``wafer-cli`` entry point
as a subcommand. Each subcommand is a thin Typer wrapper around the
existing script's ``main()`` / argparse implementation so that CLI flag
parity is preserved verbatim.

The Typer application lives in :mod:`src.cli.main` and is imported
lazily to keep ``python -m src.cli.main`` free of the ``found in
sys.modules after import of package`` runpy warning.
"""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from src.cli.main import app as _app

        return _app
    raise AttributeError(f"module 'src.cli' has no attribute {name!r}")
