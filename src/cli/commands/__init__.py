"""Subcommand modules for the unified ``wafer-cli`` Typer app.

Each module exposes a single Typer callback function that is registered
in :mod:`src.cli.main`. Keeping one file per subcommand mirrors the
existing 1:1 mapping with ``scripts/*.py``.
"""
