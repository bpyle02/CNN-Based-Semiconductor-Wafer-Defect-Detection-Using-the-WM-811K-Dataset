#!/usr/bin/env python3
"""Backwards-compat shim — use ``wafer-cli kaggle push`` instead.

This script has been migrated into the unified ``wafer-cli`` framework.
The real implementation now lives in :mod:`src.cli.commands.kaggle`.

This shim is kept so that existing CI jobs, docs, and muscle memory
(``python scripts/kaggle_push.py``) keep working — it just forwards
to the new CLI with a deprecation warning.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Ensure the repo root is importable when invoked as
# ``python scripts/kaggle_push.py`` (no editable install).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    warnings.warn(
        "scripts/kaggle_push.py is deprecated; use "
        "`wafer-cli kaggle push` (or `python -m src.cli.main kaggle push`) "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Also print to stderr so users actually see it regardless of
    # -W filters.
    print(
        "[deprecated] scripts/kaggle_push.py — forwarding to " "`wafer-cli kaggle push`.",
        file=sys.stderr,
    )

    # If the user passed --help, show the new CLI's help for the push
    # subcommand and exit 0 (preserving old help-flag UX).
    argv = sys.argv[1:]
    from typer.testing import CliRunner

    from src.cli.main import app

    if any(arg in ("-h", "--help") for arg in argv):
        runner = CliRunner()
        result = runner.invoke(app, ["kaggle", "push", "--help"])
        # Windows cp1252 can't encode Typer's box-drawing chars; write
        # bytes directly with a safe fallback.
        try:
            sys.stdout.write(result.output)
        except UnicodeEncodeError:
            sys.stdout.write(result.output.encode("ascii", errors="replace").decode("ascii"))
        return result.exit_code

    # Otherwise, invoke the real command directly.
    from src.cli.commands.kaggle import main as kaggle_main

    return kaggle_main()


if __name__ == "__main__":
    sys.exit(main())
