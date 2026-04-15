"""Shared helpers for Typer-based subcommands.

Every existing script under ``scripts/`` (and ``train.py``) already owns
its own ``argparse`` parser. Rather than duplicating those flag
definitions inside Typer (which would drift), each subcommand forwards
its extra CLI args straight through to the underlying ``main()``.

The helpers in this module patch ``sys.argv`` to the forwarded args,
invoke the target callable, and return its exit code.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def patched_argv(prog: str, args: Sequence[str]):
    """Temporarily replace ``sys.argv`` with ``[prog, *args]``."""
    saved = sys.argv
    sys.argv = [prog, *list(args)]
    try:
        yield
    finally:
        sys.argv = saved


def forward_to_main(
    main_fn: Callable[..., Optional[int]],
    prog: str,
    argv: Iterable[str],
    *,
    accepts_argv: bool = False,
) -> int:
    """Invoke an underlying script ``main()`` with the given argv.

    - ``accepts_argv=True`` means the callable signature is
      ``main(argv=None)``; we pass the list directly.
    - Otherwise we patch ``sys.argv`` so the callable's internal
      ``argparse.parse_args()`` sees the forwarded arguments.

    Returns the integer exit code (0 if the callable returned ``None``).
    """
    argv_list: List[str] = list(argv)
    if accepts_argv:
        with patched_argv(prog, argv_list):
            rc = main_fn(argv_list)
    else:
        with patched_argv(prog, argv_list):
            rc = main_fn()
    return int(rc) if rc is not None else 0


# Typer context settings used by every forwarding subcommand so that
# arbitrary ``--flag value`` pairs are passed through untouched.
FORWARD_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": ["-h", "--help"],
}
