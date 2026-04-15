"""Sequentially run a batch of analysis/deployment scripts.

Extracted from Cells 13 (analysis suite) and 14 (deployment suite) of
``docs/kaggle_quickstart.ipynb``. Each step is a (label, argv) pair; a
non-zero exit does not abort the batch — the caller inspects the return
value to decide what to do.
"""

from __future__ import annotations

import subprocess
from typing import Iterable, Sequence


def run_analysis_suite(
    scripts: Iterable[tuple[str, Sequence[str]]],
    abort_on_failure: bool = False,
) -> list[tuple[str, int]]:
    """Run each ``(label, cmd)`` in sequence, printing a banner per step.

    Parameters
    ----------
    scripts:
        Iterable of ``(label, argv)`` pairs. ``argv`` is passed to
        :func:`subprocess.call` as-is, so callers decide whether to use
        ``sys.executable`` or a bare script name.
    abort_on_failure:
        If True, stop after the first non-zero exit code. Defaults to
        False, which matches the notebook behaviour of "exit N — continuing".

    Returns
    -------
    list of ``(label, exit_code)`` in the order executed.
    """
    results: list[tuple[str, int]] = []
    for label, cmd in scripts:
        print(f"\n===== {label} =====", flush=True)
        rc = subprocess.call(list(cmd))
        if rc == 0:
            print(f"[{label}] OK")
        else:
            print(f"[{label}] exit {rc} - continuing")
        results.append((label, rc))
        if abort_on_failure and rc != 0:
            break
    return results
