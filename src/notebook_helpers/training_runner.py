"""Run ``train.py`` as a subprocess with tee-to-log streaming.

Extracted from the inner training loops in Cells 6 and 11 of
``docs/kaggle_quickstart.ipynb`` (and the Colab twin). Each model is trained
in its own subprocess so GPU/CPU memory is released cleanly between runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_train_command(
    model: str,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str = "cuda",
    extra_args: Sequence[str] | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Return the argv list that :func:`run_training_subprocess` would exec.

    Exposed so unit tests can assert on command construction without
    actually invoking ``train.py``.
    """
    cmd = [
        python_executable or sys.executable,
        "train.py",
        "--model",
        str(model),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        str(device),
        "--seed",
        str(seed),
    ]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)
    return cmd


def run_training_subprocess(
    model: str,
    epochs: int,
    batch_size: int,
    seed: int,
    log_path: str | os.PathLike[str],
    extra_args: Sequence[str] | None = None,
    device: str = "cuda",
    python_executable: str | None = None,
) -> int:
    """Run ``train.py`` for one model, tee its output to ``log_path``.

    Returns the subprocess exit code. stdout and stderr are merged and
    written to both the notebook's stdout and ``log_path`` line-by-line,
    matching the original inline loop's behaviour.
    """
    cmd = build_train_command(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        extra_args=extra_args,
        python_executable=python_executable,
    )

    log = Path(log_path)
    log.parent.mkdir(exist_ok=True, parents=True)

    with open(log, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None  # for type checkers
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return proc.wait()
