#!/usr/bin/env python3
"""
Compatibility wrapper for the canonical root-level training CLI.

The repository-maintained training entry point is ``train.py`` at the repo
root. This wrapper is kept so older docs and shell history continue to work
without preserving a second divergent implementation.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_TRAIN = REPO_ROOT / "train.py"


def _load_root_train_module():
    spec = importlib.util.spec_from_file_location("repo_root_train", ROOT_TRAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training entry point at {ROOT_TRAIN}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    logger.warning("scripts/train.py is deprecated; forwarding to the repository root train.py")
    module = _load_root_train_module()
    return int(module.main())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    raise SystemExit(main())
