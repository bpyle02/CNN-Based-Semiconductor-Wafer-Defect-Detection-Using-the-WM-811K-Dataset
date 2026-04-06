#!/usr/bin/env python3
"""
Compatibility shim for legacy setuptools tooling.

Primary packaging metadata now lives in ``pyproject.toml`` and the explicit
environment bootstrap helper lives in ``scripts/bootstrap_env.py``.
"""

from __future__ import annotations

import sys

from setuptools import setup


if __name__ == "__main__" and len(sys.argv) == 1:
    from scripts.bootstrap_env import main

    raise SystemExit(main())


setup()
