#!/usr/bin/env python3
"""Publish or update docs/kaggle_quickstart.ipynb as a Kaggle Kernel.

Reads kernel-metadata.json at the repo root and calls Kaggle's
`kernels push` under the hood, with one critical adjustment:

  - Kaggle's newer KGAT-format API tokens are BEARER tokens. The CLI
    and SDK default to Basic auth (HTTP username:password), which
    Kaggle's kernel endpoints reject with a 401 for KGAT_ tokens.
  - Setting KAGGLE_API_TOKEN before importing the SDK flips it onto
    the Bearer-auth path internally.

This script handles that transparently — all you need is a valid
~/.kaggle/kaggle.json.

Usage:
    python scripts/kaggle_push.py

Re-run the same command after any change to
docs/kaggle_quickstart.ipynb, kernel-metadata.json, or the source code
the notebook pulls. The URL stays stable so the README badge keeps
working.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            f"error: {kaggle_json} not found. Create an API token at "
            "https://www.kaggle.com/settings/account and save it there.",
            file=sys.stderr,
        )
        return 1

    creds = json.loads(kaggle_json.read_text())
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]

    # Import AFTER setting the env var — the SDK snapshots it at init time.
    from kaggle import api

    repo = Path(__file__).resolve().parents[1]
    metadata_path = repo / "kernel-metadata.json"
    if not metadata_path.exists():
        print(f"error: {metadata_path} not found.", file=sys.stderr)
        return 1

    resp = api.kernels_push(str(repo))
    error = getattr(resp, "error", None) or getattr(resp, "error_nullable", None)
    if error:
        print(f"push failed: {error}", file=sys.stderr)
        return 1

    ref = getattr(resp, "ref", "?")
    url = getattr(resp, "url", None)
    version = getattr(resp, "version_number", None)
    print(f"OK — pushed version {version}")
    print(f"  ref: {ref}")
    print(f"  url: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
