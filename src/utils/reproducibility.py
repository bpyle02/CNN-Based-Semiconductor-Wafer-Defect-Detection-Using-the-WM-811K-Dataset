"""Reproducibility manifest: SHA-256 hashes of data/config/code + env metadata.

Used by ``train.py`` to embed a per-run provenance record under
``results/metrics.json["reproducibility"]`` so experiments can be traced
back to an exact (code, data, config, environment) tuple.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

_CHUNK = 1024 * 1024  # 1 MiB — avoids materialising multi-GB cache files in RAM


def _sha256_file(path: Path) -> str:
    """Stream a file through SHA-256. Returns "missing" if the file is absent."""
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_concat(paths: Iterable[Path]) -> str:
    """Hash a stable concatenation of multiple files.

    Each file contributes ``<relative_path>\\0<file_bytes>\\0`` so the path
    identity matters — renaming a file changes the aggregate hash.
    """
    h = hashlib.sha256()
    for p in paths:
        try:
            rel = p.as_posix().encode("utf-8")
        except Exception:
            rel = str(p).encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        if p.exists() and p.is_file():
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(_CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _git_sha(cwd: Optional[Path] = None) -> str:
    """Return current HEAD SHA or 'unknown' if git is unavailable / not a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _torch_version() -> str:
    try:
        import torch  # local import so the module is importable without torch

        return str(torch.__version__)
    except Exception:
        return "unknown"


def _collect_code_files(repo_root: Path) -> list[Path]:
    """train.py + every src/**/*.py, sorted by POSIX path for determinism."""
    files: list[Path] = []
    train_py = repo_root / "train.py"
    if train_py.exists():
        files.append(train_py)
    src = repo_root / "src"
    if src.is_dir():
        files.extend(sorted(src.rglob("*.py"), key=lambda p: p.as_posix()))
    # Stable order regardless of filesystem iteration quirks
    files.sort(key=lambda p: p.as_posix())
    return files


def compute_manifest(
    data_path,
    config_path,
    extra_files: Optional[Iterable] = None,
    repo_root: Optional[Path] = None,
) -> dict:
    """Return a reproducibility manifest for the current run.

    Parameters
    ----------
    data_path:
        Path to the dataset file (e.g. ``data/LSWMD_new.pkl``). If a sibling
        ``*_cache.npz`` file exists it is hashed instead — that's the bytes
        the training loop actually reads.
    config_path:
        Path to the merged / effective YAML config. Hashed verbatim.
    extra_files:
        Optional iterable of extra file paths whose contents should be folded
        into ``code_sha256``.
    repo_root:
        Repo root used to discover ``train.py`` + ``src/**/*.py``. Defaults
        to two levels above this file (``src/utils/reproducibility.py``).

    Returns
    -------
    dict with keys: ``data_sha256``, ``config_sha256``, ``code_sha256``,
    ``torch_version``, ``python_version``, ``platform``, ``git_sha``,
    ``timestamp``.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    data_p = Path(data_path)
    # Prefer the cache actually consumed by the training loop when present
    cache_candidate = data_p.with_name("LSWMD_cache.npz")
    data_target = cache_candidate if cache_candidate.exists() else data_p
    data_sha = _sha256_file(data_target)

    config_sha = _sha256_file(Path(config_path))

    code_files = _collect_code_files(root)
    if extra_files:
        code_files.extend(Path(p) for p in extra_files)
        code_files.sort(key=lambda p: p.as_posix())
    code_sha = _sha256_concat(code_files)

    return {
        "data_sha256": data_sha,
        "config_sha256": config_sha,
        "code_sha256": code_sha,
        "torch_version": _torch_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
