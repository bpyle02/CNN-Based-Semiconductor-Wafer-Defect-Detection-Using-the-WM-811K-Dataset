"""Helpers for experiment and submission artifact manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactLocation:
    """Concrete file or directory location used for bundle manifests."""

    source: Path
    destination: Path


def _run_git(args: Sequence[str], cwd: Path = PROJECT_ROOT) -> Optional[str]:
    """Run a git command and return stripped output if available."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def compute_file_hash(path: Path) -> str:
    """Compute a SHA-256 hash for a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_tree_hash(path: Path) -> str:
    """Compute a deterministic hash for a directory tree."""
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(compute_file_hash(file_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_path(path: Path) -> dict[str, Any]:
    """Return a hash/size summary for a file or directory."""
    if not path.exists():
        return {
            "present": False,
            "path": str(path),
        }

    if path.is_dir():
        return {
            "present": True,
            "path": str(path),
            "kind": "directory",
            "sha256": compute_tree_hash(path),
            "file_count": sum(1 for item in path.rglob("*") if item.is_file()),
            "size_bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
        }

    return {
        "present": True,
        "path": str(path),
        "kind": "file",
        "sha256": compute_file_hash(path),
        "size_bytes": path.stat().st_size,
    }


def detect_latest_checkpoint(checkpoint_root: Path) -> Optional[Path]:
    """Return the newest checkpoint under a root directory, if any."""
    if not checkpoint_root.exists():
        return None
    checkpoints = sorted(
        (path for path in checkpoint_root.rglob("*.pth") if path.is_file()),
        key=lambda p: (p.stat().st_mtime, p.as_posix()),
    )
    return checkpoints[-1] if checkpoints else None


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def enrich_artifact_record(
    project_root: Path,
    submission_dir: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Add hash metadata to a copied-artifact record."""
    source = project_root / Path(record["source"])
    destination = submission_dir / Path(record["destination"])

    enriched = dict(record)
    enriched["source_abs"] = _relative_or_absolute(source.resolve(), project_root)
    enriched["destination_abs"] = _relative_or_absolute(destination.resolve(), submission_dir)
    enriched["source_hash"] = hash_path(source)
    enriched["destination_hash"] = hash_path(destination)
    return enriched


def build_experiment_manifest(
    project_root: Path,
    submission_dir: Path,
    records: Iterable[dict[str, Any]],
    validation: dict[str, Any],
    demo: dict[str, Any],
    config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Build a reproducible, JSON-serializable experiment manifest."""
    project_root = project_root.resolve()
    submission_dir = submission_dir.resolve()
    records_list = [
        enrich_artifact_record(project_root, submission_dir, record) for record in records
    ]

    config_meta = hash_path(config_path) if config_path is not None else {"present": False}
    checkpoint_meta = (
        hash_path(checkpoint_path) if checkpoint_path is not None else {"present": False}
    )

    git_commit = _run_git(["rev-parse", "HEAD"], cwd=project_root)
    git_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    git_status = _run_git(["status", "--porcelain"], cwd=project_root)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "submission_dir": str(submission_dir),
        "repository": {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "dirty": bool(git_status),
            "git_status": git_status,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "validation": validation,
        "demo": demo,
        "config": config_meta,
        "checkpoint": checkpoint_meta,
        "artifacts": records_list,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a manifest to disk as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
