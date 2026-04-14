"""Shared pytest fixtures for repository-local test state."""

import re
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def workspace_tmp_path(request):
    """Provide a deterministic repo-local temp directory per test."""
    base_dir = Path("tests/.tmp_workspace")
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid).strip("._")
    path = base_dir / safe_name

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)

    yield path

    shutil.rmtree(path, ignore_errors=True)
