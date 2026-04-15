"""Unit tests for src.utils.reproducibility.compute_manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.reproducibility import compute_manifest


EXPECTED_KEYS = {
    "data_sha256",
    "config_sha256",
    "code_sha256",
    "torch_version",
    "python_version",
    "platform",
    "git_sha",
    "timestamp",
}


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Minimal repo layout with train.py + src/ so code_sha256 has something to hash."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "mod.py").write_text("VALUE = 1\n")
    (tmp_path / "train.py").write_text("print('train')\n")
    return tmp_path


def test_compute_manifest_returns_expected_keys(tmp_path: Path, fake_repo: Path) -> None:
    data = _write(tmp_path / "data.pkl", b"some-bytes")
    cfg = _write(tmp_path / "config.yaml", b"key: value\n")

    manifest = compute_manifest(data_path=data, config_path=cfg, repo_root=fake_repo)

    assert set(manifest.keys()) == EXPECTED_KEYS
    # Hashes are 64-char hex strings (or the literal 'missing' sentinel)
    assert len(manifest["data_sha256"]) == 64
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["code_sha256"]) == 64
    # Env metadata is populated (never empty)
    assert manifest["python_version"]
    assert manifest["platform"]
    assert manifest["timestamp"]


def test_compute_manifest_is_deterministic(tmp_path: Path, fake_repo: Path) -> None:
    data = _write(tmp_path / "data.pkl", b"deterministic-bytes")
    cfg = _write(tmp_path / "config.yaml", b"k: v\n")

    m1 = compute_manifest(data_path=data, config_path=cfg, repo_root=fake_repo)
    m2 = compute_manifest(data_path=data, config_path=cfg, repo_root=fake_repo)

    # Content-derived hashes must be stable across calls with identical inputs
    assert m1["data_sha256"] == m2["data_sha256"]
    assert m1["config_sha256"] == m2["config_sha256"]
    assert m1["code_sha256"] == m2["code_sha256"]


def test_compute_manifest_different_input_different_hash(tmp_path: Path, fake_repo: Path) -> None:
    cfg = _write(tmp_path / "config.yaml", b"k: v\n")

    data_a = _write(tmp_path / "a.pkl", b"payload-A")
    data_b = _write(tmp_path / "b.pkl", b"payload-B-different-length")

    m_a = compute_manifest(data_path=data_a, config_path=cfg, repo_root=fake_repo)
    m_b = compute_manifest(data_path=data_b, config_path=cfg, repo_root=fake_repo)

    assert m_a["data_sha256"] != m_b["data_sha256"]

    # Differing config contents must also propagate
    cfg_b = _write(tmp_path / "config_b.yaml", b"k: different\n")
    m_c = compute_manifest(data_path=data_a, config_path=cfg_b, repo_root=fake_repo)
    assert m_a["config_sha256"] != m_c["config_sha256"]

    # And differing code contributes to code_sha256
    (fake_repo / "src" / "mod.py").write_text("VALUE = 2\n")
    m_d = compute_manifest(data_path=data_a, config_path=cfg, repo_root=fake_repo)
    assert m_a["code_sha256"] != m_d["code_sha256"]
