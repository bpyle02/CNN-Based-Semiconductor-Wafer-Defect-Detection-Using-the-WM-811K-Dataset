"""Unit tests for ``scripts/bump_version.py``."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bump_version.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bump_version", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["bump_version"] = module
    spec.loader.exec_module(module)
    return module


bump_version = _load_module()


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """Build a minimal project tree that ``bump_version`` can operate on."""
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nversion: "1.2.3"\ndate-released: "2026-01-01"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "__init__.py").write_text(
        '"""demo."""\n__version__ = "1.2.3"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n- thing\n",
        encoding="utf-8",
    )
    return tmp_path


def _point_module_at(repo: Path) -> None:
    bump_version.REPO_ROOT = repo
    bump_version.PYPROJECT = repo / "pyproject.toml"
    bump_version.CITATION = repo / "CITATION.cff"
    bump_version.INIT_PY = repo / "src" / "__init__.py"
    bump_version.CHANGELOG = repo / "CHANGELOG.md"


def test_check_succeeds_when_version_matches_tag(fake_repo: Path) -> None:
    _point_module_at(fake_repo)
    with mock.patch.object(bump_version, "latest_git_tag", return_value="v1.2.3"):
        runner = CliRunner()
        result = runner.invoke(bump_version.main, ["--check"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_patch_bump_increments_patch(fake_repo: Path) -> None:
    _point_module_at(fake_repo)
    with mock.patch.object(bump_version, "_git_is_dirty", return_value=False), \
         mock.patch.object(bump_version, "_git_commit") as commit_mock:
        runner = CliRunner()
        result = runner.invoke(bump_version.main, ["patch"])

    assert result.exit_code == 0, result.output
    assert 'version = "1.2.4"' in (fake_repo / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '__version__ = "1.2.4"' in (fake_repo / "src" / "__init__.py").read_text(
        encoding="utf-8"
    )
    changelog = (fake_repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [1.2.4]" in changelog
    # Body from the former Unreleased section moved down.
    assert "- thing" in changelog.split("## [1.2.4]", 1)[1]
    commit_mock.assert_called_once()
    assert commit_mock.call_args.args[0] == "1.2.4"


def test_refuses_on_dirty_tree(fake_repo: Path) -> None:
    _point_module_at(fake_repo)
    with mock.patch.object(bump_version, "_git_is_dirty", return_value=True):
        runner = CliRunner()
        result = runner.invoke(bump_version.main, ["patch"])
    assert result.exit_code != 0
    assert "dirty" in result.output.lower()
    # pyproject unchanged.
    assert 'version = "1.2.3"' in (fake_repo / "pyproject.toml").read_text(
        encoding="utf-8"
    )
