#!/usr/bin/env python3
"""Version bump utility for the wafer-defect-detection project.

Reads the current version from ``pyproject.toml``, bumps it by a semver
component, propagates the new value into ``CITATION.cff`` and
``src/__init__.py``, and rewrites ``CHANGELOG.md`` so the existing
``[Unreleased]`` section becomes the new versioned entry.

Designed to be invoked as::

    python scripts/bump_version.py patch
    python scripts/bump_version.py --check

The ``--check`` flag is used by CI on tag-push to verify that the version
recorded in ``pyproject.toml`` matches the tag that triggered the workflow.
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

try:  # Python >= 3.11
    import tomllib as _toml_reader  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - py < 3.11
    import tomli as _toml_reader  # type: ignore[import-not-found,no-redef]


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CITATION = REPO_ROOT / "CITATION.cff"
INIT_PY = REPO_ROOT / "src" / "__init__.py"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

SEMVER_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> "SemVer":
        match = SEMVER_RE.match(text.strip())
        if not match:
            raise click.ClickException(f"not a semver string: {text!r}")
        return cls(int(match["major"]), int(match["minor"]), int(match["patch"]))

    def bump(self, part: str) -> "SemVer":
        if part == "major":
            return SemVer(self.major + 1, 0, 0)
        if part == "minor":
            return SemVer(self.major, self.minor + 1, 0)
        if part == "patch":
            return SemVer(self.major, self.minor, self.patch + 1)
        raise click.ClickException(f"unknown bump part: {part}")

    def __str__(self) -> str:  # noqa: DunderStr
        return f"{self.major}.{self.minor}.{self.patch}"


def read_pyproject_version(path: Path = PYPROJECT) -> str:
    """Parse the ``[project].version`` field from ``pyproject.toml``."""
    with path.open("rb") as fh:
        data = _toml_reader.load(fh)
    try:
        return str(data["project"]["version"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise click.ClickException("pyproject.toml is missing [project].version") from exc


def latest_git_tag(cwd: Path = REPO_ROOT) -> Optional[str]:
    """Return the latest semver-looking git tag, or ``None`` if no tags."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag or None


def _git_is_dirty(cwd: Path = REPO_ROOT) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException("`git status` failed; is this a git repo?")
    return bool(result.stdout.strip())


def _replace_in_pyproject(new_version: str, path: Path = PYPROJECT) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^(version\s*=\s*)"[^"]+"',
        lambda m: f'{m.group(1)}"{new_version}"',
        text,
        count=1,
    )
    if count != 1:
        raise click.ClickException("could not find version line in pyproject.toml")
    path.write_text(updated, encoding="utf-8")


def _replace_in_citation(new_version: str, path: Path = CITATION) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^(version:\s*)"[^"]+"',
        lambda m: f'{m.group(1)}"{new_version}"',
        text,
        count=1,
    )
    if count == 0:
        # Try unquoted form.
        updated, count = re.subn(
            r"(?m)^(version:\s*)\S+",
            lambda m: f'{m.group(1)}"{new_version}"',
            text,
            count=1,
        )
    if count == 0:
        raise click.ClickException("could not find version field in CITATION.cff")
    # Update date-released to today as well if present.
    today = _dt.date.today().isoformat()
    updated = re.sub(
        r'(?m)^(date-released:\s*)"?[^"\n]+"?',
        lambda m: f'{m.group(1)}"{today}"',
        updated,
        count=1,
    )
    path.write_text(updated, encoding="utf-8")


def _replace_in_init(new_version: str, path: Path = INIT_PY) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^(__version__\s*=\s*)"[^"]+"',
        lambda m: f'{m.group(1)}"{new_version}"',
        text,
        count=1,
    )
    if count != 1:
        raise click.ClickException("could not find __version__ assignment in src/__init__.py")
    path.write_text(updated, encoding="utf-8")


_DEFAULT_SECTIONS = "### Added\n\n### Changed\n\n### Fixed\n\n"


def _rewrite_changelog(new_version: str, path: Path = CHANGELOG) -> None:
    """Demote the current [Unreleased] block into the new version section.

    The existing ``[Unreleased]`` heading is preserved (empty) at the top,
    and everything previously under it becomes the body of the newly
    inserted ``## [X.Y.Z] — YYYY-MM-DD`` section.
    """
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Locate [Unreleased] heading.
    unreleased_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"^##\s*\[Unreleased\]", line, re.IGNORECASE):
            unreleased_idx = idx
            break
    if unreleased_idx is None:
        # No Unreleased section; just prepend a new block after the header.
        today = _dt.date.today().isoformat()
        new_block = f"## [{new_version}] \u2014 {today}\n\n{_DEFAULT_SECTIONS}"
        path.write_text(new_block + text, encoding="utf-8")
        return

    # Find where the Unreleased body ends (next ## heading or EOF).
    end_idx = len(lines)
    for idx in range(unreleased_idx + 1, len(lines)):
        if lines[idx].startswith("## "):
            end_idx = idx
            break

    unreleased_body = "".join(lines[unreleased_idx + 1 : end_idx]).strip("\n")
    today = _dt.date.today().isoformat()

    new_version_block = f"## [{new_version}] \u2014 {today}\n"
    if unreleased_body.strip():
        new_version_block += "\n" + unreleased_body + "\n\n"
    else:
        new_version_block += "\n" + _DEFAULT_SECTIONS

    rebuilt = (
        "".join(lines[: unreleased_idx + 1]) + "\n" + new_version_block + "".join(lines[end_idx:])
    )
    path.write_text(rebuilt, encoding="utf-8")


def _git_commit(new_version: str, cwd: Path = REPO_ROOT) -> None:
    paths = [str(PYPROJECT), str(CITATION), str(INIT_PY), str(CHANGELOG)]
    subprocess.run(["git", "add", *paths], cwd=str(cwd), check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore(release): bump to v{new_version}"],
        cwd=str(cwd),
        check=True,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "part",
    required=False,
    type=click.Choice(["patch", "minor", "major"], case_sensitive=False),
)
@click.option(
    "--check",
    is_flag=True,
    help="Verify pyproject.toml version matches the latest git tag; exit non-zero on mismatch.",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Skip the `git commit` step (files still modified on disk).",
)
def main(part: Optional[str], check: bool, no_commit: bool) -> None:
    """Bump the project version and rewrite release metadata."""
    # Resolve paths via module globals each call so tests can monkeypatch.
    module = sys.modules[__name__]
    repo_root: Path = module.REPO_ROOT
    pyproject: Path = module.PYPROJECT
    citation: Path = module.CITATION
    init_py: Path = module.INIT_PY
    changelog: Path = module.CHANGELOG

    if check:
        current = read_pyproject_version(pyproject)
        tag = latest_git_tag(repo_root)
        if tag is None:
            click.echo(f"no git tags found; pyproject version is {current}", err=True)
            sys.exit(0 if current else 1)
        expected = tag.lstrip("v")
        if expected != current:
            click.echo(
                f"version mismatch: pyproject={current} latest-tag={tag}",
                err=True,
            )
            sys.exit(1)
        click.echo(f"OK: pyproject version {current} matches tag {tag}")
        return

    if part is None:
        raise click.UsageError("PART is required unless --check is supplied")

    if _git_is_dirty(repo_root):
        raise click.ClickException("working tree is dirty; commit or stash changes before bumping")

    current = SemVer.parse(read_pyproject_version(pyproject))
    bumped = current.bump(part.lower())
    new_version = str(bumped)

    _replace_in_pyproject(new_version, pyproject)
    _replace_in_citation(new_version, citation)
    _replace_in_init(new_version, init_py)
    _rewrite_changelog(new_version, changelog)

    if not no_commit:
        _git_commit(new_version, repo_root)

    click.echo(f"bumped {current} -> {new_version}")
    click.echo(f"next: git tag v{new_version} && git push origin v{new_version}")


if __name__ == "__main__":
    main()
