"""Smoke tests for the ``wafer-cli kaggle`` sub-app.

These tests do not touch the network and do not import the ``kaggle``
SDK — they only verify that the Typer sub-app and its nested commands
(``push``, ``delete``) are wired up and respond to ``--help``.
"""

from __future__ import annotations

from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


def test_kaggle_help_exits_zero() -> None:
    """``wafer-cli kaggle --help`` lists push + delete subcommands."""
    result = runner.invoke(app, ["kaggle", "--help"])
    assert result.exit_code == 0, result.output
    assert "push" in result.output
    assert "delete" in result.output


def test_kaggle_push_help_exits_zero() -> None:
    result = runner.invoke(app, ["kaggle", "push", "--help"])
    assert result.exit_code == 0, result.output


def test_kaggle_delete_help_exits_zero() -> None:
    result = runner.invoke(app, ["kaggle", "delete", "--help"])
    assert result.exit_code == 0, result.output
    # The --slug option should be documented.
    assert "--slug" in result.output
