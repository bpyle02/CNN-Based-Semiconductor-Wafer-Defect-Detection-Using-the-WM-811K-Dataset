"""Smoke tests for the unified ``wafer-cli`` Typer application.

These tests verify that:

1. ``wafer-cli --help`` exits 0.
2. Each subcommand's ``--help`` exits 0.
3. ``wafer-cli train --model cnn --epochs 0`` does not crash during
   argument parsing (``epochs=0`` short-circuits training).

They use ``typer.testing.CliRunner`` so no actual training / I/O runs.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()

SUBCOMMANDS = [
    "train",
    "eval",
    "benchmark",
    "distill",
    "export-onnx",
    "quantize",
    "ood",
    "calibrate",
    "bootstrap",
    "gradcam",
    "paper-figures",
    "cross-validate",
    "federated",
    "active-learn",
    "label-noise",
    "pr-ece",
]


def test_root_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    # Every registered subcommand should appear in the root help listing.
    for name in SUBCOMMANDS:
        assert name in result.output, f"{name!r} missing from root --help"


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help_exits_zero(subcommand: str) -> None:
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0, (
        f"`{subcommand} --help` exited with code {result.exit_code}: {result.output}"
    )


def test_train_parses_epochs_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``train --model cnn --epochs 0`` should not crash on argument parsing.

    We stub the heavy ``TrainingPipeline`` so the test stays hermetic
    (no torch weights, no dataset, no disk I/O). All we're verifying is
    that Typer -> argparse handoff is wired correctly and that
    ``epochs=0`` is an accepted value.
    """
    import train as train_module

    captured: dict = {}

    class _StubPipeline:
        def __init__(self, args, config):
            captured["args"] = args
            captured["config"] = config

        def run(self):  # pragma: no cover - trivial
            return 0

    monkeypatch.setattr(train_module, "TrainingPipeline", _StubPipeline)

    result = runner.invoke(app, ["train", "--model", "cnn", "--epochs", "0"])
    assert result.exit_code == 0, result.output
    assert captured.get("args") is not None, "TrainingPipeline was never constructed"
    assert captured["args"].model == "cnn"
    assert captured["args"].epochs == 0
