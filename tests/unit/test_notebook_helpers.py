"""Smoke tests for ``src.notebook_helpers``.

These verify command construction and light side effects only — no
subprocess actually runs ``train.py`` or the analysis scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.notebook_helpers import (
    analysis_runner,
    dataset,
    env_snapshot,
    training_runner,
)


# ---------------------------------------------------------------------------
# env_snapshot
# ---------------------------------------------------------------------------

def test_print_env_snapshot_emits_python_line(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Skip the (potentially slow / hanging) torch import and the
    # nvidia-smi subprocess call — neither is needed to prove the
    # helper's skeleton executes correctly.
    monkeypatch.setattr(env_snapshot, "_try_import_torch", lambda: None)
    monkeypatch.setattr(env_snapshot.platform, "platform", lambda: "stub-platform")
    monkeypatch.setattr(
        env_snapshot.subprocess,
        "check_output",
        lambda *a, **kw: "nvidia-smi stub\n",
    )
    env_snapshot.print_env_snapshot(env_vars=("PATH",))
    captured = capsys.readouterr().out
    assert "Python:" in captured
    assert "env 'PATH':" in captured


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def test_locate_or_download_reuses_existing_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "LSWMD_new.pkl"
    # Fake a >1 GB file by patching the size helper (writing a real 1 GB file
    # in a unit test would be absurd).
    target.write_bytes(b"stub")
    monkeypatch.setattr(dataset, "_size_gb", lambda p: 2.1)

    resolved = dataset.locate_or_download_dataset(target_path=target)

    assert resolved == target
    assert resolved.exists()


def test_locate_or_download_raises_when_nothing_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "LSWMD_new.pkl"
    # No file at target, no kaggle inputs, kaggle CLI raises FileNotFoundError
    # on both the initial attempt and after a fake pip install.
    monkeypatch.setattr(dataset.glob, "glob", lambda *a, **k: [])

    calls: list[list[str]] = []

    def fake_check_call(cmd, *a, **kw):
        calls.append(list(cmd))
        raise FileNotFoundError("no kaggle CLI")

    monkeypatch.setattr(dataset.subprocess, "check_call", fake_check_call)

    with pytest.raises((RuntimeError, FileNotFoundError)):
        dataset.locate_or_download_dataset(
            target_path=target,
            download_dir=tmp_path / "dl",
        )


def test_pick_pkl_rejects_small_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    small = tmp_path / "LSWMD_new.pkl"
    small.write_bytes(b"tiny")
    assert dataset._pick_pkl([str(small)]) is None

    monkeypatch.setattr(dataset, "_size_gb", lambda p: 2.0)
    assert dataset._pick_pkl([str(small)]) == str(small)

    # Wrong basename is always rejected regardless of size.
    other = tmp_path / "something_else.pkl"
    other.write_bytes(b"tiny")
    assert dataset._pick_pkl([str(other)]) is None


# ---------------------------------------------------------------------------
# training_runner
# ---------------------------------------------------------------------------

def test_build_train_command_basic() -> None:
    cmd = training_runner.build_train_command(
        model="cnn",
        epochs=10,
        batch_size=64,
        seed=42,
    )
    assert cmd[0] == sys.executable
    assert cmd[1] == "train.py"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "cnn"
    assert "--epochs" in cmd and cmd[cmd.index("--epochs") + 1] == "10"
    assert "--batch-size" in cmd and cmd[cmd.index("--batch-size") + 1] == "64"
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "42"
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "cuda"


def test_build_train_command_appends_extra_args() -> None:
    cmd = training_runner.build_train_command(
        model="ride",
        epochs=20,
        batch_size=128,
        seed=42,
        extra_args=["--synthetic", "--config", "configs/rare_class/C_drw.yaml"],
    )
    # Extra args must be appended verbatim at the tail.
    assert cmd[-3:] == ["--synthetic", "--config", "configs/rare_class/C_drw.yaml"]


def test_build_train_command_honors_custom_device_and_python() -> None:
    cmd = training_runner.build_train_command(
        model="swin",
        epochs=1,
        batch_size=32,
        seed=0,
        device="cpu",
        python_executable="/opt/py/bin/python",
    )
    assert cmd[0] == "/opt/py/bin/python"
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_run_training_subprocess_streams_to_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "train_cnn.log"
    captured_cmd: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            captured_cmd.append(cmd)
            self.stdout = iter(["epoch 1/1\n", "macro_f1=0.79\n"])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(training_runner.subprocess, "Popen", FakeProc)

    rc = training_runner.run_training_subprocess(
        model="cnn",
        epochs=1,
        batch_size=32,
        seed=42,
        log_path=log_path,
    )

    assert rc == 0
    assert captured_cmd and captured_cmd[0][1] == "train.py"
    log_text = log_path.read_text()
    assert "epoch 1/1" in log_text
    assert "macro_f1=0.79" in log_text
    # Also tee'd to stdout.
    assert "epoch 1/1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# analysis_runner
# ---------------------------------------------------------------------------

def test_run_analysis_suite_runs_each_step(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_call(cmd, *a, **kw) -> int:
        calls.append(list(cmd))
        return 0

    monkeypatch.setattr(analysis_runner.subprocess, "call", fake_call)

    results = analysis_runner.run_analysis_suite([
        ("bootstrap CI", [sys.executable, "scripts/bootstrap_ci.py"]),
        ("PR curves + ECE", [sys.executable, "scripts/pr_curves_ece.py"]),
    ])

    assert [r[0] for r in results] == ["bootstrap CI", "PR curves + ECE"]
    assert all(rc == 0 for _, rc in results)
    assert len(calls) == 2
    out = capsys.readouterr().out
    assert "===== bootstrap CI =====" in out
    assert "[bootstrap CI] OK" in out


def test_run_analysis_suite_continues_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(analysis_runner.subprocess, "call", lambda cmd, *a, **kw: 2)
    results = analysis_runner.run_analysis_suite([
        ("a", ["true"]),
        ("b", ["true"]),
    ])
    assert [rc for _, rc in results] == [2, 2]
    assert "[a] exit 2 - continuing" in capsys.readouterr().out


def test_run_analysis_suite_abort_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis_runner.subprocess, "call", lambda cmd, *a, **kw: 1)
    results = analysis_runner.run_analysis_suite(
        [("a", ["x"]), ("b", ["y"])],
        abort_on_failure=True,
    )
    assert results == [("a", 1)]
