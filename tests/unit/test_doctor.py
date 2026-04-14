"""Tests for the environment doctor script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import doctor


class FakeDistribution:
    def __init__(self, version: str, location: Path) -> None:
        self.version = version
        self._location = location

    def locate_file(self, _: str) -> Path:
        return self._location


def _fake_distribution_factory(base_dir: Path):
    def _distribution(name: str):
        if name not in {
            "numpy",
            "torch",
            "torchvision",
            "pytest",
            "fastapi",
            "pydantic",
            "PyYAML",
            "scikit-learn",
        }:
            raise doctor.metadata.PackageNotFoundError(name)
        return FakeDistribution("1.2.3", base_dir / name)

    return _distribution


def _write_config(config_path: Path, dataset_path: Path) -> None:
    config_path.write_text(
        f"""
data:
  dataset_path: "{dataset_path.as_posix()}"
  image_size: 96
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
training:
  default_model: "all"
  epochs: 5
  batch_size: 64
""",
        encoding="utf-8",
    )


def test_build_summary_reports_healthy_env(monkeypatch, workspace_tmp_path):
    config_path = workspace_tmp_path / "config.yaml"
    dataset_path = workspace_tmp_path / "LSWMD_new.pkl"
    dataset_path.write_text("dataset", encoding="utf-8")
    _write_config(config_path, dataset_path)

    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(doctor, "DEFAULT_DATASET_PATH", dataset_path)
    monkeypatch.setattr(doctor, "_check_python_compatibility", lambda: True)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "py313")
    python_parent = Path(doctor.sys.executable).resolve().parent
    monkeypatch.setenv("CONDA_PREFIX", str(python_parent))
    monkeypatch.setenv("CONDA_EXE", str((python_parent / "Scripts" / "conda.exe").resolve()))
    package_root = python_parent / "Lib" / "site-packages"

    def fake_which(name: str):
        if name == "pytest":
            return str((python_parent / "pytest.exe").resolve())
        if name == "pip":
            return str((python_parent / "pip.exe").resolve())
        return None

    monkeypatch.setattr(doctor.shutil, "which", fake_which)
    monkeypatch.setattr(doctor.metadata, "distribution", _fake_distribution_factory(package_root))

    summary = doctor.build_summary()

    assert summary.status == "ok"
    assert summary.config_exists is True
    assert summary.dataset_exists is True
    assert summary.pytest_executable.endswith("pytest.exe")
    assert any(pkg.name == "numpy" and pkg.installed for pkg in summary.package_resolution)


def test_build_summary_flags_python_pytest_split(monkeypatch, workspace_tmp_path):
    config_path = workspace_tmp_path / "config.yaml"
    dataset_path = workspace_tmp_path / "LSWMD_new.pkl"
    dataset_path.write_text("dataset", encoding="utf-8")
    _write_config(config_path, dataset_path)

    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(doctor, "DEFAULT_DATASET_PATH", dataset_path)
    monkeypatch.setattr(doctor, "_check_python_compatibility", lambda: True)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "py313")
    python_parent = Path(doctor.sys.executable).resolve().parent
    monkeypatch.setenv("CONDA_PREFIX", str(python_parent))
    monkeypatch.setenv("CONDA_EXE", str((python_parent / "Scripts" / "conda.exe").resolve()))
    package_root = python_parent / "Lib" / "site-packages"

    def fake_which(name: str):
        if name == "pytest":
            return str((workspace_tmp_path / "other" / "pytest.exe").resolve())
        if name == "pip":
            return str((workspace_tmp_path / "other" / "pip.exe").resolve())
        return None

    monkeypatch.setattr(doctor.shutil, "which", fake_which)
    monkeypatch.setattr(doctor.metadata, "distribution", _fake_distribution_factory(package_root))

    summary = doctor.build_summary()

    assert summary.status == "warning"
    assert any("environment split" in issue for issue in summary.issues)


def test_build_summary_flags_nonstandard_conda_env(monkeypatch, workspace_tmp_path):
    config_path = workspace_tmp_path / "config.yaml"
    dataset_path = workspace_tmp_path / "LSWMD_new.pkl"
    dataset_path.write_text("dataset", encoding="utf-8")
    _write_config(config_path, dataset_path)

    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(doctor, "DEFAULT_DATASET_PATH", dataset_path)
    monkeypatch.setattr(doctor, "_check_python_compatibility", lambda: True)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "py311")
    python_parent = Path(doctor.sys.executable).resolve().parent
    monkeypatch.setenv("CONDA_PREFIX", str(python_parent))
    monkeypatch.setenv("CONDA_EXE", str((python_parent / "Scripts" / "conda.exe").resolve()))
    package_root = python_parent / "Lib" / "site-packages"

    def fake_which(name: str):
        if name == "pytest":
            return str((python_parent / "pytest.exe").resolve())
        if name == "pip":
            return str((python_parent / "pip.exe").resolve())
        return None

    monkeypatch.setattr(doctor.shutil, "which", fake_which)
    monkeypatch.setattr(doctor.metadata, "distribution", _fake_distribution_factory(package_root))

    summary = doctor.build_summary()

    assert summary.status == "warning"
    assert any("Standardize on 'py313'" in issue for issue in summary.issues)


def test_json_mode_emits_machine_readable_summary(monkeypatch, capsys):
    summary = doctor.DoctorSummary(
        python_executable="python",
        python_version="3.11.0",
        supported_python_range=doctor.SUPPORTED_PYTHON_RANGE,
        platform="test-platform",
        pytest_executable="pytest",
        pip_executable="pip",
        env_prefix="C:/Python311",
        config_path="config.yaml",
        config_exists=True,
        dataset_path="data/LSWMD_new.pkl",
        dataset_exists=True,
        status="ok",
    )
    summary.package_resolution = [
        doctor.PackageStatus(name="numpy", installed=True, version="1.26.0", location="C:/pkgs/numpy")
    ]

    monkeypatch.setattr(doctor, "build_summary", lambda: summary)

    exit_code = doctor.main(["--json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["package_resolution"][0]["name"] == "numpy"
