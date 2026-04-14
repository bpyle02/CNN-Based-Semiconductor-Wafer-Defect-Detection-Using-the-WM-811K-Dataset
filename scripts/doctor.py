#!/usr/bin/env python3
"""Environment diagnostic tool for the wafer defect detection project."""

from __future__ import annotations

import argparse
import os
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("doctor")

SUPPORTED_PYTHON_RANGE = ">=3.10,<3.14"
RECOMMENDED_CONDA_ENV = "py313"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "LSWMD_new.pkl"


@dataclass
class PackageStatus:
    name: str
    installed: bool
    version: Optional[str] = None
    location: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DoctorSummary:
    python_executable: str
    python_version: str
    supported_python_range: str
    platform: str
    pytest_executable: Optional[str]
    pip_executable: Optional[str]
    env_prefix: str
    conda_env_name: Optional[str] = None
    conda_prefix: Optional[str] = None
    conda_executable: Optional[str] = None
    recommended_conda_env: str = RECOMMENDED_CONDA_ENV
    package_resolution: list[PackageStatus] = field(default_factory=list)
    config_path: str = ""
    config_exists: bool = False
    dataset_path: str = ""
    dataset_exists: bool = False
    status: str = "ok"
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _package_status(name: str) -> PackageStatus:
    """Collect package metadata without importing the package."""
    try:
        dist = metadata.distribution(name)
        location = str(Path(dist.locate_file("")).resolve())
        return PackageStatus(
            name=name,
            installed=True,
            version=dist.version,
            location=location,
        )
    except metadata.PackageNotFoundError:
        return PackageStatus(name=name, installed=False, error="not installed")
    except Exception as exc:
        return PackageStatus(name=name, installed=False, error=str(exc))


def _python_prefix(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved.parent


def _environment_root(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    try:
        resolved = Path(path).expanduser().resolve()
        parent = resolved.parent
        if parent.name.lower() in {"scripts", "bin"} and parent.parent != parent:
            return parent.parent
        return parent
    except Exception:
        return None


def _read_config_dataset_path(config_path: Path) -> Optional[Path]:
    """Read the dataset path from config.yaml if available."""
    if not config_path.exists():
        return None

    try:
        content = config_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to read config file %s: %s", config_path, exc)
        return None

    match = re.search(r"(?m)^\s*dataset_path:\s*(?P<path>.+?)\s*$", content)
    if not match:
        return None

    dataset_path = match.group("path").strip().strip('"').strip("'")
    if not dataset_path:
        return None

    dataset = Path(dataset_path)
    if not dataset.is_absolute():
        dataset = (REPO_ROOT / dataset).resolve()
    return dataset


def _check_python_compatibility() -> bool:
    major, minor = sys.version_info[:2]
    return (major, minor) >= (3, 10) and (major, minor) < (3, 14)


def _detect_path_split(pytest_executable: Optional[str]) -> Optional[str]:
    """Detect a likely python/pytest environment split."""
    if not pytest_executable:
        return None

    python_root = _environment_root(sys.executable)
    pytest_root = _environment_root(pytest_executable)
    if python_root is None or pytest_root is None:
        return None

    if python_root == pytest_root:
        return None

    return f"python lives under {python_root}, pytest lives under {pytest_root}"


def _conda_context() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Collect conda runtime details if available."""
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    return env_name, prefix, executable


def _is_under_prefix(path: str, prefix: str) -> bool:
    """Return True when path resolves inside the given prefix."""
    try:
        resolved_path = Path(path).expanduser().resolve()
        resolved_prefix = Path(prefix).expanduser().resolve()
        return resolved_path.is_relative_to(resolved_prefix)
    except Exception:
        return False


def run_command(command: Iterable[str]) -> tuple[int, str]:
    """Run a subprocess and capture output."""
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return completed.returncode, output


def build_summary() -> DoctorSummary:
    """Collect environment diagnostics."""
    pytest_executable = shutil.which("pytest")
    pip_executable = shutil.which("pip")
    conda_env_name, conda_prefix, conda_executable = _conda_context()
    config_path = DEFAULT_CONFIG_PATH
    dataset_path = _read_config_dataset_path(config_path) or DEFAULT_DATASET_PATH

    summary = DoctorSummary(
        python_executable=sys.executable,
        python_version=sys.version.replace("\n", " "),
        supported_python_range=SUPPORTED_PYTHON_RANGE,
        platform=platform.platform(),
        conda_env_name=conda_env_name,
        conda_prefix=conda_prefix,
        conda_executable=conda_executable,
        recommended_conda_env=RECOMMENDED_CONDA_ENV,
        pytest_executable=pytest_executable,
        pip_executable=pip_executable,
        env_prefix=str(_python_prefix(sys.executable)),
        config_path=str(config_path),
        config_exists=config_path.exists(),
        dataset_path=str(dataset_path),
        dataset_exists=dataset_path.exists(),
    )

    core_packages = [
        "numpy",
        "torch",
        "torchvision",
        "pytest",
        "fastapi",
        "pydantic",
        "PyYAML",
        "scikit-learn",
    ]
    summary.package_resolution = [_package_status(name) for name in core_packages]

    if not _check_python_compatibility():
        summary.status = "warning"
        summary.issues.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is outside the supported range {SUPPORTED_PYTHON_RANGE}."
        )

    if not summary.conda_executable:
        summary.status = "warning"
        summary.issues.append(
            f"Conda was not detected on PATH. Standardize on `conda run -n {RECOMMENDED_CONDA_ENV} ...` for repo commands."
        )
    elif summary.conda_env_name is None:
        summary.status = "warning"
        summary.issues.append(
            f"No active conda environment detected. Use `conda run -n {RECOMMENDED_CONDA_ENV} ...` for repo commands."
        )
    elif summary.conda_env_name != RECOMMENDED_CONDA_ENV:
        summary.status = "warning"
        summary.issues.append(
            f"Active conda env is '{summary.conda_env_name}'. Standardize on '{RECOMMENDED_CONDA_ENV}' with `conda run -n {RECOMMENDED_CONDA_ENV} ...`."
        )

    path_split = _detect_path_split(pytest_executable)
    if path_split:
        summary.status = "warning"
        summary.issues.append(f"Detected possible python/pytest environment split: {path_split}.")

    for package in summary.package_resolution:
        if not package.installed:
            summary.status = "warning"
            summary.issues.append(f"Package {package.name} is not installed.")
            continue

        if summary.conda_prefix and package.location and not _is_under_prefix(package.location, summary.conda_prefix):
            summary.notes.append(
                f"Package {package.name} resolves outside the conda prefix: {package.location}"
            )

    if not summary.config_exists:
        summary.status = "warning"
        summary.issues.append(f"Config file not found: {summary.config_path}")

    if not summary.dataset_exists:
        summary.status = "warning"
        summary.notes.append(f"Dataset not found at: {summary.dataset_path}")

    return summary


def print_human_report(summary: DoctorSummary) -> None:
    """Render a readable report to stdout."""
    logger.info("=== Wafer Defect Detection Doctor ===")
    logger.info("Python executable: %s", summary.python_executable)
    logger.info("Python version: %s", summary.python_version)
    logger.info("Supported Python range: %s", summary.supported_python_range)
    logger.info("Platform: %s", summary.platform)
    logger.info("Conda env: %s", summary.conda_env_name or "not detected")
    logger.info("Conda prefix: %s", summary.conda_prefix or "not detected")
    logger.info("Conda executable: %s", summary.conda_executable or "not found")
    logger.info("Recommended conda env: %s", summary.recommended_conda_env)
    logger.info("Recommended command prefix: conda run -n %s", summary.recommended_conda_env)
    logger.info("pytest executable: %s", summary.pytest_executable or "not found")
    logger.info("pip executable: %s", summary.pip_executable or "not found")
    logger.info("Python prefix: %s", summary.env_prefix)
    logger.info("Config: %s (%s)", summary.config_path, "found" if summary.config_exists else "missing")
    logger.info("Dataset: %s (%s)", summary.dataset_path, "found" if summary.dataset_exists else "missing")
    logger.info("Package resolution:")
    for pkg in summary.package_resolution:
        if pkg.installed:
            logger.info(
                "  - %s %s at %s",
                pkg.name,
                pkg.version or "unknown",
                pkg.location or "unknown",
            )
        else:
            logger.info("  - %s: %s", pkg.name, pkg.error or "missing")

    if summary.issues:
        logger.warning("Issues:")
        for issue in summary.issues:
            logger.warning("  - %s", issue)

    if summary.notes:
        logger.info("Notes:")
        for note in summary.notes:
            logger.info("  - %s", note)

    if summary.status == "ok":
        logger.info("Environment looks healthy.")
    else:
        logger.warning("Environment has warnings.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Environment diagnostic tool.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    summary = build_summary()
    if args.json:
        payload = asdict(summary)
        payload["package_resolution"] = [asdict(pkg) for pkg in summary.package_resolution]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_human_report(summary)

    return 0 if summary.status == "ok" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
