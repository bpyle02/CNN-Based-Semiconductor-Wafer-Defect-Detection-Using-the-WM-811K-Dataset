#!/usr/bin/env python3
"""Create a committee-facing submission bundle from the current repo state."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "SUBMISSION_FINAL"
ARCHIVE_BASENAME = PROJECT_ROOT / "SUBMISSION_FINAL"
IGNORE_NAMES = shutil.ignore_patterns(
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.pyc",
)


@dataclass(frozen=True)
class Artifact:
    """A file or directory to include in the final submission bundle."""

    source: Path
    destination: Path
    description: str
    required: bool = True


def build_artifacts() -> list[Artifact]:
    """Return the curated artifact list for the committee bundle."""
    return [
        Artifact(PROJECT_ROOT / "README.md", Path("README.md"), "Repository overview"),
        Artifact(
            PROJECT_ROOT / "DEFENSE_SUBMISSION.md",
            Path("DEFENSE_SUBMISSION.md"),
            "Operational submission guide",
        ),
        Artifact(
            PROJECT_ROOT / "requirements.txt",
            Path("requirements.txt"),
            "Dependency manifest",
        ),
        Artifact(PROJECT_ROOT / "pytest.ini", Path("pytest.ini"), "Pytest configuration"),
        Artifact(
            PROJECT_ROOT / "Makefile",
            Path("Makefile"),
            "Convenience commands",
            required=False,
        ),
        Artifact(PROJECT_ROOT / "train.py", Path("train.py"), "Primary training entry point"),
        Artifact(
            PROJECT_ROOT / "docs" / "DEFENSE_PACKET.md",
            Path("docs/DEFENSE_PACKET.md"),
            "Committee-facing defense packet",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "FINAL_STATUS_REPORT.md",
            Path("docs/FINAL_STATUS_REPORT.md"),
            "Final status summary",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "presentation.tex",
            Path("docs/presentation.tex"),
            "Slide source",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "presentation.pdf",
            Path("docs/presentation.pdf"),
            "Compiled slide deck",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "wafer_defect_detection_report.tex",
            Path("docs/wafer_defect_detection_report.tex"),
            "Report source",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "wafer_defect_detection_report.pdf",
            Path("docs/wafer_defect_detection_report.pdf"),
            "Compiled report",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "wafer_defect_detection_run.ipynb",
            Path("docs/wafer_defect_detection_run.ipynb"),
            "Notebook artifact",
        ),
        Artifact(
            PROJECT_ROOT / "scripts" / "run_defense_demo.ps1",
            Path("scripts/run_defense_demo.ps1"),
            "Demo wrapper",
        ),
        Artifact(
            PROJECT_ROOT / "scripts" / "defense_smoke_demo.py",
            Path("scripts/defense_smoke_demo.py"),
            "Inference smoke demo",
        ),
        Artifact(
            PROJECT_ROOT / "scripts" / "finalize_submission.py",
            Path("scripts/finalize_submission.py"),
            "Submission packager",
        ),
        Artifact(PROJECT_ROOT / "src", Path("src"), "Source tree"),
        Artifact(PROJECT_ROOT / "tests" / "conftest.py", Path("tests/conftest.py"), "Pytest fixtures"),
        Artifact(
            PROJECT_ROOT / "tests" / "test_improvements.py",
            Path("tests/test_improvements.py"),
            "Improvement tests",
        ),
        Artifact(
            PROJECT_ROOT / "tests" / "test_uncertainty.py",
            Path("tests/test_uncertainty.py"),
            "Uncertainty tests",
        ),
        Artifact(PROJECT_ROOT / "tests" / "unit", Path("tests/unit"), "Unit tests"),
        Artifact(
            PROJECT_ROOT / "tests" / "integration",
            Path("tests/integration"),
            "Integration tests",
        ),
        Artifact(
            PROJECT_ROOT / "docs" / "generated",
            Path("docs/generated"),
            "Generated demo artifacts",
            required=False,
        ),
    ]


def run_command(command: Sequence[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a subprocess and return its exit code and merged output."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)

    output_parts = [completed.stdout.strip(), completed.stderr.strip()]
    output = "\n".join(part for part in output_parts if part)
    return completed.returncode, output


def resolve_pytest_command() -> list[str]:
    """Prefer the active pytest executable, then fall back to python -m pytest."""
    pytest_path = shutil.which("pytest")
    if pytest_path:
        return [pytest_path, "-q"]
    return [sys.executable, "-m", "pytest", "-q"]


def summarize_pytest_output(output: str) -> str:
    """Extract the pytest summary line."""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if "passed" in stripped or "failed" in stripped or "skipped" in stripped:
            return stripped
    return "pytest output unavailable"


def run_validation(skip_tests: bool) -> dict[str, str | int]:
    """Run pytest unless skipped and return a compact summary."""
    if skip_tests:
        return {
            "status": "skipped",
            "command": "",
            "summary": "Test run skipped by request",
            "output": "",
            "exit_code": 0,
        }

    command = resolve_pytest_command()
    exit_code, output = run_command(command, PROJECT_ROOT)
    status = "passed" if exit_code == 0 else "failed"
    return {
        "status": status,
        "command": " ".join(command),
        "summary": summarize_pytest_output(output),
        "output": output,
        "exit_code": exit_code,
    }


def resolve_powershell() -> str | None:
    """Return the first available PowerShell executable."""
    return shutil.which("powershell") or shutil.which("pwsh")


def run_demo(skip_demo: bool) -> dict[str, str | int]:
    """Run the committee smoke demo when PowerShell is available."""
    if skip_demo:
        return {
            "status": "skipped",
            "command": "",
            "summary": "Demo run skipped by request",
            "output": "",
            "exit_code": 0,
        }

    powershell = resolve_powershell()
    if not powershell:
        return {
            "status": "blocked",
            "command": "",
            "summary": "PowerShell not available in this environment",
            "output": "",
            "exit_code": 127,
        }

    command = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "scripts" / "run_defense_demo.ps1"),
    ]
    exit_code, output = run_command(command, PROJECT_ROOT)
    lowered = output.lower()

    if exit_code == 0:
        status = "passed"
    elif "fastapi is not installed" in lowered:
        status = "blocked"
    else:
        status = "failed"

    if "fastapi is not installed" in lowered:
        summary = "fastapi is not installed. Install requirements.txt to run the demo."
    else:
        summary = output.splitlines()[-1].strip() if output.strip() else "Demo output unavailable"
    return {
        "status": status,
        "command": " ".join(command),
        "summary": summary,
        "output": output,
        "exit_code": exit_code,
    }


def reset_submission_dir() -> None:
    """Recreate the output directory from scratch."""
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True)


def copy_artifact(artifact: Artifact) -> dict[str, str | bool | int]:
    """Copy one artifact into the submission directory."""
    destination = SUBMISSION_DIR / artifact.destination
    source = artifact.source

    if not source.exists():
        return {
            "source": str(source.relative_to(PROJECT_ROOT)),
            "destination": str(artifact.destination),
            "description": artifact.description,
            "required": artifact.required,
            "copied": False,
            "size_bytes": 0,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=IGNORE_NAMES)
        size_bytes = 0
    else:
        shutil.copy2(source, destination)
        size_bytes = destination.stat().st_size

    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "destination": str(artifact.destination),
        "description": artifact.description,
        "required": artifact.required,
        "copied": True,
        "size_bytes": size_bytes,
    }


def collect_artifacts(artifacts: Iterable[Artifact]) -> list[dict[str, str | bool | int]]:
    """Copy all curated artifacts and return the copy records."""
    return [copy_artifact(artifact) for artifact in artifacts]


def write_validation_summary(
    validation: dict[str, str | int],
    demo: dict[str, str | int],
    records: list[dict[str, str | bool | int]],
) -> None:
    """Write a human-readable bundle summary into the submission directory."""
    required_total = sum(1 for record in records if bool(record["required"]))
    required_copied = sum(
        1 for record in records if bool(record["required"]) and bool(record["copied"])
    )

    lines = [
        "# Submission Validation Summary",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Required artifacts copied: `{required_copied}/{required_total}`",
        f"- Pytest status: `{validation['status']}`",
        f"- Pytest summary: `{validation['summary']}`",
        f"- Demo status: `{demo['status']}`",
        f"- Demo summary: `{demo['summary']}`",
        "",
        "## Required Artifacts",
        "",
    ]

    for record in records:
        if not bool(record["required"]):
            continue
        mark = "OK" if bool(record["copied"]) else "MISSING"
        lines.append(f"- [{mark}] `{record['source']}` -> `{record['destination']}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The submission bundle is intentionally curated and excludes stale supporting material.",
            "- A blocked demo usually means the environment is missing FastAPI rather than the repository being broken.",
            "- The checked-in PDFs are packaged as-is; this script does not rebuild LaTeX sources.",
            "",
        ]
    )

    (SUBMISSION_DIR / "VALIDATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    validation: dict[str, str | int],
    demo: dict[str, str | int],
    records: list[dict[str, str | bool | int]],
) -> None:
    """Write a machine-readable manifest for the bundle."""
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "validation": validation,
        "demo": demo,
        "artifacts": records,
    }
    manifest_path = SUBMISSION_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def create_archive() -> Path:
    """Create a zip archive for the submission directory."""
    archive_path = Path(
        shutil.make_archive(
            str(ARCHIVE_BASENAME),
            "zip",
            root_dir=PROJECT_ROOT,
            base_dir=SUBMISSION_DIR.name,
        )
    )
    return archive_path


def print_summary(
    validation: dict[str, str | int],
    demo: dict[str, str | int],
    records: list[dict[str, str | bool | int]],
    archive_path: Path,
) -> None:
    """Print the final console summary."""
    required_total = sum(1 for record in records if bool(record["required"]))
    required_copied = sum(
        1 for record in records if bool(record["required"]) and bool(record["copied"])
    )
    print("\n" + "=" * 72)
    print("DEFENSE SUBMISSION FINALIZATION")
    print("=" * 72)
    print(f"Required artifacts copied: {required_copied}/{required_total}")
    print(f"Pytest: {validation['status']} ({validation['summary']})")
    print(f"Demo: {demo['status']} ({demo['summary']})")
    print(f"Bundle directory: {SUBMISSION_DIR}")
    print(f"Bundle archive:   {archive_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run pytest before packaging.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Do not run the PowerShell defense smoke demo.",
    )
    return parser.parse_args()


def main() -> int:
    """Run validation and create the submission bundle."""
    args = parse_args()
    artifacts = build_artifacts()

    validation = run_validation(skip_tests=args.skip_tests)
    demo = run_demo(skip_demo=args.skip_demo)

    reset_submission_dir()
    records = collect_artifacts(artifacts)
    write_validation_summary(validation, demo, records)
    write_manifest(validation, demo, records)
    archive_path = create_archive()
    print_summary(validation, demo, records, archive_path)

    required_missing = any(
        bool(record["required"]) and not bool(record["copied"]) for record in records
    )
    tests_failed = validation["status"] == "failed"
    hard_demo_failure = demo["status"] == "failed"
    return 1 if required_missing or tests_failed or hard_demo_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
