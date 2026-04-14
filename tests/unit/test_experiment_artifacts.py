"""Tests for experiment manifest generation."""

from src.analysis.artifacts import (
    build_experiment_manifest,
    compute_file_hash,
    detect_latest_checkpoint,
    hash_path,
)


def test_build_experiment_manifest_includes_hashes(workspace_tmp_path):
    project_root = workspace_tmp_path / "project"
    submission_dir = workspace_tmp_path / "bundle"
    checkpoint_dir = project_root / "checkpoints"
    project_root.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_root / "config.yaml"
    config_path.write_text("device: cpu\nseed: 42\n", encoding="utf-8")

    checkpoint_path = checkpoint_dir / "model.pth"
    checkpoint_path.write_bytes(b"checkpoint-bytes")

    source_file = project_root / "artifact.txt"
    source_file.write_text("artifact payload", encoding="utf-8")
    destination_file = submission_dir / "artifact.txt"
    destination_file.write_text("artifact payload", encoding="utf-8")

    manifest = build_experiment_manifest(
        project_root=project_root,
        submission_dir=submission_dir,
        records=[
            {
                "source": "artifact.txt",
                "destination": "artifact.txt",
                "description": "example artifact",
                "required": True,
                "copied": True,
                "size_bytes": destination_file.stat().st_size,
            }
        ],
        validation={"status": "passed", "summary": "ok"},
        demo={"status": "skipped", "summary": "not run"},
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    assert manifest["schema_version"] == 1
    assert manifest["config"]["sha256"] == compute_file_hash(config_path)
    assert manifest["checkpoint"]["sha256"] == compute_file_hash(checkpoint_path)
    assert manifest["artifacts"][0]["source_hash"]["sha256"] == compute_file_hash(source_file)
    assert manifest["artifacts"][0]["destination_hash"]["sha256"] == compute_file_hash(destination_file)
    assert manifest["repository"]["git_commit"] is None or isinstance(manifest["repository"]["git_commit"], str)


def test_detect_latest_checkpoint_returns_newest(workspace_tmp_path):
    checkpoint_dir = workspace_tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    first = checkpoint_dir / "first.pth"
    second = checkpoint_dir / "second.pth"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    latest = detect_latest_checkpoint(checkpoint_dir)

    assert latest == second
    assert hash_path(second)["present"] is True
