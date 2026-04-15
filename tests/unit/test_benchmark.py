"""Unit tests for ``scripts.benchmark``.

These do **not** exercise any real checkpoint. We register a tiny toy model
as if it were one of the known architectures, run the benchmark pipeline
against it, and verify that timing/reporting plumbing behaves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
import torch
import torch.nn as nn

from scripts import benchmark as bm


class _ToyNet(nn.Module):
    """Minimal classifier: the cheapest thing that still exercises Conv + Linear."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture(autouse=True)
def _patch_toy_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swap ``build_model`` with one that always returns a toy net.

    We also register the synthetic name 'toy' in the known-set so the
    higher-level pipeline accepts it.
    """
    monkeypatch.setattr(bm, "build_model", lambda name, num_classes=9: _ToyNet(num_classes))
    bm.KNOWN_MODEL_NAMES.add("toy")
    try:
        yield
    finally:
        bm.KNOWN_MODEL_NAMES.discard("toy")


def test_infer_model_name_supports_both_layouts(tmp_path: Path) -> None:
    p1 = tmp_path / "cnn_best.pth"
    p2 = tmp_path / "best_cnn.pth"
    p3 = tmp_path / "random.pth"
    assert bm.infer_model_name(p1) == "cnn"
    assert bm.infer_model_name(p2) == "cnn"
    assert bm.infer_model_name(p3) is None


def test_summarize_ms_matches_known_values() -> None:
    s = bm._summarize_ms([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 100.0
    # p95 of 5 samples → index 4 → 100.0.
    assert s["p95_ms"] == 100.0


def test_benchmark_checkpoint_on_toy_model(tmp_path: Path) -> None:
    """Save a toy checkpoint, run the pipeline end-to-end, inspect outputs."""
    net = _ToyNet()
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "toy_best.pth"
    torch.save(net.state_dict(), ckpt)

    results_dir = tmp_path / "results"
    out = bm.run(
        checkpoints_dir=ckpt_dir,
        results_dir=results_dir,
        devices=["cpu"],
        warmup=2,
        iters=5,
        throughput_batch_size=4,
        throughput_batches=3,
        write=True,
    )

    assert len(out) == 1
    r = out[0]
    assert r.error is None, r.error
    assert r.model == "toy"
    assert r.params == sum(p.numel() for p in net.parameters())
    assert r.checkpoint_size_mb > 0

    dev = r.devices["cpu"]
    assert dev.error is None, dev.error
    assert dev.latency["n"] == 5
    assert dev.latency["mean_ms"] > 0
    assert dev.throughput["samples_per_sec"] > 0
    # CPU run → no CUDA memory measurement.
    assert dev.peak_cuda_memory_mb is None

    # On-disk artefacts.
    json_path = results_dir / "benchmark.json"
    md_path = results_dir / "benchmark.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["models"][0]["model"] == "toy"
    assert "Inference Benchmark" in md_path.read_text()


def test_unknown_model_produces_clean_error(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Filename doesn't match either naming pattern.
    ckpt = ckpt_dir / "weird.pth"
    torch.save(_ToyNet().state_dict(), ckpt)

    out = bm.run(
        checkpoints_dir=ckpt_dir,
        results_dir=tmp_path / "results",
        devices=["cpu"],
        warmup=1,
        iters=2,
        throughput_batch_size=2,
        throughput_batches=2,
        write=False,
    )
    # ``discover_checkpoints`` only picks up *_best.pth / best_*.pth
    # patterns, so a file named 'weird.pth' is simply not seen.
    assert out == []
