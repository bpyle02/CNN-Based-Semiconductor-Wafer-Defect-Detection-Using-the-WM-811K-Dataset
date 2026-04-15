#!/usr/bin/env python3
"""
Latency/throughput benchmark for trained wafer-defect checkpoints.

For every ``checkpoints/*_best.pth`` on disk (or a single ``--checkpoint``)
this script measures:

* Single-image latency (batch=1): 10 warm-up runs, 100 timed runs, reporting
  mean / median / p95 in milliseconds.
* Batched throughput (batch=32): 10 timed batches, reporting sustained
  samples/sec.
* Parameter count, on-disk model size (MB), and peak CUDA memory used during
  inference (CUDA runs only).

Timing strategy:
* CUDA runs use ``torch.cuda.Event`` to avoid Python-side overhead that would
  otherwise dominate the small-CNN runtime.
* CPU runs use ``time.perf_counter``.

Outputs:
    results/benchmark.json   - machine-readable
    results/benchmark.md     - markdown table suitable for the report

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --checkpoint checkpoints/cnn_best.pth --model cnn
    python -m scripts.benchmark --devices cpu
    python -m scripts.benchmark --warmup 10 --iters 100 --throughput-batches 10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

INPUT_SHAPE = (3, 96, 96)
DEFAULT_NUM_CLASSES = 9

# Infer model-name from checkpoint filename. Both ``cnn_best.pth`` and
# ``best_cnn.pth`` layouts appear in the repo history.
_NAME_PATTERNS = (
    re.compile(r"^(?P<name>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_best\.pth$"),
    re.compile(r"^best_(?P<name>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)\.pth$"),
)

KNOWN_MODEL_NAMES = {
    "cnn",
    "cnn_fpn",
    "resnet",
    "efficientnet",
    "vit",
    "swin",
    "ride",
}


def infer_model_name(path: Path) -> Optional[str]:
    """Infer a short model identifier from a checkpoint filename."""
    stem = path.name
    for pattern in _NAME_PATTERNS:
        m = pattern.match(stem)
        if m:
            return m.group("name").lower()
    return None


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(model_name: str, num_classes: int = DEFAULT_NUM_CLASSES) -> nn.Module:
    """Instantiate a model by its short name (no pretrained downloads)."""
    name = model_name.lower()

    if name == "cnn":
        from src.models import WaferCNN

        return WaferCNN(num_classes=num_classes)

    if name == "cnn_fpn":
        from src.models import WaferCNNFPN

        return WaferCNNFPN(num_classes=num_classes)

    if name == "resnet":
        from src.models import get_resnet18

        return get_resnet18(num_classes=num_classes, pretrained=False, freeze_until=None)

    if name in {"efficientnet", "effnet"}:
        from src.models import get_efficientnet_b0

        return get_efficientnet_b0(num_classes=num_classes, pretrained=False, freeze_until=None)

    if name == "vit":
        from src.models import get_vit_small

        return get_vit_small(num_classes=num_classes)

    if name == "swin":
        from src.models import get_swin_tiny

        return get_swin_tiny(num_classes=num_classes)

    if name == "ride":
        from src.models import build_ride_model

        return build_ride_model(backbone_name="cnn", num_classes=num_classes, device="cpu")

    raise ValueError(
        f"Unknown model name '{model_name}'. Expected one of: " f"{sorted(KNOWN_MODEL_NAMES)}"
    )


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path, device: str) -> None:
    """Load weights into model, tolerating wrapped / raw state_dict layouts.

    Missing/unexpected keys are logged but do not abort — benchmarking the
    architecture with random weights is still useful for latency.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            "Checkpoint %s loaded with missing=%d unexpected=%d keys",
            checkpoint_path.name,
            len(missing),
            len(unexpected),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


@contextmanager
def _cuda_memory_tracker(device: torch.device):
    """Reset and yield; caller reads ``torch.cuda.max_memory_allocated`` after."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cuda(run_once: Callable[[], None], iters: int, device: torch.device) -> List[float]:
    """Time ``run_once`` ``iters`` times on CUDA via events. Returns ms/iter."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize(device)
    for i in range(iters):
        starts[i].record()
        run_once()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [starts[i].elapsed_time(ends[i]) for i in range(iters)]


def _time_cpu(run_once: Callable[[], None], iters: int) -> List[float]:
    """Time ``run_once`` ``iters`` times on CPU via perf_counter. Returns ms/iter."""
    timings: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return timings


def _summarize_ms(samples: List[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    p95_idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean_ms": float(statistics.fmean(samples)),
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(ordered[p95_idx]),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "n": len(samples),
    }


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


@dataclass
class DeviceResult:
    device: str
    latency: Dict[str, float] = field(default_factory=dict)
    throughput: Dict[str, float] = field(default_factory=dict)
    peak_cuda_memory_mb: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ModelResult:
    model: str
    checkpoint: str
    checkpoint_size_mb: float
    params: int
    devices: Dict[str, DeviceResult] = field(default_factory=dict)
    error: Optional[str] = None


def benchmark_on_device(
    model: nn.Module,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
    throughput_batch_size: int,
    throughput_batches: int,
) -> DeviceResult:
    result = DeviceResult(device=str(device))
    try:
        model = model.to(device).eval()
        # --- Latency: batch=1 ---
        x1 = torch.randn((1, *INPUT_SHAPE), device=device)

        def _forward_single() -> None:
            with torch.inference_mode():
                model(x1)

        # Warm-up (not measured).
        for _ in range(warmup):
            _forward_single()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with _cuda_memory_tracker(device):
            if device.type == "cuda":
                timings = _time_cuda(_forward_single, iters, device)
            else:
                timings = _time_cpu(_forward_single, iters)

        result.latency = _summarize_ms(timings)

        # --- Throughput: batch=32 ---
        xb = torch.randn((throughput_batch_size, *INPUT_SHAPE), device=device)

        def _forward_batch() -> None:
            with torch.inference_mode():
                model(xb)

        # A couple of warm-ups for throughput as well.
        for _ in range(min(3, warmup)):
            _forward_batch()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with _cuda_memory_tracker(device):
            if device.type == "cuda":
                batch_ms = _time_cuda(_forward_batch, throughput_batches, device)
            else:
                batch_ms = _time_cpu(_forward_batch, throughput_batches)

        total_samples = throughput_batch_size * throughput_batches
        total_sec = sum(batch_ms) / 1000.0
        result.throughput = {
            "batch_size": throughput_batch_size,
            "batches": throughput_batches,
            "total_samples": total_samples,
            "total_sec": total_sec,
            "samples_per_sec": total_samples / total_sec if total_sec > 0 else 0.0,
            "mean_batch_ms": float(statistics.fmean(batch_ms)),
        }

        if device.type == "cuda":
            result.peak_cuda_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except Exception as exc:  # pragma: no cover - reported through JSON
        logger.exception("Benchmark failed on %s: %s", device, exc)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def benchmark_checkpoint(
    checkpoint_path: Path,
    model_name: Optional[str],
    *,
    devices: Iterable[str],
    warmup: int,
    iters: int,
    throughput_batch_size: int,
    throughput_batches: int,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> ModelResult:
    name = (model_name or infer_model_name(checkpoint_path) or "").lower()
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    result = ModelResult(
        model=name or checkpoint_path.stem,
        checkpoint=str(checkpoint_path),
        checkpoint_size_mb=size_mb,
        params=0,
    )
    if not name:
        result.error = (
            f"Cannot infer model architecture from filename '{checkpoint_path.name}'. "
            "Pass --model explicitly."
        )
        return result
    if name not in KNOWN_MODEL_NAMES:
        result.error = (
            f"Inferred model '{name}' is not a known architecture. "
            f"Known: {sorted(KNOWN_MODEL_NAMES)}"
        )
        return result

    try:
        model = build_model(name, num_classes=num_classes)
        load_checkpoint_weights(model, checkpoint_path, device="cpu")
    except Exception as exc:
        logger.exception("Failed to build/load model for %s", checkpoint_path)
        result.error = f"build/load failed: {type(exc).__name__}: {exc}"
        return result

    result.params = count_parameters(model)

    for dev_str in devices:
        torch_dev = torch.device(dev_str)
        if torch_dev.type == "cuda" and not torch.cuda.is_available():
            result.devices[dev_str] = DeviceResult(
                device=dev_str, error="CUDA requested but not available"
            )
            continue
        result.devices[dev_str] = benchmark_on_device(
            model,
            torch_dev,
            warmup=warmup,
            iters=iters,
            throughput_batch_size=throughput_batch_size,
            throughput_batches=throughput_batches,
        )

    return result


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(val: Any, spec: str = ".2f") -> str:
    if val is None:
        return "—"
    try:
        return format(val, spec)
    except (TypeError, ValueError):
        return str(val)


def render_markdown(results: List[ModelResult]) -> str:
    lines: List[str] = []
    lines.append("# Inference Benchmark\n")
    lines.append(
        "Single-image latency is batch=1 over 100 timed iterations "
        "(warm-up 10). Throughput is sustained samples/sec across 10 batches "
        "of size 32. CUDA timings use `torch.cuda.Event`; CPU timings use "
        "`time.perf_counter`.\n"
    )
    lines.append(
        "| Model | Device | Params | Ckpt MB | Mean ms | Median ms | P95 ms | Samples/s | Peak CUDA MB |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if not r.devices:
            lines.append(
                f"| {r.model} | — | {r.params:,} | {_fmt(r.checkpoint_size_mb)} | "
                f"— | — | — | — | — |"
            )
            continue
        for dev_name, dr in r.devices.items():
            if dr.error:
                lines.append(
                    f"| {r.model} | {dev_name} | {r.params:,} | "
                    f"{_fmt(r.checkpoint_size_mb)} | ERR | ERR | ERR | ERR | — |"
                )
                continue
            lat = dr.latency or {}
            tput = dr.throughput or {}
            lines.append(
                f"| {r.model} | {dev_name} | {r.params:,} | "
                f"{_fmt(r.checkpoint_size_mb)} | "
                f"{_fmt(lat.get('mean_ms'))} | "
                f"{_fmt(lat.get('median_ms'))} | "
                f"{_fmt(lat.get('p95_ms'))} | "
                f"{_fmt(tput.get('samples_per_sec'), '.1f')} | "
                f"{_fmt(dr.peak_cuda_memory_mb)} |"
            )
    # Errors footer
    errors = [(r.model, r.error) for r in results if r.error]
    if errors:
        lines.append("")
        lines.append("## Errors")
        for name, err in errors:
            lines.append(f"- `{name}`: {err}")
    return "\n".join(lines) + "\n"


def results_to_json(results: List[ModelResult]) -> Dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "models": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover_checkpoints(ckpt_dir: Path) -> List[Path]:
    """Find ``*_best.pth`` and ``best_*.pth`` under ``ckpt_dir``."""
    found = set()
    for pattern in ("*_best.pth", "best_*.pth"):
        for p in ckpt_dir.glob(pattern):
            found.add(p)
    return sorted(found)


def _default_devices() -> List[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def run(
    *,
    checkpoint: Optional[Path] = None,
    model: Optional[str] = None,
    checkpoints_dir: Path = REPO_ROOT / "checkpoints",
    results_dir: Path = REPO_ROOT / "results",
    devices: Optional[List[str]] = None,
    warmup: int = 10,
    iters: int = 100,
    throughput_batch_size: int = 32,
    throughput_batches: int = 10,
    write: bool = True,
) -> List[ModelResult]:
    """Run the full benchmark. Returns a list of ``ModelResult``."""
    devices = devices or _default_devices()

    if checkpoint is not None:
        checkpoints = [checkpoint]
    else:
        checkpoints = discover_checkpoints(checkpoints_dir)

    if not checkpoints:
        logger.error("No checkpoints discovered under %s", checkpoints_dir)
        return []

    results: List[ModelResult] = []
    for ckpt in checkpoints:
        logger.info("Benchmarking %s ...", ckpt.name)
        r = benchmark_checkpoint(
            ckpt,
            model_name=model,
            devices=devices,
            warmup=warmup,
            iters=iters,
            throughput_batch_size=throughput_batch_size,
            throughput_batches=throughput_batches,
        )
        results.append(r)

    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / "benchmark.json"
        md_path = results_dir / "benchmark.md"
        json_path.write_text(json.dumps(results_to_json(results), indent=2))
        md_path.write_text(render_markdown(results))
        logger.info("Wrote %s and %s", json_path, md_path)

    return results


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single checkpoint to benchmark (default: all *_best.pth).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Architecture name when it cannot be inferred from filename.",
    )
    p.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument(
        "--devices",
        nargs="+",
        default=None,
        choices=["cpu", "cuda"],
        help="Devices to benchmark on. Default: cpu and cuda when available.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--throughput-batch-size", type=int, default=32)
    p.add_argument("--throughput-batches", type=int, default=10)
    p.add_argument(
        "--no-write", action="store_true", help="Skip writing results/benchmark.{json,md}."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run(
        checkpoint=args.checkpoint,
        model=args.model,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        devices=args.devices,
        warmup=args.warmup,
        iters=args.iters,
        throughput_batch_size=args.throughput_batch_size,
        throughput_batches=args.throughput_batches,
        write=not args.no_write,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
