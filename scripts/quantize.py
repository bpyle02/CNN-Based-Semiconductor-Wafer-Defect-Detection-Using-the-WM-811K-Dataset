#!/usr/bin/env python3
"""
Post-training static INT8 quantization for wafer-defect checkpoints.

Uses ``torch.ao.quantization`` (static quantization with FBGEMM/QNNPACK
backend). Calibrates on a random subset of validation data (default 128
samples) and measures:

* INT8 vs FP32 accuracy on a held-out test subset
* File-size ratio (FP32 state_dict bytes vs INT8 scripted model bytes)
* CPU latency ratio (INT8 mean-ms / FP32 mean-ms) at batch=1

Output:
    checkpoints/<model>_int8.pt
    results/quantization_<model>.json

Caveats:
    Static quantization has sharp edges — it reliably quantizes classic
    conv-relu-bn CNNs but frequently fails on transformers (ViT / Swin)
    and custom gating modules (RIDE). When that happens, the error is
    captured in the JSON output and the script moves on without aborting.

Usage:
    python -m scripts.quantize --checkpoint checkpoints/cnn_best.pth --model cnn
    python -m scripts.quantize --all
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import (  # noqa: E402 - local package import
    DEFAULT_NUM_CLASSES,
    INPUT_SHAPE,
    KNOWN_MODEL_NAMES,
    build_model,
    discover_checkpoints,
    infer_model_name,
    load_checkpoint_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantResult:
    model: str
    checkpoint: str
    int8_path: Optional[str]
    fp32_size_mb: Optional[float]
    int8_size_mb: Optional[float]
    size_ratio: Optional[float]
    fp32_accuracy: Optional[float]
    int8_accuracy: Optional[float]
    accuracy_delta: Optional[float]
    fp32_latency_ms: Optional[float]
    int8_latency_ms: Optional[float]
    latency_ratio: Optional[float]
    calibration_samples: int
    test_samples: int
    ok: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Data loading (with synthetic fallback)
# ---------------------------------------------------------------------------


def _try_load_cached_tensors(
    cache_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load images+labels from ``data/LSWMD_cache.npz`` if it exists.

    The cache format is produced by ``scripts/precompute_tensors.py``. We
    treat missing data as non-fatal: quantization still calibrates fine on
    random normal inputs — accuracy will be meaningless, but the script will
    at least verify the quantization pipeline.
    """
    if not cache_path.exists():
        return None
    try:
        # ``allow_pickle=True`` is required because the shipping cache stores
        # object-dtype string labels. We only load our own data here, so
        # pickle-based unpickling is acceptable.
        npz = np.load(cache_path, allow_pickle=True)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", cache_path, exc)
        return None
    # Try common key names.
    image_keys = [k for k in ("maps", "images", "X", "wafer_maps") if k in npz.files]
    label_keys = [k for k in ("labels", "y") if k in npz.files]
    if not image_keys or not label_keys:
        logger.warning(
            "Cache %s has keys %s; no recognised image/label arrays.",
            cache_path,
            list(npz.files),
        )
        return None
    return npz[image_keys[0]], npz[label_keys[0]]


def _to_tensor_batch(images: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    """Shape ``images[indices]`` to ``(N, C, H, W)`` on CPU as float32."""
    sel = images[indices]
    arr = np.asarray(sel, dtype=np.float32)
    # Accept a variety of layouts and normalise into (N, 3, H, W).
    if arr.ndim == 3:  # (N, H, W)
        arr = np.stack([arr, arr, arr], axis=1)
    elif arr.ndim == 4:
        if arr.shape[-1] in (1, 3):  # (N, H, W, C)
            arr = np.transpose(arr, (0, 3, 1, 2))
        # else assume (N, C, H, W) already
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 3, axis=1)
    else:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    # Values in cache are typically already in [0, 1]; otherwise clamp.
    if arr.max() > 3.0:
        arr = arr / max(arr.max(), 1.0)
    return torch.from_numpy(arr.copy())


def load_calib_and_test(
    cache_path: Path,
    *,
    num_calib: int,
    num_test: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (calib_x, test_x, test_y_or_None, n_classes_or_None).

    Falls back to random tensors if cached data is unavailable. When labels
    are unavailable the returned ``test_y`` is ``None``; accuracy is then
    reported as ``None``.
    """
    rng = np.random.default_rng(seed)
    cached = _try_load_cached_tensors(cache_path)
    if cached is None:
        logger.warning(
            "No cached dataset at %s — falling back to synthetic calibration data. "
            "Accuracy numbers will be meaningless but quantization will still run.",
            cache_path,
        )
        calib = torch.randn((num_calib, *INPUT_SHAPE))
        test_x = torch.randn((num_test, *INPUT_SHAPE))
        return calib, test_x, None, None

    images, labels = cached
    # Capture the optional ``classes`` array so string labels are mapped
    # consistently with how the model was trained.
    class_order: Optional[List[str]] = None
    try:
        npz = np.load(cache_path, allow_pickle=True)
        if "classes" in npz.files:
            class_order = [str(c) for c in npz["classes"]]
    except Exception:  # pragma: no cover - only a hint, fall back below
        class_order = None
    n = len(images)
    if n < num_calib + num_test:
        logger.warning(
            "Cache has only %d samples; requested %d calib + %d test.",
            n,
            num_calib,
            num_test,
        )
    idx = rng.permutation(n)
    calib_idx = idx[:num_calib]
    test_idx = idx[num_calib : num_calib + num_test]

    calib_x = _to_tensor_batch(images, calib_idx)
    test_x = _to_tensor_batch(images, test_idx)

    # Encode labels → int64.
    lab_arr = np.asarray(labels)
    if lab_arr.dtype.kind in "USO":  # string / object-dtype → map to int
        order = class_order or sorted({str(v) for v in lab_arr})
        mapping = {v: i for i, v in enumerate(order)}
        lab_arr = np.array([mapping.get(str(v), -1) for v in lab_arr], dtype=np.int64)
    test_y = torch.as_tensor(lab_arr[test_idx], dtype=torch.long)
    return calib_x, test_x, test_y, None


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _pick_backend() -> str:
    """Pick a quantized backend available on this platform."""
    import torch.backends.quantized as q_backends

    supported = getattr(q_backends, "supported_engines", ["fbgemm"])
    for pref in ("fbgemm", "qnnpack", "x86"):
        if pref in supported:
            torch.backends.quantized.engine = pref
            return pref
    return torch.backends.quantized.engine


class _QuantWrapper(nn.Module):
    """Wrap a model with QuantStub/DeQuantStub for static quantization.

    Static PTQ requires the graph to enter/exit the quantized domain at the
    input and output tensors. Wrapping is the least-invasive way to do that
    without modifying the model's ``forward`` in-place.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        from torch.ao.quantization import DeQuantStub, QuantStub

        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


def _fuse_cnn_in_place(model: nn.Module) -> None:
    """Best-effort fuse Conv+BN+ReLU and Conv+BN patterns in WaferCNN."""
    from torch.ao.quantization import fuse_modules

    # Walk sequential children and look for [Conv2d, BatchNorm2d, ReLU] triples.
    def _fuse_sequential(seq: nn.Sequential) -> None:
        names = list(seq._modules.keys())
        mods = [seq._modules[n] for n in names]
        i = 0
        groups: List[List[str]] = []
        while i < len(mods):
            if (
                i + 2 < len(mods)
                and isinstance(mods[i], nn.Conv2d)
                and isinstance(mods[i + 1], nn.BatchNorm2d)
                and isinstance(mods[i + 2], nn.ReLU)
            ):
                groups.append([names[i], names[i + 1], names[i + 2]])
                i += 3
            elif (
                i + 1 < len(mods)
                and isinstance(mods[i], nn.Conv2d)
                and isinstance(mods[i + 1], nn.BatchNorm2d)
            ):
                groups.append([names[i], names[i + 1]])
                i += 2
            else:
                i += 1
        if groups:
            fuse_modules(seq, groups, inplace=True)

    for module in model.modules():
        if isinstance(module, nn.Sequential):
            try:
                _fuse_sequential(module)
            except Exception as exc:
                logger.debug("Fuse pass skipped a sequential: %s", exc)


@torch.inference_mode()
def _calibrate(model: nn.Module, calib_x: torch.Tensor, batch_size: int = 32) -> None:
    for i in range(0, len(calib_x), batch_size):
        model(calib_x[i : i + batch_size])


@torch.inference_mode()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> float:
    correct = 0
    total = 0
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size])
        pred = logits.argmax(dim=1)
        correct += (pred.cpu() == y[i : i + batch_size].cpu()).sum().item()
        total += pred.numel()
    return correct / total if total > 0 else 0.0


@torch.inference_mode()
def _mean_latency_ms(model: nn.Module, iters: int = 30, warmup: int = 5) -> float:
    x = torch.randn((1, *INPUT_SHAPE))
    for _ in range(warmup):
        model(x)
    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Core quantization pipeline
# ---------------------------------------------------------------------------


def quantize_checkpoint(
    model_name: str,
    checkpoint: Optional[Path],
    *,
    num_calib: int = 128,
    num_test: int = 1024,
    out_dir: Path = REPO_ROOT / "checkpoints",
    results_dir: Path = REPO_ROOT / "results",
    data_cache: Path = REPO_ROOT / "data" / "LSWMD_cache.npz",
    num_classes: int = DEFAULT_NUM_CLASSES,
    write: bool = True,
) -> QuantResult:
    name = model_name.lower()
    out_path = out_dir / f"{name}_int8.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    result = QuantResult(
        model=name,
        checkpoint=str(checkpoint) if checkpoint else "",
        int8_path=None,
        fp32_size_mb=None,
        int8_size_mb=None,
        size_ratio=None,
        fp32_accuracy=None,
        int8_accuracy=None,
        accuracy_delta=None,
        fp32_latency_ms=None,
        int8_latency_ms=None,
        latency_ratio=None,
        calibration_samples=num_calib,
        test_samples=num_test,
        ok=False,
    )

    # -- build fp32 model --
    try:
        fp32 = build_model(name, num_classes=num_classes).cpu().eval()
    except Exception as exc:
        result.error = f"build failed: {type(exc).__name__}: {exc}"
        return result

    if checkpoint is not None:
        try:
            load_checkpoint_weights(fp32, Path(checkpoint), device="cpu")
        except Exception as exc:
            result.error = f"checkpoint load failed: {type(exc).__name__}: {exc}"
            return result

    # -- data --
    calib_x, test_x, test_y, _ = load_calib_and_test(
        data_cache, num_calib=num_calib, num_test=num_test
    )

    # -- fp32 baseline metrics --
    try:
        fp32_latency = _mean_latency_ms(fp32)
        fp32_acc = _accuracy(fp32, test_x, test_y) if test_y is not None else None
    except Exception as exc:
        result.error = f"fp32 eval failed: {type(exc).__name__}: {exc}"
        return result

    # FP32 "on-disk" size — approximate as state_dict byte count.
    fp32_bytes = sum(p.numel() * p.element_size() for p in fp32.parameters())
    fp32_bytes += sum(b.numel() * b.element_size() for b in fp32.buffers())
    result.fp32_size_mb = fp32_bytes / (1024 * 1024)
    result.fp32_latency_ms = fp32_latency
    result.fp32_accuracy = fp32_acc

    # -- static quantization --
    backend = _pick_backend()
    logger.info("Quantizing %s using backend=%s", name, backend)

    try:
        from torch.ao.quantization import (
            convert,
            get_default_qconfig,
            prepare,
        )

        quantizable = copy.deepcopy(fp32).eval()
        _fuse_cnn_in_place(quantizable)
        wrapped = _QuantWrapper(quantizable).eval()
        wrapped.qconfig = get_default_qconfig(backend)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare(wrapped, inplace=False)
            _calibrate(prepared, calib_x)
            int8_model = convert(prepared, inplace=False)
    except Exception as exc:
        logger.warning("Quantization failed for %s: %s", name, exc)
        result.error = f"quantize failed: {type(exc).__name__}: {exc}"
        if write:
            (results_dir / f"quantization_{name}.json").write_text(
                json.dumps(asdict(result), indent=2)
            )
        return result

    # -- evaluate int8 --
    try:
        int8_latency = _mean_latency_ms(int8_model)
        int8_acc = _accuracy(int8_model, test_x, test_y) if test_y is not None else None
    except Exception as exc:
        logger.warning("INT8 eval failed for %s: %s", name, exc)
        result.error = f"int8 eval failed: {type(exc).__name__}: {exc}"
        if write:
            (results_dir / f"quantization_{name}.json").write_text(
                json.dumps(asdict(result), indent=2)
            )
        return result

    # -- serialise int8 model --
    try:
        scripted = torch.jit.script(int8_model)
        scripted.save(str(out_path))
    except Exception as exc:
        # Fallback: just save the state dict.
        logger.warning(
            "torch.jit.script failed for %s: %s; falling back to state_dict save", name, exc
        )
        try:
            torch.save(int8_model.state_dict(), out_path)
        except Exception as exc2:
            result.error = f"save failed: {type(exc2).__name__}: {exc2}"
            if write:
                (results_dir / f"quantization_{name}.json").write_text(
                    json.dumps(asdict(result), indent=2)
                )
            return result

    int8_size_mb = out_path.stat().st_size / (1024 * 1024)
    result.int8_path = str(out_path)
    result.int8_size_mb = int8_size_mb
    result.size_ratio = result.fp32_size_mb / int8_size_mb if int8_size_mb > 0 else None
    result.int8_latency_ms = int8_latency
    result.latency_ratio = int8_latency / fp32_latency if fp32_latency > 0 else None
    result.int8_accuracy = int8_acc
    result.accuracy_delta = (
        (int8_acc - fp32_acc) if (int8_acc is not None and fp32_acc is not None) else None
    )
    result.ok = True

    if write:
        json_path = results_dir / f"quantization_{name}.json"
        json_path.write_text(json.dumps(asdict(result), indent=2))
        logger.info("Wrote %s", json_path)

    return result


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--all", action="store_true", help="Quantize every *_best.pth under --checkpoints-dir."
    )
    p.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--data-cache", type=Path, default=REPO_ROOT / "data" / "LSWMD_cache.npz")
    p.add_argument("--num-calib", type=int, default=128)
    p.add_argument("--num-test", type=int, default=1024)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.all:
        targets = []
        for ckpt in discover_checkpoints(args.checkpoints_dir):
            name = infer_model_name(ckpt)
            if name and name in KNOWN_MODEL_NAMES:
                targets.append((name, ckpt))
            else:
                logger.warning("Skipping %s: unknown model name", ckpt)
    else:
        if args.model is None and args.checkpoint is None:
            logger.error("Pass --model, --checkpoint, or --all.")
            return 2
        name = args.model
        if name is None:
            name = infer_model_name(args.checkpoint)
            if name is None:
                logger.error("Cannot infer model from %s", args.checkpoint)
                return 2
        targets = [(name, args.checkpoint)]

    any_ok = False
    for name, ckpt in targets:
        r = quantize_checkpoint(
            name,
            ckpt,
            num_calib=args.num_calib,
            num_test=args.num_test,
            out_dir=args.checkpoints_dir,
            results_dir=args.results_dir,
            data_cache=args.data_cache,
        )
        msg = (
            f"{name}: {'OK' if r.ok else 'FAIL'}"
            + (f" size_ratio={r.size_ratio:.2f}x" if r.size_ratio else "")
            + (f" lat_ratio={r.latency_ratio:.2f}x" if r.latency_ratio else "")
            + (f" Δacc={r.accuracy_delta:+.4f}" if r.accuracy_delta is not None else "")
            + (f" err={r.error}" if r.error else "")
        )
        logger.info(msg)
        any_ok = any_ok or r.ok

    return 0 if any_ok or not targets else 1


if __name__ == "__main__":
    raise SystemExit(main())
