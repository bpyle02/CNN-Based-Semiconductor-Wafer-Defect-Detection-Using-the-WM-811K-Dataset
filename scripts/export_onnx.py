#!/usr/bin/env python3
"""
Export trained wafer-defect models to ONNX.

* ``--model <name>`` or ``--checkpoint <path>`` (at least one required).
* Writes ``checkpoints/<model>.onnx`` with dynamic batch size on both input
  and output.
* Validates the export against the original PyTorch model on a batch of
  random inputs. The mean absolute difference must be below ``--tolerance``
  (default ``1e-5``); otherwise the export is flagged.
* Reports the ONNX file size.
* Default ``opset_version=17``.

Robustness:
    Some architectures (notably ViT / Swin / RIDE with data-dependent control
    flow) can fail to trace. Those failures are caught, logged, and skipped
    so that a bulk export of the checkpoints directory still finishes.

Usage:
    python -m scripts.export_onnx --model cnn
    python -m scripts.export_onnx --checkpoint checkpoints/cnn_best.pth
    python -m scripts.export_onnx --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

DEFAULT_OPSET = 17
DEFAULT_TOLERANCE = 1e-5


@dataclass
class ExportResult:
    model: str
    onnx_path: Optional[str]
    size_mb: Optional[float]
    mean_abs_diff: Optional[float]
    max_abs_diff: Optional[float]
    tolerance: float
    ok: bool
    error: Optional[str] = None


def _validate_onnx(
    model: nn.Module,
    onnx_path: Path,
    *,
    batch_size: int = 4,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """Compare PyTorch vs onnxruntime outputs on a random batch.

    Returns a dict with ``mean_abs_diff`` and ``max_abs_diff`` in float.
    """
    import onnxruntime as ort  # imported lazily so the script runs even if

    # ``onnxruntime`` is missing (the export itself would still succeed).

    model.eval()
    x = torch.randn((batch_size, *INPUT_SHAPE), dtype=torch.float32)
    with torch.inference_mode():
        torch_out = model(x).detach().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    (onnx_out,) = sess.run(None, {input_name: x.numpy()})

    diff = np.abs(torch_out - onnx_out)
    stats = {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }
    if stats["mean_abs_diff"] > tolerance:
        logger.warning(
            "ONNX validation: mean|Δ|=%.3e > tol=%.0e for %s",
            stats["mean_abs_diff"],
            tolerance,
            onnx_path.name,
        )
    return stats


def export_single(
    model_name: str,
    *,
    checkpoint: Optional[Path] = None,
    out_path: Optional[Path] = None,
    opset: int = DEFAULT_OPSET,
    tolerance: float = DEFAULT_TOLERANCE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    validate: bool = True,
) -> ExportResult:
    """Export one model to ONNX. Errors are captured, not raised."""
    name = model_name.lower()
    out_path = out_path or (REPO_ROOT / "checkpoints" / f"{name}.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = ExportResult(
        model=name,
        onnx_path=None,
        size_mb=None,
        mean_abs_diff=None,
        max_abs_diff=None,
        tolerance=tolerance,
        ok=False,
    )

    try:
        model = build_model(name, num_classes=num_classes).cpu().eval()
    except Exception as exc:
        result.error = f"build failed: {type(exc).__name__}: {exc}"
        return result

    if checkpoint is not None:
        try:
            load_checkpoint_weights(model, Path(checkpoint), device="cpu")
        except Exception as exc:
            result.error = f"checkpoint load failed: {type(exc).__name__}: {exc}"
            return result

    dummy = torch.randn((1, *INPUT_SHAPE), dtype=torch.float32)

    # ``dynamic_axes`` maps the 0-th dim of ``input`` and ``output`` to a
    # symbolic "batch" dimension so the exported graph accepts any batch size.
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    try:
        # ``dynamo=False`` pins the legacy TorchScript-based tracer. The newer
        # dynamo exporter requires ``onnxscript`` which isn't a hard dep here,
        # and the legacy tracer is perfectly adequate for these CNNs.
        export_kwargs = dict(
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        try:
            torch.onnx.export(model, dummy, str(out_path), dynamo=False, **export_kwargs)
        except TypeError:
            # Older torch versions don't expose the ``dynamo`` kwarg.
            torch.onnx.export(model, dummy, str(out_path), **export_kwargs)
    except Exception as exc:
        logger.warning("ONNX export failed for %s: %s", name, exc)
        result.error = f"onnx.export failed: {type(exc).__name__}: {exc}"
        # Best-effort cleanup of a partial file.
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        return result

    size_mb = out_path.stat().st_size / (1024 * 1024)
    result.onnx_path = str(out_path)
    result.size_mb = size_mb

    if validate:
        try:
            stats = _validate_onnx(model, out_path, tolerance=tolerance)
            result.mean_abs_diff = stats["mean_abs_diff"]
            result.max_abs_diff = stats["max_abs_diff"]
            result.ok = stats["mean_abs_diff"] < tolerance
        except Exception as exc:
            logger.warning("ONNX validation failed for %s: %s", name, exc)
            result.error = f"validation failed: {type(exc).__name__}: {exc}"
            result.ok = False
    else:
        result.ok = True

    return result


def export_all(
    checkpoints_dir: Path = REPO_ROOT / "checkpoints",
    *,
    opset: int = DEFAULT_OPSET,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[ExportResult]:
    """Export every checkpoint under ``checkpoints_dir`` (graceful failures)."""
    results: List[ExportResult] = []
    for ckpt in discover_checkpoints(checkpoints_dir):
        name = infer_model_name(ckpt)
        if name is None or name not in KNOWN_MODEL_NAMES:
            results.append(
                ExportResult(
                    model=ckpt.stem,
                    onnx_path=None,
                    size_mb=None,
                    mean_abs_diff=None,
                    max_abs_diff=None,
                    tolerance=tolerance,
                    ok=False,
                    error=f"unknown model in filename '{ckpt.name}'",
                )
            )
            continue
        logger.info("Exporting %s -> ONNX ...", name)
        results.append(
            export_single(
                name,
                checkpoint=ckpt,
                opset=opset,
                tolerance=tolerance,
            )
        )
    return results


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", type=str, default=None, help="Architecture name (e.g. cnn, resnet, vit)."
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None, help="Checkpoint file to load before export."
    )
    p.add_argument(
        "--all", action="store_true", help="Export every *_best.pth under --checkpoints-dir."
    )
    p.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument(
        "--out", type=Path, default=None, help="Output path; defaults to checkpoints/<model>.onnx."
    )
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.all:
        results = export_all(args.checkpoints_dir, opset=args.opset, tolerance=args.tolerance)
    else:
        if args.model is None and args.checkpoint is None:
            logger.error("Pass --model, --checkpoint, or --all.")
            return 2
        name = args.model
        if name is None:
            name = infer_model_name(args.checkpoint)
            if name is None:
                logger.error("Cannot infer model from '%s'; pass --model.", args.checkpoint)
                return 2
        results = [
            export_single(
                name,
                checkpoint=args.checkpoint,
                out_path=args.out,
                opset=args.opset,
                tolerance=args.tolerance,
                validate=not args.no_validate,
            )
        ]

    for r in results:
        line = (
            f"{r.model}: {'OK' if r.ok else 'FAIL'}"
            + (f" size={r.size_mb:.2f}MB" if r.size_mb is not None else "")
            + (f" mean|Δ|={r.mean_abs_diff:.3e}" if r.mean_abs_diff is not None else "")
            + (f" err={r.error}" if r.error else "")
        )
        logger.info(line)

    # Return non-zero if any export failed, but do not abort the batch early.
    return 0 if all(r.ok or r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
