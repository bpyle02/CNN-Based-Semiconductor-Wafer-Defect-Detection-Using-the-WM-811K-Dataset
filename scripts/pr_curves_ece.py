#!/usr/bin/env python3
"""Per-class precision-recall curves + Expected Calibration Error.

For each ``checkpoints/<model>_best.pth`` (or just the one passed with
``--checkpoint``):

1. Rebuild the model with ``train.build_model`` and load its weights.
2. Reproduce the deterministic seed=42 test split (reuses
   ``scripts/evaluate_ensemble._load_splits`` — the same 70/15/15 split
   ``train.py`` uses so the test set matches published numbers).
3. Compute softmax probs on the test set.
4. Plot 9 per-class PR curves on one figure.
5. Compute ECE with **equal-mass binning** on max-confidence
   predictions, and render a calibration diagram.

Outputs per model ``<m>``:

- ``results/pr_curves_<m>.png``         (9-class PR curves)
- ``results/calibration_<m>.png``       (reliability diagram)
- ``results/calibration_<m>.json``      (ECE + per-bin stats)

Usage::

    python scripts/pr_curves_ece.py                       # auto-discover
    python scripts/pr_curves_ece.py --checkpoint checkpoints/cnn_best.pth
    python scripts/pr_curves_ece.py --device cpu --n-bins 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # headless-safe before pyplot import
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse split/build helpers from evaluate_ensemble to guarantee we see
# exactly the same test set train.py trains against.
from scripts.evaluate_ensemble import (  # noqa: E402
    DISPLAY,
    _build_and_load,
    _collect_probs,
    _discover,
    _load_splits,
)
from src.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECE with equal-mass binning
# ---------------------------------------------------------------------------


def equal_mass_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[dict]]:
    """ECE with equal-mass (equal-count) binning on max-confidence predictions.

    Equal-mass binning places roughly the same number of samples in
    each bin — more robust than equal-width when confidences pile up
    near 1.0, which is typical for trained CNNs.

    Returns ``(ece, per_bin_stats)``.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = confidences.shape[0]
    if n == 0:
        return 0.0, []

    # Sort by confidence, then slice into ~equal-count bins.
    order = np.argsort(confidences, kind="mergesort")
    conf_sorted = confidences[order]
    correct_sorted = correct[order]

    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    per_bin: List[dict] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            per_bin.append(
                {
                    "bin_index": i,
                    "count": 0,
                    "confidence": float("nan"),
                    "accuracy": float("nan"),
                    "gap": float("nan"),
                    "conf_lo": float("nan"),
                    "conf_hi": float("nan"),
                }
            )
            continue
        bin_conf = conf_sorted[lo:hi]
        bin_acc = correct_sorted[lo:hi]
        mean_conf = float(bin_conf.mean())
        mean_acc = float(bin_acc.mean())
        gap = abs(mean_conf - mean_acc)
        ece += (hi - lo) / n * gap
        per_bin.append(
            {
                "bin_index": i,
                "count": int(hi - lo),
                "confidence": mean_conf,
                "accuracy": mean_acc,
                "gap": gap,
                "conf_lo": float(bin_conf[0]),
                "conf_hi": float(bin_conf[-1]),
            }
        )
    return float(ece), per_bin


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pr_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    out_path: Path,
    model_title: str,
) -> Dict[str, float]:
    """One-vs-rest PR curve per class on a single figure."""
    n_classes = probs.shape[1]
    ap_per_class: Dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(n_classes):
        y_true = (labels == c).astype(int)
        if y_true.sum() == 0:
            ap_per_class[class_names[c]] = float("nan")
            continue
        precision, recall, _ = precision_recall_curve(y_true, probs[:, c])
        ap = average_precision_score(y_true, probs[:, c])
        ap_per_class[class_names[c]] = float(ap)
        ax.plot(recall, precision, lw=1.5, label=f"{class_names[c]} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Per-class PR curves — {model_title}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ap_per_class


def plot_reliability(
    per_bin: List[dict],
    ece: float,
    out_path: Path,
    model_title: str,
) -> None:
    """Reliability diagram: bar of accuracy vs confidence per bin."""
    confs = [b["confidence"] for b in per_bin if b["count"] > 0]
    accs = [b["accuracy"] for b in per_bin if b["count"] > 0]
    counts = [b["count"] for b in per_bin if b["count"] > 0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    if confs:
        ax1.plot(confs, accs, "o-", lw=1.5, ms=5, label=f"Observed (ECE={ece:.4f})")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Reliability diagram (equal-mass) — {model_title}")
    ax1.legend(loc="lower right")

    if confs:
        ax2.bar(confs, counts, width=0.03, alpha=0.6, color="steelblue")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single checkpoint to evaluate; omit to auto-discover all in --checkpoints-dir",
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "LSWMD_new.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve checkpoints.
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            logger.error("Checkpoint not found: %s", args.checkpoint)
            return 1
        name = args.checkpoint.stem.replace("_best", "")
        found = [(name, args.checkpoint)]
    else:
        found = _discover(args.checkpoints_dir, None)
        if not found:
            logger.error("No *_best.pth checkpoints in %s", args.checkpoints_dir)
            return 1

    logger.info("Evaluating %d model(s): %s", len(found), [n for n, _ in found])

    config = load_config(args.config)
    # Reuse train.py's deterministic split + class names.
    from src.data.dataset import KNOWN_CLASSES

    class_names = list(KNOWN_CLASSES)

    _, test_loader, num_classes = _load_splits(
        config, args.data_path, args.seed, args.batch_size, args.num_workers
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)

    for model_name, ckpt in found:
        display = DISPLAY.get(model_name, model_name)
        logger.info("--- %s (%s) ---", display, ckpt)
        model = _build_and_load(model_name, ckpt, config, num_classes, args.device)
        probs, labels = _collect_probs(model, test_loader, args.device)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

        # PR curves.
        pr_path = args.results_dir / f"pr_curves_{model_name}.png"
        ap_per_class = plot_pr_curves(probs, labels, class_names, pr_path, display)
        logger.info("Wrote %s", pr_path)

        # Equal-mass ECE + reliability diagram.
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        correct = (preds == labels).astype(np.float64)
        ece, per_bin = equal_mass_ece(confidences, correct, n_bins=args.n_bins)
        cal_png = args.results_dir / f"calibration_{model_name}.png"
        plot_reliability(per_bin, ece, cal_png, display)
        logger.info("Wrote %s", cal_png)

        cal_json = args.results_dir / f"calibration_{model_name}.json"
        cal_json.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "display_name": display,
                    "n_bins": args.n_bins,
                    "binning": "equal_mass",
                    "ece": ece,
                    "accuracy": float(correct.mean()),
                    "mean_confidence": float(confidences.mean()),
                    "per_bin": per_bin,
                    "average_precision": ap_per_class,
                    "n_samples": int(labels.shape[0]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %s (ECE=%.4f)", cal_json, ece)

    return 0


if __name__ == "__main__":
    sys.exit(main())
