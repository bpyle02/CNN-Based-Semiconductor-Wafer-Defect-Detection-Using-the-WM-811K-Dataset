#!/usr/bin/env python3
"""
Label noise estimation via confident-learning-style disagreement analysis.

Pure numpy/torch implementation (no cleanlab dependency). Flags training
samples that the model predicts with high confidence (>threshold) into a
class OTHER than their labeled class. These are candidate mislabels.

Algorithm:
    1. Train a small CNN on the train split (or load checkpoint via
       --use-checkpoint). Optionally hold out K folds for cross-validated
       probabilities; otherwise use self-predictions (with a caveat note).
    2. For every train sample, record p(pred_class) and disagreement flag
       where argmax(p) != true_label and max(p) > confidence_threshold.
    3. Rank disagreements by confidence; report top-N as likely mislabels.
    4. Estimate per-class noise rate as (flagged / class_total).

Outputs:
    results/label_noise_candidates.json
    docs/label_noise_analysis.md

Usage:
    python scripts/label_noise.py --quick
    python scripts/label_noise.py --use-checkpoint checkpoints/best_cnn.pth
    python scripts/label_noise.py --folds 3 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    KNOWN_CLASSES,
    WaferMapDataset,
    get_image_transforms,
    load_dataset,
    preprocess_wafer_maps,
    seed_worker,
)
from src.models import WaferCNN  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class NoiseCandidate:
    sample_idx: int
    true_label: str
    predicted_label: str
    confidence: float


def _train_model(
    train_ds: WaferMapDataset,
    num_classes: int,
    epochs: int,
    device: str,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> nn.Module:
    """Train a WaferCNN on the given dataset for a few epochs."""
    model = WaferCNN(num_classes=num_classes).to(device)
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        logger.info(
            "  epoch %d/%d loss=%.4f acc=%.4f",
            ep + 1,
            epochs,
            loss_sum / total,
            correct / total,
        )
    return model


@torch.no_grad()
def _predict_probs(
    model: nn.Module,
    ds: WaferMapDataset,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    all_probs: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def compute_cv_probabilities(
    maps: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    folds: int,
    epochs: int,
    device: str,
) -> np.ndarray:
    """Compute out-of-fold predicted probabilities via K-fold CV."""
    probs = np.zeros((len(labels), num_classes), dtype=np.float32)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    transform = get_image_transforms()
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(maps), 1):
        logger.info("Fold %d/%d: train=%d val=%d", fold_idx, folds, len(tr_idx), len(va_idx))
        tr_ds = WaferMapDataset(maps[tr_idx], labels[tr_idx], transform=transform)
        va_ds = WaferMapDataset(maps[va_idx], labels[va_idx], transform=None)
        model = _train_model(tr_ds, num_classes, epochs, device)
        probs[va_idx] = _predict_probs(model, va_ds, device)
    return probs


def self_probabilities(
    maps: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    epochs: int,
    device: str,
    checkpoint: Optional[Path] = None,
) -> np.ndarray:
    """Train once on all data, then predict on all data (self-confidence bias)."""
    ds_train = WaferMapDataset(maps, labels, transform=get_image_transforms())
    ds_eval = WaferMapDataset(maps, labels, transform=None)
    if checkpoint is not None and checkpoint.exists():
        logger.info("Loading checkpoint %s", checkpoint)
        model = WaferCNN(num_classes=num_classes).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        model = _train_model(ds_train, num_classes, epochs, device)
    return _predict_probs(model, ds_eval, device)


def flag_candidates(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    threshold: float,
) -> Tuple[List[NoiseCandidate], np.ndarray]:
    """Return sorted list of high-confidence disagreements + per-class counts."""
    pred_idx = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    disagree = (pred_idx != labels) & (pred_conf > threshold)

    num_classes = len(class_names)
    per_class = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        per_class[c] = int(disagree[labels == c].sum())

    candidates: List[NoiseCandidate] = []
    for idx in np.where(disagree)[0]:
        candidates.append(
            NoiseCandidate(
                sample_idx=int(idx),
                true_label=class_names[int(labels[idx])],
                predicted_label=class_names[int(pred_idx[idx])],
                confidence=float(pred_conf[idx]),
            )
        )
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates, per_class


def write_report(
    top: List[NoiseCandidate],
    per_class: np.ndarray,
    class_names: List[str],
    class_totals: np.ndarray,
    used_cv: bool,
    out_md: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Label Noise Analysis\n")
    method = (
        "K-fold cross-validated predictions (unbiased)."
        if used_cv
        else "Self-predictions (subject to self-confidence bias; "
        "over-estimates confidence on training samples)."
    )
    lines.append(f"**Method:** {method}\n")
    lines.append(
        "We flag training samples where the model predicts a class OTHER "
        "than the labeled class with confidence greater than the threshold. "
        "These are candidate mislabels for manual inspection.\n",
    )

    lines.append("\n## Per-class estimated noise rate\n")
    lines.append("| Class | Flagged | Total | Rate |")
    lines.append("|---|---:|---:|---:|")
    for i, name in enumerate(class_names):
        total = int(class_totals[i])
        rate = per_class[i] / total if total > 0 else 0.0
        lines.append(f"| {name} | {int(per_class[i])} | {total} | {rate:.2%} |")

    lines.append("\n## Top-20 highest-confidence disagreements\n")
    lines.append("| Rank | sample_idx | true_label | predicted_label | confidence |")
    lines.append("|---:|---:|---|---|---:|")
    for rank, c in enumerate(top[:20], 1):
        lines.append(
            f"| {rank} | {c.sample_idx} | {c.true_label} | "
            f"{c.predicted_label} | {c.confidence:.4f} |",
        )

    lines.append(
        "\n> Interpretation: high-confidence disagreements are candidate "
        "mislabels. A manual re-inspection of the top-ranked wafer maps is "
        "a cheap sanity check that often pays off.\n",
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Confident-learning-style noise estimation")
    parser.add_argument(
        "--folds", type=int, default=0, help="Number of CV folds (0 = self-prediction)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs per fold / self-train"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for flagging disagreements",
    )
    parser.add_argument(
        "--use-checkpoint",
        type=Path,
        default=None,
        help="Reuse existing CNN checkpoint instead of training",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick demo: 1 epoch, tiny subsample, self-predict"
    )
    parser.add_argument(
        "--subsample", type=int, default=0, help="Subsample train set to N examples (0 = all)"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--out-json", type=Path, default=REPO_ROOT / "results" / "label_noise_candidates.json"
    )
    parser.add_argument(
        "--out-md", type=Path, default=REPO_ROOT / "docs" / "label_noise_analysis.md"
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.folds = 0
        if args.subsample == 0:
            args.subsample = 2000

    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Loading WM-811K dataset...")
    df = load_dataset(args.data_path)
    mask = df["failureClass"].isin(KNOWN_CLASSES)
    df = df[mask].reset_index(drop=True)

    le = LabelEncoder()
    labels = le.fit_transform(df["failureClass"])
    class_names = le.classes_.tolist()
    maps_raw = df["waferMap"].values

    # Use train split (same 80/20 convention as active_learn.py)
    train_idx, _ = train_test_split(
        np.arange(len(labels)),
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )
    train_labels = labels[train_idx]
    train_maps_raw = [maps_raw[i] for i in train_idx]

    if args.subsample > 0 and args.subsample < len(train_labels):
        logger.info("Subsampling train set: %d -> %d", len(train_labels), args.subsample)
        sub = np.random.choice(len(train_labels), args.subsample, replace=False)
        train_labels = train_labels[sub]
        train_maps_raw = [train_maps_raw[i] for i in sub]
        train_idx = train_idx[sub]

    logger.info("Preprocessing %d wafer maps...", len(train_labels))
    maps = np.array(preprocess_wafer_maps(train_maps_raw))

    num_classes = len(class_names)
    if args.folds >= 2:
        logger.info("Computing cross-validated probabilities (%d folds)", args.folds)
        probs = compute_cv_probabilities(
            maps,
            train_labels,
            num_classes,
            folds=args.folds,
            epochs=args.epochs,
            device=args.device,
        )
        used_cv = True
    else:
        logger.info("Computing self-predictions (biased; see docs)")
        probs = self_probabilities(
            maps,
            train_labels,
            num_classes,
            epochs=args.epochs,
            device=args.device,
            checkpoint=args.use_checkpoint,
        )
        used_cv = False

    candidates, per_class = flag_candidates(
        probs,
        train_labels,
        class_names,
        args.threshold,
    )
    # Rewrite sample_idx to refer back to the original DataFrame index
    for c in candidates:
        c.sample_idx = int(train_idx[c.sample_idx])

    class_totals = np.array(
        [(train_labels == i).sum() for i in range(num_classes)],
        dtype=np.int64,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "cross_validation" if used_cv else "self_prediction",
        "folds": args.folds if used_cv else 0,
        "epochs": args.epochs,
        "threshold": args.threshold,
        "num_samples": int(len(train_labels)),
        "num_flagged": int(len(candidates)),
        "class_names": class_names,
        "per_class_flagged": per_class.tolist(),
        "per_class_total": class_totals.tolist(),
        "per_class_noise_rate": [
            float(per_class[i] / class_totals[i]) if class_totals[i] > 0 else 0.0
            for i in range(num_classes)
        ],
        "top_candidates": [asdict(c) for c in candidates[:100]],
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    write_report(candidates, per_class, class_names, class_totals, used_cv, args.out_md)
    logger.info("Wrote %s", args.out_md)
    logger.info(
        "Flagged %d / %d samples (%.2f%%)",
        len(candidates),
        len(train_labels),
        100.0 * len(candidates) / max(1, len(train_labels)),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sys.exit(main())
