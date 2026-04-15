#!/usr/bin/env python3
"""
Generate active learning curves: random sampling vs. uncertainty sampling.

Produces THE curve for the report:
    x-axis: number of labeled samples (100, 500, 1000, 2500, 5000, 10000, full)
    y-axis: test Macro F1
    two lines: random-sampling baseline + uncertainty-sampling active learning
    optional shaded std band across seeds

Both strategies reuse the same Custom CNN architecture and training protocol
(2 epochs per budget, Adam 1e-3). Active learning bootstraps from 100 random
samples, then at each subsequent budget selects the (delta) samples with
highest predictive entropy from the remaining unlabeled pool.

This is expensive (~1hr on a T4 GPU); use --quick for a smoke-test run
(3 budgets x 1 seed, subsampled pool).

Outputs:
    results/active_learning.json
    results/fig_active_learning.png
    results/fig_active_learning.pdf

Usage:
    python scripts/active_learning_curves.py --quick
    python scripts/active_learning_curves.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

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

DEFAULT_BUDGETS = [100, 500, 1000, 2500, 5000, 10000]
QUICK_BUDGETS = [100, 500, 1500]


def _make_loader(ds: WaferMapDataset, batch_size: int, shuffle: bool) -> DataLoader:
    g = torch.Generator().manual_seed(42)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        worker_init_fn=seed_worker, generator=g,
    )


def train_and_evaluate(
    maps: np.ndarray,
    labels: np.ndarray,
    idx_train: np.ndarray,
    test_ds: WaferMapDataset,
    num_classes: int,
    epochs: int,
    device: str,
    batch_size: int = 64,
) -> Tuple[nn.Module, float]:
    """Train a fresh CNN on maps[idx_train], return (model, test macro F1)."""
    tr_ds = WaferMapDataset(
        maps[idx_train], labels[idx_train], transform=get_image_transforms(),
    )
    loader = _make_loader(tr_ds, batch_size, shuffle=True)
    model = WaferCNN(num_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    return model, evaluate_f1(model, test_ds, device, batch_size)


@torch.no_grad()
def evaluate_f1(
    model: nn.Module, test_ds: WaferMapDataset, device: str, batch_size: int = 128,
) -> float:
    loader = _make_loader(test_ds, batch_size, shuffle=False)
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(f1_score(y, p, average="macro", zero_division=0))


@torch.no_grad()
def predictive_entropy(
    model: nn.Module,
    maps: np.ndarray,
    labels: np.ndarray,
    idxs: np.ndarray,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Entropy of softmax predictions on the given sample indices."""
    ds = WaferMapDataset(maps[idxs], labels[idxs], transform=None)
    loader = _make_loader(ds, batch_size, shuffle=False)
    model.eval()
    out: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        probs = torch.softmax(model(xb), dim=1)
        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        out.append(ent.cpu().numpy())
    return np.concatenate(out)


def run_random_curve(
    maps: np.ndarray,
    labels: np.ndarray,
    test_ds: WaferMapDataset,
    budgets: List[int],
    num_classes: int,
    epochs: int,
    device: str,
    seed: int,
) -> List[Tuple[int, float]]:
    """For each budget, train on a fresh random subset and record Macro F1."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    curve: List[Tuple[int, float]] = []
    for b in budgets:
        b_eff = min(b, n)
        idx = rng.choice(n, b_eff, replace=False)
        _, f1 = train_and_evaluate(
            maps, labels, idx, test_ds, num_classes, epochs, device,
        )
        logger.info("  [random seed=%d] budget=%d f1=%.4f", seed, b_eff, f1)
        curve.append((b_eff, f1))
    return curve


def run_active_curve(
    maps: np.ndarray,
    labels: np.ndarray,
    test_ds: WaferMapDataset,
    budgets: List[int],
    num_classes: int,
    epochs: int,
    device: str,
    seed: int,
) -> List[Tuple[int, float]]:
    """Active learning: bootstrap with random 100, then entropy-sample to each
    subsequent budget."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    budgets = [min(b, n) for b in budgets]

    labeled = rng.choice(n, budgets[0], replace=False)
    model, f1 = train_and_evaluate(
        maps, labels, labeled, test_ds, num_classes, epochs, device,
    )
    logger.info("  [AL seed=%d] budget=%d (bootstrap) f1=%.4f",
                seed, len(labeled), f1)
    curve: List[Tuple[int, float]] = [(len(labeled), f1)]

    for b in budgets[1:]:
        want = b - len(labeled)
        if want <= 0:
            curve.append((len(labeled), f1))
            continue
        unlabeled = np.setdiff1d(np.arange(n), labeled, assume_unique=False)
        if len(unlabeled) == 0:
            curve.append((len(labeled), f1))
            continue
        ent = predictive_entropy(model, maps, labels, unlabeled, device)
        top = unlabeled[np.argsort(-ent)[:want]]
        labeled = np.concatenate([labeled, top])
        model, f1 = train_and_evaluate(
            maps, labels, labeled, test_ds, num_classes, epochs, device,
        )
        logger.info("  [AL seed=%d] budget=%d (+%d by entropy) f1=%.4f",
                    seed, len(labeled), want, f1)
        curve.append((len(labeled), f1))
    return curve


def aggregate_curves(
    per_seed: List[List[Tuple[int, float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-seed curves -> (budgets, mean_f1, std_f1)."""
    arr = np.array(per_seed, dtype=np.float64)  # (seeds, budgets, 2)
    budgets = arr[0, :, 0]
    f1s = arr[:, :, 1]
    return budgets, f1s.mean(axis=0), f1s.std(axis=0)


def plot_curves(
    random_budgets: np.ndarray,
    random_mean: np.ndarray,
    random_std: np.ndarray,
    al_budgets: np.ndarray,
    al_mean: np.ndarray,
    al_std: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(random_budgets, random_mean, marker="o", label="Random sampling",
            color="tab:blue")
    if random_std.any():
        ax.fill_between(random_budgets, random_mean - random_std,
                        random_mean + random_std, color="tab:blue", alpha=0.15)
    ax.plot(al_budgets, al_mean, marker="s",
            label="Active learning (entropy)", color="tab:orange")
    if al_std.any():
        ax.fill_between(al_budgets, al_mean - al_std, al_mean + al_std,
                        color="tab:orange", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled training samples")
    ax.set_ylabel("Test Macro F1")
    ax.set_title("Active Learning vs. Random Sampling (WM-811K, Custom CNN)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Active learning curves")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for std band")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per budget")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Override budget list")
    parser.add_argument("--include-full", action="store_true",
                        help="Append full train-set budget to the random curve")
    parser.add_argument("--quick", action="store_true",
                        help="3 budgets x 1 seed, 4k pool subsample")
    parser.add_argument("--pool-subsample", type=int, default=0,
                        help="Subsample train pool to N (0 = full)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out-json", type=Path,
                        default=REPO_ROOT / "results" / "active_learning.json")
    parser.add_argument("--out-png", type=Path,
                        default=REPO_ROOT / "results" / "fig_active_learning.png")
    parser.add_argument("--out-pdf", type=Path,
                        default=REPO_ROOT / "results" / "fig_active_learning.pdf")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.epochs = 1
        if args.budgets is None:
            args.budgets = QUICK_BUDGETS
        if args.pool_subsample == 0:
            args.pool_subsample = 4000

    budgets = args.budgets if args.budgets is not None else list(DEFAULT_BUDGETS)

    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Loading WM-811K dataset...")
    df = load_dataset(args.data_path)
    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)
    le = LabelEncoder()
    labels_all = le.fit_transform(df["failureClass"])
    class_names = le.classes_.tolist()
    num_classes = len(class_names)
    maps_all = df["waferMap"].values

    tr_idx, te_idx = train_test_split(
        np.arange(len(labels_all)), test_size=0.20,
        stratify=labels_all, random_state=42,
    )
    train_labels = labels_all[tr_idx]
    test_labels = labels_all[te_idx]
    train_maps_raw = [maps_all[i] for i in tr_idx]
    test_maps_raw = [maps_all[i] for i in te_idx]

    if args.pool_subsample > 0 and args.pool_subsample < len(train_labels):
        logger.info("Subsampling train pool: %d -> %d",
                    len(train_labels), args.pool_subsample)
        sub = np.random.RandomState(0).choice(
            len(train_labels), args.pool_subsample, replace=False,
        )
        train_labels = train_labels[sub]
        train_maps_raw = [train_maps_raw[i] for i in sub]

    logger.info("Preprocessing train=%d test=%d ...",
                len(train_labels), len(test_labels))
    train_maps = np.array(preprocess_wafer_maps(train_maps_raw))
    test_maps = np.array(preprocess_wafer_maps(test_maps_raw))
    test_ds = WaferMapDataset(test_maps, test_labels, transform=None)

    budgets_eff = [b for b in budgets if b <= len(train_labels)]
    if args.include_full and len(train_labels) not in budgets_eff:
        budgets_eff.append(len(train_labels))
    logger.info("Budgets: %s", budgets_eff)

    random_curves: List[List[Tuple[int, float]]] = []
    al_curves: List[List[Tuple[int, float]]] = []
    # AL curve excludes the "full" budget point: uncertainty has no pool left.
    al_budgets = [b for b in budgets_eff if b < len(train_labels)] or budgets_eff

    for s in range(args.seeds):
        seed = 42 + s
        logger.info("=== Seed %d/%d ===", s + 1, args.seeds)
        logger.info("--- Random baseline ---")
        random_curves.append(run_random_curve(
            train_maps, train_labels, test_ds, budgets_eff,
            num_classes, args.epochs, args.device, seed,
        ))
        logger.info("--- Active learning ---")
        al_curves.append(run_active_curve(
            train_maps, train_labels, test_ds, al_budgets,
            num_classes, args.epochs, args.device, seed,
        ))

    rb, rm, rs = aggregate_curves(random_curves)
    ab, am, asd = aggregate_curves(al_curves)

    payload: Dict = {
        "config": {
            "seeds": args.seeds,
            "epochs_per_budget": args.epochs,
            "budgets": budgets_eff,
            "al_budgets": al_budgets,
            "pool_subsample": args.pool_subsample,
            "pool_size": int(len(train_labels)),
            "test_size": int(len(test_labels)),
        },
        "class_names": class_names,
        "random": {
            "budgets": rb.tolist(),
            "macro_f1_mean": rm.tolist(),
            "macro_f1_std": rs.tolist(),
            "per_seed": [[(int(b), float(f)) for b, f in curve]
                         for curve in random_curves],
        },
        "active": {
            "budgets": ab.tolist(),
            "macro_f1_mean": am.tolist(),
            "macro_f1_std": asd.tolist(),
            "per_seed": [[(int(b), float(f)) for b, f in curve]
                         for curve in al_curves],
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    plot_curves(rb, rm, rs, ab, am, asd, args.out_png, args.out_pdf)
    logger.info("Wrote %s and %s", args.out_png, args.out_pdf)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sys.exit(main())
