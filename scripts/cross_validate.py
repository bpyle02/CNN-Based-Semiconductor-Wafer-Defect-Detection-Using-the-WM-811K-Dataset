#!/usr/bin/env python3
"""
5-fold stratified cross-validation for wafer defect models.

For each fold we treat that fold's indices as *test*, and split the remaining
indices into train/val (90/10, stratified) so the scheduler + early stopping
have a validation signal. Per-fold test metrics (macro_f1, accuracy,
weighted_f1) are aggregated into mean + std.

Outputs:
    results/cv_<model>.json   — per-fold + aggregate metrics
    docs/cv_<model>.md        — report-ready markdown table

Usage:
    python scripts/cross_validate.py --model cnn --n-folds 5 --epochs 10
    python scripts/cross_validate.py --model ride --n-folds 5 --epochs 10 --device cuda

Budget note: 5 folds * 10 epochs * small model ~= 30 min on T4. Lower
--epochs for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Repo root on sys.path (this file lives in scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from torchvision import transforms as tv_transforms  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    KNOWN_CLASSES,
    WaferMapDataset,
    get_image_transforms,
    get_imagenet_normalize,
    load_dataset,
    preprocess_wafer_maps,
    seed_worker,
)
from src.models import (  # noqa: E402
    WaferCNN,
    build_ride_model,
    get_efficientnet_b0,
    get_resnet18,
    get_swin_tiny,
)
from src.training import train_model  # noqa: E402

logger = logging.getLogger("cross_validate")

SEED = 42
VALID_MODELS = ("cnn", "resnet", "efficientnet", "ride", "swin")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess(
    dataset_path: Path, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load + preprocess, honouring the .npz cache when present (matches train.py)."""
    cache_path = Path(dataset_path).parent / "LSWMD_cache.npz"
    maps_npy_path = cache_path.with_suffix(".maps.npy")

    if cache_path.exists():
        logger.info("Using pre-resized cache: %s", cache_path)
        cache = np.load(cache_path, allow_pickle=True)
        cached_labels_str = cache["labels"]
        if maps_npy_path.exists():
            cached_maps = np.load(maps_npy_path, mmap_mode="r")
        elif "maps" in cache.files:
            cached_maps = cache["maps"]
        else:
            raise RuntimeError(f"{cache_path} missing 'maps' key and no sidecar .npy")
        le = LabelEncoder().fit(np.array(KNOWN_CLASSES))
        labels = le.transform(cached_labels_str)
        wafer_maps = np.empty(len(cached_maps), dtype=object)
        for i in range(len(cached_maps)):
            wafer_maps[i] = cached_maps[i]
        return wafer_maps, labels, list(KNOWN_CLASSES)

    logger.info("Loading raw dataset (no cache)...")
    df = load_dataset(dataset_path)
    labeled_mask = df["failureClass"].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)
    le = LabelEncoder()
    df_clean["label_encoded"] = le.fit_transform(df_clean["failureClass"])

    wafer_raw = df_clean["waferMap"].values
    labels = df_clean["label_encoded"].values
    logger.info("Preprocessing %d maps...", len(wafer_raw))
    maps_processed = preprocess_wafer_maps([wafer_raw[i] for i in range(len(wafer_raw))])
    logger.info("Loaded: %d samples, %d classes", len(maps_processed), len(le.classes_))
    return maps_processed, labels, le.classes_.tolist()


def build_model(model_type: str, num_classes: int, device: str) -> Tuple[nn.Module, bool]:
    """Return (model, uses_imagenet_norm). uses_imagenet_norm=True for pretrained backbones."""
    if model_type == "cnn":
        return WaferCNN(num_classes=num_classes).to(device), False
    if model_type == "resnet":
        return get_resnet18(num_classes=num_classes).to(device), True
    if model_type == "efficientnet":
        return get_efficientnet_b0(num_classes=num_classes).to(device), True
    if model_type == "swin":
        return get_swin_tiny(num_classes=num_classes).to(device), False
    if model_type == "ride":
        return (
            build_ride_model(
                backbone_name="cnn",
                num_classes=num_classes,
                num_experts=3,
                reduction=4,
                device=device,
            ),
            False,
        )
    raise ValueError(f"Unknown model_type={model_type!r}; expected one of {VALID_MODELS}")


def compute_loss_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = Counter(y)
    total = len(y)
    return torch.tensor(
        [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# CV loop
# ---------------------------------------------------------------------------


def run_fold(
    fold_idx: int,
    n_folds: int,
    model_type: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    maps: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    batch_size: int,
    epochs: int,
    device: str,
    seed: int,
) -> Dict[str, float]:
    """Run a single fold: split train into train/val, train, evaluate on test fold."""
    # Carve a small stratified val slice out of the training indices so the
    # scheduler / early stopping inside train_model() still has a signal.
    inner_train_idx, inner_val_idx = train_test_split(
        train_idx,
        test_size=0.10,
        stratify=labels[train_idx],
        random_state=seed,
    )

    y_train = labels[inner_train_idx]
    y_val = labels[inner_val_idx]
    y_test = labels[test_idx]

    train_maps = maps[inner_train_idx]
    val_maps = maps[inner_val_idx]
    test_maps = maps[test_idx]

    model, uses_imagenet = build_model(model_type, len(class_names), device)

    base_aug = get_image_transforms()
    if uses_imagenet:
        imagenet_norm = get_imagenet_normalize()
        train_transform = tv_transforms.Compose([base_aug, imagenet_norm])
        eval_transform = imagenet_norm
    else:
        train_transform = base_aug
        eval_transform = None

    train_ds = WaferMapDataset(train_maps, y_train, transform=train_transform)
    val_ds = WaferMapDataset(val_maps, y_val, transform=eval_transform)
    test_ds = WaferMapDataset(test_maps, y_test, transform=eval_transform)

    g = torch.Generator().manual_seed(seed)
    loader_kw = dict(batch_size=batch_size, num_workers=0, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    loss_weights = compute_loss_weights(y_train, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    lr = 1e-3 if model_type in ("cnn", "ride") else 1e-4
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    logger.info(
        "  Fold %d/%d: train=%d val=%d test=%d (model=%s, lr=%g)",
        fold_idx + 1,
        n_folds,
        len(inner_train_idx),
        len(inner_val_idx),
        len(test_idx),
        model_type,
        lr,
    )

    t0 = time.time()
    model, _ = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        epochs=epochs,
        model_name=f"{model_type.upper()} (fold {fold_idx + 1}/{n_folds})",
        device=device,
    )
    train_time = time.time() - t0

    # Evaluate on held-out test fold
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, y in test_loader:
            inputs = inputs.to(device)
            out = model(inputs)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            targets.extend(y.numpy())

    acc = float(accuracy_score(targets, preds))
    mf1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    wf1 = float(f1_score(targets, preds, average="weighted", zero_division=0))

    logger.info(
        "    -> test_acc=%.4f test_macro_f1=%.4f test_weighted_f1=%.4f (%.1fs)",
        acc,
        mf1,
        wf1,
        train_time,
    )

    return {
        "fold": fold_idx + 1,
        "n_train": int(len(inner_train_idx)),
        "n_val": int(len(inner_val_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "train_time_sec": float(train_time),
    }


def cross_validate(
    model_type: str,
    maps: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    n_folds: int,
    batch_size: int,
    epochs: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("CV: model=%s folds=%d epochs=%d device=%s", model_type, n_folds, epochs, device)
    logger.info("=" * 70)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    per_fold: List[Dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(maps, labels)):
        r = run_fold(
            fold_idx=fold_idx,
            n_folds=n_folds,
            model_type=model_type,
            train_idx=train_idx,
            test_idx=test_idx,
            maps=maps,
            labels=labels,
            class_names=class_names,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        per_fold.append(r)

    def agg(key: str) -> Dict[str, float]:
        vals = np.array([f[key] for f in per_fold], dtype=float)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    aggregate = {
        "accuracy": agg("accuracy"),
        "macro_f1": agg("macro_f1"),
        "weighted_f1": agg("weighted_f1"),
    }

    return {
        "model": model_type,
        "n_folds": n_folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "class_names": class_names,
        "per_fold": per_fold,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_json(results: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)


def write_markdown(results: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = results["model"]
    n_folds = results["n_folds"]
    epochs = results["epochs"]
    per_fold = results["per_fold"]
    agg = results["aggregate"]

    lines: List[str] = []
    lines.append(f"# {n_folds}-Fold Stratified Cross-Validation — `{model}`")
    lines.append("")
    lines.append(
        f"- Folds: **{n_folds}**  |  Epochs/fold: **{epochs}**  |  "
        f"Batch size: **{results['batch_size']}**  |  Seed: **{results['seed']}**  |  "
        f"Device: **{results['device']}**"
    )
    lines.append("")
    lines.append("## Per-fold test metrics")
    lines.append("")
    lines.append("| Fold | Train | Val | Test | Accuracy | Macro F1 | Weighted F1 | Time (s) |")
    lines.append("|-----:|------:|----:|-----:|---------:|---------:|------------:|---------:|")
    for f in per_fold:
        lines.append(
            f"| {f['fold']} | {f['n_train']} | {f['n_val']} | {f['n_test']} | "
            f"{f['accuracy']:.4f} | {f['macro_f1']:.4f} | {f['weighted_f1']:.4f} | "
            f"{f['train_time_sec']:.1f} |"
        )
    lines.append("")
    lines.append("## Aggregate (mean ± std across folds)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|:-------|-----:|----:|----:|----:|")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        a = agg[metric]
        lines.append(
            f"| {metric} | {a['mean']:.4f} | {a['std']:.4f} | " f"{a['min']:.4f} | {a['max']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Headline:** macro F1 = **{agg['macro_f1']['mean']:.4f} "
        f"± {agg['macro_f1']['std']:.4f}** over {n_folds} folds."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified K-fold CV for wafer defect models")
    parser.add_argument("--model", choices=VALID_MODELS, default="cnn")
    parser.add_argument(
        "--n-folds",
        "--n-splits",
        dest="n_folds",
        type=int,
        default=5,
        help="Number of stratified folds (default 5)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per fold (default 10)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.warning("load_config(%s) failed (%s); using built-in defaults", args.config, exc)
        cfg = None

    device = (
        args.device
        or (getattr(cfg, "device", None) if cfg else None)
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.data_path is None:
        if cfg is not None:
            args.data_path = Path(cfg.data.dataset_path)
        else:
            args.data_path = REPO_ROOT / "data" / "LSWMD_new.pkl"
        if not args.data_path.is_absolute():
            args.data_path = REPO_ROOT / args.data_path

    set_seed(args.seed)
    logger.info("Device=%s seed=%d data=%s", device, args.seed, args.data_path)

    maps, labels, class_names = load_and_preprocess(args.data_path, seed=args.seed)

    results = cross_validate(
        model_type=args.model,
        maps=maps,
        labels=labels,
        class_names=class_names,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )

    json_path = args.results_dir / f"cv_{args.model}.json"
    md_path = args.docs_dir / f"cv_{args.model}.md"
    write_json(results, json_path)
    write_markdown(results, md_path)

    agg = results["aggregate"]
    logger.info("=" * 70)
    logger.info("CV SUMMARY — %s (%d folds, %d epochs/fold)", args.model, args.n_folds, args.epochs)
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        a = agg[metric]
        logger.info(
            "  %s: %.4f ± %.4f (min=%.4f max=%.4f)", metric, a["mean"], a["std"], a["min"], a["max"]
        )
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
