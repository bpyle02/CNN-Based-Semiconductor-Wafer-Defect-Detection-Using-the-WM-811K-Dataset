#!/usr/bin/env python3
"""Energy-based OOD detection analysis on trained wafer defect models.

Loads a trained checkpoint, splits the test set into in-distribution
(the 9 known classes) and synthetic OOD (pixel-flipped wafers, or
genuine 'unknown' samples if present in the raw pkl), fits an energy
threshold at FPR=5% on a held-out val slice, and reports AUROC,
FPR@95%TPR, and detection accuracy at the fitted threshold.

Writes results/ood_metrics.json.

Usage:
    python scripts/ood_analysis.py --checkpoint checkpoints/best_cnn.pth --model-type cnn
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_model(model_type: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a model from checkpoint."""
    from src.models import WaferCNN, get_resnet18, get_efficientnet_b0

    if model_type == "cnn":
        model = WaferCNN(num_classes=9)
    elif model_type == "resnet":
        model = get_resnet18(num_classes=9, pretrained=False, freeze_until=None)
    elif model_type == "efficientnet":
        model = get_efficientnet_b0(num_classes=9, pretrained=False, freeze_until=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def _corrupt_maps(maps, rng: np.random.Generator, flip_frac: float = 0.5):
    """Simulate OOD by flipping half of each wafer map's pixels at random.

    This destroys the spatial defect signature while preserving wafer
    geometry, yielding samples whose structure is unlike any known class.
    """
    corrupted = []
    for m in maps:
        arr = np.asarray(m).copy()
        flat = arr.reshape(-1)
        n_flip = max(1, int(flip_frac * flat.size))
        idx = rng.choice(flat.size, size=n_flip, replace=False)
        # Cycle values 0 -> 1 -> 2 -> 0 to scramble the map.
        flat[idx] = (flat[idx] + 1) % 3
        corrupted.append(arr.reshape(np.asarray(m).shape))
    return corrupted


def prepare_data(data_path: str = None, batch_size: int = 64, seed: int = 42):
    """Build in-dist val/test loaders and a synthetic-OOD test loader.

    Returns:
        (val_loader, id_test_loader, ood_test_loader)
    """
    from src.data.dataset import load_dataset, KNOWN_CLASSES
    from src.data.preprocessing import WaferMapDataset, get_image_transforms
    from sklearn.model_selection import train_test_split

    df = load_dataset(Path(data_path) if data_path else None)

    # Real "unknown" OOD pool (pre-filter, before we drop to known).
    unknown_df = df[~df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)

    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    labels = le.transform(df["failureClass"].values)
    maps = df["waferMap"].tolist()

    idx_trainval, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.15, stratify=labels, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15, stratify=labels[idx_trainval], random_state=seed
    )

    val_maps = [maps[i] for i in idx_val]
    val_labels = labels[idx_val]
    test_maps = [maps[i] for i in idx_test]
    test_labels = labels[idx_test]

    transform = get_image_transforms(augment=False)
    val_ds = WaferMapDataset(val_maps, val_labels, transform=transform)
    id_test_ds = WaferMapDataset(test_maps, test_labels, transform=transform)

    # Build OOD pool: prefer real unknowns if available, else synthesize.
    rng = np.random.default_rng(seed)
    if len(unknown_df) >= 32:
        ood_maps = unknown_df["waferMap"].tolist()
        ood_source = f"real-unknown (n={len(ood_maps)})"
    else:
        # Corrupt a copy of the in-dist test maps into synthetic OOD.
        ood_maps = _corrupt_maps(test_maps, rng=rng, flip_frac=0.5)
        ood_source = f"synthetic-pixel-flip (n={len(ood_maps)})"
    ood_labels = np.zeros(len(ood_maps), dtype=np.int64)  # dummy
    ood_ds = WaferMapDataset(ood_maps, ood_labels, transform=transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    id_test_loader = DataLoader(id_test_ds, batch_size=batch_size, shuffle=False)
    ood_test_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)

    return val_loader, id_test_loader, ood_test_loader, ood_source


@torch.no_grad()
def _collect_energies(model, loader, device: str, T: float = 1.0) -> np.ndarray:
    from src.inference.ood import energy_score

    all_e = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = model(x.to(device))
        all_e.append(energy_score(logits, T=T).cpu().numpy())
    return np.concatenate(all_e, axis=0) if all_e else np.array([])


def _auroc(id_energies: np.ndarray, ood_energies: np.ndarray) -> float:
    """AUROC where OOD (higher energy) is the positive class."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros_like(id_energies), np.ones_like(ood_energies)])
    s = np.concatenate([id_energies, ood_energies])
    return float(roc_auc_score(y, s))


def _fpr_at_tpr(id_energies: np.ndarray, ood_energies: np.ndarray, tpr: float = 0.95) -> float:
    """FPR on in-dist when the threshold catches `tpr` fraction of OOD."""
    thr = float(np.quantile(ood_energies, 1.0 - tpr))
    return float((id_energies > thr).mean())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Energy-based OOD detection analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", required=True, choices=["cnn", "resnet", "efficientnet"])
    parser.add_argument("--data-path", default=None, help="Path to LSWMD_new.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument(
        "--output", default="results/ood_metrics.json", help="Output JSON path"
    )
    args = parser.parse_args()

    from src.inference.ood import OODDetector

    logger.info(f"Loading {args.model_type} model from {args.checkpoint}")
    model = load_model(args.model_type, args.checkpoint, args.device)

    logger.info("Building in-dist val / test and OOD loaders...")
    val_loader, id_test_loader, ood_test_loader, ood_source = prepare_data(
        args.data_path, args.batch_size
    )
    logger.info(f"OOD source: {ood_source}")

    # Fit threshold at requested FPR on val set.
    detector = OODDetector(model, T=args.temperature, device=args.device)
    threshold = detector.fit(val_loader, target_fpr=args.target_fpr)
    logger.info(f"Fitted energy threshold @ FPR={args.target_fpr}: {threshold:.4f}")

    id_e = _collect_energies(model, id_test_loader, args.device, T=args.temperature)
    ood_e = _collect_energies(model, ood_test_loader, args.device, T=args.temperature)
    logger.info(
        f"Test energies: id mean={id_e.mean():.3f} std={id_e.std():.3f}; "
        f"ood mean={ood_e.mean():.3f} std={ood_e.std():.3f}"
    )

    # Metrics.
    auroc = _auroc(id_e, ood_e)
    fpr95 = _fpr_at_tpr(id_e, ood_e, tpr=0.95)

    # Detection accuracy at fitted threshold: fraction correctly flagged.
    n_id = len(id_e)
    n_ood = len(ood_e)
    tp = int((ood_e > threshold).sum())
    tn = int((id_e <= threshold).sum())
    det_acc = (tp + tn) / max(1, n_id + n_ood)
    empirical_fpr = float((id_e > threshold).mean())
    empirical_tpr = float((ood_e > threshold).mean())

    metrics = {
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "ood_source": ood_source,
        "temperature": args.temperature,
        "target_fpr": args.target_fpr,
        "fitted_threshold": threshold,
        "n_in_dist": n_id,
        "n_ood": n_ood,
        "auroc": auroc,
        "fpr_at_95_tpr": fpr95,
        "detection_accuracy": det_acc,
        "empirical_fpr": empirical_fpr,
        "empirical_tpr": empirical_tpr,
        "energy_stats": {
            "id_mean": float(id_e.mean()), "id_std": float(id_e.std()),
            "ood_mean": float(ood_e.mean()), "ood_std": float(ood_e.std()),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 54)
    logger.info("Energy-based OOD Detection Results")
    logger.info("=" * 54)
    logger.info(f"AUROC:              {auroc:.4f}")
    logger.info(f"FPR@95%TPR:         {fpr95:.4f}")
    logger.info(f"Detection accuracy: {det_acc:.4f}")
    logger.info(f"Wrote {out_path}")
    logger.info("=" * 54)

    return 0


if __name__ == "__main__":
    sys.exit(main())
