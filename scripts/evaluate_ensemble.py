#!/usr/bin/env python3
"""Evaluate single checkpoints and their ensemble on the held-out test split.

Discovers every ``checkpoints/<model>_best.pth`` on disk, loads each with its
matching architecture from ``train.py``'s ``build_model``, runs the same
deterministic test split the training loop used (seed=42, test=15%), and
reports per-model + ensemble metrics.

Three ensemble aggregations are evaluated:
  - ``averaging``: mean of softmax probabilities (the strong default)
  - ``voting``: majority vote of argmax predictions
  - ``weighted``: weights fit on the val split to maximize macro F1

Writes ``results/ensemble_metrics.json`` and a markdown summary at
``docs/ensemble_results.md``.

Usage:
    python scripts/evaluate_ensemble.py
    python scripts/evaluate_ensemble.py --models cnn resnet ride swin
    python scripts/evaluate_ensemble.py --batch-size 128 --device cuda
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
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config  # noqa: E402
from src.inference.calibration import TemperatureScaling  # noqa: E402
from src.inference.tta import predict_with_tta  # noqa: E402

logger = logging.getLogger(__name__)

DISPLAY = {
    "cnn": "Custom CNN",
    "cnn_fpn": "CNN+FPN",
    "resnet": "ResNet-18",
    "efficientnet": "EfficientNet-B0",
    "vit": "ViT-Tiny",
    "swin": "Swin-Tiny",
    "ride": "RIDE",
}


def _load_splits(config, data_path: Path, seed: int, batch_size: int, num_workers: int):
    """Reproduce train.py's deterministic split and return the test DataLoader."""
    import train as _train

    data = _train.load_and_preprocess_data(
        data_path,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        target_size=(config.data.target_size, config.data.target_size),
        seed=seed,
        synthetic=False,
    )
    from src.data.preprocessing import WaferMapDataset

    val_ds = WaferMapDataset(data["val_maps"], data["y_val"])
    test_ds = WaferMapDataset(data["test_maps"], data["y_test"])
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return (
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
        len(data["classes"]),
    )


def _discover(ckpt_dir: Path, requested: List[str] | None) -> List[Tuple[str, Path]]:
    found = []
    for ckpt in sorted(ckpt_dir.glob("*_best.pth")):
        name = ckpt.stem.replace("_best", "")
        if requested and name not in requested:
            continue
        found.append((name, ckpt))
    return found


def _build_and_load(
    model_name: str, ckpt: Path, config, num_classes: int, device: str
) -> torch.nn.Module:
    import train as _train

    model_cfg = getattr(config.model, model_name, None) or config.model
    model, _ = _train.build_model(model_name, model_cfg, num_classes, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw logits (pre-softmax) over the loader."""
    logits_chunks, labels_chunks = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).cpu().numpy()
        logits_chunks.append(logits)
        labels_chunks.append(labels.numpy())
    return np.concatenate(logits_chunks), np.concatenate(labels_chunks)


@torch.no_grad()
def _collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    use_tta: bool = False,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect softmax probabilities.

    Pipeline order when combined: raw logits -> temp-scale -> softmax -> TTA average.
    TTA is applied on the augmented inputs; each view's logits are temperature-scaled
    before softmax, then averaged.
    """
    probs_chunks, labels_chunks = [], []
    t = float(temperature) if temperature and temperature > 0 else 1.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if use_tta:
            # Wrap model to apply temperature scaling to each view's logits
            # before softmax; TTA averages the scaled softmax outputs.
            if t != 1.0:

                class _Scaled(torch.nn.Module):
                    def __init__(self, base, temp):
                        super().__init__()
                        self.base = base
                        self.temp = temp

                    def forward(self, x):
                        return self.base(x) / self.temp

                probs = predict_with_tta(_Scaled(model, t), imgs)
            else:
                probs = predict_with_tta(model, imgs)
            probs = probs.cpu().numpy()
        else:
            logits = model(imgs)
            if t != 1.0:
                logits = logits / t
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_chunks.append(probs)
        labels_chunks.append(labels.numpy())
    return np.concatenate(probs_chunks), np.concatenate(labels_chunks)


def _metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
    }


def _fit_weights(val_probs: List[np.ndarray], y_val: np.ndarray) -> np.ndarray:
    n = len(val_probs)
    stacked = np.stack(val_probs, axis=0)

    def neg_f1(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s <= 0:
            return 1.0
        w = w / s
        avg = (w[:, None, None] * stacked).sum(axis=0)
        return -f1_score(y_val, avg.argmax(axis=1), average="macro", zero_division=0)

    res = minimize(
        neg_f1,
        x0=np.ones(n) / n,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 200},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to include (default: all discovered).",
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "LSWMD_new.pkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument(
        "--tta", action="store_true", help="Apply 8-view test-time augmentation before ensembling."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit temperature scaling on the val split per model "
        "and apply to test logits before metrics.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    found = _discover(args.checkpoints_dir, args.models)
    if not found:
        logger.error("No *_best.pth checkpoints found in %s", args.checkpoints_dir)
        return 1
    logger.info("Evaluating %d models: %s", len(found), [n for n, _ in found])

    val_loader, test_loader, num_classes = _load_splits(
        config, args.data_path, args.seed, args.batch_size, args.num_workers
    )

    per_model = {}
    val_probs_all, test_probs_all, y_val_ref, y_test_ref = [], [], None, None
    temperatures: Dict[str, float] = {}
    for name, ckpt in found:
        logger.info("Loading %s from %s", name, ckpt)
        model = _build_and_load(name, ckpt, config, num_classes, args.device)

        temperature = 1.0
        if args.calibrate:
            # Fit T on raw val logits, then use it downstream.
            val_logits, y_val = _collect_logits(model, val_loader, args.device)
            ts = TemperatureScaling()
            temperature = ts.fit(val_logits, y_val)
            temperatures[name] = temperature
            logger.info("  %-12s  calibrated T=%.4f", DISPLAY.get(name, name), temperature)
            val_probs = torch.softmax(torch.from_numpy(val_logits) / temperature, dim=1).numpy()
        else:
            val_probs, y_val = _collect_probs(
                model,
                val_loader,
                args.device,
                use_tta=args.tta,
                temperature=1.0,
            )

        test_probs, y_test = _collect_probs(
            model,
            test_loader,
            args.device,
            use_tta=args.tta,
            temperature=temperature,
        )
        per_model[name] = _metrics(test_probs, y_test)
        logger.info(
            "  %-12s  acc=%.4f  macroF1=%.4f",
            DISPLAY.get(name, name),
            per_model[name]["accuracy"],
            per_model[name]["macro_f1"],
        )
        val_probs_all.append(val_probs)
        test_probs_all.append(test_probs)
        y_val_ref = y_val if y_val_ref is None else y_val_ref
        y_test_ref = y_test if y_test_ref is None else y_test_ref
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    avg_probs = np.stack(test_probs_all, axis=0).mean(axis=0)
    vote_preds = np.stack([p.argmax(axis=1) for p in test_probs_all], axis=0)
    vote_pred = np.array(
        [np.bincount(vote_preds[:, i]).argmax() for i in range(vote_preds.shape[1])]
    )
    weights = _fit_weights(val_probs_all, y_val_ref)
    weighted_probs = (weights[:, None, None] * np.stack(test_probs_all, axis=0)).sum(axis=0)

    ensemble = {
        "averaging": _metrics(avg_probs, y_test_ref),
        "voting": {
            "accuracy": float(accuracy_score(y_test_ref, vote_pred)),
            "macro_f1": float(f1_score(y_test_ref, vote_pred, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_test_ref, vote_pred, average="weighted", zero_division=0)
            ),
        },
        "weighted": {
            **_metrics(weighted_probs, y_test_ref),
            "weights": {n: float(w) for (n, _), w in zip(found, weights)},
        },
    }

    results = {
        "per_model": per_model,
        "ensemble": ensemble,
        "num_models": len(found),
        "model_names": [n for n, _ in found],
        "tta": bool(args.tta),
        "calibrate": bool(args.calibrate),
        "temperatures": temperatures,
    }
    out_json = REPO_ROOT / "results" / "ensemble_metrics.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_json)

    lines = [
        "# Ensemble evaluation\n",
        "## Per-model (test split)\n",
        "| Model | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---|---|---|",
    ]
    for n, m in per_model.items():
        lines.append(
            f"| {DISPLAY.get(n, n)} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |"
        )
    lines += [
        "",
        "## Ensemble aggregations\n",
        "| Method | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---|---|---|",
    ]
    for method, m in ensemble.items():
        lines.append(
            f"| {method} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |"
        )
    lines += ["", f"**Learned weights:** `{ensemble['weighted']['weights']}`", ""]
    out_md = REPO_ROOT / "docs" / "ensemble_results.md"
    out_md.write_text("\n".join(lines))
    logger.info("Wrote %s", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
