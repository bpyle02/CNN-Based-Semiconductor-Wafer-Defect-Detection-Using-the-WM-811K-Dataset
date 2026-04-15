#!/usr/bin/env python3
"""Grad-CAM overlays on the highest-confidence *mispredicted* test samples per class.

For every known class ``c``, finds the top-N test samples where the true label
is ``c`` but the model predicted ``y != c`` with the highest confidence
(softmax probability of the wrong predicted class). These are the model's
"most confidently wrong" failures -- the most instructive error modes.

Produces:
  - ``results/gradcam_errors_<model>.png``: a (num_classes x N) grid with the
    original greyscale wafer map and a Grad-CAM overlay (jet, alpha=0.5).
    Each cell caption: ``True: X  Pred: Y  (conf: 0.87)``.
  - ``results/gradcam_errors_<model>.json``: metadata listing the selected
    test indices, true labels, predicted labels, and confidences.

Reuses the deterministic ``seed=42`` test split from ``train.py`` via
``scripts/evaluate_ensemble.py:_load_splits`` so the visualised samples
correspond exactly to the ones that produced reported test metrics.

Auto-picks the best checkpoint by ``best_val_macro_f1`` from
``results/metrics.json`` when ``--checkpoint`` is not provided.

Usage:
    python scripts/gradcam_errors.py
    python scripts/gradcam_errors.py --checkpoint checkpoints/best_cnn.pth
    python scripts/gradcam_errors.py --model resnet --per-class 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.inference.gradcam import GradCAM  # noqa: E402
from scripts.evaluate_ensemble import _load_splits, _build_and_load  # noqa: E402

logger = logging.getLogger(__name__)

# Canonical model names recognised by train.build_model.
_KNOWN_MODELS = ("cnn_fpn", "cnn", "resnet", "efficientnet", "vit", "swin", "ride")


def _pick_best_checkpoint(ckpt_dir: Path, metrics_path: Path) -> Tuple[Path, str]:
    """Pick the single-model checkpoint with the highest ``best_val_macro_f1``.

    Falls back to highest test ``macro_f1`` if ``best_val_macro_f1`` is
    missing, and finally to the first discovered checkpoint.
    """
    checkpoints = {p.stem.replace("best_", ""): p for p in ckpt_dir.glob("best_*.pth")}
    if not checkpoints:
        raise FileNotFoundError(f"No best_*.pth checkpoints in {ckpt_dir}")

    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {}
        scored = []
        for name, ckpt in checkpoints.items():
            m = metrics.get(name)
            if not isinstance(m, dict):
                continue
            score = m.get("best_val_macro_f1")
            if score is None:
                score = m.get("macro_f1")
            if score is not None:
                scored.append((float(score), name, ckpt))
        if scored:
            scored.sort(reverse=True)
            _, name, ckpt = scored[0]
            logger.info("Auto-picked %s (val macro_f1=%.4f) from %s",
                        name, scored[0][0], ckpt)
            return ckpt, name

    # Fallback: first checkpoint alphabetically.
    name, ckpt = sorted(checkpoints.items())[0]
    logger.warning("No metrics.json scoring available; falling back to %s", ckpt)
    return ckpt, name


def _infer_model_name(checkpoint: Path) -> str:
    """Infer the architecture name from a ``best_<model>.pth`` filename."""
    stem = checkpoint.stem
    match = re.match(r"best_(.+)$", stem)
    candidate = match.group(1) if match else stem
    # Prefer longest matching known prefix (e.g. 'cnn_fpn' over 'cnn').
    for known in sorted(_KNOWN_MODELS, key=len, reverse=True):
        if candidate == known or candidate.startswith(known + "_"):
            return known
    raise ValueError(
        f"Could not infer model architecture from {checkpoint.name}. "
        f"Pass --model explicitly (one of: {_KNOWN_MODELS})."
    )


def _find_target_layer(model: nn.Module) -> nn.Module:
    """Return the last ``nn.Conv2d`` layer in ``model`` for Grad-CAM.

    Raises if the architecture has no Conv2d (e.g. pure ViT). For RIDE
    ensembles we return the last Conv2d inside any expert backbone, which
    still yields a spatially-meaningful CAM for the first forward path.
    """
    last_conv: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(
            "No nn.Conv2d layer found in model. Grad-CAM requires a "
            "convolutional target layer; transformer-only backbones "
            "(ViT without conv patch embedding) are unsupported."
        )
    return last_conv


@torch.no_grad()
def _collect_preds(model: nn.Module, loader: DataLoader, device: str
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, confidences-of-predicted-class, true labels)."""
    preds_all, confs_all, y_all = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        confs = probs[np.arange(probs.shape[0]), preds]
        preds_all.append(preds)
        confs_all.append(confs)
        y_all.append(labels.numpy())
    return (
        np.concatenate(preds_all),
        np.concatenate(confs_all),
        np.concatenate(y_all),
    )


def _select_errors(y_true: np.ndarray, y_pred: np.ndarray, confs: np.ndarray,
                   per_class: int, num_classes: int) -> List[List[int]]:
    """Per class, return indices of the top-``per_class`` highest-confidence
    mispredictions. Missing classes yield an empty list."""
    selected: List[List[int]] = []
    for cls in range(num_classes):
        mask = (y_true == cls) & (y_pred != cls)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            selected.append([])
            continue
        order = np.argsort(-confs[idxs])[:per_class]
        selected.append(idxs[order].tolist())
    return selected


def _test_dataset_from_loader(loader: DataLoader):
    return loader.dataset


def _get_raw_image(dataset, idx: int) -> np.ndarray:
    """Return an ``[H, W]`` float32 greyscale map in [0, 1] from the dataset."""
    tensor, _ = dataset[idx]
    # dataset returns (3, H, W) after stacking; channels are identical.
    arr = tensor[0].detach().cpu().numpy().astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def _get_input_tensor(dataset, idx: int) -> torch.Tensor:
    tensor, _ = dataset[idx]
    return tensor.unsqueeze(0)


def _render_grid(
    out_png: Path,
    grey_images: List[List[np.ndarray]],
    cams: List[List[np.ndarray]],
    captions: List[List[str]],
    class_names: List[str],
    per_class: int,
) -> None:
    """Render the (num_classes x per_class) overlay grid to ``out_png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(class_names)
    fig, axes = plt.subplots(
        num_classes, per_class,
        figsize=(2.2 * per_class, 2.4 * num_classes),
        squeeze=False,
    )
    for row in range(num_classes):
        for col in range(per_class):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(class_names[row], fontsize=9, rotation=0,
                              labelpad=36, ha="right", va="center")
            if row < len(grey_images) and col < len(grey_images[row]) \
                    and grey_images[row][col] is not None:
                ax.imshow(grey_images[row][col], cmap="gray",
                          vmin=0.0, vmax=1.0)
                ax.imshow(cams[row][col], cmap="jet", alpha=0.5,
                          vmin=0.0, vmax=1.0)
                ax.set_title(captions[row][col], fontsize=7)
            else:
                ax.text(0.5, 0.5, "(no errors)", ha="center", va="center",
                        fontsize=8, transform=ax.transAxes, color="gray")
                ax.set_facecolor("#f3f3f3")
    fig.suptitle("High-confidence misclassifications  (Grad-CAM, alpha=0.5)",
                 fontsize=11)
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.97))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a best_<model>.pth checkpoint. "
                             "Defaults to best-by-val-macro_f1 from results/metrics.json.")
    parser.add_argument("--model", default=None,
                        help="Architecture name (cnn, cnn_fpn, resnet, efficientnet, "
                             "vit, swin, ride). Inferred from checkpoint filename if omitted.")
    parser.add_argument("--per-class", type=int, default=5,
                        help="Top-N highest-confidence errors per true class.")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=("cuda", "cpu"))
    parser.add_argument("--data-path", type=Path,
                        default=REPO_ROOT / "data" / "LSWMD_new.pkl")
    parser.add_argument("--checkpoints-dir", type=Path,
                        default=REPO_ROOT / "checkpoints")
    parser.add_argument("--metrics-path", type=Path,
                        default=REPO_ROOT / "results" / "metrics.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (keep small to fit a T4).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve checkpoint + model name.
    if args.checkpoint is None:
        ckpt_path, inferred_name = _pick_best_checkpoint(
            args.checkpoints_dir, args.metrics_path
        )
    else:
        ckpt_path = args.checkpoint
        if not ckpt_path.exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            return 1
        inferred_name = _infer_model_name(ckpt_path)

    model_name = args.model or inferred_name
    if model_name not in _KNOWN_MODELS:
        logger.error("Unknown --model %s (expected one of %s)", model_name, _KNOWN_MODELS)
        return 1
    logger.info("Using model=%s from %s", model_name, ckpt_path)

    # Build test split.
    config = load_config(args.config)
    _, test_loader, num_classes = _load_splits(
        config, args.data_path, args.seed, args.batch_size, args.num_workers
    )
    try:
        from src.data.preprocessing import KNOWN_CLASSES as _KC
        class_names = list(_KC)
    except Exception:
        from src.data.dataset import KNOWN_CLASSES as _KC
        class_names = list(_KC)
    if len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    # Build and load model.
    model = _build_and_load(model_name, ckpt_path, config, num_classes, args.device)

    # Forward pass over the test set.
    logger.info("Collecting predictions over %d test batches...", len(test_loader))
    preds, confs, y_true = _collect_preds(model, test_loader, args.device)
    err_mask = preds != y_true
    logger.info("Test size=%d, errors=%d (%.2f%%)",
                len(y_true), int(err_mask.sum()),
                100.0 * float(err_mask.mean()))

    selected = _select_errors(y_true, preds, confs, args.per_class, num_classes)

    # Grad-CAM on the selected samples.
    target_layer = _find_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    test_ds = _test_dataset_from_loader(test_loader)

    grey_grid: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    cam_grid: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    caption_grid: List[List[str]] = [[] for _ in range(num_classes)]
    metadata = {
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "per_class": args.per_class,
        "num_classes": num_classes,
        "class_names": class_names,
        "seed": args.seed,
        "test_size": int(len(y_true)),
        "errors": int(err_mask.sum()),
        "per_class_errors": [],
    }

    try:
        for cls, idxs in enumerate(selected):
            per_class_entries = []
            for idx in idxs:
                img_tensor = _get_input_tensor(test_ds, idx).to(args.device)
                # Confidence is of the predicted class, not the true class.
                pred = int(preds[idx])
                conf = float(confs[idx])
                # Zero grads per sample -- backward() accumulates.
                model.zero_grad(set_to_none=True)
                cam, _ = gradcam.generate(
                    img_tensor, target_class=pred, device=args.device
                )
                grey = _get_raw_image(test_ds, idx)
                cam = np.asarray(cam, dtype=np.float32)
                if cam.shape != grey.shape:
                    # Defensive: GradCAM.generate resizes to input, but guard
                    # against off-by-one if the wafer map dims differ.
                    from skimage.transform import resize as _resize
                    cam = _resize(cam, grey.shape, anti_aliasing=True,
                                  preserve_range=True).astype(np.float32)
                grey_grid[cls].append(grey)
                cam_grid[cls].append(cam)
                caption_grid[cls].append(
                    f"True: {class_names[cls]}  "
                    f"Pred: {class_names[pred]}  (conf: {conf:.2f})"
                )
                per_class_entries.append({
                    "test_index": int(idx),
                    "true_label": class_names[cls],
                    "true_label_id": int(cls),
                    "pred_label": class_names[pred],
                    "pred_label_id": int(pred),
                    "confidence": conf,
                })
            # Pad the grid row up to per_class width.
            while len(grey_grid[cls]) < args.per_class:
                grey_grid[cls].append(None)  # type: ignore[arg-type]
                cam_grid[cls].append(None)   # type: ignore[arg-type]
                caption_grid[cls].append("")
            metadata["per_class_errors"].append({
                "class_id": cls,
                "class_name": class_names[cls],
                "selected": per_class_entries,
            })
    finally:
        gradcam.remove_hooks()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / f"gradcam_errors_{model_name}.png"
    out_json = args.output_dir / f"gradcam_errors_{model_name}.json"

    _render_grid(out_png, grey_grid, cam_grid, caption_grid,
                 class_names, args.per_class)
    out_json.write_text(json.dumps(metadata, indent=2))

    logger.info("Wrote %s", out_png)
    logger.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
