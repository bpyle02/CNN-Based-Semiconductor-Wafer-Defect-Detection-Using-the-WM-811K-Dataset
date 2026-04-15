#!/usr/bin/env python3
"""Regenerate paper-ready figures at 300 DPI with a consistent style.

Scans ``--results-dir`` for artefacts written by other scripts and
produces publication-grade figures in ``--output-dir``. Missing inputs
skip gracefully with a warning — each figure is independent.

Figures:
  fig_confusion_matrices  one panel per model, uniform colormap
  fig_training_curves     overlaid train/val loss + acc across models
  fig_macro_f1_comparison bar chart with 95% CI (bootstrap or seed stderr)
  fig_rare_class_study    grouped per-condition per-class F1
  fig_ensemble_comparison single-model vs ensemble scatter with diagonal
  table_results.tex       LaTeX tabular ready to \\input into the report

Class ordering is alphabetical with ``none`` last. Fonts default to serif
for paper consistency.

Usage:
    python scripts/paper_figures.py
    python scripts/paper_figures.py --results-dir results/ \\
        --output-dir results/paper/ --format pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # noqa: F401  # used opportunistically below
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # noqa: F401

logger = logging.getLogger("paper_figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
CLASS_ORDER: Tuple[str, ...] = (
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
)

MODEL_DISPLAY: Dict[str, str] = {
    "cnn": "Custom CNN",
    "cnn_fpn": "CNN+FPN",
    "resnet": "ResNet-18",
    "efficientnet": "EfficientNet-B0",
    "vit": "ViT-Small",
    "swin": "Swin-Tiny",
    "ride": "RIDE",
    "student": "Distilled Student",
    "student_distilled": "Distilled Student",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def _display(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)


def _order_classes(names: Sequence[str]) -> List[str]:
    """Return ``names`` in the canonical paper order, keeping unknowns last."""
    idx = {n: i for i, n in enumerate(CLASS_ORDER)}
    return sorted(names, key=lambda c: (idx.get(c, len(CLASS_ORDER)), c))


def _save(fig: plt.Figure, out: Path, fmt: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(f".{fmt}"), format=fmt)
    plt.close(fig)
    logger.info("  wrote %s", out.with_suffix(f".{fmt}"))


# ---------------------------------------------------------------------------
# Loaders (tolerant; return None on missing)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return None


def _load_metrics(results_dir: Path) -> Optional[dict]:
    return _load_json(results_dir / "metrics.json")


def _load_ensemble(results_dir: Path) -> Optional[dict]:
    return _load_json(results_dir / "ensemble_metrics.json")


def _load_bootstrap(results_dir: Path) -> Optional[dict]:
    return _load_json(results_dir / "bootstrap_ci.json")


def _load_history(results_dir: Path, model_name: str) -> Optional[dict]:
    """Look for per-model training history JSON in common locations."""
    candidates = [
        results_dir / f"history_{model_name}.json",
        results_dir / f"{model_name}_history.json",
        results_dir / "histories" / f"{model_name}.json",
    ]
    for path in candidates:
        h = _load_json(path)
        if h is not None:
            return h
    return None


def _load_distill(results_dir: Path) -> Optional[dict]:
    return _load_json(results_dir / "distill_metrics.json")


def _load_rare_class(results_dir: Path) -> Optional[dict]:
    # Prefer an aggregated file if present; otherwise merge per-condition JSONs.
    agg = _load_json(results_dir / "rare_class_study.json")
    if agg is not None:
        return agg
    rc_dir = results_dir / "rare_class"
    if not rc_dir.exists():
        return None
    merged: Dict[str, Dict[str, List[float]]] = {}
    for p in sorted(rc_dir.glob("*.json")):
        # Expect filenames like "<condition>_seed<N>.json".
        stem = p.stem
        if "_seed" not in stem:
            continue
        condition = stem.split("_seed")[0]
        doc = _load_json(p) or {}
        per_class = doc.get("per_class_f1") or doc.get("per_class") or {}
        bucket = merged.setdefault(condition, {})
        for cls, v in per_class.items():
            val = v.get("f1", v) if isinstance(v, dict) else v
            bucket.setdefault(cls, []).append(float(val))
    if not merged:
        return None
    # Collapse seed replicates to mean.
    return {cond: {c: float(np.mean(vs)) for c, vs in d.items()} for cond, d in merged.items()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_confusion_matrices(results_dir: Path, out_dir: Path, fmt: str) -> None:
    """One confusion matrix panel per model using a shared colormap.

    Expects per-model confusion matrices either embedded in ``metrics.json``
    under ``<model>.confusion_matrix`` or stored in
    ``results/confusion_<model>.json``.
    """
    logger.info("figure: confusion matrices")
    metrics = _load_metrics(results_dir) or {}

    model_mats: Dict[str, np.ndarray] = {}
    for name, entry in metrics.items():
        if not isinstance(entry, dict):
            continue
        cm = entry.get("confusion_matrix")
        if cm is not None:
            model_mats[name] = np.asarray(cm, dtype=float)

    for sidecar in sorted(results_dir.glob("confusion_*.json")):
        name = sidecar.stem.replace("confusion_", "")
        doc = _load_json(sidecar) or {}
        cm = doc.get("confusion_matrix") or doc.get("matrix")
        if cm is not None:
            model_mats[name] = np.asarray(cm, dtype=float)

    if not model_mats:
        logger.warning("  no confusion matrices found — skipping")
        return

    class_names = _order_classes(CLASS_ORDER)
    n = len(model_mats)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False)
    vmax = max(m.max() for m in model_mats.values() if m.size) or 1.0

    for ax, (name, cm) in zip(axes.flat, sorted(model_mats.items())):
        # Row-normalize for readability.
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        norm = cm / row_sum
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(_display(name))
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = norm[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "black")

    for ax in axes.flat[len(model_mats):]:
        ax.set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Row-normalized count")
    _save(fig, out_dir / "fig_confusion_matrices", fmt)
    # vmax variable no longer used after per-row normalization; kept for API parity.
    _ = vmax


def figure_training_curves(results_dir: Path, out_dir: Path, fmt: str) -> None:
    """Overlaid train/val loss and accuracy across all available models."""
    logger.info("figure: training curves")
    histories: Dict[str, dict] = {}
    metrics = _load_metrics(results_dir) or {}
    for name in sorted(set(metrics.keys()) | {"cnn", "resnet", "efficientnet", "swin", "ride", "vit"}):
        h = _load_history(results_dir, name)
        if h:
            histories[name] = h
    # Also pick up the distilled student if present.
    distill = _load_distill(results_dir)
    if distill and distill.get("history"):
        histories["student_distilled"] = distill["history"]

    if not histories:
        logger.warning("  no per-model history JSONs found — skipping")
        return

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for name, h in sorted(histories.items()):
        label = _display(name)
        if "train_loss" in h and h["train_loss"]:
            epochs = range(1, len(h["train_loss"]) + 1)
            ax_loss.plot(epochs, h["train_loss"], label=f"{label} (train)", linestyle="--", alpha=0.7)
        if "val_loss" in h and h["val_loss"]:
            epochs = range(1, len(h["val_loss"]) + 1)
            ax_loss.plot(epochs, h["val_loss"], label=f"{label} (val)", linestyle="-")
        acc_key = "val_acc" if "val_acc" in h else "val_accuracy"
        if acc_key in h and h[acc_key]:
            epochs = range(1, len(h[acc_key]) + 1)
            ax_acc.plot(epochs, h[acc_key], label=label)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training / validation loss")
    ax_loss.legend(loc="best", ncol=1)

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.legend(loc="best")

    _save(fig, out_dir / "fig_training_curves", fmt)


def figure_macro_f1_comparison(results_dir: Path, out_dir: Path, fmt: str) -> None:
    """Bar chart of macro F1 per model, with error bars (bootstrap CI or seed stderr)."""
    logger.info("figure: macro F1 comparison")
    metrics = _load_metrics(results_dir) or {}
    if not metrics:
        logger.warning("  no metrics.json — skipping")
        return

    models = sorted(
        [k for k, v in metrics.items() if isinstance(v, dict) and "macro_f1" in v]
    )
    if not models:
        logger.warning("  metrics.json has no macro_f1 entries — skipping")
        return

    means = np.array([metrics[m]["macro_f1"] for m in models], dtype=float)

    lower = np.full_like(means, np.nan)
    upper = np.full_like(means, np.nan)
    err_source = "none"

    boot = _load_bootstrap(results_dir)
    if boot:
        err_source = "bootstrap"
        # Best effort: accept either {model: {macro_f1: {lo, hi}}} or
        # {model: {macro_f1_ci: [lo, hi]}} or flat structures.
        for i, m in enumerate(models):
            entry = boot.get(m) or boot.get(_display(m))
            if not entry:
                continue
            ci = entry.get("macro_f1_ci") or entry.get("macro_f1", {}).get("ci")
            if isinstance(entry.get("macro_f1"), dict):
                ci = ci or (entry["macro_f1"].get("lo"), entry["macro_f1"].get("hi"))
            if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
                lower[i] = float(ci[0])
                upper[i] = float(ci[1])
    else:
        # Fall back to seed stderr when per-model `macro_f1_seeds` is present.
        have_any = False
        for i, m in enumerate(models):
            seeds = metrics[m].get("macro_f1_seeds")
            if seeds and len(seeds) > 1:
                stderr = float(np.std(seeds, ddof=1) / np.sqrt(len(seeds)))
                lower[i] = means[i] - 1.96 * stderr
                upper[i] = means[i] + 1.96 * stderr
                have_any = True
        err_source = "seed_stderr" if have_any else "none"

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    ax.bar(x, means, color="#4c72b0", edgecolor="black", linewidth=0.5)
    if np.any(~np.isnan(lower)):
        yerr = np.vstack([means - np.where(np.isnan(lower), means, lower),
                          np.where(np.isnan(upper), means, upper) - means])
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_display(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("Macro F1 (test)")
    caption = {
        "bootstrap": "error bars: 95% bootstrap CI",
        "seed_stderr": "error bars: 95% CI from seed stderr",
        "none": "no CI data available",
    }[err_source]
    ax.set_title(f"Macro F1 comparison ({caption})")
    ax.set_ylim(0, 1.0)
    _save(fig, out_dir / "fig_macro_f1_comparison", fmt)


def figure_rare_class_study(results_dir: Path, out_dir: Path, fmt: str) -> None:
    """Grouped bar chart: per-condition per-class F1."""
    logger.info("figure: rare-class study")
    rc = _load_rare_class(results_dir)
    if not rc:
        logger.warning("  no rare_class study data — skipping")
        return

    conditions = sorted(rc.keys())
    all_classes = set()
    for d in rc.values():
        all_classes.update(d.keys())
    classes = _order_classes(list(all_classes))

    data = np.full((len(conditions), len(classes)), np.nan)
    for i, cond in enumerate(conditions):
        for j, cls in enumerate(classes):
            if cls in rc[cond]:
                data[i, j] = float(rc[cond][cls])

    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 2, 4))
    width = 0.8 / max(len(conditions), 1)
    x = np.arange(len(classes))
    for i, cond in enumerate(conditions):
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(x + offset, np.nan_to_num(data[i], nan=0.0), width, label=cond)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Per-class F1")
    ax.set_title("Rare-class study: per-condition per-class F1")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Condition", loc="best")
    _save(fig, out_dir / "fig_rare_class_study", fmt)


def figure_ensemble_comparison(results_dir: Path, out_dir: Path, fmt: str) -> None:
    """Scatter: single-model macro F1 vs ensemble macro F1 with diagonal."""
    logger.info("figure: ensemble comparison")
    ens = _load_ensemble(results_dir)
    metrics = _load_metrics(results_dir) or {}
    if not ens:
        logger.warning("  no ensemble_metrics.json — skipping")
        return

    per_model = ens.get("per_model") or {}
    # Prefer the ensemble's own per-model numbers (same test split);
    # fall back to metrics.json if needed.
    points: List[Tuple[str, float, float]] = []
    ens_avg = (ens.get("ensemble") or {}).get("averaging", {}).get("macro_f1")
    if ens_avg is None:
        logger.warning("  ensemble_metrics.json has no averaging macro_f1 — skipping")
        return

    for name, m in per_model.items():
        f1 = m.get("macro_f1")
        if f1 is None and name in metrics and isinstance(metrics[name], dict):
            f1 = metrics[name].get("macro_f1")
        if f1 is not None:
            points.append((name, float(f1), float(ens_avg)))

    if not points:
        logger.warning("  no per-model macro_f1 values for scatter — skipping")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    ax.scatter(xs, ys, s=40, color="#4c72b0", edgecolor="black", zorder=3)
    for name, x, y in points:
        ax.annotate(_display(name), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    lo = float(min(xs.min(), ys.min()) - 0.05)
    hi = float(max(xs.max(), ys.max()) + 0.05)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Single-model macro F1")
    ax.set_ylabel("Ensemble (averaging) macro F1")
    ax.set_title("Single model vs. ensemble")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "fig_ensemble_comparison", fmt)


def table_results(results_dir: Path, out_dir: Path) -> None:
    """LaTeX tabular of per-model + ensemble test metrics."""
    logger.info("table: results tex")
    metrics = _load_metrics(results_dir) or {}
    ens = _load_ensemble(results_dir) or {}
    distill = _load_distill(results_dir) or {}

    rows: List[Tuple[str, float, float, float]] = []
    for name, entry in sorted(metrics.items()):
        if not isinstance(entry, dict) or "accuracy" not in entry:
            continue
        rows.append((
            _display(name),
            float(entry.get("accuracy", float("nan"))),
            float(entry.get("macro_f1", float("nan"))),
            float(entry.get("weighted_f1", float("nan"))),
        ))

    ensemble_block: List[Tuple[str, float, float, float]] = []
    for method, m in (ens.get("ensemble") or {}).items():
        ensemble_block.append((
            f"Ensemble ({method})",
            float(m.get("accuracy", float("nan"))),
            float(m.get("macro_f1", float("nan"))),
            float(m.get("weighted_f1", float("nan"))),
        ))

    student_row: Optional[Tuple[str, float, float, float]] = None
    if distill and "test" in distill:
        t = distill["test"]
        student_row = (
            "Distilled Student",
            float(t.get("accuracy", float("nan"))),
            float(t.get("macro_f1", float("nan"))),
            float(t.get("weighted_f1", float("nan"))),
        )

    if not rows and not ensemble_block and student_row is None:
        logger.warning("  no metrics available for LaTeX table — skipping")
        return

    out = out_dir / "table_results.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/paper_figures.py — do not edit by hand.",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Accuracy & Macro F1 & Weighted F1 \\\\",
        "\\midrule",
    ]
    for name, acc, mf1, wf1 in rows:
        lines.append(f"{name} & {acc:.4f} & {mf1:.4f} & {wf1:.4f} \\\\")
    if student_row is not None:
        lines.append("\\midrule")
        name, acc, mf1, wf1 = student_row
        lines.append(f"{name} & {acc:.4f} & {mf1:.4f} & {wf1:.4f} \\\\")
    if ensemble_block:
        lines.append("\\midrule")
        for name, acc, mf1, wf1 in ensemble_block:
            lines.append(f"{name} & {acc:.4f} & {mf1:.4f} & {wf1:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))
    logger.info("  wrote %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <results-dir>/paper/.")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _apply_style()

    out_dir = args.output_dir or (args.results_dir / "paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("results dir: %s", args.results_dir)
    logger.info("output  dir: %s", out_dir)

    figure_confusion_matrices(args.results_dir, out_dir, args.format)
    figure_training_curves(args.results_dir, out_dir, args.format)
    figure_macro_f1_comparison(args.results_dir, out_dir, args.format)
    figure_rare_class_study(args.results_dir, out_dir, args.format)
    figure_ensemble_comparison(args.results_dir, out_dir, args.format)
    table_results(args.results_dir, out_dir)

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
