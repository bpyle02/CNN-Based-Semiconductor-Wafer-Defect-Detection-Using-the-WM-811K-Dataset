#!/usr/bin/env python3
"""Rare-class study: compare rebalancing interventions on WM-811K.

Runs the custom CNN under five conditions x three seeds = 15 runs total,
then writes docs/rare_class_study.md with a per-condition results table,
per-class recall on the rare classes (Near-full, Scratch, Donut, Random),
and head-to-head deltas vs the unintervened baseline.

Conditions (each configured via configs/rare_class/*.yaml overlays):
    A_baseline             unweighted CE (current repo default)
    B_focal                FocalLoss (gamma=2.0), unweighted
    C_drw                  weighted CE with deferred re-weighting at epoch 10
    D_balanced_sampling    ClassBalancedSampler, unweighted CE
    E_synthetic            --synthetic flag (rule-based defect simulation)

Usage (locally or in Colab):
    python scripts/run_rare_class_study.py --epochs 20 --batch-size 128 \
        --seeds 0,1,2 --conditions all --device cuda

    python scripts/run_rare_class_study.py --conditions A_baseline,C_drw \
        --seeds 0 --epochs 5   # quick smoke test

Each run's full metrics.json is archived to results/rare_class/<condition>_seed<N>.json
so a mid-study crash doesn't lose completed runs.

Resumable: reruns skip any (condition, seed) whose archive file already
exists. Delete the archive to force a rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITION_DIR = REPO_ROOT / "configs" / "rare_class"
ARCHIVE_DIR = REPO_ROOT / "results" / "rare_class"
METRICS_JSON = REPO_ROOT / "results" / "metrics.json"

ALL_CONDITIONS = [
    "A_baseline",
    "B_focal",
    "C_drw",
    "D_balanced_sampling",
    "E_synthetic",
]

RARE_CLASSES = ["Near-full", "Scratch", "Donut", "Random"]


def _archive_path(condition: str, seed: int) -> Path:
    return ARCHIVE_DIR / f"{condition}_seed{seed}.json"


def _backup_metrics_json() -> Path | None:
    if METRICS_JSON.exists():
        backup = METRICS_JSON.with_suffix(".json.study_backup")
        shutil.copy2(METRICS_JSON, backup)
        return backup
    return None


def _restore_metrics_json(backup: Path | None) -> None:
    if backup is not None and backup.exists():
        shutil.move(str(backup), str(METRICS_JSON))


def _extract_cnn_metrics(metrics_json: Path) -> dict:
    if not metrics_json.exists():
        raise RuntimeError(f"train.py did not produce {metrics_json}")
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    if "cnn" not in payload:
        raise RuntimeError(f"{metrics_json} has no 'cnn' entry (keys: {list(payload.keys())})")
    return payload["cnn"]


def run_one(condition: str, seed: int, epochs: int, batch_size: int, device: str) -> dict:
    """Train CNN under one condition/seed; return its metrics dict."""
    overlay = CONDITION_DIR / f"{condition}.yaml"
    if not overlay.exists():
        raise FileNotFoundError(f"Condition overlay missing: {overlay}")

    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--model",
        "cnn",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--seed",
        str(seed),
        "--config",
        "config.yaml",
        "--config",
        str(overlay),
    ]
    if condition == "E_synthetic":
        cmd.append("--synthetic")

    # Make sure this run's CNN entry is the only one that ends up in
    # results/metrics.json. Clear it before the subprocess so the merge
    # path in train.py (post-83ba643) doesn't carry old entries forward.
    if METRICS_JSON.exists():
        METRICS_JSON.unlink()

    log_path = ARCHIVE_DIR / f"{condition}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Run %s seed=%d — launching subprocess", condition, seed)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            logf.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()

    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"{condition} seed={seed} failed with exit code {rc}; see {log_path}")

    cnn_metrics = _extract_cnn_metrics(METRICS_JSON)
    cnn_metrics["runner_wall_clock_sec"] = elapsed
    cnn_metrics["condition"] = condition
    cnn_metrics["seed"] = seed
    cnn_metrics["log_path"] = str(log_path.relative_to(REPO_ROOT))
    return cnn_metrics


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    import statistics

    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _rare_class_recall(run: dict, cls: str) -> float:
    per_class = run.get("per_class") or {}
    entry = per_class.get(cls)
    if not entry:
        return float("nan")
    return float(entry.get("recall", float("nan")))


def _render_markdown(
    results: Dict[str, Dict[int, dict]], output: Path, conditions: List[str], seeds: List[int]
) -> None:
    lines: List[str] = []
    lines.append("# Rare-class Study on WM-811K\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    lines.append(
        "Each condition is the custom CNN trained from scratch on the standard "
        "70/15/15 stratified split, then evaluated on the test set. "
        f"{len(seeds)} seeds per condition.\n"
    )

    # Headline table: macro F1, weighted F1, accuracy mean ± std
    lines.append("## Aggregate test-set metrics (mean ± std across seeds)\n")
    lines.append("| Condition | Macro F1 | Weighted F1 | Accuracy | ECE | Wall-clock |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in conditions:
        runs = list(results.get(cond, {}).values())
        if not runs:
            lines.append(f"| {cond} | — | — | — | — | — |")
            continue
        m_f1 = _mean_std([r["macro_f1"] for r in runs])
        w_f1 = _mean_std([r["weighted_f1"] for r in runs])
        acc = _mean_std([r["accuracy"] for r in runs])
        ece = _mean_std([r["ece"] for r in runs])
        wc = _mean_std([r["runner_wall_clock_sec"] / 60.0 for r in runs])
        lines.append(
            f"| {cond} | {m_f1[0]:.4f} ± {m_f1[1]:.4f} "
            f"| {w_f1[0]:.4f} ± {w_f1[1]:.4f} "
            f"| {acc[0]:.4f} ± {acc[1]:.4f} "
            f"| {ece[0]:.4f} ± {ece[1]:.4f} "
            f"| {wc[0]:.1f} ± {wc[1]:.1f} min |"
        )

    # Rare-class recall table
    lines.append("\n## Per-class recall on rare classes (mean across seeds)\n")
    header = "| Condition | " + " | ".join(RARE_CLASSES) + " |"
    sep = "|---|" + "---:|" * len(RARE_CLASSES)
    lines.append(header)
    lines.append(sep)
    for cond in conditions:
        runs = list(results.get(cond, {}).values())
        if not runs:
            lines.append(f"| {cond} | " + " | ".join(["—"] * len(RARE_CLASSES)) + " |")
            continue
        cells: List[str] = []
        for cls in RARE_CLASSES:
            vals = [_rare_class_recall(r, cls) for r in runs]
            vals = [v for v in vals if not (v != v)]  # drop nan
            if not vals:
                cells.append("—")
            else:
                m, s = _mean_std(vals)
                cells.append(f"{m:.3f} ± {s:.3f}")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    # Delta vs baseline
    baseline_runs = list(results.get("A_baseline", {}).values())
    if baseline_runs:
        bl_macro = _mean_std([r["macro_f1"] for r in baseline_runs])[0]
        lines.append("\n## Macro F1 delta vs A_baseline\n")
        lines.append("| Condition | ΔMacro F1 |")
        lines.append("|---|---:|")
        for cond in conditions:
            runs = list(results.get(cond, {}).values())
            if not runs or cond == "A_baseline":
                continue
            m_f1 = _mean_std([r["macro_f1"] for r in runs])[0]
            delta = m_f1 - bl_macro
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {cond} | {sign}{delta:.4f} |")

    # Methodology footer
    lines.append("\n## Methodology notes\n")
    lines.append(
        "- **Seeds**: small sample, so we report mean ± std rather than "
        "formal significance tests. With 3 seeds, paired Wilcoxon has "
        "minimum achievable p ≈ 0.25 — not useful. Treat Δ > 2σ as "
        "directionally informative, not as p < 0.05.\n"
        "- **Rare classes** are those with < 1000 training samples: "
        "Near-full (149), Scratch (1193), Donut (555), Random (866).\n"
        "- **Baseline (A)** is the current repo default post-`83ba643`: "
        "unweighted cross-entropy, no DRW, no balanced sampling.\n"
        "- **DRW (C)** switches from unweighted to weighted CE at epoch 10 "
        "(Cao et al. 2019, arXiv:1906.07413).\n"
        "- **Focal (B)** uses γ=2.0 without class weighting; modulating "
        "factor pushes gradient toward hard examples.\n"
        "- **Balanced sampling (D)** uses `ClassBalancedSampler` "
        "(`src/data/preprocessing.py:132`) with inverse-frequency "
        "oversampling of minority classes.\n"
        "- **Synthetic (E)** uses `--synthetic` which generates "
        "rule-based parametric defect wafers until classes are "
        "approximately balanced (`src/augmentation/synthetic.py`).\n"
    )

    lines.append("\n## Per-seed detail\n")
    lines.append("| Condition | Seed | Macro F1 | Weighted F1 | Accuracy | ECE | Epochs | Log |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for cond in conditions:
        for seed in seeds:
            r = results.get(cond, {}).get(seed)
            if r is None:
                lines.append(f"| {cond} | {seed} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {cond} | {seed} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} "
                f"| {r['accuracy']:.4f} | {r['ece']:.4f} | "
                f"{r.get('epochs_ran', '?')} | `{r.get('log_path', '?')}` |"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (default: 0,1,2)")
    parser.add_argument(
        "--conditions", default="all", help="Comma-separated conditions or 'all' (default: all)"
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "rare_class_study.md")
    parser.add_argument("--force", action="store_true", help="Re-run even if archive files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    seeds = [int(s) for s in args.seeds.split(",") if s]
    conditions = (
        ALL_CONDITIONS if args.conditions == "all" else [c for c in args.conditions.split(",") if c]
    )

    for cond in conditions:
        if cond not in ALL_CONDITIONS:
            parser.error(f"Unknown condition: {cond!r}. Valid: {ALL_CONDITIONS}")

    logger.info(
        "Study design: %d conditions x %d seeds = %d runs",
        len(conditions),
        len(seeds),
        len(conditions) * len(seeds),
    )
    logger.info("Conditions: %s", conditions)
    logger.info("Seeds: %s", seeds)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    pre_study_backup = _backup_metrics_json()

    results: Dict[str, Dict[int, dict]] = {}
    try:
        for cond in conditions:
            results.setdefault(cond, {})
            for seed in seeds:
                arc = _archive_path(cond, seed)
                if arc.exists() and not args.force:
                    logger.info(
                        "Skip %s seed=%d (archived at %s)", cond, seed, arc.relative_to(REPO_ROOT)
                    )
                    results[cond][seed] = json.loads(arc.read_text(encoding="utf-8"))
                    continue
                try:
                    run_metrics = run_one(
                        cond,
                        seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                except Exception as exc:
                    logger.error("Run %s seed=%d failed: %s", cond, seed, exc)
                    continue
                arc.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
                results[cond][seed] = run_metrics
                logger.info(
                    "Run %s seed=%d OK — macro F1 %.4f, acc %.4f (%.1f min)",
                    cond,
                    seed,
                    run_metrics["macro_f1"],
                    run_metrics["accuracy"],
                    run_metrics["runner_wall_clock_sec"] / 60.0,
                )

        _render_markdown(results, args.output, conditions, seeds)
    finally:
        _restore_metrics_json(pre_study_backup)

    logger.info("Study complete. Results table: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
