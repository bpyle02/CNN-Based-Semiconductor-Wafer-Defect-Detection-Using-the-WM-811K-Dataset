#!/usr/bin/env python3
"""Compare a fresh metrics.json against the committed baseline.

Exits 0 if every model in the baseline still meets the tolerance thresholds,
exits 1 otherwise. Designed for CI: a PR that accidentally degrades the
reported CNN accuracy by >2 points or macro F1 by >3 points should fail the
check and require either a fix or an explicit baseline refresh (documented
in CHANGELOG.md and acknowledged in the PR description).

Usage:
    python scripts/check_metrics.py \
        --baseline results/metrics.baseline.json \
        --current results/metrics.json

Tolerances are intentionally loose on macro F1 because the rare-class recall
(Near-full, Scratch) is very noisy across runs and seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Per-metric absolute drop tolerances. A drop GREATER than this fails.
DEFAULT_TOLERANCES: Dict[str, float] = {
    "accuracy": 0.02,  # 2 percentage points
    "macro_f1": 0.03,  # 3 percentage points (rare-class noise)
    "weighted_f1": 0.02,
}


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)


def _compare_model(
    model_name: str,
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    tolerances: Dict[str, float],
) -> list[str]:
    failures: list[str] = []
    for metric, tol in tolerances.items():
        if metric not in baseline:
            continue
        b = float(baseline[metric])
        if metric not in current:
            failures.append(f"{model_name}.{metric}: baseline={b:.4f}, current is MISSING")
            continue
        c = float(current[metric])
        drop = b - c
        if drop > tol:
            failures.append(
                f"{model_name}.{metric}: baseline={b:.4f} current={c:.4f} "
                f"drop={drop:+.4f} (tolerance={tol:+.4f})"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/metrics.baseline.json"),
        help="Committed baseline metrics (default: results/metrics.baseline.json)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("results/metrics.json"),
        help="Fresh metrics to check (default: results/metrics.json)",
    )
    parser.add_argument(
        "--tolerance-accuracy",
        type=float,
        default=DEFAULT_TOLERANCES["accuracy"],
    )
    parser.add_argument(
        "--tolerance-macro-f1",
        type=float,
        default=DEFAULT_TOLERANCES["macro_f1"],
    )
    parser.add_argument(
        "--tolerance-weighted-f1",
        type=float,
        default=DEFAULT_TOLERANCES["weighted_f1"],
    )
    args = parser.parse_args(argv)

    tolerances = {
        "accuracy": args.tolerance_accuracy,
        "macro_f1": args.tolerance_macro_f1,
        "weighted_f1": args.tolerance_weighted_f1,
    }

    baseline = _load(args.baseline)
    current = _load(args.current)

    all_failures: list[str] = []
    for model_name, model_baseline in baseline.items():
        if not isinstance(model_baseline, dict):
            continue
        if model_name not in current:
            all_failures.append(
                f"{model_name}: baseline expects this model but it is MISSING from current"
            )
            continue
        all_failures.extend(
            _compare_model(model_name, model_baseline, current[model_name], tolerances)
        )

    if all_failures:
        print("FAIL: metrics regression detected\n")
        for line in all_failures:
            print(f"  - {line}")
        print(
            "\nEither fix the regression or, if the change was intentional, "
            "refresh results/metrics.baseline.json in the same PR and update "
            "CHANGELOG.md."
        )
        return 1

    print(
        f"OK: all {len(baseline)} model(s) within tolerance "
        f"(accuracy ±{tolerances['accuracy']:.2f}, "
        f"macro_f1 ±{tolerances['macro_f1']:.2f}, "
        f"weighted_f1 ±{tolerances['weighted_f1']:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
