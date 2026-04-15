#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for rare-class study metrics.

Two operating modes:

1. **Archive mode** (``--metrics-dir``): walk archived
   ``<condition>_seed<N>.json`` produced by
   ``scripts/run_rare_class_study.py``. Because those archives store
   *summary* metrics (macro F1, accuracy) rather than per-sample
   predictions, we fall back to a non-parametric bootstrap **over the 3
   seeds**. With N=3 this is a small-sample CI — it is reported honestly
   as such, following the methodology footer in
   ``docs/rare_class_study.md``.

2. **Per-sample mode** (``--predictions``): JSON or ``.npz`` with
   per-sample predictions + labels. This gives the true test-set
   bootstrap (N ~ 25k) and is strongly preferred. JSON schema:

       {"preds": [...], "labels": [...], "probs": [[...], ...]}

   NPZ schema: keys ``preds`` (int, shape [N]), ``labels`` (int, shape
   [N]); optional ``probs`` (float, shape [N, C]).

Outputs:

- ``results/bootstrap_ci.json``  (machine-readable)
- ``docs/bootstrap_ci.md``       (markdown table, paste into report)

No new dependencies — uses ``sklearn.utils.resample``.

Usage::

    python scripts/bootstrap_ci.py --metrics-dir results/rare_class/
    python scripts/bootstrap_ci.py --predictions results/preds_cnn.npz \\
        --n-bootstrap 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "rare_class"
DEFAULT_CI_JSON = REPO_ROOT / "results" / "bootstrap_ci.json"
DEFAULT_CI_MD = REPO_ROOT / "docs" / "bootstrap_ci.md"

ARCHIVE_RE = re.compile(r"^(?P<cond>.+)_seed(?P<seed>\d+)\.json$")


# ---------------------------------------------------------------------------
# Bootstrap primitives
# ---------------------------------------------------------------------------


def bootstrap_metric(
    values: np.ndarray,
    statistic,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Resample ``values`` with replacement; return (mean, ci_lo, ci_hi).

    ``statistic`` is a callable mapping ``np.ndarray -> float`` that is
    evaluated on each bootstrap replicate. Uses percentile CI.
    """
    values = np.asarray(values)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(n_bootstrap, dtype=np.float64)
    n = values.shape[0]
    seed_seq = rng.integers(0, 2**31 - 1, size=n_bootstrap)
    for i in range(n_bootstrap):
        sample = resample(values, replace=True, n_samples=n, random_state=int(seed_seq[i]))
        stats[i] = statistic(sample)
    point = float(statistic(values))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return point, lo, hi


def bootstrap_paired(
    preds: np.ndarray,
    labels: np.ndarray,
    statistic,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Paired bootstrap on (preds, labels): resample indices jointly.

    ``statistic`` receives ``(preds_sample, labels_sample)`` and returns
    a scalar. This preserves the pairing between each prediction and
    its true label, which is the correct resampling scheme for
    classification metrics.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.shape != labels.shape:
        raise ValueError(f"preds {preds.shape} vs labels {labels.shape} shape mismatch")
    n = preds.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(n_bootstrap, dtype=np.float64)
    idx_seeds = rng.integers(0, 2**31 - 1, size=n_bootstrap)
    all_idx = np.arange(n)
    for i in range(n_bootstrap):
        idx = resample(all_idx, replace=True, n_samples=n, random_state=int(idx_seeds[i]))
        stats[i] = statistic(preds[idx], labels[idx])
    point = float(statistic(preds, labels))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Archive mode (fallback: bootstrap over seeds)
# ---------------------------------------------------------------------------


def _discover_archives(metrics_dir: Path) -> Dict[str, Dict[int, Path]]:
    """Return ``{condition: {seed: path}}`` for every archive JSON."""
    found: Dict[str, Dict[int, Path]] = {}
    for p in sorted(metrics_dir.glob("*.json")):
        m = ARCHIVE_RE.match(p.name)
        if not m:
            continue
        cond = m.group("cond")
        seed = int(m.group("seed"))
        found.setdefault(cond, {})[seed] = p
    return found


def _load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_bootstrap(
    runs: List[dict],
    key: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap mean of ``key`` across seeds."""
    vals = np.array([float(r[key]) for r in runs if key in r], dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    mean, lo, hi = bootstrap_metric(vals, np.mean, n_bootstrap, rng)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(vals.size)}


def bootstrap_from_archive(metrics_dir: Path, n_bootstrap: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    archives = _discover_archives(metrics_dir)
    if not archives:
        raise FileNotFoundError(
            f"No archive JSON files matching <cond>_seed<N>.json in {metrics_dir}"
        )

    results: Dict[str, Any] = {
        "mode": "seed_bootstrap",
        "source": str(metrics_dir),
        "conditions": {},
    }
    for cond, seed_paths in sorted(archives.items()):
        runs = [_load_run(p) for _, p in sorted(seed_paths.items())]
        cond_out: Dict[str, Any] = {"seeds": sorted(seed_paths.keys()), "n_seeds": len(runs)}
        for metric_key, out_prefix in [
            ("macro_f1", "macro_f1"),
            ("accuracy", "accuracy"),
            ("weighted_f1", "weighted_f1"),
            ("ece", "ece"),
        ]:
            stat = _seed_bootstrap(runs, metric_key, n_bootstrap, rng)
            cond_out[f"{out_prefix}_mean"] = stat["mean"]
            cond_out[f"{out_prefix}_ci_lo"] = stat["ci_lo"]
            cond_out[f"{out_prefix}_ci_hi"] = stat["ci_hi"]
        results["conditions"][cond] = cond_out
    return results


# ---------------------------------------------------------------------------
# Per-sample mode (true test-set bootstrap)
# ---------------------------------------------------------------------------


def _load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=False)
        return np.asarray(d["preds"]).astype(int), np.asarray(d["labels"]).astype(int)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        preds = np.asarray(payload["preds"]).astype(int)
        labels = np.asarray(payload["labels"]).astype(int)
        return preds, labels
    raise ValueError(f"Unsupported predictions format: {path.suffix} (expected .json or .npz)")


def bootstrap_from_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
    name: str = "model",
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    def _macro_f1(p, y):
        return f1_score(y, p, average="macro", zero_division=0)

    def _weighted_f1(p, y):
        return f1_score(y, p, average="weighted", zero_division=0)

    def _acc(p, y):
        return accuracy_score(y, p)

    metrics: Dict[str, Any] = {"n_samples": int(preds.shape[0])}
    for stat_name, stat in [
        ("macro_f1", _macro_f1),
        ("weighted_f1", _weighted_f1),
        ("accuracy", _acc),
    ]:
        mean, lo, hi = bootstrap_paired(preds, labels, stat, n_bootstrap, rng)
        metrics[f"{stat_name}_mean"] = mean
        metrics[f"{stat_name}_ci_lo"] = lo
        metrics[f"{stat_name}_ci_hi"] = hi
    return {"mode": "per_sample_bootstrap", "conditions": {name: metrics}}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    if any(np.isnan(v) for v in (mean, lo, hi)):
        return "—"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def render_markdown(results: Dict[str, Any]) -> str:
    mode = results.get("mode", "unknown")
    lines: List[str] = []
    lines.append("# Bootstrap 95% Confidence Intervals\n")
    if mode == "seed_bootstrap":
        lines.append(
            "_Bootstrap over **N=3 seeds** (archived summary metrics — no per-sample "
            "predictions available). These intervals reflect seed-level variance and are "
            "wider than a true test-set bootstrap would produce. Treat as directional._\n"
        )
    elif mode == "per_sample_bootstrap":
        lines.append(
            "_Paired bootstrap over **per-sample test-set predictions** "
            "(preds resampled jointly with labels)._\n"
        )
    lines.append("Format: `mean [CI_lo, CI_hi]` at 95% level.\n")
    lines.append("| Condition | Macro F1 | Weighted F1 | Accuracy | ECE |")
    lines.append("|---|---|---|---|---|")
    for cond, m in sorted(results.get("conditions", {}).items()):
        mf1 = _fmt_ci(
            m.get("macro_f1_mean", float("nan")),
            m.get("macro_f1_ci_lo", float("nan")),
            m.get("macro_f1_ci_hi", float("nan")),
        )
        wf1 = _fmt_ci(
            m.get("weighted_f1_mean", float("nan")),
            m.get("weighted_f1_ci_lo", float("nan")),
            m.get("weighted_f1_ci_hi", float("nan")),
        )
        acc = _fmt_ci(
            m.get("accuracy_mean", float("nan")),
            m.get("accuracy_ci_lo", float("nan")),
            m.get("accuracy_ci_hi", float("nan")),
        )
        ece = _fmt_ci(
            m.get("ece_mean", float("nan")),
            m.get("ece_ci_lo", float("nan")),
            m.get("ece_ci_hi", float("nan")),
        )
        lines.append(f"| {cond} | {mf1} | {wf1} | {acc} | {ece} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Archive dir produced by run_rare_class_study.py",
    )
    src.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Per-sample JSON or .npz with {preds, labels}",
    )
    parser.add_argument(
        "--name",
        default="model",
        help="Name for the --predictions condition in output (default: model)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_CI_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_CI_MD)
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    if args.predictions is not None:
        preds, labels = _load_predictions(args.predictions)
        logger.info("Per-sample mode: %d samples from %s", preds.size, args.predictions)
        results = bootstrap_from_predictions(
            preds, labels, args.n_bootstrap, args.seed, name=args.name
        )
    else:
        logger.info("Archive mode: scanning %s", args.metrics_dir)
        results = bootstrap_from_archive(args.metrics_dir, args.n_bootstrap, args.seed)

    results["n_bootstrap"] = args.n_bootstrap
    results["seed"] = args.seed

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(results), encoding="utf-8")
    logger.info("Wrote %s", args.out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
