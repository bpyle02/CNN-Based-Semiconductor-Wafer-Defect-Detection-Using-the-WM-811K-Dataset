"""Tests for scripts/bootstrap_ci.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bootstrap_ci.py"


def _load_module():
    """Import scripts/bootstrap_ci.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("bootstrap_ci", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bootstrap_ci"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bci():
    return _load_module()


def test_bootstrap_metric_recovers_known_mean_ci(bci):
    """For a large normal sample the 95% percentile CI should bracket
    the true population mean with high probability."""
    rng_data = np.random.default_rng(1234)
    # Known population: mean=5.0, std=2.0, n=500
    values = rng_data.normal(loc=5.0, scale=2.0, size=500)
    rng_boot = np.random.default_rng(42)
    mean, lo, hi = bci.bootstrap_metric(values, np.mean, n_bootstrap=2000, rng=rng_boot)

    # Point estimate matches np.mean exactly.
    assert mean == pytest.approx(float(values.mean()), abs=1e-12)
    # CI must bracket the true mean.
    assert lo < 5.0 < hi
    # Width should be on the order of 2 * 1.96 * sigma/sqrt(n) = ~0.35.
    # Allow a generous factor to keep the test non-flaky.
    assert 0.1 < (hi - lo) < 1.0
    assert lo < mean < hi


def test_bootstrap_metric_degenerate_constant(bci):
    """Bootstrap of a constant vector is a constant: CI collapses."""
    values = np.full(50, 0.7)
    rng = np.random.default_rng(0)
    mean, lo, hi = bci.bootstrap_metric(values, np.mean, n_bootstrap=200, rng=rng)
    assert mean == pytest.approx(0.7, abs=1e-12)
    assert lo == pytest.approx(0.7, abs=1e-12)
    assert hi == pytest.approx(0.7, abs=1e-12)


def test_bootstrap_paired_accuracy_toy(bci):
    """On a perfectly-classified paired sample, accuracy CI is exactly [1, 1]."""
    preds = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    labels = preds.copy()

    from sklearn.metrics import accuracy_score

    def stat(p, y):
        return accuracy_score(y, p)

    rng = np.random.default_rng(42)
    mean, lo, hi = bci.bootstrap_paired(preds, labels, stat, n_bootstrap=500, rng=rng)
    assert mean == pytest.approx(1.0, abs=1e-12)
    assert lo == pytest.approx(1.0, abs=1e-12)
    assert hi == pytest.approx(1.0, abs=1e-12)


def test_bootstrap_paired_non_trivial_ci_brackets_point(bci):
    """Realistic case: CI must bracket the point estimate."""
    rng_data = np.random.default_rng(7)
    labels = rng_data.integers(0, 3, size=200)
    # Predictions agree 80% of the time.
    preds = labels.copy()
    flip_mask = rng_data.random(200) < 0.2
    preds[flip_mask] = (preds[flip_mask] + 1) % 3

    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(42)
    mean, lo, hi = bci.bootstrap_paired(
        preds, labels, lambda p, y: accuracy_score(y, p), n_bootstrap=1000, rng=rng
    )
    assert 0.7 < mean < 0.9
    assert lo <= mean <= hi
    # CI width should be a few percentage points, not 0 and not huge.
    assert 0.01 < (hi - lo) < 0.2


def test_archive_mode_end_to_end(tmp_path, bci):
    """bootstrap_from_archive should read <cond>_seed<N>.json and produce CI per condition."""
    mdir = tmp_path / "rare_class"
    mdir.mkdir()

    # Two conditions x 3 seeds of toy metrics.
    payloads = {
        ("A", 0): {"macro_f1": 0.50, "weighted_f1": 0.70, "accuracy": 0.80, "ece": 0.10},
        ("A", 1): {"macro_f1": 0.52, "weighted_f1": 0.71, "accuracy": 0.81, "ece": 0.09},
        ("A", 2): {"macro_f1": 0.48, "weighted_f1": 0.69, "accuracy": 0.79, "ece": 0.11},
        ("B", 0): {"macro_f1": 0.60, "weighted_f1": 0.75, "accuracy": 0.82, "ece": 0.08},
        ("B", 1): {"macro_f1": 0.61, "weighted_f1": 0.76, "accuracy": 0.83, "ece": 0.07},
        ("B", 2): {"macro_f1": 0.59, "weighted_f1": 0.74, "accuracy": 0.81, "ece": 0.09},
    }
    for (cond, seed), p in payloads.items():
        (mdir / f"{cond}_seed{seed}.json").write_text(json.dumps(p), encoding="utf-8")

    result = bci.bootstrap_from_archive(mdir, n_bootstrap=500, seed=42)
    assert result["mode"] == "seed_bootstrap"
    assert set(result["conditions"].keys()) == {"A", "B"}

    a = result["conditions"]["A"]
    # With 3 points {0.48, 0.50, 0.52}, mean == 0.50 exactly.
    assert a["macro_f1_mean"] == pytest.approx(0.50, abs=1e-12)
    assert a["macro_f1_ci_lo"] <= 0.50 <= a["macro_f1_ci_hi"]
    # Bootstrap over 3 points cannot extend beyond [min, max].
    assert a["macro_f1_ci_lo"] >= 0.48 - 1e-12
    assert a["macro_f1_ci_hi"] <= 0.52 + 1e-12

    b = result["conditions"]["B"]
    assert b["macro_f1_mean"] == pytest.approx(0.60, abs=1e-12)
    # Sanity: B > A on macro F1.
    assert b["macro_f1_mean"] > a["macro_f1_mean"]


def test_render_markdown_contains_header_and_rows(bci):
    results = {
        "mode": "seed_bootstrap",
        "conditions": {
            "A": {
                "macro_f1_mean": 0.5, "macro_f1_ci_lo": 0.48, "macro_f1_ci_hi": 0.52,
                "weighted_f1_mean": 0.7, "weighted_f1_ci_lo": 0.69, "weighted_f1_ci_hi": 0.71,
                "accuracy_mean": 0.8, "accuracy_ci_lo": 0.79, "accuracy_ci_hi": 0.81,
                "ece_mean": 0.1, "ece_ci_lo": 0.09, "ece_ci_hi": 0.11,
            },
        },
    }
    md = bci.render_markdown(results)
    assert "Bootstrap 95% Confidence Intervals" in md
    assert "| A |" in md
    assert "0.5000 [0.4800, 0.5200]" in md
