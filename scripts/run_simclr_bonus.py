#!/usr/bin/env python3
"""SimCLR bonus condition for the rare-class study.

Intended to run AFTER scripts/run_rare_class_study.py has produced
docs/rare_class_study.md. This script:

  1. Pretrains a CNN backbone self-supervised with SimCLR on the
     labeled WM-811K subset (the simclr_pretrain.py module uses labeled
     data but discards the labels, which still gives a useful
     contrastive signal without touching the ~638k unlabeled wafers).
  2. Fine-tunes the pretrained backbone with standard supervised CE
     across the same 3 seeds used for the core study.
  3. Appends a "F_simclr" section to docs/rare_class_study.md with the
     same format as the other conditions.

Running separately from the core study keeps a runaway pretraining
step from eating into core results: if the core study hasn't already
written rare_class_study.md, this script refuses to run.

Usage (Colab):
    python scripts/run_simclr_bonus.py --pretrain-epochs 10 \
        --finetune-epochs 20 --batch-size 128 --device cuda \
        --seeds 0,1,2
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
STUDY_DOC = REPO_ROOT / "docs" / "rare_class_study.md"
ARCHIVE_DIR = REPO_ROOT / "results" / "rare_class"
METRICS_JSON = REPO_ROOT / "results" / "metrics.json"
SIMCLR_CKPT = REPO_ROOT / "checkpoints" / "simclr_backbone.pth"

CONDITION = "F_simclr"

RARE_CLASSES = ["Near-full", "Scratch", "Donut", "Random"]


def _archive_path(seed: int) -> Path:
    return ARCHIVE_DIR / f"{CONDITION}_seed{seed}.json"


def _require_core_study() -> None:
    if not STUDY_DOC.exists():
        sys.stderr.write(
            f"Refusing to run: {STUDY_DOC.relative_to(REPO_ROOT)} does not exist.\n"
            "Run scripts/run_rare_class_study.py first so the bonus result "
            "is appended to an existing study, not produced in isolation.\n"
        )
        sys.exit(2)


def _run(cmd: List[str], log_path: Path) -> None:
    """Stream subprocess output to both a log file and parent stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    if rc != 0:
        raise RuntimeError(f"{cmd[1]} failed (exit {rc}); see {log_path}")


def pretrain(pretrain_epochs: int, batch_size: int, device: str) -> Path:
    if SIMCLR_CKPT.exists():
        logger.info(
            "SimCLR checkpoint already at %s — skipping pretrain",
            SIMCLR_CKPT.relative_to(REPO_ROOT),
        )
        return SIMCLR_CKPT

    log_path = ARCHIVE_DIR / f"{CONDITION}_pretrain.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "simclr_pretrain.py"),
        "--epochs",
        str(pretrain_epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--output-path",
        str(SIMCLR_CKPT),
    ]
    t0 = time.time()
    _run(cmd, log_path)
    logger.info("SimCLR pretrain finished in %.1f min", (time.time() - t0) / 60.0)
    return SIMCLR_CKPT


def finetune(seed: int, epochs: int, batch_size: int, device: str, pretrained: Path) -> dict:
    """Fine-tune CNN from the SimCLR-pretrained backbone for one seed."""
    if METRICS_JSON.exists():
        METRICS_JSON.unlink()

    log_path = ARCHIVE_DIR / f"{CONDITION}_seed{seed}.log"
    cmd = [
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
        str(REPO_ROOT / "configs" / "rare_class" / "A_baseline.yaml"),
        "--pretrained-checkpoint",
        str(pretrained),
    ]
    t0 = time.time()
    _run(cmd, log_path)
    elapsed = time.time() - t0

    payload = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    metrics = payload["cnn"]
    metrics["runner_wall_clock_sec"] = elapsed
    metrics["condition"] = CONDITION
    metrics["seed"] = seed
    metrics["log_path"] = str(log_path.relative_to(REPO_ROOT))
    metrics["pretrained_checkpoint"] = str(pretrained.relative_to(REPO_ROOT))
    return metrics


def append_to_study(runs: Dict[int, dict]) -> None:
    """Append a F_simclr section to docs/rare_class_study.md."""
    if not runs:
        logger.warning("No SimCLR runs succeeded; not modifying %s", STUDY_DOC)
        return

    import statistics

    runs_list = list(runs.values())
    macro_mean = statistics.mean(r["macro_f1"] for r in runs_list)
    macro_std = statistics.stdev(r["macro_f1"] for r in runs_list) if len(runs_list) > 1 else 0.0
    acc_mean = statistics.mean(r["accuracy"] for r in runs_list)
    acc_std = statistics.stdev(r["accuracy"] for r in runs_list) if len(runs_list) > 1 else 0.0

    section = ["\n## Bonus: F_simclr — SimCLR pretraining + CE fine-tune\n"]
    section.append(
        "Self-supervised SimCLR contrastive pretraining on the labeled "
        "WM-811K subset (labels discarded for the contrastive step), "
        "followed by standard CE fine-tuning on the same 70/15/15 split "
        "as the other conditions.\n"
    )
    section.append(f"**Runs collected:** {len(runs_list)} of 3 seeds.\n")
    section.append("| Metric | Value (mean ± std) |")
    section.append("|---|---:|")
    section.append(f"| Macro F1 | {macro_mean:.4f} ± {macro_std:.4f} |")
    section.append(f"| Accuracy | {acc_mean:.4f} ± {acc_std:.4f} |")
    section.append(
        f"| Wall-clock per run | "
        f"{statistics.mean(r['runner_wall_clock_sec'] for r in runs_list)/60.0:.1f} min |"
    )

    section.append("\n### Rare-class recall (mean across seeds)\n")
    section.append("| Class | Recall |")
    section.append("|---|---:|")
    for cls in RARE_CLASSES:
        vals = []
        for r in runs_list:
            per = (r.get("per_class") or {}).get(cls)
            if per and "recall" in per:
                vals.append(float(per["recall"]))
        if vals:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            section.append(f"| {cls} | {m:.3f} ± {s:.3f} |")
        else:
            section.append(f"| {cls} | — |")

    section.append("\n### Per-seed detail\n")
    section.append("| Seed | Macro F1 | Accuracy | Log |")
    section.append("|---:|---:|---:|---|")
    for seed, r in sorted(runs.items()):
        section.append(
            f"| {seed} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
            f"`{r.get('log_path', '?')}` |"
        )

    section.append(
        "\n_This bonus condition runs after the core five-condition "
        "study, so a slow SimCLR pretrain never blocks the core "
        "deliverable._\n"
    )

    existing = STUDY_DOC.read_text(encoding="utf-8")
    # Strip any prior F_simclr section (re-run idempotence).
    marker = "\n## Bonus: F_simclr"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"

    STUDY_DOC.write_text(existing + "\n".join(section) + "\n", encoding="utf-8")
    logger.info("Appended F_simclr section to %s", STUDY_DOC.relative_to(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    _require_core_study()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]

    logger.info(
        "SimCLR bonus: pretrain %d epochs, fine-tune %d epochs x %d seeds",
        args.pretrain_epochs,
        args.finetune_epochs,
        len(seeds),
    )

    try:
        backbone = pretrain(args.pretrain_epochs, args.batch_size, args.device)
    except Exception as exc:
        logger.error("SimCLR pretrain failed: %s", exc)
        logger.error("Core rare-class study is unaffected.")
        return 1

    results: Dict[int, dict] = {}
    for seed in seeds:
        arc = _archive_path(seed)
        if arc.exists() and not args.force:
            logger.info("Skip F_simclr seed=%d (archived)", seed)
            results[seed] = json.loads(arc.read_text(encoding="utf-8"))
            continue
        try:
            metrics = finetune(seed, args.finetune_epochs, args.batch_size, args.device, backbone)
        except Exception as exc:
            logger.error("F_simclr seed=%d failed: %s", seed, exc)
            continue
        arc.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        results[seed] = metrics
        logger.info("F_simclr seed=%d OK — macro F1 %.4f", seed, metrics["macro_f1"])

    append_to_study(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
