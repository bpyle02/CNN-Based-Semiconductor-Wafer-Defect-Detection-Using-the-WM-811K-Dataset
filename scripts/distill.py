#!/usr/bin/env python3
"""Knowledge distillation: train a small student to mimic an ensemble teacher.

Loads each ensemble member from ``checkpoints/best_<name>.pth`` via
``train.build_model``, freezes them in eval mode, and averages their
temperature-softened softmax probabilities as the teacher distribution.
The student (default: Custom CNN) is trained with:

    loss = alpha * T^2 * KL(student_softmax_T || teacher_softmax_T)
         + (1 - alpha) * CE(student_logits, hard_labels)

Reuses ``train.load_and_preprocess_data`` and
``src.data.preprocessing.WaferMapDataset`` — no data-pipeline reimplementation.

Outputs:
  - checkpoints/student_distilled_best.pth
  - results/distill_metrics.json
  - results/distill.log

Usage:
    python scripts/distill.py
    python scripts/distill.py --teacher-models cnn,resnet \
        --student-arch cnn --epochs 20 --batch-size 128 \
        --temperature 4.0 --alpha 0.7 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.preprocessing import WaferMapDataset  # noqa: E402

logger = logging.getLogger("distill")

DEFAULT_TEACHERS = ("cnn", "resnet", "efficientnet", "ride", "swin")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def discover_checkpoints(ckpt_dir: Path, requested: Sequence[str] | None) -> List[Tuple[str, Path]]:
    """Find ``best_<name>.pth`` (or ``<name>_best.pth``) in ``ckpt_dir``.

    If ``requested`` is provided, keep only those names that have a file on
    disk. Unrequested or missing names are skipped with a warning.
    """
    found: Dict[str, Path] = {}
    for ckpt in sorted(ckpt_dir.glob("*.pth")):
        stem = ckpt.stem
        if stem.startswith("best_"):
            name = stem[len("best_") :]
        elif stem.endswith("_best"):
            name = stem[: -len("_best")]
        else:
            continue
        if name == "student_distilled":
            continue
        found[name] = ckpt

    if requested:
        result: List[Tuple[str, Path]] = []
        for name in requested:
            if name in found:
                result.append((name, found[name]))
            else:
                logger.warning(
                    "Requested teacher '%s' has no checkpoint in %s; skipping", name, ckpt_dir
                )
        return result

    return [(n, p) for n, p in found.items()]


# ---------------------------------------------------------------------------
# Teacher / student construction
# ---------------------------------------------------------------------------
def build_and_load(model_name: str, ckpt: Path, config, num_classes: int, device: str) -> nn.Module:
    import train as _train

    model_cfg = getattr(config.model, model_name, None) or config.model
    model, _ = _train.build_model(model_name, model_cfg, num_classes, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("[%s] missing keys: %d", model_name, len(missing))
    if unexpected:
        logger.warning("[%s] unexpected keys: %d", model_name, len(unexpected))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------
@torch.no_grad()
def teacher_soft_targets(
    teachers: List[nn.Module],
    imgs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return averaged temperature-softened softmax over all teachers."""
    probs = None
    for t in teachers:
        logits = t(imgs)
        # RIDE and some models can return tuples — normalize to tensor.
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        p = F.softmax(logits / temperature, dim=1)
        probs = p if probs is None else probs + p
    return probs / len(teachers)


def distill_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total_loss, kd_loss, ce_loss)."""
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    # batchmean KL matches Hinton's formulation (averaged over batch).
    kd = F.kl_div(log_student, teacher_probs, reduction="batchmean") * (temperature**2)
    ce = F.cross_entropy(student_logits, labels)
    total = alpha * kd + (1.0 - alpha) * ce
    return total, kd, ce


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    acc = float(accuracy_score(labels_arr, preds))
    macro_f1 = float(f1_score(labels_arr, preds, average="macro", zero_division=0))
    return acc, macro_f1, preds, labels_arr


def train_student(
    student: nn.Module,
    teachers: List[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    temperature: float,
    alpha: float,
    lr: float,
    weight_decay: float,
    ckpt_out: Path,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_kd": [],
        "train_ce": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        student.train()
        running = {"loss": 0.0, "kd": 0.0, "ce": 0.0, "n": 0}
        start = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            teacher_probs = teacher_soft_targets(teachers, imgs, temperature)

            optimizer.zero_grad(set_to_none=True)
            logits = student(imgs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss, kd, ce = distill_loss(logits, teacher_probs, labels, temperature, alpha)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running["loss"] += float(loss) * bs
            running["kd"] += float(kd) * bs
            running["ce"] += float(ce) * bs
            running["n"] += bs

        scheduler.step()
        n = max(running["n"], 1)
        tr_loss = running["loss"] / n
        tr_kd = running["kd"] / n
        tr_ce = running["ce"] / n
        val_acc, val_f1, _, _ = evaluate(student, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["train_kd"].append(tr_kd)
        history["train_ce"].append(tr_ce)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        dur = time.time() - start
        logger.info(
            "epoch %02d/%d  loss=%.4f (kd=%.4f ce=%.4f)  val_acc=%.4f  val_macroF1=%.4f  [%.1fs]",
            epoch,
            epochs,
            tr_loss,
            tr_kd,
            tr_ce,
            val_acc,
            val_f1,
            dur,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_acc,
                    "temperature": temperature,
                    "alpha": alpha,
                },
                ckpt_out,
            )
            logger.info("  ↳ new best val_macro_f1=%.4f — saved %s", val_f1, ckpt_out)

    return history


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def _parse_teacher_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--teacher-models",
        type=str,
        default=",".join(DEFAULT_TEACHERS),
        help=f"Comma-separated teacher names (default: {','.join(DEFAULT_TEACHERS)}). "
        "Use 'auto' to discover all *_best.pth / best_*.pth in checkpoints/.",
    )
    parser.add_argument(
        "--student-arch",
        type=str,
        default="cnn",
        help="Student architecture name understood by train.build_model (default: cnn).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "LSWMD_new.pkl")
    parser.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    # Logging: console + file
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / "distill.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")],
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)

    # -- Teachers --
    requested = (
        None
        if args.teacher_models.strip().lower() == "auto"
        else _parse_teacher_list(args.teacher_models)
    )
    teacher_pairs = discover_checkpoints(args.checkpoints_dir, requested)
    if not teacher_pairs:
        logger.error("No teacher checkpoints found in %s", args.checkpoints_dir)
        return 1
    logger.info("Teachers: %s", [n for n, _ in teacher_pairs])

    # -- Data (reuse train.py pipeline) --
    import train as _train

    data = _train.load_and_preprocess_data(
        args.data_path,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        target_size=(config.data.target_size, config.data.target_size),
        seed=args.seed,
        synthetic=False,
    )
    class_names = data["class_names"]
    num_classes = len(class_names)

    train_ds = WaferMapDataset(data["train_maps"], data["y_train"])
    val_ds = WaferMapDataset(data["val_maps"], data["y_val"])
    test_ds = WaferMapDataset(data["test_maps"], data["y_test"])

    dl_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(args.device == "cuda")
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # -- Build teachers --
    teachers: List[nn.Module] = []
    for name, ckpt in teacher_pairs:
        logger.info("Loading teacher %s from %s", name, ckpt)
        teachers.append(build_and_load(name, ckpt, config, num_classes, args.device))

    # -- Build student --
    student_cfg = getattr(config.model, args.student_arch, None) or config.model
    student, student_display = _train.build_model(
        args.student_arch, student_cfg, num_classes, args.device
    )
    student.train()
    n_params = count_params(student)
    logger.info("Student: %s  params=%s", student_display, f"{n_params:,}")

    # -- Train --
    ckpt_out = args.checkpoints_dir / "student_distilled_best.pth"
    history = train_student(
        student=student,
        teachers=teachers,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_out=ckpt_out,
    )

    # -- Reload best student and evaluate on test --
    best_state = torch.load(ckpt_out, map_location=args.device, weights_only=False)
    student.load_state_dict(best_state["model_state_dict"])
    test_acc, test_f1, preds, labels_arr = evaluate(student, test_loader, args.device)

    size_mb = file_size_mb(ckpt_out)
    metrics = {
        "student_arch": args.student_arch,
        "student_display_name": student_display,
        "teachers": [n for n, _ in teacher_pairs],
        "num_teachers": len(teacher_pairs),
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "test": {
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "weighted_f1": float(f1_score(labels_arr, preds, average="weighted", zero_division=0)),
        },
        "param_count": int(n_params),
        "checkpoint_path": str(ckpt_out),
        "checkpoint_size_mb": size_mb,
        "best_epoch": int(best_state.get("epoch", -1)),
        "best_val_macro_f1": float(best_state.get("val_macro_f1", -1.0)),
        "history": history,
        "class_names": list(class_names),
    }

    out_json = args.results_dir / "distill_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", out_json)

    # -- Final report --
    logger.info("=" * 60)
    logger.info("DISTILLATION COMPLETE")
    logger.info("  student           : %s", student_display)
    logger.info("  test accuracy     : %.4f", test_acc)
    logger.info("  test macro F1     : %.4f", test_f1)
    logger.info("  parameter count   : %s", f"{n_params:,}")
    logger.info("  checkpoint size   : %.2f MB", size_mb)
    logger.info("  checkpoint path   : %s", ckpt_out)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
