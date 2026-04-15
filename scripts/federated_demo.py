#!/usr/bin/env python3
"""
Federated learning demo: 3-fab FedAvg vs. centralized baseline.

Simulates realistic fab specialization by partitioning the WM-811K train
set into 3 "fabs":
    Fab A: 50% of Center + 50% of Donut samples
    Fab B: 50% of Loc + 50% of Scratch samples
    Fab C: everything else

Runs 5 rounds of FedAvg (Custom CNN, 2 local epochs per round), compares
final global-model test accuracy vs. a centralized baseline (a single
client with ALL the data trained for the same total epoch budget).

Outputs:
    results/federated_demo.json
    docs/federated_demo.md

Usage:
    python scripts/federated_demo.py --quick
    python scripts/federated_demo.py --rounds 5 --local-epochs 2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    KNOWN_CLASSES,
    WaferMapDataset,
    get_image_transforms,
    load_dataset,
    preprocess_wafer_maps,
    seed_worker,
)
from src.federated.fed_avg import (  # noqa: E402
    FedAvgConfig,
    FedAveragingClient,
    FedAveragingServer,
)
from src.models import WaferCNN  # noqa: E402

logger = logging.getLogger(__name__)

# Fab specialization mapping: class -> fab_id that gets 50% of those samples
PRIMARY_FAB_FOR_CLASS = {
    "Center": 0,   # Fab A
    "Donut": 0,    # Fab A
    "Loc": 1,      # Fab B
    "Scratch": 1,  # Fab B
}


def partition_by_fab(
    labels: np.ndarray,
    class_names: List[str],
    seed: int = 42,
) -> List[np.ndarray]:
    """Partition sample indices into 3 fabs with specialization.

    Fab A gets 50% of Center + 50% of Donut.
    Fab B gets 50% of Loc + 50% of Scratch.
    Fab C gets everything else (including the remaining 50% of the four
    "specialized" classes so they still exist across fabs).
    """
    rng = np.random.RandomState(seed)
    fabs: List[List[int]] = [[], [], []]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    specialized_idx = {
        name_to_idx[n]: fab for n, fab in PRIMARY_FAB_FOR_CLASS.items()
        if n in name_to_idx
    }

    for cls_idx in range(len(class_names)):
        idxs = np.where(labels == cls_idx)[0]
        rng.shuffle(idxs)
        if cls_idx in specialized_idx:
            split = len(idxs) // 2
            fabs[specialized_idx[cls_idx]].extend(idxs[:split].tolist())
            fabs[2].extend(idxs[split:].tolist())  # remainder -> Fab C
        else:
            fabs[2].extend(idxs.tolist())

    return [np.array(sorted(f), dtype=np.int64) for f in fabs]


def describe_partition(
    fab_indices: List[np.ndarray],
    labels: np.ndarray,
    class_names: List[str],
) -> List[Dict[str, int]]:
    rows = []
    for fab_id, idxs in enumerate(fab_indices):
        row = {"fab": chr(ord("A") + fab_id), "total": int(len(idxs))}
        for c, name in enumerate(class_names):
            row[name] = int((labels[idxs] == c).sum())
        rows.append(row)
    return rows


def make_loader(
    ds: WaferMapDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    g = torch.Generator().manual_seed(42)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        worker_init_fn=seed_worker, generator=g,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = correct / max(total, 1)
    macro_f1 = float(f1_score(y, p, average="macro", zero_division=0))
    return acc, macro_f1


def train_centralized(
    train_ds: WaferMapDataset,
    test_loader: DataLoader,
    num_classes: int,
    total_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> Tuple[nn.Module, float, float, float]:
    """Centralized baseline: single trainer over full dataset."""
    model = WaferCNN(num_classes=num_classes).to(device)
    loader = make_loader(train_ds, batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for ep in range(total_epochs):
        seen, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            seen += yb.size(0)
        logger.info("  [centralized] epoch %d/%d loss=%.4f acc=%.4f",
                    ep + 1, total_epochs, loss_sum / seen, correct / seen)
    elapsed = time.time() - t0
    acc, f1 = evaluate(model, test_loader, device)
    return model, acc, f1, elapsed


def train_federated(
    train_ds: WaferMapDataset,
    fab_indices: List[np.ndarray],
    test_loader: DataLoader,
    num_classes: int,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> Tuple[nn.Module, float, float, float, List[Dict]]:
    """3-fab FedAvg training loop."""
    fab_loaders = [
        make_loader(Subset(train_ds, idxs.tolist()), batch_size, shuffle=True)
        for idxs in fab_indices
    ]
    global_model = WaferCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    clients = []
    for fab_id, loader in enumerate(fab_loaders):
        client_model = WaferCNN(num_classes=num_classes)
        clients.append(FedAveragingClient(
            client_id=fab_id,
            train_loader=loader,
            model=client_model,
            criterion=criterion,
            local_epochs=local_epochs,
            learning_rate=lr,
            device=device,
        ))

    cfg = FedAvgConfig(
        num_rounds=rounds,
        clients_per_round=len(clients),
        local_epochs=local_epochs,
        learning_rate=lr,
        device=device,
        seed=42,
        verbose=False,
    )
    server = FedAveragingServer(
        model=global_model, clients=clients, config=cfg, test_loader=test_loader,
    )

    t0 = time.time()
    round_history: List[Dict] = []
    for r in range(rounds):
        loss, acc, test_acc = server.train_round(r)
        round_acc, round_f1 = evaluate(server.get_global_model(), test_loader, device)
        round_history.append({
            "round": r + 1,
            "train_loss_avg": float(loss),
            "train_acc_avg": float(acc),
            "test_accuracy": float(round_acc),
            "test_macro_f1": float(round_f1),
        })
        logger.info("  [fedavg] round %d/%d train_loss=%.4f test_acc=%.4f test_f1=%.4f",
                    r + 1, rounds, loss, round_acc, round_f1)
    elapsed = time.time() - t0
    final_acc, final_f1 = evaluate(server.get_global_model(), test_loader, device)
    return server.get_global_model(), final_acc, final_f1, elapsed, round_history


def write_report(
    out_md: Path,
    partition_rows: List[Dict[str, int]],
    cent: Dict,
    fed: Dict,
    round_history: List[Dict],
) -> None:
    lines: List[str] = []
    lines.append("# Federated Learning Demo: 3-Fab FedAvg vs. Centralized\n")
    lines.append(
        "Simulates fab specialization where each fab primarily sees one or "
        "two defect families. Demonstrates FedAvg viability when no single "
        "fab may share raw wafer maps (IP/regulatory constraints).\n",
    )

    lines.append("\n## Data partition\n")
    class_keys = [k for k in partition_rows[0].keys() if k not in ("fab", "total")]
    header = "| Fab | Total | " + " | ".join(class_keys) + " |"
    sep = "|---|---:|" + "|".join(["---:"] * len(class_keys)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in partition_rows:
        cells = [f"{row[k]}" for k in class_keys]
        lines.append(f"| {row['fab']} | {row['total']} | " + " | ".join(cells) + " |")

    lines.append("\n## Comparison\n")
    lines.append("| Setup | Test Accuracy | Test Macro F1 | Wall time (s) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Centralized | {cent['accuracy']:.4f} | "
        f"{cent['macro_f1']:.4f} | {cent['wall_time_s']:.1f} |",
    )
    lines.append(
        f"| FedAvg (3 fabs) | {fed['accuracy']:.4f} | "
        f"{fed['macro_f1']:.4f} | {fed['wall_time_s']:.1f} |",
    )
    gap_acc = cent["accuracy"] - fed["accuracy"]
    gap_f1 = cent["macro_f1"] - fed["macro_f1"]
    lines.append(
        f"\n**Gap:** centralized outperforms FedAvg by "
        f"{gap_acc * 100:.2f} acc pts and {gap_f1 * 100:.2f} F1 pts. "
        "Expected: 2-5 pts gap trades a small accuracy cost for the ability "
        "to train across fabs without sharing raw wafer maps.\n",
    )

    lines.append("\n## Per-round test metrics\n")
    lines.append("| Round | train loss (avg) | test accuracy | test macro F1 |")
    lines.append("|---:|---:|---:|---:|")
    for r in round_history:
        lines.append(
            f"| {r['round']} | {r['train_loss_avg']:.4f} | "
            f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} |",
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="3-fab FedAvg vs. centralized demo")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subsample", type=int, default=0,
                        help="Subsample train set to N examples (0 = all)")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny demo: 2 rounds, 1 local epoch, 2k subsample")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out-json", type=Path,
                        default=REPO_ROOT / "results" / "federated_demo.json")
    parser.add_argument("--out-md", type=Path,
                        default=REPO_ROOT / "docs" / "federated_demo.md")
    args = parser.parse_args()

    if args.quick:
        args.rounds = 2
        args.local_epochs = 1
        if args.subsample == 0:
            args.subsample = 2000

    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Loading WM-811K dataset...")
    df = load_dataset(args.data_path)
    df = df[df["failureClass"].isin(KNOWN_CLASSES)].reset_index(drop=True)
    le = LabelEncoder()
    labels_all = le.fit_transform(df["failureClass"])
    class_names = le.classes_.tolist()
    num_classes = len(class_names)
    maps_all = df["waferMap"].values

    tr_idx, te_idx = train_test_split(
        np.arange(len(labels_all)), test_size=0.20,
        stratify=labels_all, random_state=42,
    )
    train_labels = labels_all[tr_idx]
    test_labels = labels_all[te_idx]
    train_maps_raw = [maps_all[i] for i in tr_idx]
    test_maps_raw = [maps_all[i] for i in te_idx]

    if args.subsample > 0 and args.subsample < len(train_labels):
        logger.info("Subsampling train: %d -> %d", len(train_labels), args.subsample)
        sub = np.random.RandomState(0).choice(
            len(train_labels), args.subsample, replace=False,
        )
        train_labels = train_labels[sub]
        train_maps_raw = [train_maps_raw[i] for i in sub]

    logger.info("Preprocessing train (%d) + test (%d) ...",
                len(train_labels), len(test_labels))
    train_maps = np.array(preprocess_wafer_maps(train_maps_raw))
    test_maps = np.array(preprocess_wafer_maps(test_maps_raw))

    transform = get_image_transforms()
    train_ds = WaferMapDataset(train_maps, train_labels, transform=transform)
    test_ds = WaferMapDataset(test_maps, test_labels, transform=None)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    fab_indices = partition_by_fab(train_labels, class_names, seed=42)
    partition_rows = describe_partition(fab_indices, train_labels, class_names)
    for row in partition_rows:
        logger.info("Fab %s: %s", row["fab"], row)

    # Match epoch budget: centralized runs for rounds * local_epochs total epochs
    total_epochs = args.rounds * args.local_epochs

    logger.info("=== Centralized baseline (%d epochs) ===", total_epochs)
    _, c_acc, c_f1, c_time = train_centralized(
        train_ds, test_loader, num_classes,
        total_epochs=total_epochs, lr=args.lr,
        batch_size=args.batch_size, device=args.device,
    )

    logger.info("=== FedAvg (%d rounds x %d local epochs x 3 fabs) ===",
                args.rounds, args.local_epochs)
    _, f_acc, f_f1, f_time, round_hist = train_federated(
        train_ds, fab_indices, test_loader, num_classes,
        rounds=args.rounds, local_epochs=args.local_epochs,
        lr=args.lr, batch_size=args.batch_size, device=args.device,
    )

    cent = {"accuracy": c_acc, "macro_f1": c_f1, "wall_time_s": c_time,
            "epochs": total_epochs}
    fed = {"accuracy": f_acc, "macro_f1": f_f1, "wall_time_s": f_time,
           "rounds": args.rounds, "local_epochs": args.local_epochs,
           "num_fabs": 3}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(json.dumps({
        "config": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "subsample": args.subsample,
        },
        "class_names": class_names,
        "partition": partition_rows,
        "centralized": cent,
        "federated": fed,
        "fedavg_round_history": round_hist,
    }, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    write_report(args.out_md, partition_rows, cent, fed, round_hist)
    logger.info("Wrote %s", args.out_md)
    logger.info("Centralized acc=%.4f f1=%.4f | FedAvg acc=%.4f f1=%.4f",
                c_acc, c_f1, f_acc, f_f1)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sys.exit(main())
