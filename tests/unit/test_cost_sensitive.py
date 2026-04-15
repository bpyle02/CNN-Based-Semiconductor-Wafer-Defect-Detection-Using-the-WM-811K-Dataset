"""Unit tests for CostSensitiveCE loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.training.losses import (
    CostSensitiveCE,
    build_classification_loss,
    build_cost_matrix_wm811k,
)


def test_identity_cost_matches_vanilla_ce():
    """cost_matrix = one-hot(y) at position y (0 on diag, 0 off-diag except
    row-sum = 1 at target column) reduces to standard cross-entropy.

    More simply: with cost[i, j] = 1 iff j == i else 0, the per-sample loss
    becomes -log p_y, exactly vanilla CE (reduction='mean').
    """
    torch.manual_seed(0)
    num_classes = 9
    batch = 64
    logits = torch.randn(batch, num_classes)
    targets = torch.randint(0, num_classes, (batch,))

    identity_cost = torch.eye(num_classes)  # cost[y, y] = 1, else 0
    loss_cs = CostSensitiveCE(cost_matrix=identity_cost, reduction="mean")
    vanilla = F.cross_entropy(logits, targets, reduction="mean")

    assert torch.allclose(loss_cs(logits, targets), vanilla, atol=1e-6)

    # Also verify via the builder path.
    loss_via_builder = build_classification_loss(
        "CostSensitiveCE", cost_matrix=identity_cost
    )
    assert torch.allclose(loss_via_builder(logits, targets), vanilla, atol=1e-6)


def test_custom_cost_penalizes_specified_pairs():
    """Raising cost[i, j] for a specific (true, pred) pair must strictly
    increase the loss when predictions favor that wrong class for true=i."""
    num_classes = 3
    batch = 8

    # All samples have true label 0; logits strongly favor predicting class 1.
    targets = torch.zeros(batch, dtype=torch.long)
    logits = torch.zeros(batch, num_classes)
    logits[:, 1] = 5.0  # peak on wrong class 1
    logits[:, 2] = 0.0  # low on class 2

    # Baseline: uniform off-diagonal cost = 1.
    base_cost = 1.0 - torch.eye(num_classes)  # 0 diag, 1 elsewhere
    base_loss = CostSensitiveCE(cost_matrix=base_cost)(logits, targets)

    # Penalize class 0 -> class 1 mistakes 10x more.
    pen_cost = base_cost.clone()
    pen_cost[0, 1] = 10.0
    pen_loss = CostSensitiveCE(cost_matrix=pen_cost)(logits, targets)

    assert pen_loss > base_loss, (
        f"Penalizing (0, 1) should raise loss; base={base_loss:.4f} "
        f"pen={pen_loss:.4f}"
    )

    # Sanity-check the WM-811K builder puts 10x on Near-full rows.
    class_names = [
        "Center", "Donut", "Edge-Loc", "Edge-Ring",
        "Loc", "Near-full", "Random", "Scratch", "none",
    ]
    m = build_cost_matrix_wm811k(class_names)
    nf = class_names.index("Near-full")
    # Diagonal must be zero; off-diagonal of Near-full row must be 10.
    assert float(m[nf, nf]) == 0.0
    off = [float(m[nf, j]) for j in range(len(class_names)) if j != nf]
    assert all(v == 10.0 for v in off), off
