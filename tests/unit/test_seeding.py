"""Tests for src/utils/seeding.py (centralized seed management)."""

import os
import random

import numpy as np
import pytest
import torch

from src.utils.seeding import seed_everything


def test_seed_everything_makes_torch_randn_reproducible():
    """Calling seed_everything with the same seed must produce identical torch.randn outputs."""
    seed_everything(1234)
    a = torch.randn(8, 8)
    seed_everything(1234)
    b = torch.randn(8, 8)
    assert torch.equal(a, b)


def test_different_seeds_produce_different_outputs():
    seed_everything(1)
    a = torch.randn(16)
    seed_everything(2)
    b = torch.randn(16)
    assert not torch.equal(a, b)


def test_seed_everything_also_seeds_python_random_and_numpy():
    seed_everything(7)
    py_a = random.random()
    np_a = np.random.rand(4)

    seed_everything(7)
    py_b = random.random()
    np_b = np.random.rand(4)

    assert py_a == py_b
    assert np.array_equal(np_a, np_b)


def test_seed_everything_sets_pythonhashseed():
    seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_seed_everything_default_seed():
    seed_everything()
    a = torch.randn(4)
    seed_everything(42)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_seed_everything_rejects_non_int():
    with pytest.raises(ValueError):
        seed_everything("42")  # type: ignore[arg-type]


def test_seed_everything_rejects_negative():
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seed_everything_rejects_too_large():
    with pytest.raises(ValueError):
        seed_everything(2**32)


def test_seed_everything_is_idempotent():
    seed_everything(555)
    state1 = torch.get_rng_state()
    seed_everything(555)
    state2 = torch.get_rng_state()
    assert torch.equal(state1, state2)
