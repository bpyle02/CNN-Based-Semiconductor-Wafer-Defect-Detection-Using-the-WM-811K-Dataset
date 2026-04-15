"""Tests for src/training/distributed.py.

We can't spawn real NCCL/gloo process groups in CI, so we focus on the
helpers that are usable without an initialized process group, and exercise
``setup_distributed``/``wrap_model_distributed`` only to verify they don't
raise on entry with a CPU/gloo single-process config.
"""

import os

import pytest
import torch
import torch.nn as nn

from src.training import distributed as dist_mod


def test_is_distributed_false_by_default():
    assert dist_mod.is_distributed() is False


def test_get_rank_returns_zero_when_not_distributed():
    assert dist_mod.get_rank() == 0


def test_get_world_size_returns_one_when_not_distributed():
    assert dist_mod.get_world_size() == 1


def test_synchronize_is_noop_when_not_distributed():
    # Should not raise when no process group is initialized.
    dist_mod.synchronize()


def test_reduce_tensor_is_identity_when_not_distributed():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = dist_mod.reduce_tensor(t.clone(), world_size=1)
    assert torch.equal(out, t)


def test_wrap_model_dataparallel_single_device_returns_same_model():
    # With a single device_id, DataParallel should not be applied.
    model = nn.Linear(4, 4)
    wrapped = dist_mod.wrap_model_dataparallel(model, device_ids=[0])
    assert wrapped is model


def test_setup_distributed_cpu_single_process():
    """Initialize a gloo process group with world_size=1 on CPU and tear it down."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29555"
    try:
        dist_mod.setup_distributed(rank=0, world_size=1, backend="gloo")
        assert dist_mod.is_distributed() is True
        assert dist_mod.get_rank() == 0
        assert dist_mod.get_world_size() == 1
        # synchronize should also work now
        dist_mod.synchronize()
    finally:
        if torch.distributed.is_initialized():
            dist_mod.cleanup_distributed()
