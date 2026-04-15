"""Canonical DataLoader factories.

Consolidates the ~34 duplicated DataLoader construction sites across
``train.py``/``scripts/`` behind three helpers:

    create_train_loader(maps, labels, *, config, sampler=None, generator=None)
    create_val_loader(maps, labels, *, config)
    create_test_loader(maps, labels, *, config)

Defaults (``num_workers``, ``pin_memory``, ``persistent_workers``,
``prefetch_factor``, ``worker_init_fn=seed_worker``) come from ``config.data``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import WaferMapDataset, seed_worker


def _resolve_data_cfg(config: Any):
    """Extract the data sub-config from a full Config or accept a DataConfig."""
    if config is None:
        return None
    if hasattr(config, "data"):
        return config.data
    return config


def _base_kwargs(data_cfg, *, device: str = "cpu") -> dict:
    num_workers = getattr(data_cfg, "num_workers", 0) if data_cfg is not None else 0
    pin = bool(getattr(data_cfg, "pin_memory", False)) if data_cfg is not None else False
    kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin and str(device).startswith("cuda"),
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        # persistent_workers + prefetch_factor require num_workers > 0. They
        # cut per-epoch worker-respawn overhead (~10-20% on Colab Pro) so the
        # GPU doesn't stall between epochs waiting for DataLoader warmup.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def _as_dataset(maps, labels, transform) -> Dataset:
    if isinstance(maps, Dataset):
        return maps
    return WaferMapDataset(maps, labels, transform=transform)


def create_train_loader(
    maps,
    labels,
    *,
    config: Any,
    batch_size: Optional[int] = None,
    transform: Any = None,
    sampler=None,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> DataLoader:
    """Construct the training DataLoader with canonical defaults.

    If ``sampler`` is provided, it is used (no shuffle). Otherwise shuffle=True.
    """
    data_cfg = _resolve_data_cfg(config)
    bs = batch_size
    if bs is None and config is not None and hasattr(config, "training"):
        bs = getattr(config.training, "batch_size", None)
    if bs is None:
        bs = 64

    kwargs = _base_kwargs(data_cfg, device=device)
    kwargs["batch_size"] = bs
    if generator is not None:
        kwargs["generator"] = generator

    dataset = _as_dataset(maps, labels, transform)
    if sampler is not None:
        return DataLoader(dataset, sampler=sampler, **kwargs)
    return DataLoader(dataset, shuffle=True, **kwargs)


def create_val_loader(
    maps,
    labels,
    *,
    config: Any,
    batch_size: Optional[int] = None,
    transform: Any = None,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> DataLoader:
    """Construct the validation DataLoader (no shuffle)."""
    data_cfg = _resolve_data_cfg(config)
    bs = batch_size
    if bs is None and config is not None and hasattr(config, "training"):
        bs = getattr(config.training, "batch_size", None)
    if bs is None:
        bs = 64

    kwargs = _base_kwargs(data_cfg, device=device)
    kwargs["batch_size"] = bs
    if generator is not None:
        kwargs["generator"] = generator

    dataset = _as_dataset(maps, labels, transform)
    return DataLoader(dataset, shuffle=False, **kwargs)


def create_test_loader(
    maps,
    labels,
    *,
    config: Any,
    batch_size: Optional[int] = None,
    transform: Any = None,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> DataLoader:
    """Construct the test DataLoader (no shuffle)."""
    return create_val_loader(
        maps,
        labels,
        config=config,
        batch_size=batch_size,
        transform=transform,
        generator=generator,
        device=device,
    )


__all__ = [
    "create_train_loader",
    "create_val_loader",
    "create_test_loader",
]
