"""Utility functions for reproducibility and calculating parameter/runtime metrics."""

import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Return the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())


class EpochTimer:
    """Records elapsed time for an epoch in seconds."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


class PeakMemoryTracker:
    """Records peak memory usage for a given epoch in megabytes."""

    def __enter__(self):
        self.peak_memory_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            self.peak_memory_mb = 0.0
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        elif torch.backends.mps.is_available():
            self.peak_memory_mb = torch.mps.current_allocated_memory() / 1024 ** 2
        else:
            self.peak_memory_mb = 0.0
