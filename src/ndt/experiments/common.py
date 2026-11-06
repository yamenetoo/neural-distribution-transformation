"""Shared experiment configuration and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

__all__ = ["Config", "quantile_grid", "empirical_quantile"]


@dataclass
class Config:
    """Knobs shared by every experiment."""

    quick: bool = False
    jobs: int = 1
    device: str = "cpu"
    seed: int = 20261015
    m: int = 64
    n_train: int = 4000
    mc: int = 100_000
    reps: int = 40
    alpha: float = 0.05

    #: filled in by the runner so every row records where it came from
    meta: Dict[str, object] = field(default_factory=dict)

    def scaled(self, name: str, full, quick):
        """Pick ``quick`` or ``full`` according to the mode."""
        return quick if self.quick else full


def quantile_grid(n: int, rng, offset: float = 0.0) -> np.ndarray:
    """The deterministic grid ``u_j = j/(n+1)`` (optionally shifted)."""
    return (np.arange(1, n + 1) + offset) / (n + 1)


def empirical_quantile(sorted_samples: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``Q_n(u) = X_(ceil(n u))`` for a sorted sample."""
    n = sorted_samples.size
    idx = np.ceil(n * np.asarray(u, dtype=float)).astype(int) - 1
    return sorted_samples[np.clip(idx, 0, n - 1)]


def summarise(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }
