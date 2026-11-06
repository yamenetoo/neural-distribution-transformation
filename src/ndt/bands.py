"""Distribution-free confidence bands (Corollary 4.9 of the paper).

The band uses only empirical distribution functions and the DKW--Massart
inequality, so it needs no density, continuity or moment assumptions:

    Q_Yhat(F_hat(x) - t_X - t_Y)  <=  T(x)  <=  Q_Yhat(F_hat(x) + t_X + t_Y)

with ``t_X = sqrt(log(4/alpha) / (2 n_X))`` and the same for ``Y``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["dkw_radius", "joint_radii", "Empirical", "plugin_band", "band_covers"]


def dkw_radius(n: int, alpha: float = 0.05) -> float:
    """DKW--Massart half-width: ``P(sup |F_n - F| > t) <= 2 exp(-2 n t^2)``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def joint_radii(n_x: int, n_y: int, alpha: float = 0.05):
    """Radii for a simultaneous band over two independent samples.

    Splits ``alpha`` evenly and applies a union bound, giving the paper's
    ``sqrt(log(4/alpha) / (2n))``.
    """
    return dkw_radius(n_x, alpha / 2.0), dkw_radius(n_y, alpha / 2.0)


class Empirical:
    """Empirical distribution and quantile functions with extended conventions."""

    def __init__(self, samples):
        self.sorted = np.sort(np.asarray(samples, dtype=float).ravel())
        self.n = self.sorted.size

    def cdf(self, y):
        return np.searchsorted(self.sorted, np.asarray(y, dtype=float), side="right") / self.n

    def ppf(self, u):
        """Generalised inverse; returns -inf at 0 and +inf above 1."""
        u = np.asarray(u, dtype=float)
        out = np.empty_like(u)
        finite = (u > 0.0) & (u <= 1.0)
        idx = np.ceil(self.n * u[finite]).astype(int) - 1
        out[finite] = self.sorted[np.clip(idx, 0, self.n - 1)]
        out[u <= 0.0] = -np.inf
        out[u > 1.0] = np.inf
        return out


def plugin_band(x, x_sample, y_sample, alpha: float = 0.05):
    """Simultaneous band for ``T(x)`` at the points ``x``."""
    fx = Empirical(x_sample)
    fy = Empirical(y_sample)
    t_x, t_y = joint_radii(fx.n, fy.n, alpha=alpha)
    rank = fx.cdf(x)
    return fy.ppf(rank - t_x - t_y), fy.ppf(rank + t_x + t_y), (t_x, t_y)


def band_covers(band, truth):
    """Per-point coverage and the simultaneous (all-points) indicator."""
    lo, hi, _ = band
    truth = np.asarray(truth, dtype=float)
    covered = (lo <= truth) & (truth <= hi)
    return covered, bool(np.all(covered))
