"""Error metrics for a fitted transport map.

All metrics compare the fitted network ``N`` against the true map ``T`` or against
the target law ``G``; none of them assumes a density unless stated.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

#: ``np.trapezoid`` is the NumPy 2 name; keep working on older NumPy too
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "sup_error",
    "mse_error",
    "ks_distance",
    "w1_to_law",
    "w2_quantile",
    "evaluate",
]


def sup_error(model, task, lo: float, hi: float, n: int = 20001) -> float:
    """``sup_{x in [lo,hi]} |N(x) - T(x)|`` on a fine grid."""
    x = np.linspace(lo, hi, n)
    return float(np.max(np.abs(model.predict(x) - task.T(x))))


def mse_error(model, task, x_sample, lo: float, hi: float) -> float:
    """``E[(N(X) - T(X))^2 1{X in [lo,hi]}]`` estimated from a sample of X."""
    x_sample = np.asarray(x_sample, dtype=float)
    inside = (x_sample >= lo) & (x_sample <= hi)
    if not np.any(inside):
        return float("nan")
    diff = model.predict(x_sample[inside]) - task.T(x_sample[inside])
    return float(np.mean(diff ** 2))


def ks_distance(samples, cdf) -> float:
    """Kolmogorov--Smirnov statistic between a sample and a distribution function."""
    samples = np.asarray(samples, dtype=float)
    return float(stats.kstest(samples, cdf).statistic)


def w1_to_law(samples, task, lo: float, hi: float, n: int = 40001) -> float:
    """Wasserstein-1 distance between the law of N(X) and G, by integrating |ECDF - G|.

    Restricted to ``[lo, hi]``; the tails are not represented.
    """
    samples = np.sort(np.asarray(samples, dtype=float))
    grid = np.linspace(lo, hi, n)
    ecdf = np.searchsorted(samples, grid, side="right") / samples.size
    return float(_trapz(np.abs(ecdf - task.cdf_y(grid)), grid))


def w2_quantile(model, task, u_lo: float = 1e-4, u_hi: float = 1 - 1e-4,
                n: int = 20001) -> float:
    """``W2^2`` between the law of N(X) and G, from the quantile representation.

    Uses ``W2^2 = int_0^1 (Q_A(u) - Q_B(u))^2 du`` with ``Q_A = N o Q_X``.
    """
    u = np.linspace(u_lo, u_hi, n)
    qx = task.ppf_x(u)
    qy = task.ppf_y(u)
    diff = model.predict(qx) - qy
    return float(np.sqrt(_trapz(diff ** 2, u)))


def evaluate(model, task, lo: float, hi: float, x_mc, tail_quantile: float = 1e-4):
    """The standard per-network report used by the experiment modules."""
    y_hat = model.predict(x_mc)
    return {
        "sup": sup_error(model, task, lo, hi),
        "mse": mse_error(model, task, x_mc, lo, hi),
        "ks": ks_distance(y_hat, task.cdf_y),
        "w1": w1_to_law(y_hat, task, lo, hi),
        "w2": w2_quantile(model, task, u_lo=tail_quantile, u_hi=1 - tail_quantile),
        "monotone": model.is_monotone(np.linspace(lo, hi, 5001)),
    }
