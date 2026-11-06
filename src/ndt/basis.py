"""Core linear-algebra pieces: knot placement, design matrix, evaluation.

The monotone spline network (MSN) is

    N(x) = c + sum_j s_j [ relu(x - t_j) - relu(x - t_{j+1}) ],   s_j >= 0,

with knots t_0 < ... < t_m. Each bracketed term equals
``clip(x, t_j, t_{j+1}) - t_j``, which is the basis function used in the design
matrix. Because every s_j is non-negative, N is non-decreasing for every
parameter value, and because the basis is linear in the parameters the fit is a
convex non-negative least-squares problem.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_knots", "design_matrix", "ramp_predict", "relu_predict", "is_monotone"]


def make_knots(x, m, scheme: str = "quantile", target=None):
    """Knots for ``m`` intervals.

    Parameters
    ----------
    x : array_like
        Training inputs (or any sample from the source law).
    m : int
        Number of intervals; ``m + 1`` knots are returned (duplicates removed).
    scheme : {"quantile", "uniform", "equal-increment"}
        ``quantile`` puts knots at empirical quantiles of ``x``;
        ``uniform`` spaces them evenly on the range of ``x``;
        ``equal-increment`` picks them so that the *target* increases by a
        constant amount on each interval (needs ``target`` and a monotone map).
    target : callable, optional
        The true monotone map, required for ``equal-increment``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if m < 1:
        raise ValueError("m must be >= 1")

    if scheme == "quantile":
        knots = np.quantile(x, np.linspace(0.0, 1.0, m + 1))
    elif scheme == "uniform":
        knots = np.linspace(x.min(), x.max(), m + 1)
    elif scheme == "equal-increment":
        if target is None:
            raise ValueError("scheme='equal-increment' requires target=...")
        grid = np.linspace(x.min(), x.max(), 4097)
        values = np.asarray(target(grid), dtype=float)
        # T is non-decreasing, so invert it by interpolation on (value, grid).
        keep = np.concatenate([[True], np.diff(values) > 0])
        levels = np.linspace(values[keep][0], values[keep][-1], m + 1)
        knots = np.interp(levels, values[keep], grid[keep])
    else:
        raise ValueError("unknown knot scheme %r" % (scheme,))

    return np.unique(np.asarray(knots, dtype=float))


def design_matrix(x, knots):
    """Columns: the free-sign intercept pair ``[1, -1]``, then one ramp per interval.

    Two intercept columns let NNLS produce a coefficient of either sign: it
    returns non-negative weights for both, and the intercept is their difference.
    """
    x = np.asarray(x, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    lo, hi = knots[:-1], knots[1:]
    ramps = np.clip(x[:, None], lo[None, :], hi[None, :]) - lo[None, :]
    return np.hstack([np.ones((x.size, 1)), -np.ones((x.size, 1)), ramps])


def _coef_split(coef):
    coef = np.asarray(coef, dtype=float).ravel()
    return coef[0] - coef[1], coef[2:]


def ramp_predict(x, knots, coef):
    """Fast O(log m) evaluation: the network is piecewise linear through the knots."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float).ravel()
    c, s = _coef_split(coef)
    if knots.size < 2:
        return np.full_like(x, c, dtype=float)
    values = np.concatenate([[c], c + np.cumsum(s * np.diff(knots))])
    # np.interp clamps outside the knots, which is exactly the ReLU form's
    # behaviour: constant below t_0 and above t_m.
    return np.interp(x, knots, values)


def relu_predict(x, knots, coef):
    """Literal evaluation as a sum of ReLUs; used to verify the fast path."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float).ravel()
    c, s = _coef_split(coef)
    out = np.full(x.shape, c, dtype=float)
    for j, weight in enumerate(s):
        if weight == 0.0:
            continue
        out += weight * (np.maximum(x - knots[j], 0.0) - np.maximum(x - knots[j + 1], 0.0))
    return out


def is_monotone(x, knots, coef, tol: float = 1e-9) -> bool:
    """Check non-decreasingness on the grid ``x``."""
    values = ramp_predict(np.sort(np.asarray(x, dtype=float)), knots, coef)
    return bool(np.all(np.diff(values) >= -tol))
