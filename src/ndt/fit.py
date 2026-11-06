"""Fitting: convex non-negative least squares, optionally over many replications."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import nnls

from .basis import design_matrix, make_knots
from .network import MSN

__all__ = ["fit_msn", "fit_msn_batch", "solve_nnls"]


def solve_nnls(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares with a generous iteration cap."""
    coef, _ = nnls(A, y, maxiter=50 * A.shape[1])
    return coef


def fit_msn(x, y, m: int = 64, scheme: str = "quantile", target=None) -> MSN:
    """Fit a monotone spline network with ``m`` intervals.

    Parameters
    ----------
    x, y : array_like
        Quantile-coupled pairs ``y_i = T(x_i)`` (in the parametric setting) or
        empirical quantiles of two independent samples (in the data-driven one).
    m : int
        Number of intervals ("knots" in the paper's notation).
    scheme : {"quantile", "uniform", "equal-increment"}
        Knot placement; see :func:`ndt.basis.make_knots`.
    target : callable, optional
        Required for the ``equal-increment`` scheme.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    knots = make_knots(x, m, scheme=scheme, target=target)
    coef = solve_nnls(design_matrix(x, knots), y)
    return MSN(knots, coef)


def fit_msn_batch(pairs: Iterable[tuple], m: int = 64, scheme: str = "quantile",
                  target=None, n_jobs: int = 1) -> list:
    """Fit one network per ``(x, y)`` pair, optionally in parallel.

    Uses processes, so the scipy NNLS solver releases the GIL independently in
    each worker. For a single fit the overhead dominates; this pays off once
    there are tens of replications.
    """
    pairs = list(pairs)
    if not pairs:
        return []

    if n_jobs in (1, None):
        return [fit_msn(x, y, m=m, scheme=scheme, target=target) for x, y in pairs]

    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover - depends on the install
        return [fit_msn(x, y, m=m, scheme=scheme, target=target) for x, y in pairs]

    return Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(fit_msn)(x, y, m=m, scheme=scheme, target=target) for x, y in pairs
    )
