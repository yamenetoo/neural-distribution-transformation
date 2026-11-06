"""The problem zoo: transport problems with a known target map.

Every task exposes ``cdf_x``, ``ppf_x``, ``cdf_y``, ``ppf_y`` and samplers, so the
true map ``T = ppf_y o cdf_x`` is available in closed form for the error metrics.
``A1`` deliberately has an **atom** in the target law: the paper's theory allows
atoms, and this task checks that claim numerically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy import stats

__all__ = ["Task", "TASKS", "get_task", "TASK_KEYS"]


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    source: str
    target: str
    cdf_x: Callable
    ppf_x: Callable
    cdf_y: Callable
    ppf_y: Callable
    sample_x: Callable
    sample_y: Callable
    has_atom: bool = False
    tail_quantile: float = 0.01

    def T(self, x):
        """The quantile coupling, ``Q_Y(F_X(x))``."""
        return self.ppf_y(self.cdf_x(np.asarray(x, dtype=float)))

    def range_on(self, lo_q: float, hi_q: float):
        """The evaluation interval ``K = [Q_X(lo_q), Q_X(hi_q)]``."""
        return tuple(float(v) for v in self.ppf_x(np.array([lo_q, hi_q])))


def _std_normal(rng, n):
    return rng.normal(size=n)


def _chi2_map(x):
    # Q_Y(u) for chi^2_1 is (Phi^{-1}((1+u)/2))^2, hence T(x) != x^2.
    return stats.norm.ppf((1.0 + stats.norm.cdf(x)) / 2.0) ** 2


def _mixture_sample(rng, n, w=0.5, mu=2.0, sd=0.5):
    left = rng.random(n) < w
    return np.where(left, rng.normal(-mu, sd, n), rng.normal(mu, sd, n))


def _mixture_cdf(x, w=0.5, mu=2.0, sd=0.5):
    return w * stats.norm.cdf(x, -mu, sd) + (1.0 - w) * stats.norm.cdf(x, mu, sd)


def _mixture_ppf(u, w=0.5, mu=2.0, sd=0.5):
    """Bisection inversion on a bounded interval (vectorised)."""
    u = np.asarray(u, dtype=float)
    lo = np.full_like(u, -10.0)
    hi = np.full_like(u, 10.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        left = _mixture_cdf(mid, w, mu, sd) < u
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    return 0.5 * (lo + hi)


# --- atom task -------------------------------------------------------------
_ATOM_W = 0.3
_ATOM_AT = 1.0


def _atom_cdf(y):
    y = np.asarray(y, dtype=float)
    return (1.0 - _ATOM_W) * stats.norm.cdf(y) + _ATOM_W * (y >= _ATOM_AT)


def _atom_ppf(u):
    u = np.asarray(u, dtype=float)
    p_at = (1.0 - _ATOM_W) * stats.norm.cdf(_ATOM_AT)
    out = np.empty_like(u)
    below = u <= p_at
    inside = (u > p_at) & (u <= p_at + _ATOM_W)
    above = u > p_at + _ATOM_W
    out[below] = stats.norm.ppf(u[below] / (1.0 - _ATOM_W))
    out[inside] = _ATOM_AT
    out[above] = stats.norm.ppf((u[above] - _ATOM_W) / (1.0 - _ATOM_W))
    return out


def _atom_sample(rng, n):
    return _atom_ppf(rng.random(n))


TASKS: Dict[str, Task] = {
    "P1": Task(
        key="P1", name="Gaussian to log-normal", source="N(0,1)", target="LogNormal(0,1)",
        cdf_x=stats.norm.cdf, ppf_x=stats.norm.ppf,
        cdf_y=stats.lognorm(1).cdf, ppf_y=stats.lognorm(1).ppf,
        sample_x=_std_normal, sample_y=lambda rng, n: stats.lognorm(1).rvs(n, random_state=rng),
    ),
    "P2": Task(
        key="P2", name="Uniform to exponential", source="Unif(0,1)", target="Exp(1)",
        cdf_x=stats.uniform.cdf, ppf_x=stats.uniform.ppf,
        cdf_y=stats.expon.cdf, ppf_y=stats.expon.ppf,
        sample_x=lambda rng, n: rng.random(n),
        sample_y=lambda rng, n: rng.exponential(size=n),
    ),
    "P3": Task(
        key="P3", name="Gaussian to chi-square(1)", source="N(0,1)", target="chi2(1)",
        cdf_x=stats.norm.cdf, ppf_x=stats.norm.ppf,
        cdf_y=stats.chi2(1).cdf, ppf_y=stats.chi2(1).ppf,
        sample_x=_std_normal,
        sample_y=lambda rng, n: rng.chisquare(1, size=n),
    ),
    "P4": Task(
        key="P4", name="Gaussian shift and scale", source="N(0,1)", target="N(3,2^2)",
        cdf_x=stats.norm.cdf, ppf_x=stats.norm.ppf,
        cdf_y=stats.norm(3, 2).cdf, ppf_y=stats.norm(3, 2).ppf,
        sample_x=_std_normal, sample_y=lambda rng, n: rng.normal(3, 2, n),
    ),
    "N1": Task(
        key="N1", name="Bimodal mixture to Gaussian", source="0.5N(-2,.25)+0.5N(2,.25)",
        target="N(0,1)",
        cdf_x=_mixture_cdf, ppf_x=_mixture_ppf,
        cdf_y=stats.norm.cdf, ppf_y=stats.norm.ppf,
        sample_x=_mixture_sample, sample_y=_std_normal, tail_quantile=0.05,
    ),
    "N2": Task(
        key="N2", name="Gamma to uniform", source="Gamma(2,1)", target="Unif(0,1)",
        cdf_x=stats.gamma(2).cdf, ppf_x=stats.gamma(2).ppf,
        cdf_y=stats.uniform.cdf, ppf_y=stats.uniform.ppf,
        sample_x=lambda rng, n: rng.gamma(2.0, size=n),
        sample_y=lambda rng, n: rng.random(n), tail_quantile=0.05,
    ),
    "N3": Task(
        key="N3", name="Beta to Gamma", source="Beta(2,5)", target="Gamma(3,1)",
        cdf_x=stats.beta(2, 5).cdf, ppf_x=stats.beta(2, 5).ppf,
        cdf_y=stats.gamma(3).cdf, ppf_y=stats.gamma(3).ppf,
        sample_x=lambda rng, n: rng.beta(2, 5, n),
        sample_y=lambda rng, n: rng.gamma(3.0, size=n), tail_quantile=0.05,
    ),
    "A1": Task(
        key="A1", name="Gaussian to Gaussian + atom", source="N(0,1)",
        target="0.7N(0,1) + 0.3*delta_1",
        cdf_x=stats.norm.cdf, ppf_x=stats.norm.ppf,
        cdf_y=_atom_cdf, ppf_y=_atom_ppf,
        sample_x=_std_normal, sample_y=_atom_sample, has_atom=True, tail_quantile=0.05,
    ),
}

TASK_KEYS = tuple(TASKS)


def get_task(key: str) -> Task:
    try:
        return TASKS[key]
    except KeyError:
        raise KeyError("unknown task %r; available: %s" % (key, ", ".join(TASK_KEYS)))
