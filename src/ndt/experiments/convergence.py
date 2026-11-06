"""Experiment B: learning T from two independent, unpaired samples.

Reports the uniform error over replications and a least-squares slope of
``log(sup error)`` on ``log(n)``, with a bootstrap interval over replications.
"""
from __future__ import annotations

import numpy as np

from ..fit import fit_msn_batch
from ..tasks import get_task
from .common import Config, empirical_quantile, quantile_grid

__all__ = ["run"]

TASKS_FULL = ("N1", "N2", "N3", "A1")
TASKS_QUICK = ("N1", "A1")
NS_FULL = (300, 1000, 3000, 10_000, 30_000)
NS_QUICK = (300, 3000, 30_000)


def _slope(ns, values):
    return float(np.polyfit(np.log(ns), np.log(values), 1)[0])


def run(cfg: Config):
    rows = []
    tasks = TASKS_QUICK if cfg.quick else TASKS_FULL
    ns = NS_QUICK if cfg.quick else NS_FULL
    reps = 5 if cfg.quick else cfg.reps
    n_train = 2000 if cfg.quick else cfg.n_train
    alpha = cfg.alpha

    for key in tasks:
        task = get_task(key)
        lo, hi = task.range_on(0.05, 0.95)
        grid = np.linspace(lo, hi, 20001)
        truth = task.T(grid)
        u = quantile_grid(n_train, np.random.default_rng(0))

        per_n = {}
        for n in ns:
            rng = np.random.default_rng(cfg.seed + n)
            pairs = []
            for _ in range(reps):
                xs = np.sort(task.sample_x(rng, n))
                ys = np.sort(task.sample_y(rng, n))
                pairs.append((empirical_quantile(xs, u), empirical_quantile(ys, u)))

            models = fit_msn_batch(pairs, m=cfg.m, n_jobs=cfg.jobs)
            sups = [float(np.max(np.abs(model.predict(grid) - truth))) for model in models]
            per_n[n] = sups
            rows.append(
                dict(experiment="convergence", task=key, task_name=task.name, n=n,
                     reps=reps, m=cfg.m, mean=float(np.mean(sups)),
                     sd=float(np.std(sups, ddof=1)) if reps > 1 else 0.0,
                     median=float(np.median(sups)), **cfg.meta)
            )

        # log-log slope with a bootstrap interval over replications
        means = [float(np.mean(per_n[n])) for n in ns]
        point = _slope(ns, means)
        rng = np.random.default_rng(cfg.seed)
        boot = []
        for _ in range(500):
            sampled = [float(np.mean(rng.choice(per_n[n], size=reps, replace=True))) for n in ns]
            if all(v > 0 for v in sampled):
                boot.append(_slope(ns, sampled))
        lo_ci, hi_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
            if len(boot) > 10 else (float("nan"), float("nan"))

        rows.append(
            dict(experiment="convergence_slope", task=key, task_name=task.name,
                 slope=point, slope_lo=lo_ci, slope_hi=hi_ci, reps=reps, m=cfg.m,
                 alpha=alpha, **cfg.meta)
        )
    return rows
