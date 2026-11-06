"""Experiment C: ablations over knot placement and training-grid size."""
from __future__ import annotations

import numpy as np

from ..fit import fit_msn
from ..metrics import sup_error
from ..tasks import get_task
from .common import Config, empirical_quantile, quantile_grid

__all__ = ["run"]

SCHEMES = ("quantile", "uniform", "equal-increment")
MS = (16, 64, 256)
GRID_SIZES = (250, 1000, 4000, 16_000)


def run(cfg: Config):
    rows = []
    ms = (16, 64) if cfg.quick else MS
    grids = (250, 4000) if cfg.quick else GRID_SIZES
    schemes = ("quantile", "equal-increment") if cfg.quick else SCHEMES
    tasks = ("N1",) if cfg.quick else ("N1", "P1")

    # --- knot placement (known laws) ------------------------------------
    for key in tasks:
        task = get_task(key)
        lo, hi = task.range_on(task.tail_quantile, 1.0 - task.tail_quantile)
        u = quantile_grid(cfg.n_train, np.random.default_rng(0))
        x_train, y_train = task.ppf_x(u), task.ppf_y(u)
        for scheme in schemes:
            for m in ms:
                model = fit_msn(x_train, y_train, m=m, scheme=scheme, target=task.T)
                rows.append(
                    dict(experiment="ablation_knots", task=key, task_name=task.name,
                         scheme=scheme, m=m, n_train=cfg.n_train,
                         sup=sup_error(model, task, lo, hi),
                         n_intervals=model.n_intervals, **cfg.meta)
                )

    # --- training-grid size (two unpaired samples) ----------------------
    key = "N1"
    task = get_task(key)
    lo, hi = task.range_on(0.05, 0.95)
    n = 3000
    rng = np.random.default_rng(cfg.seed)
    xs = np.sort(task.sample_x(rng, n))
    ys = np.sort(task.sample_y(rng, n))
    for n_grid in grids:
        u = quantile_grid(n_grid, rng)
        model = fit_msn(empirical_quantile(xs, u), empirical_quantile(ys, u), m=cfg.m)
        rows.append(
            dict(experiment="ablation_grid", task=key, task_name=task.name,
                 n_grid=n_grid, n=3000, m=cfg.m, sup=sup_error(model, task, lo, hi),
                 **cfg.meta)
        )
    return rows
