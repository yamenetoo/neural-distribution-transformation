"""Experiment A: approximation fidelity when both laws are known."""
from __future__ import annotations

import numpy as np

from ..fit import fit_msn
from ..metrics import evaluate, ks_distance
from ..tasks import TASK_KEYS, get_task
from .common import Config, quantile_grid

__all__ = ["run"]

MS_FULL = (8, 16, 32, 64, 128, 256)
MS_QUICK = (16, 64)


def run(cfg: Config):
    rows = []
    ms = MS_QUICK if cfg.quick else MS_FULL

    for key in TASK_KEYS:
        task = get_task(key)
        rng = np.random.default_rng(cfg.seed)
        n_train = 2000 if cfg.quick else cfg.n_train
        n_mc = 20_000 if cfg.quick else cfg.mc

        u = quantile_grid(n_train, rng)
        x_train = task.ppf_x(u)
        y_train = task.ppf_y(u)

        lo, hi = task.range_on(task.tail_quantile, 1.0 - task.tail_quantile)
        grid = np.linspace(lo, hi, 20001)
        x_mc = task.sample_x(rng, n_mc)

        # Monte Carlo noise floor: the exact map applied to the same draws
        floor = ks_distance(task.T(x_mc), task.cdf_y)

        for m in ms:
            model = fit_msn(x_train, y_train, m=m)
            rep = evaluate(model, task, lo, hi, x_mc)
            # literal ReLU evaluation must agree with the fast path
            sample = grid[::997]
            relu_gap = float(np.max(np.abs(model.predict_relu(sample) - model.predict(sample))))
            rows.append(
                dict(rep, experiment="fidelity", task=key, task_name=task.name,
                     m=m, n_train=n_train, n_mc=n_mc, ks_floor=floor,
                     relu_gap=relu_gap, has_atom=task.has_atom, **cfg.meta)
            )
    return rows
