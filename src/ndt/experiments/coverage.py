"""Experiment D: empirical coverage of the distribution-free confidence band.

This is a direct check of Corollary 4.9. For each replication we draw two samples,
build the band, and ask how often it contains the true map ``T`` -- both pointwise
and simultaneously over the whole grid.
"""
from __future__ import annotations

import numpy as np

from ..bands import band_covers, joint_radii, plugin_band
from ..tasks import get_task
from .common import Config

__all__ = ["run"]

TASKS_FULL = ("N1", "A1")
TASKS_QUICK = ("N1",)
NS_FULL = (500, 2000, 10_000)
NS_QUICK = (500, 10_000)


def run(cfg: Config):
    rows = []
    tasks = TASKS_QUICK if cfg.quick else TASKS_FULL
    ns = NS_QUICK if cfg.quick else NS_FULL
    reps = 20 if cfg.quick else 200

    for key in tasks:
        task = get_task(key)
        lo, hi = task.range_on(0.05, 0.95)
        grid = np.linspace(lo, hi, 401)
        truth = task.T(grid)

        for n in ns:
            rng = np.random.default_rng(cfg.seed + 7 * n)
            point, simultaneous, widths = [], [], []

            for _ in range(reps):
                xs = task.sample_x(rng, n)
                ys = task.sample_y(rng, n)
                band = plugin_band(grid, xs, ys, alpha=cfg.alpha)
                covered, all_covered = band_covers(band, truth)
                point.append(float(np.mean(covered)))
                simultaneous.append(1.0 if all_covered else 0.0)
                width = band[1] - band[0]
                finite = np.isfinite(width)
                if np.any(finite):
                    widths.append(float(np.median(width[finite])))

            t_x, t_y = joint_radii(n, n, alpha=cfg.alpha)
            rows.append(
                dict(experiment="coverage", task=key, task_name=task.name, n=n, reps=reps,
                     alpha=cfg.alpha, nominal=1.0 - cfg.alpha,
                     point_coverage=float(np.mean(point)),
                     point_coverage_sd=float(np.std(point, ddof=1)),
                     simultaneous_rate=float(np.mean(simultaneous)),
                     median_width=float(np.median(widths)) if widths else float("nan"),
                     t_x=t_x, t_y=t_y, **cfg.meta)
            )
    return rows
