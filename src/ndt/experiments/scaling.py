"""Experiment F: evaluation throughput and parallel fitting.

Two costs matter once a network is fitted: evaluating it at many points (vectorised,
GPU-friendly) and fitting many networks (embarrassingly parallel over replications).
"""
from __future__ import annotations

import time

import numpy as np

from ..fit import fit_msn, fit_msn_batch
from ..network import MSN, resolve_device, torch_predict, torch_predict_many
from ..tasks import get_task
from .common import Config, quantile_grid

__all__ = ["run"]

SIZES_FULL = (100_000, 1_000_000, 10_000_000)
SIZES_QUICK = (100_000, 1_000_000)


def _time(fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def run(cfg: Config):
    rows = []
    device = resolve_device(cfg.device)
    task = get_task("P3")
    u = quantile_grid(cfg.n_train, np.random.default_rng(0))
    model = fit_msn(task.ppf_x(u), task.ppf_y(u), m=cfg.m)

    sizes = SIZES_QUICK if cfg.quick else SIZES_FULL
    repeat = 2 if cfg.quick else 3

    # ---- single-network evaluation: numpy vs device --------------------
    for n in sizes:
        x = np.linspace(-2.3, 2.3, n)
        t_numpy = _time(lambda: model.predict(x), repeat=repeat)
        rows.append(dict(experiment="scaling_eval", backend="numpy", device="cpu", n=n,
                         seconds=t_numpy, speedup=1.0, n_models=1, **cfg.meta))
        try:
            torch_predict(model, x[:1000], device=device)  # warm up / init context
            t_dev = _time(lambda: torch_predict(model, x, device=device), repeat=repeat)
            rows.append(dict(experiment="scaling_eval", backend="torch", device=device, n=n,
                             seconds=t_dev, speedup=t_numpy / t_dev, n_models=1, **cfg.meta))
        except ImportError:
            pass

    # ---- many networks on the same inputs ------------------------------
    n_points = 200_000 if cfg.quick else 1_000_000
    n_models = 16 if cfg.quick else 64
    rng = np.random.default_rng(cfg.seed)
    knots = model.knots
    models = [MSN(knots, np.concatenate([[0.0, 0.0], rng.random(knots.size - 1) * 0.01]))
              for _ in range(n_models)]
    x = np.linspace(-2.3, 2.3, n_points)

    t_loop = _time(lambda: np.vstack([m.predict(x) for m in models]), repeat=repeat)
    rows.append(dict(experiment="scaling_many", backend="numpy-loop", device="cpu",
                     n=n_points, n_models=n_models, seconds=t_loop, speedup=1.0, **cfg.meta))
    try:
        torch_predict_many(models, x[:1000], device=device)  # warm up
        t_batched = _time(lambda: torch_predict_many(models, x, device=device), repeat=repeat)
        rows.append(dict(experiment="scaling_many", backend="torch-batched", device=device,
                         n=n_points, n_models=n_models, seconds=t_batched,
                         speedup=t_loop / t_batched, **cfg.meta))
    except ImportError:
        pass

    # ---- parallel fitting ---------------------------------------------
    n_fits = 16 if cfg.quick else 64
    pairs = [(task.ppf_x(u) + 0.01 * j, task.ppf_y(u) + 0.01 * j) for j in range(n_fits)]
    job_grid = (1, 4) if cfg.quick else (1, 2, 4, 8, 24)

    baseline = None
    for jobs in job_grid:
        t_par = _time(lambda: fit_msn_batch(pairs, m=cfg.m, n_jobs=jobs), repeat=1)
        baseline = t_par if baseline is None else baseline
        rows.append(dict(experiment="scaling_fit", backend="joblib", device="cpu",
                         n_jobs=jobs, n_fits=n_fits, seconds=t_par,
                         speedup=baseline / t_par, **cfg.meta))
    return rows
