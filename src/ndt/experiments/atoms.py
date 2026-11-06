"""Experiment E: a target law with an atom.

The theory allows atoms in the target, where the quantile coupling has a jump. This
experiment checks that the fitted network reproduces the jump and puts the right
amount of mass on the atom, which no density-based method can represent.
"""
from __future__ import annotations

import numpy as np

from ..fit import fit_msn
from ..metrics import ks_distance, sup_error
from ..tasks import get_task
from .common import Config, quantile_grid

__all__ = ["run"]

ATOM = 1.0
ATOM_MASS = 0.3


def run(cfg: Config):
    rows = []
    task = get_task("A1")
    ms = (16, 64) if cfg.quick else (16, 64, 256)
    n_mc = 20_000 if cfg.quick else cfg.mc

    rng = np.random.default_rng(cfg.seed)
    u = quantile_grid(2000 if cfg.quick else cfg.n_train, rng)
    x_train, y_train = task.ppf_x(u), task.ppf_y(u)
    lo, hi = task.range_on(task.tail_quantile, 1.0 - task.tail_quantile)

    x_mc = task.sample_x(rng, n_mc)
    y_exact = task.T(x_mc)
    mass_exact = float(np.mean(np.isclose(y_exact, ATOM, atol=1e-12)))

    for m in ms:
        model = fit_msn(x_train, y_train, m=m)
        y_hat = model.predict(x_mc)
        mass_hat = float(np.mean(np.isclose(y_hat, ATOM, atol=1e-9)))
        rows.append(
            dict(experiment="atoms", task="A1", task_name=task.name, m=m,
                 atom_value=ATOM, atom_mass_theory=ATOM_MASS,
                 atom_mass_exact=mass_exact, atom_mass_fitted=mass_hat,
                 ks=ks_distance(y_hat, task.cdf_y), ks_exact=ks_distance(y_exact, task.cdf_y),
                 sup=sup_error(model, task, lo, hi),
                 monotone=model.is_monotone(np.linspace(lo, hi, 5001)),
                 n_mc=n_mc, **cfg.meta)
        )
    return rows
