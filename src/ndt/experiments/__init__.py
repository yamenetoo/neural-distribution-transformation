"""The experiment suite.

Each module exposes ``run(cfg) -> list[dict]``; the runner serialises the rows.
"""
from __future__ import annotations

from . import ablation, atoms, convergence, coverage, fidelity, scaling
from .common import Config

EXPERIMENTS = {
    "fidelity": fidelity.run,
    "convergence": convergence.run,
    "ablation": ablation.run,
    "coverage": coverage.run,
    "atoms": atoms.run,
    "scaling": scaling.run,
}

__all__ = ["Config", "EXPERIMENTS", "run_one"]


def run_one(name: str, cfg: Config):
    """Run a single experiment by name."""
    try:
        runner = EXPERIMENTS[name]
    except KeyError:
        raise KeyError("unknown experiment %r; available: %s"
                       % (name, ", ".join(sorted(EXPERIMENTS))))
    return runner(cfg)
