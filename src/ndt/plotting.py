"""Figures from the experiment CSV files."""
from __future__ import annotations

import csv
import os

__all__ = ["make_all"]


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def convergence_figure(path, out_dir):
    plt = _mpl()
    rows = [r for r in _read(path) if r["experiment"] == "convergence"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = sorted((int(r["n"]), _num(r["mean"]), _num(r["sd"]))
                     for r in rows if r["task"] == task)
        n = [s[0] for s in sub]
        mean = [s[1] for s in sub]
        sd = [s[2] for s in sub]
        ax.errorbar(n, mean, yerr=sd, marker="o", capsize=2, label=task)
    ns = sorted({int(r["n"]) for r in rows})
    if len(ns) > 1:
        ref = [0.25 * (n / ns[0]) ** -0.5 * max(1.0, _num(rows[0].get("m"), 1) / 64)
               for n in ns]
        ax.plot(ns, ref, "k--", lw=1, label=r"$n^{-1/2}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size $n=n_X=n_Y$")
    ax.set_ylabel(r"$\sup_K|\hat N - T|$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    path_out = os.path.join(out_dir, "convergence.png")
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    return path_out


def fidelity_figure(path, out_dir):
    plt = _mpl()
    rows = [r for r in _read(path) if r["experiment"] == "fidelity"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for task in sorted({r["task"] for r in rows}):
        sub = sorted((int(r["m"]), _num(r["sup"])) for r in rows if r["task"] == task)
        ax.plot([s[0] for s in sub], [s[1] for s in sub], marker="o", label=task)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("knots $m$")
    ax.set_ylabel(r"$\sup_K|\hat N - T|$")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    path_out = os.path.join(out_dir, "fidelity.png")
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    return path_out


def coverage_figure(path, out_dir):
    plt = _mpl()
    rows = [r for r in _read(path) if r["experiment"] == "coverage"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for task in sorted({r["task"] for r in rows}):
        sub = sorted((int(r["n"]), _num(r["point_coverage"]), _num(r["simultaneous_rate"]))
                     for r in rows if r["task"] == task)
        ax.plot([s[0] for s in sub], [s[1] for s in sub], marker="o", label="%s point" % task)
        ax.plot([s[0] for s in sub], [s[2] for s in sub], marker="s", ls="--",
                label="%s simultaneous" % task)
    nominal = 1.0 - _num(rows[0].get("alpha"), 0.05)
    ax.axhline(nominal, color="k", lw=1, ls=":", label="nominal %.2f" % nominal)
    ax.set_xscale("log")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path_out = os.path.join(out_dir, "coverage.png")
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    return path_out


def scaling_figure(path, out_dir):
    plt = _mpl()
    rows = [r for r in _read(path) if r["experiment"] == "scaling_eval"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (backend, device) in sorted({(r["backend"], r["device"]) for r in rows}):
        sub = sorted((int(r["n"]), _num(r["seconds"])) for r in rows
                     if r["backend"] == backend and r["device"] == device)
        ax.plot([s[0] for s in sub], [s[1] for s in sub], marker="o",
                label="%s (%s)" % (backend, device))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("evaluation points")
    ax.set_ylabel("seconds (median)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    path_out = os.path.join(out_dir, "scaling.png")
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    return path_out


def make_all(results_dir: str):
    """Render every figure that the available CSVs support."""
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    produced = []
    for name, fn in (("convergence", convergence_figure), ("fidelity", fidelity_figure),
                     ("coverage", coverage_figure), ("scaling", scaling_figure)):
        path = os.path.join(results_dir, name + ".csv")
        if os.path.exists(path):
            try:
                out = fn(path, fig_dir)
            except Exception as exc:  # pragma: no cover - plotting is best effort
                print("  (figure %s skipped: %s)" % (name, exc))
                out = None
            if out:
                produced.append(out)
    return produced
