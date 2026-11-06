"""Reproducible numerical illustrations for the NDT article (Section 5).

Estimator: monotone spline network (MSN) -- a one-hidden-layer ReLU network
N(x) = c + sum_j s_j [relu(x - t_j) - relu(x - t_{j+1})],  s_j >= 0,
fitted by non-negative least squares on quantile-coupled pairs
(x_j, y_j) = (Q_X(u_j), Q_Y(u_j)),  u_j = j/(N+1).
"""
import json
import time

import numpy as np
from scipy import stats
from scipy.optimize import nnls

rng = np.random.default_rng(20261015)


# ---------------------------------------------------------------- estimator
def design(x, knots):
    # columns: [1, -1, clip(x, t_j, t_{j+1}) - t_j for each interval]
    lo, hi = knots[:-1], knots[1:]
    B = np.clip(x[:, None], lo[None, :], hi[None, :]) - lo[None, :]
    return np.hstack([np.ones((len(x), 1)), -np.ones((len(x), 1)), B])


def fit_msn(x, y, K):
    # knots at empirical quantiles of the training inputs
    knots = np.quantile(x, np.linspace(0, 1, K + 1))
    knots = np.unique(knots)
    A = design(x, knots)
    coef, _ = nnls(A, y, maxiter=50 * A.shape[1])
    return knots, coef


def eval_msn(x, model):
    knots, coef = model
    c = coef[0] - coef[1]
    s = coef[2:]
    # equivalent piecewise-linear evaluation (O(log K) per point)
    vals = np.concatenate([[c], c + np.cumsum(s * np.diff(knots))])
    return np.interp(x, knots, vals)


def check_relu_form(model, x):
    """Evaluate the same network literally as a sum of ReLUs (for verification)."""
    knots, coef = model
    c = coef[0] - coef[1]
    s = coef[2:]
    relu = lambda z: np.maximum(z, 0)
    out = np.full_like(x, c)
    for j in range(len(s)):
        out += s[j] * (relu(x - knots[j]) - relu(x - knots[j + 1]))
    return out


# ---------------------------------------------------------------- tasks
def chi2_map(x):
    return stats.norm.ppf((1 + stats.norm.cdf(x)) / 2) ** 2


mix_w, mix_m, mix_s = 0.5, 2.0, 0.5


def mix_cdf(x):
    return 0.5 * stats.norm.cdf(x, -mix_m, mix_s) + 0.5 * stats.norm.cdf(x, mix_m, mix_s)


def mix_sample(n):
    z = rng.random(n) < 0.5
    return np.where(z, rng.normal(-mix_m, mix_s, n), rng.normal(mix_m, mix_s, n))


def mix_ppf(u):
    # numerical inversion (bisection, vectorised) -- used only for ground truth
    lo, hi = np.full_like(u, -10.0), np.full_like(u, 10.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        left = mix_cdf(mid) < u
        lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
    return 0.5 * (lo + hi)


tasks = {
    "P1": dict(Qx=stats.norm.ppf, Qy=stats.lognorm(1).ppf, T=np.exp,
               sx=lambda n: rng.normal(size=n), Fy=stats.lognorm(1).cdf),
    "P2": dict(Qx=lambda u: u, Qy=stats.expon.ppf, T=lambda x: -np.log1p(-x),
               sx=lambda n: rng.random(n), Fy=stats.expon.cdf),
    "P3": dict(Qx=stats.norm.ppf, Qy=stats.chi2(1).ppf, T=chi2_map,
               sx=lambda n: rng.normal(size=n), Fy=stats.chi2(1).cdf),
    "N1": dict(Qx=mix_ppf, Qy=stats.norm.ppf, T=lambda x: stats.norm.ppf(mix_cdf(x)),
               sx=mix_sample, Fy=stats.norm.cdf),
    "N2": dict(Qx=stats.gamma(2).ppf, Qy=lambda u: u, T=stats.gamma(2).cdf,
               sx=lambda n: rng.gamma(2.0, size=n), Fy=stats.uniform.cdf),
}


def w1_to_target(ysamp, Fy, lo, hi):
    grid = np.linspace(lo, hi, 20001)
    ecdf = np.searchsorted(np.sort(ysamp), grid, side="right") / len(ysamp)
    return np.trapezoid(np.abs(ecdf - Fy(grid)), grid)


results = {}

# ------------------------------------------- Experiment A: parametric fidelity
N_train, Ks, delta = 4000, [16, 64, 256], 0.01
u_train = np.arange(1, N_train + 1) / (N_train + 1)
for name, t in tasks.items():
    x_tr, y_tr = t["Qx"](u_train), t["Qy"](u_train)
    a, b = t["Qx"](np.array([delta, 1 - delta]))
    xs = np.linspace(a, b, 200001)
    xs_mc = t["sx"](100000)
    xs_mc_in = xs_mc[(xs_mc >= a) & (xs_mc <= b)]
    row = {}
    # noise floor: exact map applied to the same Monte-Carlo inputs
    y_exact = t["T"](xs_mc)
    row["KS_exact"] = stats.kstest(y_exact, t["Fy"]).statistic
    for K in Ks:
        m = fit_msn(x_tr, y_tr, K)
        err = eval_msn(xs, m) - t["T"](xs)
        yhat = eval_msn(xs_mc, m)
        mono = np.all(np.diff(eval_msn(xs, m)) >= -1e-12)
        relu_gap = np.max(np.abs(check_relu_form(m, xs[::997]) - eval_msn(xs[::997], m)))
        row[K] = dict(
            sup=float(np.max(np.abs(err))),
            mse=float(np.mean((eval_msn(xs_mc_in, m) - t["T"](xs_mc_in)) ** 2)),
            KS=float(stats.kstest(yhat, t["Fy"]).statistic),
            monotone=bool(mono), relu_gap=float(relu_gap),
        )
    results[name] = row
    print(name, json.dumps(row, default=float))

# --------------------------- Experiment B: non-parametric consistency (N1, N2)
reps, ns, K = 20, [300, 1000, 3000, 10000, 30000], 64
conv = {}
for name in ["N1", "N2"]:
    t = tasks[name]
    a, b = t["Qx"](np.array([0.05, 0.95]))
    xs = np.linspace(a, b, 20001)
    # target sampler: Y = T(X') with independent X'
    conv[name] = {}
    for n in ns:
        sups = []
        for r in range(reps):
            xsamp, ysamp = np.sort(t["sx"](n)), np.sort(t["T"](t["sx"](n)))
            # empirical quantile functions Q_n(u) = X_(ceil(n u))
            Qxn = lambda u: xsamp[np.clip(np.ceil(n * u).astype(int) - 1, 0, n - 1)]
            Qyn = lambda u: ysamp[np.clip(np.ceil(n * u).astype(int) - 1, 0, n - 1)]
            m = fit_msn(Qxn(u_train), Qyn(u_train), K)
            sups.append(np.max(np.abs(eval_msn(xs, m) - t["T"](xs))))
        conv[name][n] = (float(np.mean(sups)), float(np.std(sups)))
        print(name, n, conv[name][n])
results["conv"] = conv

# slope of log(sup err) vs log(n)
for name in conv:
    lx = np.log(ns)
    ly = np.log([conv[name][n][0] for n in ns])
    results[f"slope_{name}"] = float(np.polyfit(lx, ly, 1)[0])
    print("slope", name, results[f"slope_{name}"])

# ------------------------------------------- Experiment C: evaluation cost
x_tr, y_tr = tasks["P3"]["Qx"](u_train), tasks["P3"]["Qy"](u_train)
m = fit_msn(x_tr, y_tr, 256)
xq = rng.normal(size=10**6)
timing = {}
for label, f in [("scipy_chi2_ppf", lambda x: stats.chi2(1).ppf(stats.norm.cdf(x))),
                 ("closed_form", chi2_map),
                 ("msn_K256", lambda x: eval_msn(x, m))]:
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); f(xq); ts.append(time.perf_counter() - t0)
    timing[label] = float(np.median(ts))
    print(label, timing[label])
results["timing"] = timing

# ------------------------------------- worked numbers used in the text
ex = {}
ex["neurons_equal_increment_eps0.1"] = int(np.ceil((np.e**3 - np.e**-3) / 0.1))
ex["neurons_lipschitz_eps0.1"] = int(np.ceil(np.e**3 * 6 / 0.1))
alpha, n = 0.05, 10**4
ex["dkw_t"] = float(np.sqrt(np.log(4 / alpha) / (2 * n)))  # two CDFs, union bound
# chi2 map values
ex["chi2_T"] = {str(v): float(chi2_map(np.array(v))) for v in [-2.0, -1.0, 0.0, 1.0, 2.0]}
results["examples"] = ex
print(ex)

# ------------------------------------- coordinates for TikZ figures
xg = np.linspace(-3, 3, 61)
np.savetxt("fig_chi2.dat", np.c_[xg, xg**2, chi2_map(xg)], header="x xsq T", comments="")
# equal-increment monotone ReLU interpolant of exp on [-3,3], eps = 2.5
eps = 2.5
levels = np.arange(np.exp(-3), np.exp(3), eps)
tk = np.log(levels)
tk = np.unique(np.r_[tk, 3.0])
np.savetxt("fig_interp.dat", np.c_[tk, np.exp(tk)], header="t v", comments="")
np.savetxt("fig_exp.dat", np.c_[xg, np.exp(xg)], header="x v", comments="")
with open("fig_conv.dat", "w") as fh:
    fh.write("n N1 N2\n")
    for nn in ns:
        fh.write(f"{nn} {conv['N1'][nn][0]} {conv['N2'][nn][0]}\n")

with open("results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=float)
