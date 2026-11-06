# Neural Distribution Transformation — replication code

Code that regenerates every number, table and data-driven figure in

> **Neural Distribution Transformation: Monotone Neural Approximation of Quantile
> Couplings with Uniform Error Bounds and Distribution-Free Confidence Bands**
> Mohamad Yamen AL Mohamad, Ali Akbar Heydari, Hossein Jabbari Khamnei
> Department of Statistics, University of Tabriz

This repository contains the experiment script only; the manuscript source is not
included.

## The estimator

`experiments.py` fits the paper's *monotone spline network* (MSN):

```
N(x) = c + sum_j s_j [ relu(x - t_j) - relu(x - t_{j+1}) ],   s_j >= 0
```

It is non-decreasing for every parameter value, and for fixed knots the fit is a convex
non-negative least-squares problem, solved with `scipy.optimize.nnls` on quantile-coupled
pairs `(x_j, y_j) = (Q_X(u_j), Q_Y(u_j))` with `u_j = j/(N+1)`.

## Requirements

Python 3 with NumPy (>= 2, for `np.trapezoid`) and SciPy. No GPU.

```sh
pip install -r requirements.txt
```

## Running

```sh
python experiments.py
```

Takes about 30 s on a single CPU core. It writes `results.json` and the figure coordinates
`fig_chi2.dat`, `fig_interp.dat`, `fig_exp.dat`, `fig_conv.dat`.

## What it reproduces

| Experiment | Output | Paper artifact |
| --- | --- | --- |
| A — fidelity with known quantile functions | uniform error, MSE, KS distance for P1–P3, N1–N2 at 16/64/256 knots | Table 1 |
| B — learning from two unpaired samples | uniform error vs sample size, 20 replications | Table 2, Figure 1 |
| C — evaluation cost | timings for `10^6` evaluations | Table 3 |

**Problems.** P1 `N(0,1) -> LogNormal(0,1)`, `T(x) = e^x`; P2 `Unif -> Exp(1)`,
`T(x) = -log(1-x)`; P3 `N(0,1) -> chi^2_1`, `T(x) = (Phi^-1((1+Phi(x))/2))^2` (not `x^2`);
N1 `0.5N(-2,0.25) + 0.5N(2,0.25) -> N(0,1)`; N2 `Gamma(2,1) -> Unif`.

**Reproducibility.** The seed is fixed (`20261015`), so every accuracy number is
bit-identical between runs; only the Experiment C timings vary, as they must. Verified on
two independent runs: all values in Tables 1–2 and Figure 1 match to machine precision.

The timings in Table 3 are hardware dependent. The script reports the machine it runs on,
so the two-machine table in the paper cannot be regenerated from a single run.

## License

MIT — see [LICENSE](LICENSE).
