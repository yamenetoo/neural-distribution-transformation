# Neural Distribution Transformation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamenetoo/neural-distribution-transformation/blob/main/notebooks/colab_ndt_demo.ipynb)

Python package and experiment suite for

> **Neural Distribution Transformation: Monotone Neural Approximation of Quantile
> Couplings with Uniform Error Bounds and Distribution-Free Confidence Bands**
> Mohamad Yamen AL Mohamad, Ali Akbar Heydari, Hossein Jabbari Khamnei
> Department of Statistics, University of Tabriz

The **manuscript is not in this repository** — this is the code only.

## What the method is

The quantile coupling `T = Q_Y o F_X` is the canonical map that transports the law of
`X` onto that of `Y`. It is learned here by a *monotone spline network* (MSN):

```
N(x) = c + sum_j s_j [ relu(x - t_j) - relu(x - t_{j+1}) ],   s_j >= 0
```

- **monotone for every parameter value** — monotonicity is structural, not a penalty;
- **convex training** — for fixed knots the fit is a non-negative least-squares problem;
- **uniform guarantees** — a deterministic perturbation bound, a distribution-free
  simultaneous confidence band, and an end-to-end sup-norm error `O_P(n^(-1/2))`.

## Install

```sh
pip install -e ".[full]"     # package + torch, joblib, matplotlib, pandas
pip install -e .             # core only: numpy + scipy
```

## Quickstart

```python
import numpy as np
from ndt import fit_msn, get_task, sup_error

task = get_task("P1")                       # N(0,1) -> LogNormal(0,1), T(x) = exp(x)
u = np.arange(1, 4001) / 4001
model = fit_msn(task.ppf_x(u), task.ppf_y(u), m=64)

lo, hi = task.range_on(0.01, 0.99)
print(sup_error(model, task, lo, hi))
print(model.is_monotone(np.linspace(lo, hi, 5001)))
```

## Experiment suite

```sh
python -m ndt env                                  # CPU, GPU, versions
python -m ndt run --experiments all --out results  # full suite
python -m ndt run --experiments coverage --quick   # one experiment, fast
python -m ndt run --experiments all --device cuda --jobs 16 --figures
```

| Experiment | What it measures | Paper claim it checks |
| --- | --- | --- |
| `fidelity` | sup error, MSE, KS, W1, W2 on 8 problems × knot counts | approximation bound; KS at the noise floor |
| `convergence` | uniform error vs sample size, 40 reps, bootstrap slope | `n^-1/2` rate |
| `ablation` | knot placement (quantile / uniform / equal-increment), grid size | range-based unit count |
| `coverage` | empirical coverage of the DKW band, pointwise and simultaneous | Corollary 4.9 |
| `atoms` | atom mass reproduced, jump in `T` | theory allows atoms in the target |
| `scaling` | evaluation throughput, parallel fitting speed-up | practicality |

Each writes a CSV to `results/`; `--figures` also renders the figure set into
`results/figures/`.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the protocol of each one.

## Colab

`notebooks/colab_ndt_demo.ipynb` installs the package, fits a network, runs the suite
and displays the figures. Open it in Colab and pick *Runtime > Change runtime type >
GPU* to exercise the CUDA path:

```
https://colab.research.google.com/github/yamenetoo/neural-distribution-transformation/blob/main/notebooks/colab_ndt_demo.ipynb
```

## GPU and parallelism

Two independent axes, both exercised by `scaling`:

- **Device (GPU).** Fitting is convex NNLS and stays on the CPU. *Evaluation* and
  *metric computation* are vectorised and run on CUDA (or MPS, or CPU) through
  `ndt.torch_predict` / `ndt.torch_predict_many`, which batch over points and, in the
  many-network case, over networks in a single kernel. `--device auto` picks the best
  available device and falls back cleanly when PyTorch is absent.
- **Parallelism (CPU).** Each replication is an independent fit, so `fit_msn_batch`
  fans them out over processes with joblib; `--jobs` sets the worker count
  (default: cores - 1).

```python
from ndt import torch_predict, torch_predict_many

y = torch_predict(model, x, device="cuda")        # one network, many points
Y = torch_predict_many(models, x, device="cuda")  # many networks, one launch
```

## Repository layout

```
src/ndt/
  basis.py        knots, design matrix, ramp and ReLU evaluation
  network.py      MSN, torch/GPU evaluation, device resolution
  fit.py          convex NNLS fit, batched/parallel fitting
  tasks.py        8 transport problems (incl. one with an atom)
  metrics.py      sup, MSE, KS, W1, W2, monotonicity
  bands.py        DKW radii, empirical quantiles, confidence band, coverage
  parallel.py     environment description, worker defaults
  experiments/    fidelity, convergence, ablation, coverage, atoms, scaling
  runner.py       CLI (python -m ndt)
  plotting.py     figures from the CSVs
tests/            pytest suite (core algebra, statistics, GPU parity, CLI)
notebooks/        Colab demo
experiments.py    the original single-file replication script of the paper
```

`experiments.py` is kept unchanged for provenance: it is the script that produced the
paper's tables. The package supersedes it and reproduces the same numbers.

## Tests

```sh
pytest -q
```

Covers the design matrix against the literal ReLU form, monotonicity for random
weights (and that the check is not vacuous), exactness of the affine case, the
chi-square map that is *not* `x^2`, atom mass, DKW radii, band coverage, GPU/NumPy
parity, and the CLI.

## Reproducibility

The seed is fixed (`20261015`) in `Config`, so every accuracy number is bit-identical
between runs; only the timing experiment varies, as it must. Absolute timings are
hardware dependent, so the `scaling` experiment reports its own machine.

## Citation

```bibtex
@article{almohamad2026ndt,
  title   = {Neural Distribution Transformation: Monotone Neural Approximation of
             Quantile Couplings with Uniform Error Bounds and Distribution-Free
             Confidence Bands},
  author  = {AL Mohamad, Mohamad Yamen and Heydari, Ali Akbar and Jabbari Khamnei, Hossein},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
