# Experiment protocols

Each experiment answers a specific claim in the manuscript. All of them write a CSV with
one row per configuration into `--out` (default `results/`), plus `environment.json`
(machine, CPU count, GPU, versions) and `summary.json` (rows and wall time per module).

Run everything with

```sh
python -m ndt run --experiments all --out results --device auto --jobs 16 --figures
```

Add `--quick` for a reduced sweep that finishes in well under a minute on a laptop.

**Status:** the suite is written and self-contained, but has *not* been executed on the
author's machine yet. Numbers are produced on your run; nothing here is a reported
result.

---

## A. `fidelity` — approximation with known laws

**Claim.** With both quantile functions known, the MSN reaches the sup-norm bound of
Theorem 3.2/3.5 and reproduces the target law up to Monte Carlo noise.

**Protocol.** 8 problems (`P1`–`P4`, `N1`–`N3`, `A1`) × `m ∈ {8,16,32,64,128,256}`
(quick: `{16,64}`). Training pairs `(Q_X(u_j), Q_Y(u_j))`, `u_j = j/4001`, `N = 4000`
points. Evaluation on `K = [Q_X(0.01), Q_X(0.99)]` with a 20 001-point grid.

**Columns.** `task`, `m`, `sup`, `mse`, `ks`, `w1`, `w2`, `monotone`, `ks_floor`,
`relu_gap`, `n_train`, `n_mc`.

**Notes.** `ks_floor` is the KS distance of the *exact* map on the same 10^5 draws, so
it is the noise floor a fitted map cannot beat. `relu_gap` is the discrepancy between
the fast piecewise-linear evaluation and the literal sum of ReLUs; it should be near
machine epsilon and confirms the model really is a ReLU network.

**Paper use.** Table of sup/MSE/KS versus `m`; the "fidelity at the noise floor"
sentence, now backed by `ks` vs `ks_floor` on all eight problems.

---

## B. `convergence` — two unpaired samples

**Claim.** The uniform error decays as `n^(-1/2)` (Theorem 4.11).

**Protocol.** Tasks `N1`, `N2`, `N3`, `A1`; `n ∈ {300, 1000, 3000, 10^4, 3·10^4}`;
40 replications, 5 in `--quick`. Per replication: two independent samples of size `n`,
empirical quantile functions evaluated on the `N = 4000` grid, one fit with `m = 64`,
sup error on `K = [Q_X(0.05), Q_X(0.95)]`. Fits are fanned out with joblib.

**Columns.** per `n`: `mean`, `sd`, `median`, `reps`; then one `convergence_slope` row
per task with `slope`, `slope_lo`, `slope_hi` (bootstrap over replications, 500 draws).

**Paper use.** Replaces the single-task slope with a slope *and* an interval, for four
problems. Expect `slope ≈ -0.5` with an interval that contains it.

---

## C. `ablation` — what the knots buy

**Claims.** (i) Unit count is governed by the *range* of the map, not its Lipschitz
constant (Corollary 3.3); (ii) equal-increment knots equalise the error.

**Protocol.**

1. *Knot placement.* `N1` and `P1` × `{quantile, uniform, equal-increment}` ×
   `m ∈ {16, 64, 256}`. The equal-increment scheme inverts the true `T` to place knots
   at equal target increments.
2. *Grid size.* `N1`, `m = 64`, `N ∈ {250, 1000, 4000, 16000}` training pairs from
   samples of size 3000.

**Columns.** `scheme`, `m`, `sup`, `n_intervals` for the first; `n_grid`, `sup` for the
second.

**Paper use.** A table showing quantile knots are near-optimal for quantile-shaped
problems and that `1/(N+1)` is negligible once `N ≥ 1000`.

---

## D. `coverage` — does the band hold its promise?

**Claim.** Corollary 4.9: the band covers `T(x)` simultaneously for all `x` with
probability at least `1 - alpha`, distribution-free.

**Protocol.** Tasks `N1` and `A1`; `n ∈ {500, 2000, 10^4}`; 200 replications. For each
replication build the band on a 401-point grid over `K = [Q_X(0.05), Q_X(0.95)]` with
`alpha = 0.05`, then record pointwise coverage and whether *every* grid point was
covered.

**Columns.** `point_coverage`, `point_coverage_sd`, `simultaneous_rate`,
`median_width`, `t_x`, `t_y`, `nominal`.

**Expectation.** Pointwise coverage comfortably above 0.95 — the DKW bound is
conservative, and more so at small `n`.

**Paper use.** A new table and figure. This is the empirical validation the paper
currently lacks: the band is stated but never checked.

---

## E. `atoms` — a target law with a point mass

**Claim.** The theory allows atoms in `Y`; the quantile coupling has a jump there and
the map still pushes `P_X` forward correctly.

**Protocol.** Task `A1` is `0.7·N(0,1) + 0.3·δ_1`. Fit with `m ∈ {16, 64, 256}` on 4000
known quantile pairs, evaluate on 10^5 draws: fraction of outputs equal to the atom
value, KS against the true (discontinuous) CDF, sup error, monotonicity.

**Columns.** `atom_value`, `atom_mass_theory`, `atom_mass_exact`, `atom_mass_fitted`,
`ks`, `ks_exact`, `sup`, `monotone`.

**Paper use.** A short paragraph arguing that a piecewise-linear monotone network
represents a jump *natively*, which no density-based flow can do — a concrete
advantage worth stating in the introduction and in Table 1 of related work.

---

## F. `scaling` — evaluation throughput and parallel fitting

**Claim.** The MSN is cheap to evaluate and the experiment loop parallelises.

**Protocol.**

1. *Single network.* `10^5`–`10^7` evaluation points, NumPy vs the device backend
   (CUDA if present), median of 3 runs.
2. *Many networks.* 64 networks × `10^6` points, loop vs `torch_predict_many`
   (single batched launch).
3. *Parallel fitting.* 64 independent fits, `n_jobs ∈ {1, 2, 4, 8, 24}`.

**Columns.** `backend`, `device`, `n`, `n_models`, `seconds`, `speedup`, `n_jobs`.

**Expectation.** Large speed-ups for evaluation at 10^7 points on a GPU. For fitting,
process start-up on Windows can dominate at small problem sizes — report the measured
numbers as they are rather than assuming ideal scaling.

**Paper use.** Replaces the current two-machine timing table with a scaling study, and
lets the reproducibility section state the hardware precisely.
