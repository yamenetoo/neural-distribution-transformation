"""Core algebra: basis functions, monotonicity, fitting."""
import numpy as np
import pytest

from ndt import MSN, design_matrix, fit_msn, make_knots, ramp_predict, relu_predict
from ndt.network import random_msn


def test_design_matrix_columns_match_relu_form():
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(size=50))
    knots = make_knots(x, 8)
    A = design_matrix(x, knots)
    assert A.shape == (50, knots.size + 1)

    coef = np.concatenate([[1.5, 0.25], rng.random(knots.size - 1)])
    fast = ramp_predict(x, knots, coef)
    literal = relu_predict(x, knots, coef)
    np.testing.assert_allclose(fast, literal, atol=1e-12)


def test_extrapolation_is_constant():
    """The ReLU form is flat outside the outermost knots."""
    knots = np.linspace(0.0, 1.0, 6)
    coef = np.concatenate([[2.0, 0.0], np.full(5, 0.3)])
    left = ramp_predict(np.array([-5.0, -1.0]), knots, coef)
    right = ramp_predict(np.array([2.0, 10.0]), knots, coef)
    assert np.allclose(left, 2.0)
    assert np.allclose(right, ramp_predict(np.array([1.0]), knots, coef)[0] + 0.0)


def test_random_weights_are_monotone():
    """Monotonicity must hold for every parameter value, not just fitted ones."""
    rng = np.random.default_rng(1)
    x = np.linspace(-3, 3, 2001)
    for _ in range(20):
        model = random_msn(int(rng.integers(1, 12)), rng)
        assert model.is_monotone(x)


def test_negative_weights_break_monotonicity():
    """A sanity check that the test above is not vacuous."""
    knots = np.linspace(0.0, 1.0, 4)
    coef = np.concatenate([[0.0, 0.0], [1.0, -0.5, 0.2]])
    model = MSN(knots, coef)
    assert not model.is_monotone(np.linspace(0, 1, 501))


def test_knot_schemes():
    x = np.sort(np.random.default_rng(2).normal(size=500))
    q = make_knots(x, 8, scheme="quantile")
    u = make_knots(x, 8, scheme="uniform")
    assert q.size >= 2 and u.size >= 2
    # quantile knots are not uniformly spaced on the x-axis
    assert not np.allclose(np.diff(q), np.diff(q)[0])
    np.testing.assert_allclose(np.diff(u), np.diff(u)[0])

    with pytest.raises(ValueError):
        make_knots(x, 8, scheme="equal-increment")
    with pytest.raises(ValueError):
        make_knots(x, 8, scheme="nope")


def test_equal_increment_equalises_target_increments():
    x = np.linspace(-2, 2, 4001)
    knots = make_knots(x, 10, scheme="equal-increment", target=lambda z: z ** 3)
    values = knots ** 3
    # knots are recovered by inverting T on a finite grid, so the increments are
    # equal only up to the interpolation error of that inversion
    np.testing.assert_allclose(np.diff(values), np.diff(values)[0], rtol=0.02, atol=0.02)


def test_fit_recovers_affine_map_exactly():
    """With one interval the network is a straight line, so an affine T is exact."""
    task_x = np.sort(np.random.default_rng(3).normal(size=400))
    y = 3.0 + 2.0 * task_x
    model = fit_msn(task_x, y, m=1)
    x = np.linspace(task_x.min(), task_x.max(), 101)
    np.testing.assert_allclose(model.predict(x), 3.0 + 2.0 * x, atol=1e-8)


def test_fit_is_convex_least_squares_and_monotone():
    rng = np.random.default_rng(4)
    x = np.linspace(-3, 3, 4000)
    y = np.exp(x)
    model = fit_msn(x, y, m=64)
    assert np.all(model.coef >= 0)          # NNLS returns non-negative weights
    assert model.is_monotone(np.linspace(-3, 3, 4001))
    # exp ranges over [0.05, 20] on [-3, 3]; a 64-knot fit is within a few percent
    assert np.max(np.abs(model.predict(x) - y)) < 0.5


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_msn(np.zeros(5), np.zeros(4), m=4)


def test_msn_rejects_bad_coefficient_count():
    with pytest.raises(ValueError):
        MSN(np.linspace(0, 1, 4), np.zeros(3))
