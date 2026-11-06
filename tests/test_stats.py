"""Tasks, metrics and confidence bands."""
import numpy as np
import pytest
from scipy import stats

from ndt import Empirical, band_covers, dkw_radius, fit_msn, get_task, joint_radii
from ndt.bands import plugin_band
from ndt.metrics import ks_distance, sup_error, w2_quantile
from ndt.tasks import TASKS


def test_quantile_coupling_matches_closed_forms():
    p1 = get_task("P1")
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(p1.T(x), np.exp(x), rtol=1e-10)

    p2 = get_task("P2")
    np.testing.assert_allclose(p2.T(np.array([0.25, 0.5])),
                               -np.log1p(-np.array([0.25, 0.5])), rtol=1e-10)

    p4 = get_task("P4")
    np.testing.assert_allclose(p4.T(x), 3.0 + 2.0 * x, rtol=1e-10)


def test_chi_square_map_is_not_x_squared():
    """The paper's cautionary example: T for chi^2_1 is not x -> x^2."""
    p3 = get_task("P3")
    x = np.array([1.0, 2.0])
    t = p3.T(x)
    assert not np.allclose(t, x ** 2)
    np.testing.assert_allclose(t[0], 1.9869967133159565, rtol=1e-9)
    np.testing.assert_allclose(t[1], 5.187483801768254, rtol=1e-9)


def test_atom_task_has_the_right_mass():
    task = get_task("A1")
    rng = np.random.default_rng(0)
    y = task.sample_y(rng, 200_000)
    assert abs(np.mean(np.isclose(y, 1.0)) - 0.3) < 0.01
    # the quantile function is flat across the atom: every level in that range
    # maps to the atom itself
    lo = task.ppf_y(np.array([0.55]))[0]
    hi = task.ppf_y(np.array([0.65]))[0]
    assert lo < 1.0 < hi


def test_cdf_and_ppf_roundtrip():
    for key, task in TASKS.items():
        u = np.array([0.1, 0.35, 0.6, 0.9])
        if task.has_atom:
            # the atom makes F(Q(u)) >= u rather than equal
            assert np.all(task.cdf_y(task.ppf_y(u)) >= u - 1e-12)
        else:
            np.testing.assert_allclose(task.cdf_y(task.ppf_y(u)), u, atol=1e-9)


def test_sup_error_is_zero_for_affine_target():
    task = get_task("P4")
    x = np.linspace(-2.5, 2.5, 500)
    model = fit_msn(x, task.T(x), m=1)
    assert sup_error(model, task, -2.5, 2.5, n=501) < 1e-8


def test_ks_distance_of_exact_sample_is_small():
    task = get_task("P1")
    rng = np.random.default_rng(1)
    y = task.sample_y(rng, 100_000)
    assert ks_distance(y, task.cdf_y) < 0.01


def test_w2_is_zero_for_matching_laws():
    task = get_task("P4")
    # the fitted map has the right law, so its W2 to G is tiny
    x = np.linspace(-2.5, 2.5, 2000)
    model = fit_msn(x, task.T(x), m=1)
    assert w2_quantile(model, task, u_lo=0.001, u_hi=0.999, n=4001) < 1e-6


def test_dkw_radius_formula_and_monotonicity():
    assert dkw_radius(10_000, 0.05) == pytest.approx(
        np.sqrt(np.log(2 / 0.05) / (2 * 10_000)))
    assert dkw_radius(100, 0.05) > dkw_radius(1000, 0.05)
    # the paper's two-sample radius is sqrt(log(4/alpha)/(2n))
    t_x, t_y = joint_radii(10_000, 10_000, 0.05)
    assert t_x == pytest.approx(np.sqrt(np.log(4 / 0.05) / (2 * 10_000)))
    assert t_x == t_y


def test_empirical_quantile_conventions():
    e = Empirical(np.arange(1.0, 11.0))
    assert e.ppf(np.array([0.0]))[0] == -np.inf
    assert e.ppf(np.array([1.5]))[0] == np.inf
    np.testing.assert_allclose(e.ppf(np.array([0.1, 0.55])), [1.0, 6.0])
    assert e.cdf(np.array([0.0]))[0] == 0.0
    assert e.cdf(np.array([10.0]))[0] == 1.0


def test_band_contains_truth_for_large_samples():
    task = get_task("N1")
    rng = np.random.default_rng(2)
    xs = task.sample_x(rng, 20_000)
    ys = task.sample_y(rng, 20_000)
    grid = np.linspace(*task.range_on(0.05, 0.95), 101)
    band = plugin_band(grid, xs, ys, alpha=0.05)
    covered, _ = band_covers(band, task.T(grid))
    assert covered.mean() > 0.93


def test_band_coverage_is_at_least_nominal():
    """The DKW band is conservative, so coverage should never fall far below 1 - alpha.

    It is *widest* at small n, so coverage is expected to be highest there and to
    approach the nominal level as n grows -- not to increase with n.
    """
    task = get_task("N1")
    rng = np.random.default_rng(3)
    grid = np.linspace(*task.range_on(0.05, 0.95), 101)
    truth = task.T(grid)
    for n in (200, 20_000):
        covered = []
        for _ in range(20):
            band = plugin_band(grid, task.sample_x(rng, n), task.sample_y(rng, n))
            covered.append(np.mean(band_covers(band, truth)[0]))
        assert np.mean(covered) > 0.93, (n, np.mean(covered))


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        get_task("ZZ")
