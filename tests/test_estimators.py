"""Tests for :mod:`galaxydisksize.estimators`."""

import numpy as np
import pytest

from galaxydisksize import estimators


@pytest.fixture
def linear_data():
    """Synthetic data on a known line with small scatter and errors."""
    rng = np.random.default_rng(0)
    log_x = rng.uniform(0.0, 1.5, size=150)
    log_y = 0.7 * log_x + 0.69 + rng.normal(0.0, 0.04, size=log_x.size)
    log_x_err = np.full_like(log_x, 0.02)
    log_y_err = np.full_like(log_y, 0.03)
    return log_x, log_y, log_x_err, log_y_err


@pytest.mark.parametrize("method", ["OLS(Y|X)", "Theil-Sen", "Bisector", "ODR", "York"])
def test_dependency_free_estimators_recover_slope(method, linear_data):
    """Estimators with no optional dependency recover the true slope."""
    log_x, log_y, log_x_err, log_y_err = linear_data
    result = estimators.fit_linear(method, log_x, log_y, log_x_err, log_y_err)
    assert result["slope"] == pytest.approx(0.7, abs=0.06)
    assert result["intercept"] == pytest.approx(0.69, abs=0.06)
    assert result["n"] == 150


def test_bayesian_estimator_recovers_relation(linear_data):
    """The Bayesian fit recovers slope, intercept, and scatter, and is reported."""
    log_x, log_y, log_x_err, log_y_err = linear_data
    result = estimators.fit_linear("Bayesian", log_x, log_y, log_x_err, log_y_err, seed=1)
    assert result["slope"] == pytest.approx(0.7, abs=0.05)
    assert result["intercept"] == pytest.approx(0.69, abs=0.05)
    assert "sigma_int" in result and "slope_p16" in result


def test_fit_bayesian_linear_reproducible(linear_data):
    """Same seed gives identical posteriors."""
    log_x, log_y, _, log_y_err = linear_data
    first = estimators.fit_bayesian_linear(log_x, log_y, log_y_err, seed=5, n_steps=600, n_burn=200)
    second = estimators.fit_bayesian_linear(
        log_x, log_y, log_y_err, seed=5, n_steps=600, n_burn=200
    )
    np.testing.assert_array_equal(first.samples, second.samples)


def test_available_estimators_subset_of_order():
    """available_estimators returns a subset of the canonical order."""
    available = estimators.available_estimators()
    assert set(available).issubset(set(estimators.ESTIMATOR_ORDER))
    # The dependency-free estimators are always available.
    for always in ("OLS(Y|X)", "Bayesian", "Theil-Sen", "Bisector", "ODR", "York"):
        assert always in available


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown estimator"):
        estimators.fit_linear("nonsense", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
