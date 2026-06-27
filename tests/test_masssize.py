"""Tests for :mod:`galaxydisksize.masssize`."""

import numpy as np
import pytest

from galaxydisksize.masssize import fit_mass_size, predict_log_diameter


def test_predict_log_diameter_is_linear():
    """The prediction is the straight line ``m * x + b``."""
    x = np.array([8.0, 9.0, 10.0])
    np.testing.assert_allclose(predict_log_diameter(x, 0.5, -3.3), 0.5 * x - 3.3, rtol=1e-12)


def test_fit_recovers_known_relation():
    """The MCMC fit recovers the slope and intercept of synthetic data."""
    rng = np.random.default_rng(7)
    true_slope, true_intercept, true_scatter = 0.5, -3.3, 0.05
    log_mass = rng.uniform(8.0, 10.5, size=300)
    log_diameter = (
        true_slope * log_mass + true_intercept + rng.normal(0.0, true_scatter, size=log_mass.size)
    )
    fit = fit_mass_size(log_mass, log_diameter, seed=7, n_steps=600, n_burn=200)

    assert fit.n_data == 300
    assert fit.slope == pytest.approx(true_slope, abs=0.02)
    assert fit.intercept == pytest.approx(true_intercept, abs=0.2)
    assert fit.scatter == pytest.approx(true_scatter, abs=0.02)


def test_fit_is_reproducible_with_seed():
    """Two fits with the same seed give identical posteriors."""
    rng = np.random.default_rng(3)
    log_mass = rng.uniform(8.0, 10.5, size=120)
    log_diameter = 0.5 * log_mass - 3.3 + rng.normal(0.0, 0.05, size=log_mass.size)
    first = fit_mass_size(log_mass, log_diameter, seed=11, n_steps=400, n_burn=100)
    second = fit_mass_size(log_mass, log_diameter, seed=11, n_steps=400, n_burn=100)
    np.testing.assert_array_equal(first.samples, second.samples)


def test_fit_rejects_too_few_points():
    """Fewer than three valid points raises a clear error."""
    with pytest.raises(ValueError, match="at least 3"):
        fit_mass_size([9.0, 10.0], [1.0, 1.5])
