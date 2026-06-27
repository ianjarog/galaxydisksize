"""Tests for :mod:`galaxydisksize.residual`."""

import numpy as np
import pytest

from galaxydisksize.residual import deficit_fraction, fit_baseline, size_residual


def test_residual_zero_on_baseline():
    """A galaxy exactly on the baseline has zero residual."""
    log_d25 = np.array([0.5, 1.0, 1.5])
    slope, intercept = 0.7, 0.69
    log_dhi = slope * log_d25 + intercept
    np.testing.assert_allclose(size_residual(log_dhi, log_d25, slope, intercept), 0.0, atol=1e-12)


def test_fit_baseline_recovers_slope_and_intercept():
    """Ordinary-least-squares baseline recovers the generating line."""
    rng = np.random.default_rng(5)
    log_d25 = rng.uniform(0.0, 1.5, size=200)
    log_dhi = 0.7 * log_d25 + 0.69 + rng.normal(0.0, 0.05, size=log_d25.size)
    slope, intercept, scatter = fit_baseline(log_d25, log_dhi)
    assert slope == pytest.approx(0.7, abs=0.03)
    assert intercept == pytest.approx(0.69, abs=0.03)
    assert scatter == pytest.approx(0.05, abs=0.01)


def test_deficit_fraction_factor_of_two():
    """A residual of -0.301 dex (a factor of two) is a 50% deficit."""
    assert deficit_fraction(-np.log10(2.0)) == pytest.approx(0.5)
    assert deficit_fraction(0.0) == pytest.approx(0.0)
