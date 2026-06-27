"""Tests for :mod:`galaxydisksize.survival`."""

import numpy as np
import pytest
from scipy import stats

from galaxydisksize.survival import gehan_test, kaplan_meier_left_censored


def test_km_reduces_to_empirical_median_without_censoring():
    """With no upper limits the KM median equals the sample median."""
    rng = np.random.default_rng(0)
    residuals = rng.normal(-0.2, 0.15, size=201)
    is_limit = np.zeros_like(residuals, dtype=bool)
    km = kaplan_meier_left_censored(residuals, is_limit)
    np.testing.assert_allclose(km.median, np.median(residuals), atol=1e-9)


def test_km_median_unconstrained_when_tail_is_censored():
    """When the truncated tail is all upper limits the KM median is unconstrained.

    The detections are the least-truncated members (residual near zero) and the
    upper limits are the most-truncated; the survival function then plateaus
    above 0.5 and the median falls in the unconstrained censored tail (nan), as
    happens for the full HCG sample.
    """
    residuals = np.array([-0.10, -0.15, -0.50, -0.60, -0.70])
    is_limit = np.array([False, False, True, True, True])
    km = kaplan_meier_left_censored(residuals, is_limit)
    assert np.isnan(km.median)


def test_km_fraction_below_matches_detection_only_case():
    """Fraction below a threshold matches the empirical CDF without censoring."""
    residuals = np.array([-0.4, -0.2, -0.1, 0.05, 0.3])
    is_limit = np.zeros_like(residuals, dtype=bool)
    km = kaplan_meier_left_censored(residuals, is_limit)
    assert km.fraction_below(0.0) == pytest.approx(np.mean(residuals < 0.0))


def test_gehan_agrees_with_wilcoxon_without_censoring():
    """Uncensored Gehan p-value tracks the Mann-Whitney/Wilcoxon p-value."""
    rng = np.random.default_rng(1)
    sample_a = rng.normal(0.0, 1.0, size=40)
    sample_b = rng.normal(0.6, 1.0, size=45)
    result = gehan_test(
        sample_a,
        np.zeros_like(sample_a, dtype=bool),
        sample_b,
        np.zeros_like(sample_b, dtype=bool),
    )
    wilcoxon = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    # Both are normal-approximation tests of the same hypothesis; the p-values
    # agree to better than a factor of two.
    assert result.p_value == np.float64(result.p_value)  # finite
    assert abs(np.log10(result.p_value) - np.log10(wilcoxon.pvalue)) < 0.3
