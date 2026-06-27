"""Tests for the unit conversions in :mod:`galaxydisksize.surface_density`."""

import numpy as np

from galaxydisksize import surface_density as sd


def test_arcsec_kpc_round_trip():
    """arcsec -> kpc -> arcsec returns the original angular size."""
    angle = np.array([10.0, 30.0, 120.0])
    distance = 85.0
    recovered = sd.kpc_to_arcsec(sd.arcsec_to_kpc(angle, distance), distance)
    np.testing.assert_allclose(recovered, angle, rtol=1e-12)


def test_arcsec_to_kpc_known_value():
    """30 arcsec at 100 Mpc is about 14.5 kpc (small-angle approximation)."""
    np.testing.assert_allclose(sd.arcsec_to_kpc(30.0, 100.0), 14.5444, rtol=1e-3)


def test_column_density_moment0_round_trip():
    """Column density <-> moment-0 is invertible for fixed beam axes."""
    column_density = np.array([1.0e20, 5.0e20, 1.2e21])
    bmaj, bmin = 25.0, 18.0
    moment0 = sd.column_density_to_moment0(column_density, bmaj, bmin)
    recovered = sd.moment0_to_column_density(moment0, bmaj, bmin)
    np.testing.assert_allclose(recovered, column_density, rtol=1e-12)


def test_constant_surface_density_implies_slope_half():
    """A constant mean surface density gives a size-mass slope of 0.5.

    If ``Sigma`` is fixed then ``M_HI ~ D_HI**2``, so ``log10(D_HI)`` is a
    straight line in ``log10(M_HI)`` with slope 0.5.
    """
    diameter_kpc = np.array([10.0, 20.0, 40.0, 80.0])
    fixed_sigma = 3.0  # M_sun / pc^2
    radius_pc = 0.5 * diameter_kpc * 1.0e3
    hi_mass = fixed_sigma * np.pi * radius_pc**2

    # Recovered surface density is constant ...
    np.testing.assert_allclose(
        sd.mean_surface_density(hi_mass, diameter_kpc), fixed_sigma, rtol=1e-12
    )
    # ... and the size-mass slope is 0.5.
    slope = np.polyfit(np.log10(hi_mass), np.log10(diameter_kpc), 1)[0]
    np.testing.assert_allclose(slope, 0.5, atol=1e-12)
