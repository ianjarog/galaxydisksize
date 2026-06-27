"""Unit conversions for HI moment-0 maps and the HI size-mass relation.

All functions accept either Python floats or NumPy arrays and return the same
shape. Quantities follow the conventions used throughout the study:

* HI column density ``N_HI`` is in atoms per square centimetre.
* Moment-0 intensity is in Jy beam^-1 m s^-1.
* HI mass ``M_HI`` is in solar masses; ``log_mhi`` is its base-10 logarithm.
* The HI diameter ``D_HI`` is measured at the 1 M_sun pc^-2 isophote
  (``N_HI = 1.249e20`` atoms cm^-2), the definition adopted for the size-mass
  relation in this work.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Conversion constant between HI column density (atoms cm^-2) and moment-0
# intensity (Jy beam^-1 m s^-1) for a Gaussian beam, with the beam axes in
# arcseconds: N_HI = 1.104e21 * I_mom0 / (bmaj * bmin).
COLUMN_DENSITY_PER_INTENSITY = 1.104e21

# HI column density of the 1 M_sun pc^-2 isophote that defines D_HI
# (1 M_sun pc^-2 = 1.249e20 atoms cm^-2).
N_HI_ONE_SOLAR = 1.249e20

# Small-angle conversion: 1 arcsec at distance D (Mpc) subtends
# 1e-3 * D / (180/pi * 3600) kpc; the denominator below is 180/pi * 3600 / 1e3.
_ARCSEC_PER_KPC_AT_1MPC = 180.0 / np.pi * 3600.0 / 1.0e3  # = 206.2648...


def moment0_to_column_density(
    intensity: ArrayLike, bmaj_arcsec: float, bmin_arcsec: float
) -> NDArray[np.float64]:
    """Convert moment-0 intensity to HI column density.

    Parameters
    ----------
    intensity : array_like
        Moment-0 intensity in Jy beam^-1 m s^-1.
    bmaj_arcsec, bmin_arcsec : float
        FWHM of the major and minor axes of the restoring beam, in arcseconds.

    Returns
    -------
    numpy.ndarray
        HI column density in atoms cm^-2.

    See Also
    --------
    column_density_to_moment0 : The inverse conversion.
    """
    intensity = np.asarray(intensity, dtype=float)
    return intensity * COLUMN_DENSITY_PER_INTENSITY / (bmaj_arcsec * bmin_arcsec)


def column_density_to_moment0(
    column_density: ArrayLike, bmaj_arcsec: float, bmin_arcsec: float
) -> NDArray[np.float64]:
    """Convert HI column density to moment-0 intensity.

    Parameters
    ----------
    column_density : array_like
        HI column density in atoms cm^-2.
    bmaj_arcsec, bmin_arcsec : float
        FWHM of the major and minor axes of the restoring beam, in arcseconds.

    Returns
    -------
    numpy.ndarray
        Moment-0 intensity in Jy beam^-1 m s^-1.

    See Also
    --------
    moment0_to_column_density : The inverse conversion.
    """
    column_density = np.asarray(column_density, dtype=float)
    return column_density * (bmaj_arcsec * bmin_arcsec) / COLUMN_DENSITY_PER_INTENSITY


def arcsec_to_kpc(angle_arcsec: ArrayLike, distance_mpc: ArrayLike) -> NDArray[np.float64]:
    """Convert an angular size to a physical size at a given distance.

    Parameters
    ----------
    angle_arcsec : array_like
        Angular size in arcseconds.
    distance_mpc : array_like
        Distance in megaparsecs.

    Returns
    -------
    numpy.ndarray
        Physical size in kiloparsecs, using the small-angle approximation.

    Examples
    --------
    >>> float(arcsec_to_kpc(30.0, 100.0))  # doctest: +ELLIPSIS
    14.54...
    """
    angle_arcsec = np.asarray(angle_arcsec, dtype=float)
    distance_mpc = np.asarray(distance_mpc, dtype=float)
    return angle_arcsec * distance_mpc / _ARCSEC_PER_KPC_AT_1MPC


def kpc_to_arcsec(size_kpc: ArrayLike, distance_mpc: ArrayLike) -> NDArray[np.float64]:
    """Convert a physical size to an angular size at a given distance.

    Parameters
    ----------
    size_kpc : array_like
        Physical size in kiloparsecs.
    distance_mpc : array_like
        Distance in megaparsecs.

    Returns
    -------
    numpy.ndarray
        Angular size in arcseconds, using the small-angle approximation.

    See Also
    --------
    arcsec_to_kpc : The inverse conversion.
    """
    size_kpc = np.asarray(size_kpc, dtype=float)
    distance_mpc = np.asarray(distance_mpc, dtype=float)
    return size_kpc * _ARCSEC_PER_KPC_AT_1MPC / distance_mpc


def mean_surface_density(hi_mass_msun: ArrayLike, diameter_kpc: ArrayLike) -> NDArray[np.float64]:
    r"""Mean HI surface density within the HI disc.

    Computes the mass-weighted mean surface density assuming a circular disc of
    the given diameter,

    .. math::

        \langle \Sigma_{\mathrm{HI}} \rangle
            = \frac{M_{\mathrm{HI}}}{(\pi/4)\, D_{\mathrm{HI}}^{2}} .

    A size-mass relation with slope ``0.5`` (``D_HI ~ M_HI^0.5``) is equivalent
    to a constant mean surface density, which is the physical interpretation used
    in the discussion of disc truncation.

    Parameters
    ----------
    hi_mass_msun : array_like
        HI mass in solar masses.
    diameter_kpc : array_like
        HI diameter in kiloparsecs.

    Returns
    -------
    numpy.ndarray
        Mean HI surface density in M_sun pc^-2.
    """
    hi_mass_msun = np.asarray(hi_mass_msun, dtype=float)
    diameter_kpc = np.asarray(diameter_kpc, dtype=float)
    radius_pc = 0.5 * diameter_kpc * 1.0e3
    area_pc2 = np.pi * radius_pc**2
    return hi_mass_msun / area_pc2
