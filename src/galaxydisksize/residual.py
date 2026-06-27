r"""Size residuals against an HI-to-optical baseline.

The disc-truncation diagnostic is the residual of the HI diameter about a
baseline relation fitted between the HI diameter and the optical diameter
``D_25``,

.. math::

    \\Delta = \\log_{10} D_{\\mathrm{HI}}
        - \\left( m\\,\\log_{10} D_{25} + b \\right) .

A galaxy with a truncated HI disc sits below the baseline (``Delta < 0``). The
baseline is fitted on a reference (isolated) sample, then applied unchanged to
the sample under study so the two share a common zero point.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_baseline(
    log_optical_diameter: ArrayLike, log_hi_diameter: ArrayLike
) -> tuple[float, float, float]:
    """Fit the HI-to-optical baseline by ordinary least squares.

    Parameters
    ----------
    log_optical_diameter : array_like
        Base-10 logarithm of the optical diameter ``D_25`` in kpc.
    log_hi_diameter : array_like
        Base-10 logarithm of the HI diameter ``D_HI`` in kpc.

    Returns
    -------
    slope : float
        Baseline slope ``m``.
    intercept : float
        Baseline intercept ``b``.
    scatter : float
        Standard deviation of the residuals about the fit, in dex.

    Notes
    -----
    Non-finite pairs are dropped before fitting. This is the simple baseline
    used to define the reference zero point; the full Bayesian size-mass fit is
    in :mod:`galaxydisksize.masssize`.
    """
    x = np.asarray(log_optical_diameter, dtype=float)
    y = np.asarray(log_hi_diameter, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        raise ValueError(f"need at least 2 valid data points, got {x.size}")
    slope, intercept = np.polyfit(x, y, 1)
    scatter = float(np.std(y - (slope * x + intercept)))
    return float(slope), float(intercept), scatter


def size_residual(
    log_hi_diameter: ArrayLike,
    log_optical_diameter: ArrayLike,
    slope: float,
    intercept: float,
) -> NDArray[np.float64]:
    """Size residual about the HI-to-optical baseline.

    Parameters
    ----------
    log_hi_diameter : array_like
        Base-10 logarithm of the HI diameter ``D_HI`` in kpc.
    log_optical_diameter : array_like
        Base-10 logarithm of the optical diameter ``D_25`` in kpc.
    slope, intercept : float
        Baseline parameters ``m`` and ``b``, typically from
        :func:`fit_baseline` on a reference sample.

    Returns
    -------
    numpy.ndarray
        The residual ``Delta`` in dex. Negative values indicate an HI disc that
        is small for its optical size (truncated).
    """
    log_hi_diameter = np.asarray(log_hi_diameter, dtype=float)
    log_optical_diameter = np.asarray(log_optical_diameter, dtype=float)
    return log_hi_diameter - (slope * log_optical_diameter + intercept)


def deficit_fraction(residual: ArrayLike) -> NDArray[np.float64]:
    """Fractional size deficit implied by a residual.

    Parameters
    ----------
    residual : array_like
        Size residual ``Delta`` in dex.

    Returns
    -------
    numpy.ndarray
        The fractional shortfall in diameter, ``1 - 10**Delta``. A residual of
        ``-0.301`` dex (a factor of two) returns ``0.5``.
    """
    residual = np.asarray(residual, dtype=float)
    return 1.0 - 10.0**residual
