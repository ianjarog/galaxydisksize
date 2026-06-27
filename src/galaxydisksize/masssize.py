r"""Bayesian fit of the HI size-mass relation with intrinsic scatter.

The relation is parametrised in log space as

.. math::

    \\log_{10} D_{\\mathrm{HI}} = m\\,\\log_{10} M_{\\mathrm{HI}} + b ,

with a Gaussian intrinsic scatter ``sigma`` orthogonal-free (added in
quadrature to the measurement uncertainties in the diameter direction). A slope
``m`` close to ``0.5`` corresponds to a constant mean HI surface density
(see :func:`galaxydisksize.surface_density.mean_surface_density`).

The posterior is sampled with :mod:`emcee`. The fit is intentionally generic so
it can be applied to any sample of HI masses and diameters, not only the one in
the accompanying paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimators import fit_bayesian_linear


@dataclass
class MassSizeFit:
    """Posterior samples of an HI size-mass fit.

    Attributes
    ----------
    samples : numpy.ndarray
        Array of shape ``(n_samples, 3)`` holding the posterior draws of
        ``(slope, intercept, scatter)`` after burn-in and thinning.
    n_data : int
        Number of galaxies used in the fit.

    Notes
    -----
    Point estimates are exposed as the posterior medians (:attr:`slope`,
    :attr:`intercept`, :attr:`scatter`); use :meth:`percentiles` for credible
    intervals.
    """

    samples: NDArray[np.float64]
    n_data: int

    @property
    def slope(self) -> float:
        """Posterior median slope ``m``."""
        return float(np.median(self.samples[:, 0]))

    @property
    def intercept(self) -> float:
        """Posterior median intercept ``b``."""
        return float(np.median(self.samples[:, 1]))

    @property
    def scatter(self) -> float:
        """Posterior median intrinsic scatter ``sigma`` in dex."""
        return float(np.median(self.samples[:, 2]))

    def percentiles(
        self, levels: tuple[float, float, float] = (16.0, 50.0, 84.0)
    ) -> dict[str, NDArray[np.float64]]:
        """Posterior percentiles of each parameter.

        Parameters
        ----------
        levels : tuple of float, optional
            Percentile levels to report, by default the 16th, 50th, and 84th
            (median and 1-sigma-equivalent credible interval).

        Returns
        -------
        dict
            Mapping ``{"slope": ..., "intercept": ..., "scatter": ...}`` where
            each value is an array of the requested percentiles.
        """
        return {
            "slope": np.percentile(self.samples[:, 0], levels),
            "intercept": np.percentile(self.samples[:, 1], levels),
            "scatter": np.percentile(self.samples[:, 2], levels),
        }

    def predict_log_diameter(self, log_hi_mass: ArrayLike) -> NDArray[np.float64]:
        """Predict ``log10(D_HI)`` at given ``log10(M_HI)`` using the medians.

        Parameters
        ----------
        log_hi_mass : array_like
            Base-10 logarithm of the HI mass in solar masses.

        Returns
        -------
        numpy.ndarray
            Predicted base-10 logarithm of the HI diameter in kpc.
        """
        return predict_log_diameter(log_hi_mass, self.slope, self.intercept)


def predict_log_diameter(
    log_hi_mass: ArrayLike, slope: float, intercept: float
) -> NDArray[np.float64]:
    """Evaluate the size-mass relation ``m * log10(M_HI) + b``.

    Parameters
    ----------
    log_hi_mass : array_like
        Base-10 logarithm of the HI mass in solar masses.
    slope, intercept : float
        Relation parameters ``m`` and ``b``.

    Returns
    -------
    numpy.ndarray
        Predicted base-10 logarithm of the HI diameter in kpc.
    """
    log_hi_mass = np.asarray(log_hi_mass, dtype=float)
    return slope * log_hi_mass + intercept


def fit_mass_size(
    log_hi_mass: ArrayLike,
    log_diameter: ArrayLike,
    log_hi_mass_err: ArrayLike | None = None,
    log_diameter_err: ArrayLike | None = None,
    *,
    n_walkers: int = 32,
    n_steps: int = 4000,
    n_burn: int = 1000,
    thin: int = 10,
    seed: int | None = None,
) -> MassSizeFit:
    """Fit the HI size-mass relation with intrinsic scatter via MCMC.

    Parameters
    ----------
    log_hi_mass : array_like
        Base-10 logarithm of the HI mass in solar masses.
    log_diameter : array_like
        Base-10 logarithm of the HI diameter in kpc.
    log_hi_mass_err, log_diameter_err : array_like, optional
        One-sigma uncertainties on ``log_hi_mass`` and ``log_diameter``. Missing
        errors default to zero (so only the intrinsic scatter absorbs the
        spread).
    n_walkers : int, optional
        Number of ensemble walkers, by default 32.
    n_steps : int, optional
        Number of MCMC steps per walker, by default 4000.
    n_burn : int, optional
        Number of initial steps discarded as burn-in, by default 1000.
    thin : int, optional
        Keep one of every ``thin`` post-burn-in samples, by default 10.
    seed : int, optional
        Seed for the random initial walker positions and the sampler, for
        reproducibility.

    Returns
    -------
    MassSizeFit
        Posterior samples of ``(slope, intercept, scatter)``.

    Raises
    ------
    ImportError
        If :mod:`emcee` is not installed.
    ValueError
        If fewer than three valid data points are supplied.

    Notes
    -----
    This is a domain-specific convenience around
    :func:`galaxydisksize.estimators.fit_bayesian_linear`, which performs the
    actual fit; it adds input validation and the size-mass prediction helper.
    """
    x = np.asarray(log_hi_mass, dtype=float)
    y = np.asarray(log_diameter, dtype=float)
    x_err = None if log_hi_mass_err is None else np.asarray(log_hi_mass_err, dtype=float)
    y_err = None if log_diameter_err is None else np.asarray(log_diameter_err, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if x_err is not None:
        valid &= np.isfinite(x_err)
    if y_err is not None:
        valid &= np.isfinite(y_err)
    x, y = x[valid], y[valid]
    x_err = None if x_err is None else x_err[valid]
    y_err = None if y_err is None else y_err[valid]
    if x.size < 3:
        raise ValueError(f"need at least 3 valid data points, got {x.size}")

    # With no measurement errors, pass a negligible floor so the intrinsic
    # scatter absorbs the spread (matching the historical y-only behaviour).
    fit = fit_bayesian_linear(
        x,
        y,
        np.zeros_like(y) if y_err is None else y_err,
        x_err,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        thin=thin,
        seed=42 if seed is None else seed,
    )
    return MassSizeFit(samples=fit.samples, n_data=fit.n_data)
