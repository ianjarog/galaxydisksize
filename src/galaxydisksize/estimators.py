r"""Linear-regression estimators for ``log y = m log x + b``.

Choosing the estimator for the HI-to-optical baseline matters: if the residuals
of the adopted fit correlate with :math:`\log D_{25}`, that bias propagates into
every residual measured against the baseline. The study therefore compares nine
estimators and adopts the one that is unbiased with the smallest scatter (the
Bayesian fit). This module provides them as a single, uniform interface:

* simple estimators with no error model: :func:`fit_ols`, :func:`fit_theil_sen`,
  :func:`fit_bisector`;
* errors-in-variables estimators: :func:`fit_odr`, :func:`fit_york`,
  :func:`fit_bces`, :func:`fit_linmix`, :func:`fit_hyperfit`;
* the Bayesian fit with intrinsic scatter, :func:`fit_bayesian_linear`, which is
  the adopted baseline estimator and the engine used elsewhere in the package.

:func:`fit_linear` dispatches by name and returns the slope, intercept, scatter,
and the residual-trend diagnostics used to judge each estimator.

``bces``, ``linmix``, and ``hyperfit`` are optional third-party packages; the
estimators that need them raise an informative :class:`ImportError` if missing,
and :func:`available_estimators` reports which are installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

# Canonical display/order of the estimators compared in the paper (Table 6).
ESTIMATOR_ORDER: tuple[str, ...] = (
    "OLS(Y|X)",
    "Bayesian",
    "Theil-Sen",
    "Bisector",
    "ODR",
    "BCES(Y|X)",
    "York",
    "linmix (Kelly 2007)",
    "HYPER-FIT",
)


def _clean_errors(
    errors: ArrayLike | None, like: NDArray[np.float64], floor: float
) -> NDArray[np.float64]:
    """Return usable per-point errors: finite values floored, else a 0.05 default."""
    if errors is None:
        return np.full_like(like, 0.05)
    errors = np.asarray(errors, dtype=float)
    return np.where(np.isfinite(errors), np.maximum(errors, floor), 0.05)


def fit_ols(log_x: ArrayLike, log_y: ArrayLike) -> tuple[float, float]:
    """Ordinary least squares of ``log_y`` on ``log_x`` (OLS(Y|X))."""
    slope, intercept, _, _, _ = stats.linregress(np.asarray(log_x), np.asarray(log_y))
    return float(slope), float(intercept)


def fit_theil_sen(log_x: ArrayLike, log_y: ArrayLike) -> tuple[float, float]:
    """Theil-Sen median-slope estimator (robust to outliers)."""
    slope, intercept, _, _ = stats.theilslopes(np.asarray(log_y), np.asarray(log_x), 0.95)
    return float(slope), float(intercept)


def fit_bisector(log_x: ArrayLike, log_y: ArrayLike) -> tuple[float, float]:
    """Ordinary-least-squares bisector of the Y|X and X|Y regressions."""
    log_x = np.asarray(log_x)
    log_y = np.asarray(log_y)
    slope_yx, _, _, _, _ = stats.linregress(log_x, log_y)
    slope_xy_inv, _, _, _, _ = stats.linregress(log_y, log_x)
    slope_xy = 1.0 / slope_xy_inv
    slope = ((slope_yx * slope_xy - 1.0) + np.sqrt((1.0 + slope_yx**2) * (1.0 + slope_xy**2))) / (
        slope_yx + slope_xy
    )
    intercept = float(np.mean(log_y) - slope * np.mean(log_x))
    return float(slope), intercept


def fit_odr(
    log_x: ArrayLike, log_y: ArrayLike, log_x_err: ArrayLike | None, log_y_err: ArrayLike | None
) -> tuple[float, float]:
    """Orthogonal distance regression (errors in both coordinates)."""
    from scipy.odr import ODR, Model, RealData

    log_x = np.asarray(log_x)
    log_y = np.asarray(log_y)
    x_err = _clean_errors(log_x_err, log_x, floor=1e-3)
    y_err = _clean_errors(log_y_err, log_y, floor=1e-3)
    slope0, intercept0 = fit_ols(log_x, log_y)
    odr = ODR(
        RealData(log_x, log_y, sx=x_err, sy=y_err),
        Model(lambda beta, x: beta[0] * x + beta[1]),
        beta0=[slope0, intercept0],
    )
    output = odr.run()
    return float(output.beta[0]), float(output.beta[1])


def fit_york(
    log_x: ArrayLike,
    log_y: ArrayLike,
    log_x_err: ArrayLike | None,
    log_y_err: ArrayLike | None,
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """York (2004) errors-in-variables estimator (uncorrelated errors)."""
    log_x = np.asarray(log_x, dtype=float)
    log_y = np.asarray(log_y, dtype=float)
    sx = _clean_errors(log_x_err, log_x, floor=1e-6)
    sy = _clean_errors(log_y_err, log_y, floor=1e-6)
    correlation = np.zeros_like(log_x)
    slope, _ = fit_ols(log_x, log_y)
    weight_x = 1.0 / sx**2
    weight_y = 1.0 / sy**2
    alpha = np.sqrt(weight_x * weight_y)
    x_bar = float(np.mean(log_x))
    y_bar = float(np.mean(log_y))
    for _ in range(max_iter):
        weights = (
            weight_x
            * weight_y
            / (weight_x + slope**2 * weight_y - 2.0 * slope * correlation * alpha)
        )
        x_bar = float(np.sum(weights * log_x) / np.sum(weights))
        y_bar = float(np.sum(weights * log_y) / np.sum(weights))
        u = log_x - x_bar
        v = log_y - y_bar
        beta = weights * (
            (u / weight_y) + slope * (v / weight_x) - ((slope * u + v) * correlation / alpha)
        )
        new_slope = float(np.sum(weights * beta * v) / np.sum(weights * beta * u))
        if not np.isfinite(new_slope):
            break
        if abs(new_slope - slope) < tol:
            slope = new_slope
            break
        slope = new_slope
    return float(slope), float(y_bar - slope * x_bar)


def fit_bces(
    log_x: ArrayLike, log_y: ArrayLike, log_x_err: ArrayLike | None, log_y_err: ArrayLike | None
) -> tuple[float, float]:
    """BCES(Y|X) estimator (Akritas & Bershady 1996). Requires the ``bces`` package."""
    try:
        from bces.bces import bces as bces_regression
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("fit_bces requires the 'bces' package (`pip install bces`).") from exc
    log_x = np.asarray(log_x)
    log_y = np.asarray(log_y)
    x_err = _clean_errors(log_x_err, log_x, floor=1e-6)
    y_err = _clean_errors(log_y_err, log_y, floor=1e-6)
    slopes, intercepts, _, _, _ = bces_regression(log_x, x_err, log_y, y_err, np.zeros_like(log_x))
    return float(slopes[0]), float(intercepts[0])


def fit_linmix(
    log_x: ArrayLike,
    log_y: ArrayLike,
    log_x_err: ArrayLike | None,
    log_y_err: ArrayLike | None,
    *,
    seed: int = 42,
) -> tuple[float, float]:
    """linmix Bayesian estimator (Kelly 2007). Requires the ``linmix`` package."""
    try:
        import linmix
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fit_linmix requires the 'linmix' package (`pip install linmix`)."
        ) from exc
    log_x = np.asarray(log_x)
    log_y = np.asarray(log_y)
    x_sig = _clean_errors(log_x_err, log_x, floor=1e-6)
    y_sig = _clean_errors(log_y_err, log_y, floor=1e-6)
    model = linmix.LinMix(log_x, log_y, xsig=x_sig, ysig=y_sig, parallelize=False, seed=seed)
    model.run_mcmc(miniter=800, maxiter=1600, silent=True)
    return float(np.median(model.chain["beta"])), float(np.median(model.chain["alpha"]))


def fit_hyperfit(
    log_x: ArrayLike, log_y: ArrayLike, log_x_err: ArrayLike | None, log_y_err: ArrayLike | None
) -> tuple[float, float]:
    """Hyper-Fit estimator (Robotham & Obreschkow 2015). Requires ``hyperfit``."""
    try:
        from hyperfit.linfit import LinFit
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fit_hyperfit requires the 'hyperfit' package (`pip install hyperfit`)."
        ) from exc
    log_x = np.asarray(log_x)
    log_y = np.asarray(log_y)
    data = np.vstack([log_x, log_y])
    covariance = np.zeros((2, 2, len(log_x)))
    covariance[0, 0, :] = _clean_errors(log_x_err, log_x, floor=1e-6) ** 2
    covariance[1, 1, :] = _clean_errors(log_y_err, log_y, floor=1e-6) ** 2
    fit = LinFit(data, covariance, vertaxis=1)
    bounds = (
        (float(log_x.min() - 1.0), float(log_x.max() + 1.0)),
        (float(log_y.min() - 1.0), float(log_y.max() + 1.0)),
        (1e-6, 1.0),
    )
    coords, _, _ = fit.optimize(bounds=bounds, tol=1e-6, verbose=False)
    return float(coords[0]), float(coords[1])


@dataclass
class BayesianLinearFit:
    """Posterior of a Bayesian linear fit with intrinsic scatter.

    Attributes
    ----------
    slope, intercept, scatter : float
        Posterior medians of the slope ``m``, intercept ``b``, and intrinsic
        scatter ``sigma`` (in dex).
    samples : numpy.ndarray
        Array of shape ``(n_samples, 3)`` of posterior draws of
        ``(slope, intercept, scatter)`` (scatter already exponentiated).
    n_data : int
        Number of points fitted.
    """

    slope: float
    intercept: float
    scatter: float
    samples: NDArray[np.float64]
    n_data: int

    def percentiles(
        self, levels: tuple[float, ...] = (16.0, 50.0, 84.0)
    ) -> dict[str, NDArray[np.float64]]:
        """Posterior percentiles of slope, intercept, and scatter."""
        return {
            "slope": np.percentile(self.samples[:, 0], levels),
            "intercept": np.percentile(self.samples[:, 1], levels),
            "scatter": np.percentile(self.samples[:, 2], levels),
        }


def fit_bayesian_linear(
    log_x: ArrayLike,
    log_y: ArrayLike,
    log_y_err: ArrayLike | None = None,
    log_x_err: ArrayLike | None = None,
    *,
    n_walkers: int = 50,
    n_steps: int = 3000,
    n_burn: int = 800,
    thin: int = 10,
    seed: int = 42,
) -> BayesianLinearFit:
    r"""Bayesian linear fit ``log_y = m log_x + b`` with intrinsic scatter.

    Samples the posterior with :mod:`emcee` under uniform priors on the slope,
    intercept, and :math:`\ln\sigma`. The likelihood uses the marginalized
    errors-in-variables form, projecting the x-uncertainty onto y through the
    slope: :math:`\mathrm{var} = \sigma^2 + \sigma_y^2 + m^2 \sigma_x^2`. This is
    the adopted baseline estimator of the study.

    Parameters
    ----------
    log_x, log_y : array_like
        Base-10 logarithms of the two quantities.
    log_y_err, log_x_err : array_like, optional
        One-sigma log-space uncertainties. ``log_y_err`` defaults to 0.05 where
        missing; ``log_x_err=None`` recovers a y-only likelihood.
    n_walkers, n_steps, n_burn, thin : int, optional
        Sampler settings: number of walkers (50), steps (3000), burn-in (800),
        and thinning (10).
    seed : int, optional
        Seed for reproducible initialization and sampling, by default 42.

    Returns
    -------
    BayesianLinearFit
        Posterior medians and samples of ``(slope, intercept, scatter)``.

    Raises
    ------
    ImportError
        If :mod:`emcee` is not installed.
    """
    try:
        import emcee
    except ImportError as exc:  # pragma: no cover - exercised only without emcee
        raise ImportError("fit_bayesian_linear requires 'emcee' (`pip install emcee`).") from exc

    log_x = np.asarray(log_x, dtype=float)
    log_y = np.asarray(log_y, dtype=float)
    y_err = _clean_errors(log_y_err, log_y, floor=1e-6)
    if log_x_err is None:
        x_err = np.zeros_like(log_x)
    else:
        log_x_err = np.asarray(log_x_err, dtype=float)
        x_err = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 0.0), 0.0)

    def log_probability(theta: NDArray[np.float64]) -> float:
        slope, intercept, ln_scatter = theta
        if not (-5 < slope < 5 and -10 < intercept < 10 and -10 < ln_scatter < 1):
            return -np.inf
        variance = np.clip(np.exp(2.0 * ln_scatter) + y_err**2 + slope**2 * x_err**2, 1e-20, None)
        model = slope * log_x + intercept
        return float(-0.5 * np.sum((log_y - model) ** 2 / variance + np.log(2 * np.pi * variance)))

    slope0, intercept0 = np.polyfit(log_x, log_y, 1)
    rng = np.random.default_rng(seed)
    initial = np.empty((n_walkers, 3))
    initial[:, 0] = slope0 + 1e-4 * rng.standard_normal(n_walkers)
    initial[:, 1] = intercept0 + 1e-4 * rng.standard_normal(n_walkers)
    initial[:, 2] = -1.0 + 1e-4 * rng.standard_normal(n_walkers)

    np.random.seed(seed)  # noqa: NPY002 - emcee draws proposals from NumPy's global generator
    sampler = emcee.EnsembleSampler(n_walkers, 3, log_probability)
    sampler.run_mcmc(initial, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_burn, thin=thin, flat=True)
    # Exponentiate ln(scatter) so the stored samples are the scatter itself.
    samples = np.column_stack([chain[:, 0], chain[:, 1], np.exp(chain[:, 2])])
    medians = np.median(samples, axis=0)
    return BayesianLinearFit(
        slope=float(medians[0]),
        intercept=float(medians[1]),
        scatter=float(medians[2]),
        samples=samples,
        n_data=int(log_x.size),
    )


# Estimators that take only (log_x, log_y); the rest also take the error arrays.
_NO_ERROR_ESTIMATORS = {
    "OLS(Y|X)": fit_ols,
    "Theil-Sen": fit_theil_sen,
    "Bisector": fit_bisector,
}
_ERROR_ESTIMATORS = {
    "ODR": fit_odr,
    "York": fit_york,
    "BCES(Y|X)": fit_bces,
    "HYPER-FIT": fit_hyperfit,
}
_OPTIONAL_ESTIMATORS = {
    "BCES(Y|X)": "bces",
    "linmix (Kelly 2007)": "linmix",
    "HYPER-FIT": "hyperfit",
}


def available_estimators() -> tuple[str, ...]:
    """Return the estimators whose optional dependencies are installed.

    Returns
    -------
    tuple of str
        The subset of :data:`ESTIMATOR_ORDER` that can run in this environment.
    """
    import importlib.util

    available = []
    for name in ESTIMATOR_ORDER:
        package = _OPTIONAL_ESTIMATORS.get(name)
        if package is not None and importlib.util.find_spec(package) is None:
            continue
        available.append(name)
    return tuple(available)


def fit_linear(
    method: str,
    log_x: ArrayLike,
    log_y: ArrayLike,
    log_x_err: ArrayLike | None = None,
    log_y_err: ArrayLike | None = None,
    *,
    seed: int = 42,
) -> dict[str, float]:
    """Fit with a named estimator and report fit and residual-trend diagnostics.

    Parameters
    ----------
    method : str
        Estimator name, one of :data:`ESTIMATOR_ORDER`.
    log_x, log_y : array_like
        Base-10 logarithms of the two quantities.
    log_x_err, log_y_err : array_like, optional
        One-sigma log-space uncertainties (used by the errors-in-variables
        estimators).
    seed : int, optional
        Seed for the stochastic estimators, by default 42.

    Returns
    -------
    dict
        ``method``, ``slope``, ``intercept``, ``scatter`` and the residual-trend
        diagnostics (Pearson/Spearman of residuals vs ``log_x``, the trend slope,
        and the mean residual in the low- and high-``log_x`` halves). The
        Bayesian fit additionally reports its scatter and credible intervals.
    """
    log_x = np.asarray(log_x, dtype=float)
    log_y = np.asarray(log_y, dtype=float)
    extras: dict[str, float] = {}

    if method == "Bayesian":
        fit = fit_bayesian_linear(log_x, log_y, log_y_err, log_x_err, seed=seed)
        slope, intercept = fit.slope, fit.intercept
        pct = fit.percentiles()
        extras = {
            "sigma_int": fit.scatter,
            "slope_p16": float(pct["slope"][0]),
            "slope_p84": float(pct["slope"][2]),
            "intercept_p16": float(pct["intercept"][0]),
            "intercept_p84": float(pct["intercept"][2]),
            "sigma_int_p16": float(pct["scatter"][0]),
            "sigma_int_p84": float(pct["scatter"][2]),
        }
    elif method in _NO_ERROR_ESTIMATORS:
        slope, intercept = _NO_ERROR_ESTIMATORS[method](log_x, log_y)
    elif method == "linmix (Kelly 2007)":
        slope, intercept = fit_linmix(log_x, log_y, log_x_err, log_y_err, seed=seed)
    elif method in _ERROR_ESTIMATORS:
        slope, intercept = _ERROR_ESTIMATORS[method](log_x, log_y, log_x_err, log_y_err)
    else:
        raise ValueError(f"unknown estimator: {method!r}")

    residuals = log_y - (intercept + slope * log_x)
    pearson = stats.pearsonr(log_x, residuals)
    spearman = stats.spearmanr(log_x, residuals)
    trend = stats.linregress(log_x, residuals)
    low = log_x <= np.median(log_x)
    result = {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "scatter": float(np.std(residuals, ddof=2)),
        "residual_pearson_r": float(pearson.statistic),
        "residual_pearson_p": float(pearson.pvalue),
        "residual_spearman_rho": float(spearman.statistic),
        "residual_spearman_p": float(spearman.pvalue),
        "residual_trend_slope": float(trend.slope),
        "low_bin_mean": float(np.mean(residuals[low])),
        "high_bin_mean": float(np.mean(residuals[~low])),
        "n": int(log_x.size),
    }
    result.update(extras)
    return result
