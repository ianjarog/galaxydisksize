"""Survival statistics for left-censored (upper-limit) size residuals.

Non-detected galaxies carry an *upper limit* on the HI diameter, hence an upper
limit on the size residual ``Delta`` (Section :mod:`galaxydisksize.residual`).
These are left-censored data. This module provides:

* :func:`kaplan_meier_left_censored` -- the Kaplan-Meier estimator of the
  residual distribution (Feigelson & Nelson 1985, ApJ 293, 192), and
* :func:`gehan_test` -- the Gehan generalised Wilcoxon two-sample test
  (Gehan 1965) for comparing two censored residual distributions.

The implementation maps the left-censored problem onto the standard
right-censored machinery by working with ``U = -Delta``: an upper limit
``Delta <= Dlim`` becomes a lower limit ``U >= -Dlim``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass
class KaplanMeier:
    """Kaplan-Meier estimate of a left-censored residual distribution.

    Instances are returned by :func:`kaplan_meier_left_censored`. The step
    function is stored on the transformed variable ``U = -Delta`` so that the
    distribution can be queried through the convenience methods rather than by
    manipulating ``u_steps`` and ``survival`` directly.

    Attributes
    ----------
    u_steps : numpy.ndarray
        Sorted unique values of ``U = -Delta`` at which the survival function
        is evaluated.
    survival : numpy.ndarray
        Right-continuous Kaplan-Meier survival function ``S_U(t) = P(U > t)``
        evaluated at ``u_steps``.
    median : float
        Median of ``Delta``. ``numpy.nan`` if the median lies in the
        unconstrained censored tail (i.e. ``S_U`` never drops to 0.5), which
        happens when more than half of the sample are upper limits.
    mean : float
        Kaplan-Meier mean of ``Delta``, restricted to the observed support.
    """

    u_steps: NDArray[np.float64]
    survival: NDArray[np.float64]
    median: float
    mean: float

    def fraction_below(self, threshold: float) -> float:
        """Estimated probability that ``Delta`` is below a threshold.

        Parameters
        ----------
        threshold : float
            Residual value ``x`` at which to evaluate ``P(Delta < x)``.

        Returns
        -------
        float
            ``P(Delta < x) = P(U > -x) = S_U(-x)``.
        """
        u = -threshold
        idx = np.where(self.u_steps <= u)[0]
        return float(self.survival[idx[-1]]) if idx.size else 1.0

    def quantile(self, probability: float) -> tuple[float, bool]:
        """Smallest ``Delta`` whose cumulative probability reaches ``probability``.

        Parameters
        ----------
        probability : float
            Target cumulative probability ``P(Delta <= x)`` in the open
            interval ``(0, 1)``.

        Returns
        -------
        value : float
            The estimated quantile of ``Delta``.
        constrained : bool
            ``True`` if the quantile is determined by the data, ``False`` if it
            falls in the unconstrained censored tail (in which case ``value`` is
            the relevant support boundary and should be read as a bound).
        """
        grid = -self.u_steps[::-1]
        cumulative = self.survival[::-1]
        reached = np.where(cumulative >= probability)[0]
        if reached.size == 0:
            return float(grid[0]), False
        return float(grid[reached[0]]), True


def kaplan_meier_left_censored(residuals: ArrayLike, is_upper_limit: ArrayLike) -> KaplanMeier:
    """Kaplan-Meier estimator for left-censored residuals.

    Parameters
    ----------
    residuals : array_like
        Size residuals ``Delta``. For upper limits this is the residual
        evaluated at the limiting (beam-size) diameter.
    is_upper_limit : array_like of bool
        ``True`` where the corresponding residual is an upper limit
        (left-censored), ``False`` for a detection (an exact value, i.e. an
        "event" in survival-analysis terms).

    Returns
    -------
    KaplanMeier
        The estimated survival function and summary statistics.

    Notes
    -----
    Internally the estimator works on ``U = -Delta``, on which upper limits
    become right-censored. The product-limit estimator is applied to ``U`` and
    the results are mapped back to ``Delta``.
    """
    delta = np.asarray(residuals, dtype=float)
    is_limit = np.asarray(is_upper_limit, dtype=bool)
    u = -delta  # right-censored at U for upper limits

    unique_u = np.unique(u)
    survival = 1.0
    u_steps: list[float] = []
    survival_steps: list[float] = []
    for t in unique_u:
        at_t = u == t
        n_events = int(np.sum(at_t & ~is_limit))  # detections at t
        n_at_risk = int(np.sum(u >= t))  # subjects with U >= t
        if n_events > 0 and n_at_risk > 0:
            survival *= 1.0 - n_events / n_at_risk
        u_steps.append(float(t))
        survival_steps.append(survival)

    u_array = np.asarray(u_steps, dtype=float)
    s_array = np.asarray(survival_steps, dtype=float)

    below_half = np.where(s_array <= 0.5)[0]
    median = float(-u_array[below_half[0]]) if below_half.size else np.nan

    # KM mean of U via the area under S_U, restricted to the observed support,
    # then negated to return to Delta.
    u_min = u_array[0]
    padded_u = np.concatenate(([u_min], u_array))
    padded_s = np.concatenate(([1.0], s_array))
    area = float(np.sum((padded_u[1:] - padded_u[:-1]) * padded_s[:-1]))
    mean = -(u_min + area)

    return KaplanMeier(u_steps=u_array, survival=s_array, median=median, mean=mean)


@dataclass
class GehanResult:
    """Result of a Gehan generalised Wilcoxon two-sample test.

    Attributes
    ----------
    statistic : float
        The Gehan sum-of-scores statistic ``W`` for the first sample.
    z : float
        Standardised statistic ``W / sqrt(var)``.
    p_value : float
        Two-sided p-value from the normal approximation.
    variance : float
        Permutation variance of ``W``.
    """

    statistic: float
    z: float
    p_value: float
    variance: float


def gehan_test(
    residuals_a: ArrayLike,
    is_limit_a: ArrayLike,
    residuals_b: ArrayLike,
    is_limit_b: ArrayLike,
) -> GehanResult:
    """Gehan generalised Wilcoxon test comparing two censored samples.

    Upper limits (left-censored) are handled through ``U = -Delta``
    (right-censored). Each subject is scored by the number of pooled subjects it
    is definitely greater than minus the number it is definitely less than;
    censoring makes some comparisons indeterminate (ties). The test reduces to
    the Wilcoxon rank-sum test when no value is censored.

    Parameters
    ----------
    residuals_a, residuals_b : array_like
        Size residuals ``Delta`` for samples A and B.
    is_limit_a, is_limit_b : array_like of bool
        Upper-limit flags for samples A and B.

    Returns
    -------
    GehanResult
        Test statistic, standardised value, two-sided p-value, and variance.
    """
    u = np.concatenate(
        [-np.asarray(residuals_a, dtype=float), -np.asarray(residuals_b, dtype=float)]
    )
    censored = np.concatenate(
        [np.asarray(is_limit_a, dtype=bool), np.asarray(is_limit_b, dtype=bool)]
    )
    group = np.concatenate(
        [np.zeros(len(residuals_a), dtype=int), np.ones(len(residuals_b), dtype=int)]
    )
    n_total = len(u)

    # Pairwise Gehan scores on right-censored U. A censored value c_i means the
    # true value is at least u_i, so some comparisons cannot be resolved.
    score = np.zeros(n_total)
    for i in range(n_total):
        running = 0
        u_i, censored_i = u[i], censored[i]
        for j in range(n_total):
            if i == j:
                continue
            u_j, censored_j = u[j], censored[j]
            if not censored_i and not censored_j:
                if u_i > u_j:
                    running += 1
                elif u_i < u_j:
                    running -= 1
            elif censored_i and not censored_j:
                if u_i >= u_j:
                    running += 1
            elif not censored_i and censored_j:
                if u_j >= u_i:
                    running -= 1
            # both censored: indeterminate, contributes nothing
        score[i] = running

    statistic = float(np.sum(score[group == 0]))
    n_a = int(np.sum(group == 0))
    n_b = n_total - n_a
    variance = n_a * n_b / (n_total * (n_total - 1)) * float(np.sum(score**2))
    z = statistic / np.sqrt(variance) if variance > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z)))
    return GehanResult(statistic=statistic, z=z, p_value=p_value, variance=variance)
