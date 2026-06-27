"""galaxydisksize: measure and model the atomic-hydrogen disc size of galaxies.

The package collects the reusable, well-tested pieces of the analysis behind the
HI disc-truncation study of Hickson compact groups (HCGs) and isolated AMIGA
galaxies:

``surface_density``
    Unit conversions between HI column density, moment-0 intensity, HI mass, and
    disc diameter, plus the mean HI surface density implied by the size-mass
    relation.
``masssize``
    Bayesian linear fit of the HI size-mass relation
    ``log10(D_HI) = m * log10(M_HI) + b`` with intrinsic scatter.
``residual``
    Size residuals ``Delta = log10(D_HI) - (m * log10(D_25) + b)`` against a
    fitted HI-to-optical baseline, used as the truncation diagnostic.
``survival``
    Kaplan-Meier estimator for left-censored (upper-limit) residuals and the
    Gehan generalised Wilcoxon two-sample test.
``io``
    Loaders for the measurement tables and group configuration files.

The heavyweight, paper-exact reduction scripts that regenerate every figure and
table of the manuscript live under ``workflow/scripts`` and are orchestrated by
Snakemake; this library holds the parts meant to be reused on new data.
"""

from __future__ import annotations

from . import ellipse, estimators, io, masssize, residual, surface_density, survival
from .ellipse import EllipseFitter, EllipseParameters, conic_to_geometric, fit_ellipse_conic
from .estimators import BayesianLinearFit, fit_bayesian_linear, fit_linear
from .masssize import MassSizeFit, fit_mass_size, predict_log_diameter
from .residual import fit_baseline, size_residual
from .survival import KaplanMeier, gehan_test, kaplan_meier_left_censored

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # submodules
    "ellipse",
    "estimators",
    "io",
    "masssize",
    "residual",
    "surface_density",
    "survival",
    # ellipse fitting
    "EllipseFitter",
    "EllipseParameters",
    "fit_ellipse_conic",
    "conic_to_geometric",
    # regression estimators
    "BayesianLinearFit",
    "fit_bayesian_linear",
    "fit_linear",
    # most-used callables, re-exported for convenience
    "MassSizeFit",
    "fit_mass_size",
    "predict_log_diameter",
    "fit_baseline",
    "size_residual",
    "KaplanMeier",
    "kaplan_meier_left_censored",
    "gehan_test",
]
