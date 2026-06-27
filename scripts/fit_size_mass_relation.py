#!/usr/bin/env python3
"""
Statistical consistency test for the HI mass-size relation.

Tests whether AMIGA (resolved 35) and HCGs follow the same mass-size relation
as the literature sample (Wang+16 + MIGHTEE).  Three complementary approaches:

1. Bayesian model comparison (BIC): single relation vs separate relations
2. Residual analysis: KS, Anderson-Darling, and t-tests on residuals
3. Posterior overlap: fraction of AMIGA+HCG posteriors within literature
   credible region

If consistency is confirmed, the combined relation can be used to infer HI
diameters for the larger AMIGA sample.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import corner
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats
from scipy.special import log_ndtr
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_FIGURES_DIR = PROJECT_ROOT / "figures"
ANALYSIS_PRODUCTS_DIR = PROJECT_ROOT / "products"
ANALYSIS_LATEX_DIR = PROJECT_ROOT / "latex"

MPLCONFIGDIR = ANALYSIS_PRODUCTS_DIR / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

CENTRAL_68_PERCENTILES = (16, 50, 84)
CENTRAL_95_PERCENTILES = (2.5, 97.5)

# Standard M_HI (x-axis) uncertainty, following the Jones et al. (2018, 2023)
# prescription: a 10% absolute flux-calibration term added in quadrature to the
# (usually negligible) statistical rms term. 0.10/ln(10) = 0.043 dex.
# Distance uncertainty is neglected (Jones+2023); for the size-mass relation it
# moves points along the slope-0.5 relation and does not bias the slope.
CALIB_LOGMHI_ERR = 0.10 / np.log(10.0)


def _log_mhi_err(hi_mass=None, hi_mass_err=None, n=None):
    """Per-point error on log10(M_HI): sqrt(statistical^2 + calibration^2)."""
    calib = CALIB_LOGMHI_ERR
    if hi_mass is None or hi_mass_err is None:
        return np.full(int(n), calib) if n is not None else calib
    hi_mass = np.asarray(hi_mass, float)
    hi_mass_err = np.asarray(hi_mass_err, float)
    stat = np.where(
        np.isfinite(hi_mass_err) & (hi_mass > 0),
        hi_mass_err / (hi_mass * np.log(10.0)),
        0.0,
    )
    return np.sqrt(stat**2 + calib**2)


# ---------------------------------------------------------------------------
# Utility: tee output to file + console
# ---------------------------------------------------------------------------


class _TeeStream:
    def __init__(self, file_handle, original_stream=None, echo=True):
        self._fh = file_handle
        self._orig = original_stream
        self._echo = echo and (original_stream is not None)

    def write(self, s):
        self._fh.write(s)
        self._fh.flush()
        if self._echo:
            self._orig.write(s)
            self._orig.flush()

    def flush(self):
        self._fh.flush()
        if self._echo:
            self._orig.flush()


@contextmanager
def _redirect_output(report_path, echo_to_console=True, mode="w"):
    report_dir = os.path.dirname(os.path.abspath(report_path))
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)

    fh = open(report_path, mode, encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(fh, old_out, echo=echo_to_console)
    sys.stderr = _TeeStream(fh, old_err, echo=echo_to_console)

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n===== Report started: {ts} =====\n")

    try:
        yield fh
    finally:
        ts2 = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            print(f"\n===== Report finished: {ts2} =====\n")
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()


def ensure_directories() -> None:
    for directory in (
        ANALYSIS_FIGURES_DIR,
        ANALYSIS_PRODUCTS_DIR,
        ANALYSIS_LATEX_DIR,
        MPLCONFIGDIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def configure_fonts(font_dirs: list[str]) -> None:
    font = "tex gyre heros"
    if font_dirs:
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
    mpl.rcParams["font.sans-serif"] = font
    mpl.rc("mathtext", fontset="custom", it=font + ":italic")
    mpl.rc("font", size=18)


def style_axes(ax) -> None:
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8, width=1.2, pad=10)
    ax.tick_params(which="minor", length=4, width=1, pad=10)


# ---------------------------------------------------------------------------
# Data loaders (copied from plot_size_mass_all_surveys.py for
# self-containment, matching project convention)
# ---------------------------------------------------------------------------


def load_wang_mass_size_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dash_count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("---"):
                dash_count += 1
                continue
            if dash_count < 2 or not stripped or stripped.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            try:
                dhi_kpc = float(parts[1])
                log_mhi = float(parts[2])
            except ValueError:
                continue

            if dhi_kpc <= 0:
                continue

            chunk = line[82:].strip()
            split_chunk = re.split(r"\s{2,}", chunk)
            sample_name = split_chunk[0] if split_chunk else parts[8]

            rows.append(
                {
                    "source_group": "Wang+16",
                    "source_subgroup": sample_name,
                    "log_mhi": log_mhi,
                    "log_dhi": float(np.log10(dhi_kpc)),
                    "yerr_log_dhi": 0.0,
                    "xerr_log_mhi": CALIB_LOGMHI_ERR,
                    "is_upper_limit": False,
                }
            )

    return pd.DataFrame(rows)


def load_mightee_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                log_mhi = float(parts[0])
                log_dhi = float(parts[1])
            except ValueError:
                continue
            rows.append(
                {
                    "source_group": "MIGHTEE",
                    "source_subgroup": parts[2],
                    "log_mhi": log_mhi,
                    "log_dhi": log_dhi,
                    "yerr_log_dhi": 0.0,
                    "xerr_log_mhi": CALIB_LOGMHI_ERR,
                    "is_upper_limit": False,
                }
            )
    return pd.DataFrame(rows)


def load_amiga_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"hi_mass", "hi_diameter_kpc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"AMIGA CSV missing required columns: {sorted(missing)}")

    hi_mass = pd.to_numeric(df["hi_mass"], errors="coerce")
    dhi = pd.to_numeric(df["hi_diameter_kpc"], errors="coerce")
    if "hi_diameter_err_kpc" in df.columns:
        dhi_err = pd.to_numeric(df["hi_diameter_err_kpc"], errors="coerce")
        mask = hi_mass.notna() & dhi.notna() & (hi_mass > 0) & (dhi > 0)
    else:
        mask = hi_mass.notna() & dhi.notna() & (hi_mass > 0) & (dhi > 0)
        dhi_err = pd.Series(np.nan, index=df.index)

    subset = df.loc[mask].copy()
    subset_err = dhi_err.loc[mask]
    yerr = np.where(
        subset_err.notna().to_numpy(),
        subset_err.to_numpy(float)
        / (
            pd.to_numeric(subset["hi_diameter_kpc"], errors="coerce").to_numpy(float) * np.log(10.0)
        ),
        0.0,
    )
    hi_mass_arr = pd.to_numeric(subset["hi_mass"], errors="coerce").to_numpy(float)
    hi_mass_err_arr = (
        pd.to_numeric(subset["hi_mass_err"], errors="coerce").to_numpy(float)
        if "hi_mass_err" in subset.columns
        else None
    )
    return pd.DataFrame(
        {
            "source_group": "AMIGA",
            "source_subgroup": "resolved",
            "log_mhi": np.log10(hi_mass_arr),
            "log_dhi": np.log10(
                pd.to_numeric(subset["hi_diameter_kpc"], errors="coerce").to_numpy(float)
            ),
            "yerr_log_dhi": yerr,
            "xerr_log_mhi": _log_mhi_err(hi_mass_arr, hi_mass_err_arr, n=len(subset)),
            "is_upper_limit": subset_err.isna().to_numpy(),
        }
    )


def load_hcg_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"hi_mass", "hi_diameter_kpc", "phase"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"HCG CSV missing required columns: {sorted(missing)}")

    hi_mass = pd.to_numeric(df["hi_mass"], errors="coerce")
    dhi = pd.to_numeric(df["hi_diameter_kpc"], errors="coerce")
    if "hi_diameter_err_kpc" in df.columns:
        dhi_err = pd.to_numeric(df["hi_diameter_err_kpc"], errors="coerce")
        mask = hi_mass.notna() & dhi.notna() & (hi_mass > 0) & (dhi > 0)
    else:
        mask = hi_mass.notna() & dhi.notna() & (hi_mass > 0) & (dhi > 0)
        dhi_err = pd.Series(np.nan, index=df.index)

    subset = df.loc[mask].copy()
    subset_err = dhi_err.loc[mask]
    yerr = np.where(
        subset_err.notna().to_numpy(),
        subset_err.to_numpy(float)
        / (
            pd.to_numeric(subset["hi_diameter_kpc"], errors="coerce").to_numpy(float) * np.log(10.0)
        ),
        0.0,
    )
    hi_mass_arr = pd.to_numeric(subset["hi_mass"], errors="coerce").to_numpy(float)
    hi_mass_err_arr = (
        pd.to_numeric(subset["hi_mass_err"], errors="coerce").to_numpy(float)
        if "hi_mass_err" in subset.columns
        else None
    )
    return pd.DataFrame(
        {
            "source_group": "HCG",
            "source_subgroup": subset["phase"].astype(str).str.strip().to_numpy(),
            "log_mhi": np.log10(hi_mass_arr),
            "log_dhi": np.log10(
                pd.to_numeric(subset["hi_diameter_kpc"], errors="coerce").to_numpy(float)
            ),
            "yerr_log_dhi": yerr,
            "xerr_log_mhi": _log_mhi_err(hi_mass_arr, hi_mass_err_arr, n=len(subset)),
            "is_upper_limit": subset_err.isna().to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# Bayesian fitting (extended to return max log-likelihood for BIC)
# ---------------------------------------------------------------------------


def _build_log_functions(x_det, y_det, yerr_det, x_ul, y_ul, yerr_ul, xerr_det=None, xerr_ul=None):
    """Return (log_prior, log_likelihood, log_posterior) closures.

    Implements the Jones et al. (2018) Eq. 14 errors-in-variables likelihood in
    its marginalized (effective-variance) form: the x-uncertainty is projected
    onto the y-direction through the slope and added in quadrature,
        var_eff = sigma_int^2 + sigma_y^2 + m^2 * sigma_x^2 .
    Passing xerr=None (or zeros) recovers the previous y-only likelihood.
    """
    xerr_det = np.zeros_like(x_det) if xerr_det is None else np.asarray(xerr_det, float)
    xerr_ul = np.zeros_like(x_ul) if xerr_ul is None else np.asarray(xerr_ul, float)

    def log_prior(theta):
        m, b, lnf = theta
        if -5 < m < 5 and -10 < b < 10 and -10 < lnf < 1:
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        m, b, lnf = theta
        sig2_int = np.exp(2.0 * lnf)

        mu_det = m * x_det + b
        var_det = np.clip(sig2_int + yerr_det**2 + (m**2) * xerr_det**2, 1e-20, None)
        ll_det = -0.5 * np.sum((y_det - mu_det) ** 2 / var_det + np.log(2 * np.pi * var_det))

        if x_ul.size:
            mu_ul = m * x_ul + b
            std_ul = np.clip(np.sqrt(sig2_int + yerr_ul**2 + (m**2) * xerr_ul**2), 1e-10, None)
            z = (y_ul - mu_ul) / std_ul
            ll_ul = np.sum(log_ndtr(z))
        else:
            ll_ul = 0.0

        return ll_det + ll_ul

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    return log_prior, log_likelihood, log_posterior


def bayesian_fit(x_det, y_det, yerr_det, x_ul, y_ul, yerr_ul, xerr_det=None, xerr_ul=None):
    """Run MCMC and return (samples, max_log_likelihood)."""
    _, log_likelihood, log_posterior = _build_log_functions(
        x_det, y_det, yerr_det, x_ul, y_ul, yerr_ul, xerr_det, xerr_ul
    )

    ndim, nwalkers = 3, 100
    m0, b0 = np.polyfit(x_det, y_det, 1)
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = m0 + 1e-4 * np.random.randn(nwalkers)
    pos[:, 1] = b0 + 1e-4 * np.random.randn(nwalkers)
    pos[:, 2] = -1.0 + 1e-4 * np.random.randn(nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(pos, 4000, progress=True)

    samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    log_prob = sampler.get_log_prob(discard=1000, thin=15, flat=True)

    # max log-likelihood: evaluate likelihood (not posterior) at MAP sample
    map_idx = np.argmax(log_prob)
    map_theta = samples[map_idx]
    max_ll = log_likelihood(map_theta)

    return samples, max_ll


def compute_bayesian_fit(df: pd.DataFrame) -> dict:
    """Fit the mass-size relation to DETECTIONS ONLY.

    Upper limits (beam-limited diameters) are excluded from the fit: this
    relation is used to infer D_HI for the single-dish AMIGA galaxies, so it
    must not be anchored by non-detections.
    """
    det = df[~df["is_upper_limit"]].copy()
    ul = det.iloc[0:0].copy()  # detection-only: no censored points enter the fit

    x_det = det["log_mhi"].to_numpy(float)
    y_det = det["log_dhi"].to_numpy(float)
    yerr_det = det["yerr_log_dhi"].to_numpy(float)
    x_ul = ul["log_mhi"].to_numpy(float)
    y_ul = ul["log_dhi"].to_numpy(float)
    yerr_ul = ul["yerr_log_dhi"].to_numpy(float)
    xerr_det = det["xerr_log_mhi"].to_numpy(float) if "xerr_log_mhi" in det.columns else None
    xerr_ul = ul["xerr_log_mhi"].to_numpy(float) if "xerr_log_mhi" in ul.columns else None

    samples, max_ll = bayesian_fit(x_det, y_det, yerr_det, x_ul, y_ul, yerr_ul, xerr_det, xerr_ul)
    percentiles = np.percentile(samples, CENTRAL_68_PERCENTILES, axis=0)
    percentiles_95 = np.percentile(samples, CENTRAL_95_PERCENTILES, axis=0)
    slope_med = float(percentiles[1, 0])
    intercept_med = float(percentiles[1, 1])
    scatter_med = float(np.exp(percentiles[1, 2]))

    det_residuals = y_det - (slope_med * x_det + intercept_med)
    pearson_r, pearson_p = stats.pearsonr(x_det, y_det)

    return {
        "method": "Bayesian",
        "slope": slope_med,
        "intercept": intercept_med,
        "scatter": scatter_med,
        "r_value": float(pearson_r),
        "p_value": float(pearson_p),
        "n": int(len(det)),
        "n_det": int(len(det)),
        "n_ul": int(len(ul)),
        "slope_p16": float(percentiles[0, 0]),
        "slope_p84": float(percentiles[2, 0]),
        "slope_p025": float(percentiles_95[0, 0]),
        "slope_p975": float(percentiles_95[1, 0]),
        "intercept_p16": float(percentiles[0, 1]),
        "intercept_p84": float(percentiles[2, 1]),
        "intercept_p025": float(percentiles_95[0, 1]),
        "intercept_p975": float(percentiles_95[1, 1]),
        "scatter_p16": float(np.exp(percentiles[0, 2])),
        "scatter_p84": float(np.exp(percentiles[2, 2])),
        "scatter_p025": float(np.exp(percentiles_95[0, 2])),
        "scatter_p975": float(np.exp(percentiles_95[1, 2])),
        "posterior_samples": samples,
        "max_log_likelihood": max_ll,
        "residual_scatter_det": float(np.std(det_residuals, ddof=2)),
    }


# ---------------------------------------------------------------------------
# Test 1: Bayesian model comparison via BIC
# ---------------------------------------------------------------------------


def run_bayesian_model_comparison(
    fit_combined: dict,
    fit_literature: dict,
    fit_test: dict,
    n_combined: int,
    n_literature: int,
    n_test: int,
) -> dict:
    """Compare single-relation (M1) vs two-relation (M2) models using BIC."""
    k1 = 3  # single relation: slope, intercept, ln(scatter)
    k2 = 6  # two separate relations

    bic_m1 = -2.0 * fit_combined["max_log_likelihood"] + k1 * np.log(n_combined)
    bic_m2 = (
        -2.0 * fit_literature["max_log_likelihood"]
        - 2.0 * fit_test["max_log_likelihood"]
        + k2 * np.log(n_combined)
    )
    delta_bic = bic_m2 - bic_m1  # positive favors M1 (single relation)

    if delta_bic > 10:
        interpretation = "Very strong evidence for single relation (delta_BIC > 10)"
    elif delta_bic > 6:
        interpretation = "Strong evidence for single relation (delta_BIC > 6)"
    elif delta_bic > 2:
        interpretation = "Positive evidence for single relation (delta_BIC > 2)"
    elif delta_bic > 0:
        interpretation = "Weak evidence for single relation (0 < delta_BIC < 2)"
    else:
        interpretation = "Evidence favors separate relations (delta_BIC < 0)"

    return {
        "bic_m1_single": float(bic_m1),
        "bic_m2_separate": float(bic_m2),
        "delta_bic": float(delta_bic),
        "interpretation": interpretation,
        "max_ll_combined": float(fit_combined["max_log_likelihood"]),
        "max_ll_literature": float(fit_literature["max_log_likelihood"]),
        "max_ll_test": float(fit_test["max_log_likelihood"]),
    }


# ---------------------------------------------------------------------------
# Test 1b: Heteroscedastic BIC — shared slope+intercept, separate scatter
# ---------------------------------------------------------------------------


def _heteroscedastic_log_likelihood(
    theta,
    x_det_lit,
    y_det_lit,
    yerr_det_lit,
    x_ul_lit,
    y_ul_lit,
    yerr_ul_lit,
    x_det_test,
    y_det_test,
    yerr_det_test,
    x_ul_test,
    y_ul_test,
    yerr_ul_test,
):
    """Log-likelihood for shared slope+intercept with group-specific scatter."""
    m, b, lnf_lit, lnf_test = theta
    sig2_lit = np.exp(2.0 * lnf_lit)
    sig2_test = np.exp(2.0 * lnf_test)

    # Literature detections
    mu = m * x_det_lit + b
    var = np.clip(sig2_lit + yerr_det_lit**2, 1e-20, None)
    ll = -0.5 * np.sum((y_det_lit - mu) ** 2 / var + np.log(2 * np.pi * var))

    # Literature upper limits
    if x_ul_lit.size:
        mu_ul = m * x_ul_lit + b
        std_ul = np.clip(np.sqrt(sig2_lit + yerr_ul_lit**2), 1e-10, None)
        ll += np.sum(log_ndtr((y_ul_lit - mu_ul) / std_ul))

    # Test detections
    mu = m * x_det_test + b
    var = np.clip(sig2_test + yerr_det_test**2, 1e-20, None)
    ll += -0.5 * np.sum((y_det_test - mu) ** 2 / var + np.log(2 * np.pi * var))

    # Test upper limits
    if x_ul_test.size:
        mu_ul = m * x_ul_test + b
        std_ul = np.clip(np.sqrt(sig2_test + yerr_ul_test**2), 1e-10, None)
        ll += np.sum(log_ndtr((y_ul_test - mu_ul) / std_ul))

    return ll


def fit_heteroscedastic_map(lit_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Find MAP for shared slope+intercept, separate scatter. Return max LL."""
    from scipy.optimize import minimize

    def _extract(df):
        det = df[~df["is_upper_limit"]]
        ul = df[df["is_upper_limit"]]
        return (
            det["log_mhi"].to_numpy(float),
            det["log_dhi"].to_numpy(float),
            det["yerr_log_dhi"].to_numpy(float),
            ul["log_mhi"].to_numpy(float),
            ul["log_dhi"].to_numpy(float),
            ul["yerr_log_dhi"].to_numpy(float),
        )

    lit_data = _extract(lit_df)
    test_data = _extract(test_df)

    def neg_ll(theta):
        m, b, lnf_lit, lnf_test = theta
        if not (-5 < m < 5 and -10 < b < 10 and -10 < lnf_lit < 1 and -10 < lnf_test < 1):
            return 1e10
        return -_heteroscedastic_log_likelihood(theta, *lit_data, *test_data)

    # Initialize from polyfit on all detections
    all_x = np.concatenate([lit_data[0], test_data[0]])
    all_y = np.concatenate([lit_data[1], test_data[1]])
    m0, b0 = np.polyfit(all_x, all_y, 1)

    best = None
    for lnf_lit0, lnf_test0 in [(-2.0, -2.0), (-3.0, -2.5), (-2.5, -2.0), (-2.0, -2.5)]:
        res = minimize(
            neg_ll,
            [m0, b0, lnf_lit0, lnf_test0],
            method="Nelder-Mead",
            options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res

    max_ll = -best.fun
    m, b, lnf_lit, lnf_test = best.x

    return {
        "max_log_likelihood": float(max_ll),
        "slope": float(m),
        "intercept": float(b),
        "scatter_lit": float(np.exp(lnf_lit)),
        "scatter_test": float(np.exp(lnf_test)),
    }


def run_heteroscedastic_bic(
    fit_heteroscedastic: dict,
    fit_literature: dict,
    fit_test: dict,
    n_combined: int,
) -> dict:
    """Compare heteroscedastic (shared relation, 4 params) vs fully separate (6 params).

    Positive delta_BIC means the heteroscedastic model (shared slope+intercept)
    is preferred — i.e. no evidence that the relation itself differs.
    """
    k_hetero = 4  # m, b, sigma_lit, sigma_test
    k_separate = 6  # m_lit, b_lit, sigma_lit, m_test, b_test, sigma_test

    bic_hetero = -2.0 * fit_heteroscedastic["max_log_likelihood"] + k_hetero * np.log(n_combined)
    bic_separate = (
        -2.0 * fit_literature["max_log_likelihood"]
        - 2.0 * fit_test["max_log_likelihood"]
        + k_separate * np.log(n_combined)
    )
    delta_bic = bic_separate - bic_hetero  # positive favors shared relation

    if delta_bic > 10:
        interpretation = "Very strong evidence for shared relation (delta_BIC > 10)"
    elif delta_bic > 6:
        interpretation = "Strong evidence for shared relation (delta_BIC > 6)"
    elif delta_bic > 2:
        interpretation = "Positive evidence for shared relation (delta_BIC > 2)"
    elif delta_bic > -2:
        interpretation = "No significant preference (|delta_BIC| < 2)"
    else:
        interpretation = "Evidence favors separate relations (delta_BIC < -2)"

    return {
        "bic_hetero": float(bic_hetero),
        "bic_separate": float(bic_separate),
        "delta_bic": float(delta_bic),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Test 1c: Posterior parameter differences
# ---------------------------------------------------------------------------


def run_posterior_differences(
    literature_samples: np.ndarray,
    test_samples: np.ndarray,
) -> dict:
    """Compute distributions of parameter differences from independent posteriors."""
    n = min(len(literature_samples), len(test_samples))
    idx_lit = np.random.choice(len(literature_samples), n, replace=False)
    idx_test = np.random.choice(len(test_samples), n, replace=False)

    delta_slope = test_samples[idx_test, 0] - literature_samples[idx_lit, 0]
    delta_intercept = test_samples[idx_test, 1] - literature_samples[idx_lit, 1]
    delta_scatter = np.exp(test_samples[idx_test, 2]) - np.exp(literature_samples[idx_lit, 2])

    results: dict = {}
    for name, arr in [
        ("slope", delta_slope),
        ("intercept", delta_intercept),
        ("scatter", delta_scatter),
    ]:
        p16, med, p84 = np.percentile(arr, CENTRAL_68_PERCENTILES)
        lo95, hi95 = np.percentile(arr, CENTRAL_95_PERCENTILES)
        zero_in_95 = bool(lo95 <= 0 <= hi95)
        # Fraction of posterior mass consistent with zero (within ±ε)
        float(np.mean(np.abs(arr) < np.std(arr) * 0.1))  # within 10% of std
        results[f"delta_{name}_median"] = float(med)
        results[f"delta_{name}_p16"] = float(p16)
        results[f"delta_{name}_p84"] = float(p84)
        results[f"delta_{name}_95ci"] = (float(lo95), float(hi95))
        results[f"delta_{name}_zero_in_95ci"] = zero_in_95
        # Tension in sigma: |median| / half-width of 68% CI
        sigma_tension = abs(med) / max((p84 - p16) / 2, 1e-10)
        results[f"delta_{name}_tension_sigma"] = float(sigma_tension)

    return results


# ---------------------------------------------------------------------------
# Test 2: Residual analysis
# ---------------------------------------------------------------------------


def _compute_residuals(df: pd.DataFrame, slope: float, intercept: float) -> np.ndarray:
    """Residuals for detections only."""
    det = df[~df["is_upper_limit"]]
    return det["log_dhi"].to_numpy(float) - (slope * det["log_mhi"].to_numpy(float) + intercept)


def run_residual_analysis(
    literature_df: pd.DataFrame,
    amiga_df: pd.DataFrame,
    hcg_df: pd.DataFrame,
    fit_literature: dict,
) -> dict:
    """KS, Anderson-Darling, and t-tests on residuals from the literature fit."""
    slope = fit_literature["slope"]
    intercept = fit_literature["intercept"]

    res_lit = _compute_residuals(literature_df, slope, intercept)
    res_amiga = _compute_residuals(amiga_df, slope, intercept)
    res_hcg = _compute_residuals(hcg_df, slope, intercept)
    res_test = np.concatenate([res_amiga, res_hcg])

    results: dict = {}

    # KS tests
    ks_amiga = stats.ks_2samp(res_lit, res_amiga)
    ks_hcg = stats.ks_2samp(res_lit, res_hcg)
    ks_combined = stats.ks_2samp(res_lit, res_test)
    results["ks_amiga"] = {
        "statistic": float(ks_amiga.statistic),
        "p_value": float(ks_amiga.pvalue),
    }
    results["ks_hcg"] = {"statistic": float(ks_hcg.statistic), "p_value": float(ks_hcg.pvalue)}
    results["ks_combined"] = {
        "statistic": float(ks_combined.statistic),
        "p_value": float(ks_combined.pvalue),
    }

    # Anderson-Darling k-sample tests
    ad_amiga = stats.anderson_ksamp([res_lit, res_amiga])
    ad_hcg = stats.anderson_ksamp([res_lit, res_hcg])
    ad_combined = stats.anderson_ksamp([res_lit, res_test])
    results["ad_amiga"] = {
        "statistic": float(ad_amiga.statistic),
        "p_value": float(ad_amiga.pvalue),
    }
    results["ad_hcg"] = {"statistic": float(ad_hcg.statistic), "p_value": float(ad_hcg.pvalue)}
    results["ad_combined"] = {
        "statistic": float(ad_combined.statistic),
        "p_value": float(ad_combined.pvalue),
    }

    # One-sample t-tests (H0: mean residual = 0)
    t_amiga = stats.ttest_1samp(res_amiga, 0.0)
    t_hcg = stats.ttest_1samp(res_hcg, 0.0)
    t_combined = stats.ttest_1samp(res_test, 0.0)
    results["ttest_amiga"] = {
        "statistic": float(t_amiga.statistic),
        "p_value": float(t_amiga.pvalue),
        "mean_offset": float(np.mean(res_amiga)),
        "std_offset": float(np.std(res_amiga, ddof=1)),
    }
    results["ttest_hcg"] = {
        "statistic": float(t_hcg.statistic),
        "p_value": float(t_hcg.pvalue),
        "mean_offset": float(np.mean(res_hcg)),
        "std_offset": float(np.std(res_hcg, ddof=1)),
    }
    results["ttest_combined"] = {
        "statistic": float(t_combined.statistic),
        "p_value": float(t_combined.pvalue),
        "mean_offset": float(np.mean(res_test)),
        "std_offset": float(np.std(res_test, ddof=1)),
    }

    # Store residual arrays for plotting
    results["_res_lit"] = res_lit
    results["_res_amiga"] = res_amiga
    results["_res_hcg"] = res_hcg

    return results


# ---------------------------------------------------------------------------
# Test 3: Posterior overlap
# ---------------------------------------------------------------------------


def run_posterior_overlap(
    literature_samples: np.ndarray,
    test_samples: np.ndarray,
) -> dict:
    """Fraction of test posterior samples within literature credible regions."""
    results: dict = {}

    # Marginal overlap for each parameter
    param_names = ["slope", "intercept", "scatter"]
    for i, name in enumerate(param_names):
        lit_vals = literature_samples[:, i].copy()
        test_vals = test_samples[:, i].copy()
        if i == 2:  # convert ln(scatter) -> scatter
            lit_vals = np.exp(lit_vals)
            test_vals = np.exp(test_vals)

        lit_lo, lit_hi = np.percentile(lit_vals, [2.5, 97.5])
        frac_inside = float(np.mean((test_vals >= lit_lo) & (test_vals <= lit_hi)))
        results[f"{name}_lit_95ci"] = (float(lit_lo), float(lit_hi))
        results[f"{name}_test_frac_in_lit_95ci"] = frac_inside

    # 2D overlap (slope, intercept)
    lit_2d = literature_samples[:, :2]
    test_2d = test_samples[:, :2]
    try:
        kde_lit = gaussian_kde(lit_2d.T)
        lit_densities = kde_lit(lit_2d.T)
        threshold_95 = np.percentile(lit_densities, 5)  # 95% HPD
        test_densities = kde_lit(test_2d.T)
        frac_2d = float(np.mean(test_densities >= threshold_95))
    except np.linalg.LinAlgError:
        frac_2d = np.nan
    results["slope_intercept_2d_frac_in_lit_95hpd"] = frac_2d

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_residual_diagnostics(
    literature_df: pd.DataFrame,
    amiga_df: pd.DataFrame,
    hcg_df: pd.DataFrame,
    residual_results: dict,
    fit_literature: dict,
    output_path: Path,
) -> None:
    """2x2 panel: residual histograms, residuals vs mass, QQ-plot."""
    res_lit = residual_results["_res_lit"]
    res_amiga = residual_results["_res_amiga"]
    res_hcg = residual_results["_res_hcg"]
    scatter = fit_literature["scatter"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Top-left: literature vs AMIGA residuals ---
    ax = axes[0, 0]
    bins = np.linspace(-0.6, 0.6, 30)
    ax.hist(res_lit, bins=bins, density=True, alpha=0.5, color="gray", label="Literature")
    ax.hist(res_amiga, bins=bins, density=True, alpha=0.6, color="#fe9a01", label="AMIGA")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    p_ks = residual_results["ks_amiga"]["p_value"]
    ax.set_title(f"AMIGA vs Literature (KS p={p_ks:.3f})", fontsize=14)
    ax.set_xlabel("Residual (dex)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(fontsize=11)
    style_axes(ax)

    # --- Top-right: literature vs HCG residuals ---
    ax = axes[0, 1]
    ax.hist(res_lit, bins=bins, density=True, alpha=0.5, color="gray", label="Literature")
    ax.hist(res_hcg, bins=bins, density=True, alpha=0.6, color="blue", label="HCG")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    p_ks = residual_results["ks_hcg"]["p_value"]
    ax.set_title(f"HCG vs Literature (KS p={p_ks:.3f})", fontsize=14)
    ax.set_xlabel("Residual (dex)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(fontsize=11)
    style_axes(ax)

    # --- Bottom-left: residuals vs log(M_HI) ---
    ax = axes[1, 0]
    lit_det = literature_df[~literature_df["is_upper_limit"]]
    amiga_det = amiga_df[~amiga_df["is_upper_limit"]]
    hcg_det = hcg_df[~hcg_df["is_upper_limit"]]

    ax.scatter(lit_det["log_mhi"], res_lit, c="gray", s=8, alpha=0.4, label="Literature")
    ax.scatter(
        amiga_det["log_mhi"], res_amiga, c="#fe9a01", s=50, marker="*", zorder=5, label="AMIGA"
    )
    ax.scatter(hcg_det["log_mhi"], res_hcg, c="blue", s=25, marker="o", zorder=5, label="HCG")

    ax.get_xlim()
    ax.axhline(0, color="k", ls="-", lw=1.2)
    ax.axhline(scatter, color="gray", ls="--", lw=0.8, label=rf"$\pm 1\sigma$ ({scatter:.2f} dex)")
    ax.axhline(-scatter, color="gray", ls="--", lw=0.8)
    ax.axhline(
        3 * scatter, color="gray", ls=":", lw=0.8, label=rf"$\pm 3\sigma$ ({3 * scatter:.2f} dex)"
    )
    ax.axhline(-3 * scatter, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\log\,(M_{\rm HI} / M_\odot)$", fontsize=13)
    ax.set_ylabel("Residual (dex)", fontsize=13)
    ax.set_title("Residuals vs HI mass", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    style_axes(ax)

    # --- Bottom-right: QQ-plot ---
    ax = axes[1, 1]
    res_test = np.sort(np.concatenate([res_amiga, res_hcg]))
    # Quantiles of test residuals against literature distribution
    n_test = len(res_test)
    quantiles = np.linspace(0, 1, n_test + 2)[1:-1]
    lit_quantiles = np.quantile(res_lit, quantiles)
    ax.scatter(lit_quantiles, res_test, c="teal", s=20, zorder=3)
    qq_lo = min(lit_quantiles.min(), res_test.min()) - 0.05
    qq_hi = max(lit_quantiles.max(), res_test.max()) + 0.05
    ax.plot([qq_lo, qq_hi], [qq_lo, qq_hi], "k--", lw=1.2)
    ax.set_xlabel("Literature quantiles (dex)", fontsize=13)
    ax.set_ylabel("AMIGA+HCG quantiles (dex)", fontsize=13)
    ax.set_title("Q-Q plot", fontsize=14)
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_posterior_comparison(
    literature_samples: np.ndarray,
    test_samples: np.ndarray,
    output_path: Path,
) -> None:
    """Overlaid corner plots: literature (blue) vs AMIGA+HCG (orange)."""
    labels = ["Slope", "Zero point", "Scatter"]

    lit_plot = np.copy(literature_samples)
    lit_plot[:, 2] = np.exp(lit_plot[:, 2])
    test_plot = np.copy(test_samples)
    test_plot[:, 2] = np.exp(test_plot[:, 2])

    fig = corner.corner(
        lit_plot,
        labels=labels,
        color="steelblue",
        show_titles=False,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        label_kwargs={"fontsize": 14, "family": "tex gyre heros"},
    )

    corner.corner(
        test_plot,
        labels=labels,
        color="darkorange",
        show_titles=False,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        fig=fig,
        label_kwargs={"fontsize": 14, "family": "tex gyre heros"},
    )

    # Manual legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=2, label="Literature (Wang+16 + MIGHTEE)"),
        Line2D([0], [0], color="darkorange", lw=2, label="AMIGA + HCG"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=12, frameon=False)

    for axc in fig.get_axes():
        for label in axc.get_xticklabels() + axc.get_yticklabels():
            label.set_fontsize(12)

    fig.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_parameter_comparison(
    fit_combined: dict,
    fit_literature: dict,
    fit_test: dict,
    output_path: Path,
) -> None:
    """3-panel comparison of slope, intercept, scatter with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    labels = ["Combined", "Literature", "AMIGA+HCG"]
    colors = ["black", "steelblue", "darkorange"]
    x_pos = [0, 1, 2]

    params = [
        ("slope", "Slope"),
        ("intercept", "Zero point"),
        ("scatter", "Scatter (dex)"),
    ]

    for ax, (key, ylabel) in zip(axes, params):
        for i, (fit, label, color) in enumerate(
            zip([fit_combined, fit_literature, fit_test], labels, colors)
        ):
            med = fit[key]
            lo = fit[f"{key}_p16"]
            hi = fit[f"{key}_p84"]
            ax.errorbar(
                x_pos[i],
                med,
                yerr=[[med - lo], [hi - med]],
                fmt="o",
                color=color,
                markersize=10,
                capsize=6,
                capthick=2,
                elinewidth=2,
                label=label,
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim(-0.5, 2.5)
        style_axes(ax)

    axes[0].legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def write_consistency_summary(
    summary_path: Path,
    bic_results: dict,
    residual_results: dict,
    overlap_results: dict,
    fit_combined: dict,
    fit_literature: dict,
    fit_test: dict,
    combined_df: pd.DataFrame,
    hetero_bic_results: dict | None = None,
    diff_results: dict | None = None,
    fit_hetero: dict | None = None,
    table_fits: list[tuple[str, dict]] | None = None,
) -> None:
    """Write all results to a JSON summary file."""
    # Strip non-serializable arrays from residual results
    residual_clean = {k: v for k, v in residual_results.items() if not k.startswith("_")}

    def _fit_summary(fit: dict, label: str) -> dict:
        return {
            f"{label}_slope": fit["slope"],
            f"{label}_slope_p16": fit["slope_p16"],
            f"{label}_slope_p84": fit["slope_p84"],
            f"{label}_slope_p025": fit["slope_p025"],
            f"{label}_slope_p975": fit["slope_p975"],
            f"{label}_intercept": fit["intercept"],
            f"{label}_intercept_p16": fit["intercept_p16"],
            f"{label}_intercept_p84": fit["intercept_p84"],
            f"{label}_intercept_p025": fit["intercept_p025"],
            f"{label}_intercept_p975": fit["intercept_p975"],
            f"{label}_scatter": fit["scatter"],
            f"{label}_scatter_p16": fit["scatter_p16"],
            f"{label}_scatter_p84": fit["scatter_p84"],
            f"{label}_scatter_p025": fit["scatter_p025"],
            f"{label}_scatter_p975": fit["scatter_p975"],
            f"{label}_r_value": fit["r_value"],
            f"{label}_n": fit["n"],
            f"{label}_n_det": fit["n_det"],
            f"{label}_n_ul": fit["n_ul"],
        }

    counts = combined_df.groupby("source_group").size().to_dict()

    payload = {
        "n_amiga": int(counts.get("AMIGA", 0)),
        "n_hcg": int(counts.get("HCG", 0)),
        "n_wang16": int(counts.get("Wang+16", 0)),
        "n_mightee": int(counts.get("MIGHTEE", 0)),
        "n_total": int(len(combined_df)),
        **_fit_summary(fit_combined, "combined"),
        **_fit_summary(fit_literature, "literature"),
        **_fit_summary(fit_test, "test"),
        "bic": bic_results,
        "residual_tests": residual_clean,
        "posterior_overlap": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in overlap_results.items()
        },
    }

    if hetero_bic_results is not None:
        payload["hetero_bic"] = hetero_bic_results
    if diff_results is not None:
        payload["posterior_differences"] = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in diff_results.items()
        }
    if fit_hetero is not None:
        payload["fit_heteroscedastic"] = fit_hetero
    if table_fits is not None:
        payload["table_consistency_fits"] = {
            label: _fit_summary(fit, re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower())
            for label, fit in table_fits
        }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _latex_fit_value(fit: dict, key: str, precision: int = 3) -> str:
    med = fit[key]
    lo = fit[f"{key}_p16"]
    hi = fit[f"{key}_p84"]
    upper = hi - med
    lower = med - lo
    return f"${med:.{precision}f}^{{+{upper:.{precision}f}}}_{{-{lower:.{precision}f}}}$"


def _build_consistency_fits_table(
    table_fits: list[tuple[str, dict]],
) -> str:
    table_rows = "\n\\addlinespace[0.35em]\n".join(
        " ".join(
            [
                label,
                "&",
                _latex_fit_value(fit, "slope"),
                "&",
                _latex_fit_value(fit, "intercept"),
                "&",
                _latex_fit_value(fit, "scatter"),
                "&",
                f"{fit['n']} \\\\",
            ]
        )
        for label, fit in table_fits
    )
    return (
        "\\begin{table}\n"
        "\\centering\n"
        "\\caption{Bayesian fit parameters for the \\HI\\ mass--size relation.\\label{table:consistency_fits}}\n"
        "\\begingroup\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule \\toprule\n"
        "Sample & $m$ & $b$ & $\\sigint$ Scatter & $N$ \\\\\n"
        "\\midrule\n"
        f"{table_rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\endgroup\n"
        "\\tablefoot{Columns: slope~$m$, intercept~$b$, intrinsic scatter~$\\sigint$, and sample size~$N$.\n"
        "Quoted uncertainties are the central 68\\% credible intervals "
        "(16th--84th percentiles).}\n"
        "\\end{table}"
    )


def update_paper_consistency_table(
    paper_path: Path,
    table_fits: list[tuple[str, dict]],
) -> None:
    """Replace table:consistency_fits in the manuscript with latest fit values."""
    text = paper_path.read_text(encoding="utf-8")
    new_table = _build_consistency_fits_table(table_fits)

    label = "\\label{table:consistency_fits}"
    label_idx = text.find(label)
    if label_idx == -1:
        raise ValueError(f"Could not find {label} in {paper_path}")

    begin_idx = text.rfind("\\begin{table}", 0, label_idx)
    end_idx = text.find("\\end{table}", label_idx)
    if begin_idx == -1 or end_idx == -1:
        raise ValueError(f"Could not locate full consistency-fits table in {paper_path}")
    end_idx += len("\\end{table}")

    updated = text[:begin_idx] + new_table + text[end_idx:]
    paper_path.write_text(updated, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether AMIGA and HCG galaxies follow the same HI mass-size "
            "relation as the literature (Wang+16 + MIGHTEE)."
        )
    )
    parser.add_argument(
        "--amiga-file",
        default=str(DATA_DIR / "isolated_galaxies_results.csv"),
        help="Resolved AMIGA CSV file.",
    )
    parser.add_argument(
        "--hcg-file",
        default=str(DATA_DIR / "interacting_galaxies_results.csv"),
        help="HCG CSV file.",
    )
    parser.add_argument(
        "--wang-file",
        default=str(DATA_DIR / "wang-surveys-table_original.txt"),
        help="Wang+16 fixed-width mass-size compilation file.",
    )
    parser.add_argument(
        "--mightee-file",
        default=str(DATA_DIR / "MIGHTEE_D_HI_M_HI_rajohnson22.txt"),
        help="MIGHTEE mass-size table.",
    )
    parser.add_argument(
        "--output-prefix",
        default="mass_size_consistency_test",
        help="Prefix for output figure filenames.",
    )
    parser.add_argument(
        "--summary-file",
        default=str(ANALYSIS_PRODUCTS_DIR / "mass_size_consistency_test_summary.json"),
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--report-file",
        default=str(ANALYSIS_PRODUCTS_DIR / "mass_size_consistency_test_report.txt"),
        help="Full text report path.",
    )
    parser.add_argument(
        "--paper-file",
        default=str(ANALYSIS_LATEX_DIR / "hi_disk_size_environments.tex"),
        help="Manuscript file whose table:consistency_fits will be refreshed.",
    )
    parser.add_argument(
        "--no-paper-update",
        action="store_true",
        help="Do not update table:consistency_fits in the manuscript.",
    )
    parser.add_argument(
        "--force-paper-update",
        action="store_true",
        help="(Deprecated) Write the table inline into the manuscript. Table is now "
        "an \\input of the autogen file; use scripts/write_size_mass_macros.py instead.",
    )
    parser.add_argument(
        "--font-dir",
        action="append",
        default=[d for d in [os.environ.get("GALAXYDISKSIZE_FONT_DIR")] if d],
        help="Custom font directory.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Do not echo captured report output to the console.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_directories()
    args = parse_args()
    configure_fonts(args.font_dir)

    with _redirect_output(args.report_file, echo_to_console=(not args.no_console)):
        # ---- Load data ----
        amiga_df = load_amiga_data(Path(args.amiga_file))
        hcg_df = load_hcg_data(Path(args.hcg_file))
        wang_df = load_wang_mass_size_table(Path(args.wang_file))
        mightee_df = load_mightee_table(Path(args.mightee_file))

        literature_df = pd.concat([wang_df, mightee_df], ignore_index=True)
        test_df = pd.concat([amiga_df, hcg_df], ignore_index=True)
        combined_df = pd.concat([literature_df, test_df], ignore_index=True)

        print("=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        for group in ["AMIGA", "HCG", "Wang+16", "MIGHTEE"]:
            n = int(np.sum(combined_df["source_group"] == group))
            n_det = int(
                np.sum((combined_df["source_group"] == group) & (~combined_df["is_upper_limit"]))
            )
            n_ul = n - n_det
            print(f"  {group:10s}: {n:4d} total  ({n_det} det, {n_ul} UL)")
        print(f"  Literature (Wang+16 + MIGHTEE): {len(literature_df)}")
        print(f"  Test sample (AMIGA + HCG):      {len(test_df)}")
        print(f"  Combined:                        {len(combined_df)}")

        # ---- MCMC fits ----
        fit_specs = [
            ("AMIGA", amiga_df),
            ("HCGs", hcg_df),
            ("AMIGA + HCGs", test_df),
            ("MIGHTEE", mightee_df),
            ("Wang2016", wang_df),
            ("MIGHTEE + Wang2016", literature_df),
            ("Combined sample", combined_df),
        ]
        table_fits: list[tuple[str, dict]] = []
        for idx, (label, df_fit) in enumerate(fit_specs, start=1):
            print("\n" + "=" * 60)
            print(f"MCMC FIT {idx}/{len(fit_specs)}: {label}")
            print("=" * 60)
            table_fits.append((label, compute_bayesian_fit(df_fit)))

        fit_lookup = dict(table_fits)
        fit_lookup["AMIGA"]
        fit_lookup["HCGs"]
        fit_test = fit_lookup["AMIGA + HCGs"]
        fit_lookup["MIGHTEE"]
        fit_lookup["Wang2016"]
        fit_literature = fit_lookup["MIGHTEE + Wang2016"]
        fit_combined = fit_lookup["Combined sample"]

        # ---- Print fit comparison ----
        print("\n" + "=" * 60)
        print("FIT PARAMETER COMPARISON")
        print("=" * 60)
        for label, fit in table_fits:
            print(
                f"  {label:12s}: slope = {fit['slope']:.4f} "
                f"[{fit['slope_p16']:.4f}, {fit['slope_p84']:.4f}]  "
                f"intercept = {fit['intercept']:.4f} "
                f"[{fit['intercept_p16']:.4f}, {fit['intercept_p84']:.4f}]  "
                f"scatter = {fit['scatter']:.4f} "
                f"[{fit['scatter_p16']:.4f}, {fit['scatter_p84']:.4f}]"
            )

        # ---- Test 1: BIC ----
        print("\n" + "=" * 60)
        print("TEST 1: BAYESIAN MODEL COMPARISON (BIC)")
        print("=" * 60)
        bic_results = run_bayesian_model_comparison(
            fit_combined,
            fit_literature,
            fit_test,
            n_combined=len(combined_df),
            n_literature=len(literature_df),
            n_test=len(test_df),
        )
        print(f"  BIC (single relation, M1):   {bic_results['bic_m1_single']:.2f}")
        print(f"  BIC (separate relations, M2): {bic_results['bic_m2_separate']:.2f}")
        print(f"  delta_BIC (M2 - M1):          {bic_results['delta_bic']:.2f}")
        print(f"  Interpretation: {bic_results['interpretation']}")

        # ---- Test 1b: Heteroscedastic BIC ----
        print("\n" + "=" * 60)
        print("TEST 1b: HETEROSCEDASTIC BIC (shared relation, separate scatter)")
        print("=" * 60)
        fit_hetero = fit_heteroscedastic_map(literature_df, test_df)
        print(
            f"  Heteroscedastic MAP: slope={fit_hetero['slope']:.4f}, "
            f"intercept={fit_hetero['intercept']:.4f}"
        )
        print(
            f"  scatter_lit={fit_hetero['scatter_lit']:.4f}, "
            f"scatter_test={fit_hetero['scatter_test']:.4f}"
        )
        print(f"  max log-likelihood = {fit_hetero['max_log_likelihood']:.2f}")

        hetero_bic_results = run_heteroscedastic_bic(
            fit_hetero,
            fit_literature,
            fit_test,
            n_combined=len(combined_df),
        )
        print(f"  BIC (shared relation, 4 params): {hetero_bic_results['bic_hetero']:.2f}")
        print(f"  BIC (separate relations, 6 params): {hetero_bic_results['bic_separate']:.2f}")
        print(f"  delta_BIC (separate - shared):   {hetero_bic_results['delta_bic']:.2f}")
        print(f"  Interpretation: {hetero_bic_results['interpretation']}")

        # ---- Test 1c: Posterior parameter differences ----
        print("\n" + "=" * 60)
        print("TEST 1c: POSTERIOR PARAMETER DIFFERENCES")
        print("=" * 60)
        diff_results = run_posterior_differences(
            fit_literature["posterior_samples"],
            fit_test["posterior_samples"],
        )
        for name in ["slope", "intercept", "scatter"]:
            med = diff_results[f"delta_{name}_median"]
            p16 = diff_results[f"delta_{name}_p16"]
            p84 = diff_results[f"delta_{name}_p84"]
            ci95 = diff_results[f"delta_{name}_95ci"]
            zero_ok = diff_results[f"delta_{name}_zero_in_95ci"]
            tension = diff_results[f"delta_{name}_tension_sigma"]
            print(
                f"  Delta {name:12s}: {med:+.4f} [{p16:+.4f}, {p84:+.4f}]  "
                f"95% CI: [{ci95[0]:+.4f}, {ci95[1]:+.4f}]  "
                f"zero in 95% CI: {zero_ok}  tension: {tension:.1f}sigma"
            )

        # ---- Test 2: Residual analysis ----
        print("\n" + "=" * 60)
        print("TEST 2: RESIDUAL ANALYSIS")
        print("=" * 60)
        print("  (Residuals computed from literature-only fit)")
        residual_results = run_residual_analysis(literature_df, amiga_df, hcg_df, fit_literature)

        for sample_label, key_prefix in [
            ("AMIGA", "amiga"),
            ("HCG", "hcg"),
            ("AMIGA+HCG", "combined"),
        ]:
            print(f"\n  --- {sample_label} ---")
            ks = residual_results[f"ks_{key_prefix}"]
            ad = residual_results[f"ad_{key_prefix}"]
            tt = residual_results[f"ttest_{key_prefix}"]
            print(f"    KS test:  D = {ks['statistic']:.4f},  p = {ks['p_value']:.4f}")
            print(f"    AD test:  A = {ad['statistic']:.4f},  p = {ad['p_value']:.4f}")
            print(
                f"    t-test:   t = {tt['statistic']:.4f},  p = {tt['p_value']:.4f}  "
                f"(mean offset = {tt['mean_offset']:+.4f} +/- {tt['std_offset']:.4f} dex)"
            )

        # ---- Test 3: Posterior overlap ----
        print("\n" + "=" * 60)
        print("TEST 3: POSTERIOR OVERLAP")
        print("=" * 60)
        overlap_results = run_posterior_overlap(
            fit_literature["posterior_samples"],
            fit_test["posterior_samples"],
        )
        for name in ["slope", "intercept", "scatter"]:
            ci = overlap_results[f"{name}_lit_95ci"]
            frac = overlap_results[f"{name}_test_frac_in_lit_95ci"]
            print(
                f"  {name:12s}: lit 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]  "
                f"test fraction inside = {frac:.2%}"
            )
        frac_2d = overlap_results["slope_intercept_2d_frac_in_lit_95hpd"]
        print(f"  2D (slope, intercept) fraction in lit 95% HPD = {frac_2d:.2%}")

        # ---- Generate figures ----
        prefix = args.output_prefix

        residuals_path = ANALYSIS_FIGURES_DIR / f"{prefix}_residuals.pdf"
        plot_residual_diagnostics(
            literature_df,
            amiga_df,
            hcg_df,
            residual_results,
            fit_literature,
            residuals_path,
        )
        print(f"\nResidual diagnostics saved to: {residuals_path}")

        posteriors_path = ANALYSIS_FIGURES_DIR / f"{prefix}_posteriors.pdf"
        plot_posterior_comparison(
            fit_literature["posterior_samples"],
            fit_test["posterior_samples"],
            posteriors_path,
        )
        print(f"Posterior comparison saved to: {posteriors_path}")

        parameters_path = ANALYSIS_FIGURES_DIR / f"{prefix}_parameters.pdf"
        plot_parameter_comparison(
            fit_combined,
            fit_literature,
            fit_test,
            parameters_path,
        )
        print(f"Parameter comparison saved to: {parameters_path}")

        # ---- Write summary JSON ----
        write_consistency_summary(
            Path(args.summary_file),
            bic_results,
            residual_results,
            overlap_results,
            fit_combined,
            fit_literature,
            fit_test,
            combined_df,
            hetero_bic_results=hetero_bic_results,
            diff_results=diff_results,
            fit_hetero=fit_hetero,
            table_fits=table_fits,
        )
        print(f"Summary JSON saved to: {args.summary_file}")

        # NOTE: Table:consistency_fits is now an \input of the autogen file
        # (latex/autogen/table_consistency_fits.tex), produced by
        # scripts/write_size_mass_macros.py. The in-place paper update below would
        # clobber that \input, so it is disabled by default; run write_size_mass_macros
        # after this script to refresh the table + macros.
        if not args.no_paper_update and args.force_paper_update:
            update_paper_consistency_table(
                Path(args.paper_file),
                table_fits,
            )
            print(f"Updated table:consistency_fits in: {args.paper_file}")
        else:
            print(
                "[table] left for autogen: run scripts/write_size_mass_macros.py "
                "to refresh table_consistency_fits.tex + macros_mass_size.tex"
            )

        # ---- Conclusion ----
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)

        # Primary criterion: heteroscedastic BIC (tests slope+intercept only)
        hetero_favors_shared = hetero_bic_results["delta_bic"] > 0
        slope_consistent = diff_results["delta_slope_zero_in_95ci"]
        intercept_consistent = diff_results["delta_intercept_zero_in_95ci"]

        if hetero_favors_shared and slope_consistent and intercept_consistent:
            print(
                "  The heteroscedastic BIC test and posterior parameter\n"
                "  differences confirm that AMIGA and HCG follow the same\n"
                "  mass-size relation (slope and intercept) as the literature.\n"
                "  The difference in intrinsic scatter does not affect the\n"
                "  relation itself. The combined relation can be used to\n"
                "  infer HI diameters for the larger AMIGA sample."
            )
        else:
            warnings = []
            if not hetero_favors_shared:
                warnings.append(
                    f"Heteroscedastic BIC favors separate relations "
                    f"(delta_BIC = {hetero_bic_results['delta_bic']:.1f})"
                )
            if not slope_consistent:
                warnings.append("Slope difference: zero outside 95% CI")
            if not intercept_consistent:
                warnings.append("Intercept difference: zero outside 95% CI")
            print("  WARNING: Some tests indicate potential inconsistency:")
            for w in warnings:
                print(f"    - {w}")
            print("  Inspect the diagnostic plots for further investigation.")

    print(f"Full text report saved to: {args.report_file}")


if __name__ == "__main__":
    main()
