#!/usr/bin/env python3
"""
Correlation + Residuals Analysis for HI Disk Sizes
===================================================
Implements the Perea et al. (1997) / Lisenfeld et al. (2007) methodology:
  1. Establish a correlation (D_HI vs D_25) for the isolated (AMIGA) sample
  2. Fit a regression line to define the "isolated galaxy baseline"
  3. Overlay the HCG galaxies and quantify their offset/residuals

Author: Roger Ianjamasimanana
Date: 27-06-2026
"""

import datetime

# -----------------------------------------------------------------------------
# Report logging helpers
# -----------------------------------------------------------------------------
# These helpers capture anything printed to stdout/stderr into a report file.
# This is useful when running long analyses on clusters where terminal output
# is truncated or lost.
import os
import sys
from contextlib import contextmanager

import figure_style  # noqa: E402
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import font_manager

# Optional imports
from scipy import stats

figure_style.apply()


class _TeeStream:
    """Write to a file and optionally also echo to the original stream."""

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
    """Redirect sys.stdout/sys.stderr to a report file (with optional echo)."""
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
            # In case stdout is already broken during shutdown.
            pass
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()


try:
    from scipy.odr import ODR, Model, RealData

    _HAS_ODR = True
except ImportError:
    _HAS_ODR = False


class CorrelationAnalysis:
    """
    Implements the correlation + residuals approach for comparing
    HI disk sizes between isolated and interacting galaxies.
    """

    def __init__(self, font_dirs=None):
        """Initialize with optional custom font directories."""
        self.font_dirs = font_dirs or []
        self.font = "tex gyre heros"
        self.configure_fonts()

        # Store results
        self.amiga_data = None
        self.hcg_data = None
        self.fit_results = {}

    def configure_fonts(self):
        """Configure matplotlib fonts to match your style."""
        if self.font_dirs:
            font_files = font_manager.findSystemFonts(fontpaths=self.font_dirs)
            for font_file in font_files:
                font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.sans-serif"] = self.font
        mpl.rc("mathtext", fontset="custom", it=self.font + ":italic")
        mpl.rc("font", size=20)

    def load_data(self, amiga_file, hcg_file):
        """
        Load data from CSV files.

        Expected CSV columns for the isolated (AMIGA) sample:
            - hi_diameter_kpc       : HI diameter in kpc  (D_HI)
            - optical_diameter_kpc  : optical diameter in kpc (D_25)
            Optional:
            - galaxy                : galaxy identifier
            - hi_diameter_err_kpc   : error on D_HI
            - hi_mass               : HI mass (linear, solar masses)
            - log_stellar_mass      : log10 stellar mass

        Expected CSV columns for the interacting (HCG) sample:
            - hi_diameter_kpc           : HI diameter in kpc (D_HI)
            - optical_diameter_kpc      : optical diameter in kpc (D_25)
            - phase                     : evolutionary phase label
            Optional:
            - galaxy                    : galaxy identifier
            - hi_diameter_err_kpc       : error on D_HI
            - optical_diameter_err_kpc  : error on D_25
            - log_stellar_mass          : log10 stellar mass
            - hi_mass                   : HI mass (linear, solar masses)
        """
        # --- Isolated (AMIGA) sample ---
        df_amiga = pd.read_csv(amiga_file)

        required_amiga = ["hi_diameter_kpc", "optical_diameter_kpc"]
        missing = [c for c in required_amiga if c not in df_amiga.columns]
        if missing:
            raise ValueError(
                f"Isolated-sample CSV is missing columns: {missing}. "
                f"Available columns: {list(df_amiga.columns)}"
            )

        self.amiga_data = {
            "D_HI": df_amiga["hi_diameter_kpc"].to_numpy(dtype=float),
            "D_25": df_amiga["optical_diameter_kpc"].to_numpy(dtype=float),
            "name": (
                df_amiga["galaxy"].to_numpy(dtype=str)
                if "galaxy" in df_amiga.columns
                else np.array([f"CIG_{i}" for i in range(len(df_amiga))])
            ),
        }

        if "hi_diameter_err_kpc" in df_amiga.columns:
            self.amiga_data["D_HI_err"] = df_amiga["hi_diameter_err_kpc"].to_numpy(dtype=float)
        if "hi_mass" in df_amiga.columns:
            self.amiga_data["hi_mass"] = df_amiga["hi_mass"].to_numpy(dtype=float)
            valid = self.amiga_data["hi_mass"][
                np.isfinite(self.amiga_data["hi_mass"]) & (self.amiga_data["hi_mass"] > 0)
            ]
            if len(valid) > 0:
                print(
                    f"  AMIGA hi_mass: min={valid.min():.3g}, "
                    f"max={valid.max():.3g}, median={np.median(valid):.3g}"
                    f"  (looks {'LOG' if valid.max() < 20 else 'LINEAR'})"
                )
        if "log_stellar_mass" in df_amiga.columns:
            self.amiga_data["log_stellar_mass"] = df_amiga["log_stellar_mass"].to_numpy(dtype=float)
            valid = self.amiga_data["log_stellar_mass"][
                np.isfinite(self.amiga_data["log_stellar_mass"])
                & (self.amiga_data["log_stellar_mass"] > 0)
            ]
            if len(valid) > 0:
                print(
                    f"  AMIGA log_stellar_mass: min={valid.min():.3g}, "
                    f"max={valid.max():.3g}, median={np.median(valid):.3g}"
                )

        # --- Interacting (HCG) sample ---
        df_hcg = pd.read_csv(hcg_file)

        required_hcg = [
            "hi_diameter_kpc",
            "optical_diameter_kpc",
            "phase",
        ]
        missing = [c for c in required_hcg if c not in df_hcg.columns]
        if missing:
            raise ValueError(
                f"Interacting-sample CSV is missing columns: {missing}. "
                f"Available columns: {list(df_hcg.columns)}"
            )

        self.hcg_data = {
            "D_HI": df_hcg["hi_diameter_kpc"].to_numpy(dtype=float),
            "D_25": df_hcg["optical_diameter_kpc"].to_numpy(dtype=float),
            "phase": df_hcg["phase"].astype(str).str.strip().to_numpy(),
        }

        if "hi_diameter_err_kpc" in df_hcg.columns:
            self.hcg_data["D_HI_err"] = df_hcg["hi_diameter_err_kpc"].to_numpy(dtype=float)
        if "optical_diameter_err_kpc" in df_hcg.columns:
            self.hcg_data["D_25_err"] = df_hcg["optical_diameter_err_kpc"].to_numpy(dtype=float)
        if "log_stellar_mass" in df_hcg.columns:
            self.hcg_data["log_stellar_mass"] = df_hcg["log_stellar_mass"].to_numpy(dtype=float)
            valid = self.hcg_data["log_stellar_mass"][
                np.isfinite(self.hcg_data["log_stellar_mass"])
                & (self.hcg_data["log_stellar_mass"] > 0)
            ]
            if len(valid) > 0:
                print(
                    f"  HCG log_stellar_mass: min={valid.min():.3g}, "
                    f"max={valid.max():.3g}, median={np.median(valid):.3g}"
                )
        if "hi_mass" in df_hcg.columns:
            self.hcg_data["hi_mass"] = df_hcg["hi_mass"].to_numpy(dtype=float)
            valid = self.hcg_data["hi_mass"][
                np.isfinite(self.hcg_data["hi_mass"]) & (self.hcg_data["hi_mass"] > 0)
            ]
            if len(valid) > 0:
                print(
                    f"  HCG hi_mass: min={valid.min():.3g}, "
                    f"max={valid.max():.3g}, median={np.median(valid):.3g}"
                    f"  (looks {'LOG' if valid.max() < 20 else 'LINEAR'})"
                )

        # Clean data (remove NaN, zero, negative values)
        self._clean_data()

        print(f"Loaded {len(self.amiga_data['D_HI'])} AMIGA galaxies")
        print(f"Loaded {len(self.hcg_data['D_HI'])} HCG galaxies")

    def _clean_data(self):
        """Remove invalid data points."""
        # Clean AMIGA
        mask = (
            np.isfinite(self.amiga_data["D_HI"])
            & np.isfinite(self.amiga_data["D_25"])
            & (self.amiga_data["D_HI"] > 0)
            & (self.amiga_data["D_25"] > 0)
        )
        for key in self.amiga_data:
            self.amiga_data[key] = self.amiga_data[key][mask]

        # Clean HCG. A member whose HI diameter carries no error
        # (hi_diameter_err_kpc is NaN) is beam-limited / unresolved
        # (D_HI < beam): it is not a clean detection and is handled as a
        # left-censored upper limit in the residual/KM analysis, so it must be
        # excluded from the detection-only sample here (consistent with the
        # mass-size fit, which also drops it).
        mask = (
            np.isfinite(self.hcg_data["D_HI"])
            & np.isfinite(self.hcg_data["D_25"])
            & (self.hcg_data["D_HI"] > 0)
            & (self.hcg_data["D_25"] > 0)
        )
        if "D_HI_err" in self.hcg_data:
            mask = mask & np.isfinite(self.hcg_data["D_HI_err"])
        for key in self.hcg_data:
            self.hcg_data[key] = self.hcg_data[key][mask]

    # ========== REGRESSION METHODS ==========

    def fit_ols(self, x, y):
        """Ordinary Least Squares regression in log-log space."""
        log_x = np.log10(x)
        log_y = np.log10(y)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

        # Calculate scatter (standard deviation of residuals)
        y_pred = intercept + slope * log_x
        residuals = log_y - y_pred
        scatter = np.std(residuals, ddof=2)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "scatter": scatter,
            "method": "OLS",
        }

    def fit_bisector(self, x, y):
        """
        Bisector regression (Isobe et al. 1990).
        This is the method used by Lisenfeld et al. (2007) / Perea et al. (1997).
        Symmetric treatment of both variables.
        """
        log_x = np.log10(x)
        log_y = np.log10(y)

        # OLS(Y|X)
        slope_yx, intercept_yx, _, _, _ = stats.linregress(log_x, log_y)

        # OLS(X|Y) then invert
        slope_xy_inv, intercept_xy_inv, _, _, _ = stats.linregress(log_y, log_x)
        slope_xy = 1.0 / slope_xy_inv
        -intercept_xy_inv / slope_xy_inv

        # Bisector slope (geometric mean of the two slopes)
        # Following Isobe et al. 1990 formula
        b1, b2 = slope_yx, slope_xy

        # Bisector formula
        slope_bis = (b1 * b2 - 1 + np.sqrt((1 + b1**2) * (1 + b2**2))) / (b1 + b2)

        # Intercept through centroid
        mean_x = np.mean(log_x)
        mean_y = np.mean(log_y)
        intercept_bis = mean_y - slope_bis * mean_x

        # Calculate scatter
        y_pred = intercept_bis + slope_bis * log_x
        residuals = log_y - y_pred
        scatter = np.std(residuals, ddof=2)

        # Correlation coefficient
        r_value = np.corrcoef(log_x, log_y)[0, 1]

        return {
            "slope": slope_bis,
            "intercept": intercept_bis,
            "slope_yx": slope_yx,
            "slope_xy": slope_xy,
            "r_value": r_value,
            "r_squared": r_value**2,
            "scatter": scatter,
            "method": "Bisector",
        }

    def fit_odr(self, x, y, x_err=None, y_err=None):
        """
        Orthogonal Distance Regression (accounts for errors in both variables).
        """
        if not _HAS_ODR:
            print("Warning: scipy.odr not available, falling back to OLS")
            return self.fit_ols(x, y)

        log_x = np.log10(x)
        log_y = np.log10(y)

        # Default errors if not provided (assume 10% fractional error)
        if x_err is None:
            log_x_err = np.full_like(log_x, 0.05)
        else:
            log_x_err = x_err / (x * np.log(10))

        if y_err is None:
            log_y_err = np.full_like(log_y, 0.05)
        else:
            log_y_err = y_err / (y * np.log(10))

        def linear(B, x):
            return B[0] * x + B[1]

        model = Model(linear)
        data = RealData(log_x, log_y, sx=log_x_err, sy=log_y_err)

        # Initial guess from OLS
        ols = self.fit_ols(x, y)
        odr = ODR(data, model, beta0=[ols["slope"], ols["intercept"]])
        output = odr.run()

        slope, intercept = output.beta
        slope_err, intercept_err = output.sd_beta

        # Calculate scatter
        y_pred = intercept + slope * log_x
        residuals = log_y - y_pred
        scatter = np.std(residuals, ddof=2)

        r_value = np.corrcoef(log_x, log_y)[0, 1]

        return {
            "slope": slope,
            "intercept": intercept,
            "slope_err": slope_err,
            "intercept_err": intercept_err,
            "r_value": r_value,
            "r_squared": r_value**2,
            "scatter": scatter,
            "method": "ODR",
        }

    # ========== RESIDUAL ANALYSIS ==========

    def compute_residuals(self, D_HI, D_25, fit_result):
        """
        Compute residuals from the fitted relation.

        Residual = log(D_HI_observed) - log(D_HI_expected)

        Where D_HI_expected = 10^(intercept + slope * log(D_25))

        Positive residual = larger HI disk than expected (extended)
        Negative residual = smaller HI disk than expected (truncated)
        """
        log_D_25 = np.log10(D_25)
        log_D_HI_obs = np.log10(D_HI)
        log_D_HI_exp = fit_result["intercept"] + fit_result["slope"] * log_D_25

        residuals = log_D_HI_obs - log_D_HI_exp

        return residuals

    def analyze_residuals(self, residuals, label="Sample"):
        """Compute statistics on residuals."""
        n = len(residuals)
        mean = np.mean(residuals)
        median = np.median(residuals)
        std = np.std(residuals, ddof=1)
        mad = 1.4826 * np.median(np.abs(residuals - median))

        # Bootstrap confidence interval for mean
        n_bootstrap = 10000
        bootstrap_means = np.array(
            [np.mean(np.random.choice(residuals, size=n, replace=True)) for _ in range(n_bootstrap)]
        )
        ci_lo, ci_hi = np.percentile(bootstrap_means, [2.5, 97.5])

        # Fraction truncated (residual < 0)
        n_truncated = np.sum(residuals < 0)
        f_truncated = n_truncated / n

        # Fraction extended (residual > scatter threshold)
        scatter_threshold = 0.1  # ~25% larger than expected in linear space
        n_extended = np.sum(residuals > scatter_threshold)
        f_extended = n_extended / n

        result = {
            "n": n,
            "mean": mean,
            "median": median,
            "std": std,
            "mad": mad,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_truncated": n_truncated,
            "f_truncated": f_truncated,
            "n_extended": n_extended,
            "f_extended": f_extended,
        }

        print(f"\n=== Residual Statistics: {label} ===")
        print(f"  N = {n}")
        print(f"  Mean offset = {mean:.3f} dex (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
        print(f"  Median offset = {median:.3f} dex")
        print(f"  Scatter (std) = {std:.3f} dex")
        print(f"  Scatter (MAD) = {mad:.3f} dex")
        print(f"  Truncated (Δ<0): {n_truncated}/{n} = {100 * f_truncated:.1f}%")
        print(f"  Extended (Δ>0.1): {n_extended}/{n} = {100 * f_extended:.1f}%")

        return result

    def compare_residuals(self, res_amiga, res_hcg):
        """Statistical tests comparing AMIGA and HCG residuals."""
        print("\n=== Statistical Comparison of Residuals ===")

        # Mann-Whitney U test (non-parametric)
        stat_mw, p_mw = stats.mannwhitneyu(res_amiga, res_hcg, alternative="two-sided")
        print(f"Mann-Whitney U test: U={stat_mw:.1f}, p={p_mw:.2e}")

        # Kolmogorov-Smirnov test
        stat_ks, p_ks = stats.ks_2samp(res_amiga, res_hcg)
        print(f"Kolmogorov-Smirnov test: D={stat_ks:.3f}, p={p_ks:.2e}")

        # Welch's t-test
        stat_t, p_t = stats.ttest_ind(res_amiga, res_hcg, equal_var=False)
        print(f"Welch's t-test: t={stat_t:.2f}, p={p_t:.2e}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(res_amiga, ddof=1) + np.var(res_hcg, ddof=1)) / 2)
        cohens_d = (np.mean(res_amiga) - np.mean(res_hcg)) / pooled_std
        print(f"Cohen's d effect size: {cohens_d:.2f}")

        # Offset between samples
        offset = np.mean(res_hcg) - np.mean(res_amiga)
        print(f"\nMean offset (HCG - AMIGA): {offset:.3f} dex")
        print(
            f"  → HCG galaxies have HI disks that are 10^{offset:.2f} = {10**offset:.2f}x "
            f"{'smaller' if offset < 0 else 'larger'} than AMIGA at fixed D_25"
        )

        return {
            "mw_stat": stat_mw,
            "mw_p": p_mw,
            "ks_stat": stat_ks,
            "ks_p": p_ks,
            "t_stat": stat_t,
            "t_p": p_t,
            "cohens_d": cohens_d,
            "offset": offset,
        }

    # ========== MAIN ANALYSIS ==========

    def run_full_analysis(self, method="bisector"):
        """
        Run the complete Perea et al. style analysis.

        Parameters
        ----------
        method : str
            Regression method: 'ols', 'bisector', or 'odr'
        """
        print("=" * 60)
        print("CORRELATION + RESIDUALS ANALYSIS")
        print("Following Perea et al. (1997) / Lisenfeld et al. (2007)")
        print("=" * 60)

        # 1. Fit the AMIGA (isolated) baseline
        print("\n1. ESTABLISHING THE ISOLATED GALAXY BASELINE (AMIGA)")
        print("-" * 50)

        if method == "ols":
            fit = self.fit_ols(self.amiga_data["D_25"], self.amiga_data["D_HI"])
        elif method == "bisector":
            fit = self.fit_bisector(self.amiga_data["D_25"], self.amiga_data["D_HI"])
        elif method == "odr":
            fit = self.fit_odr(self.amiga_data["D_25"], self.amiga_data["D_HI"])
        else:
            raise ValueError(f"Unknown method: {method}")

        self.fit_results["amiga"] = fit

        print(f"\nFit method: {fit['method']}")
        print(f"Relation: log(D_HI) = {fit['slope']:.3f} × log(D_25) + {fit['intercept']:.3f}")
        print(f"  → D_HI ∝ D_25^{fit['slope']:.2f}")
        print(f"Correlation: r = {fit['r_value']:.3f}, r² = {fit['r_squared']:.3f}")
        print(f"Intrinsic scatter: {fit['scatter']:.3f} dex")

        # 2. Compute residuals for both samples
        print("\n2. COMPUTING RESIDUALS FROM THE AMIGA BASELINE")
        print("-" * 50)

        res_amiga = self.compute_residuals(self.amiga_data["D_HI"], self.amiga_data["D_25"], fit)
        res_hcg = self.compute_residuals(self.hcg_data["D_HI"], self.hcg_data["D_25"], fit)

        self.fit_results["residuals_amiga"] = res_amiga
        self.fit_results["residuals_hcg"] = res_hcg

        # 3. Analyze residuals
        stats_amiga = self.analyze_residuals(res_amiga, "AMIGA (isolated)")
        stats_hcg = self.analyze_residuals(res_hcg, "HCG (interacting)")

        self.fit_results["stats_amiga"] = stats_amiga
        self.fit_results["stats_hcg"] = stats_hcg

        # 4. Compare samples
        comparison = self.compare_residuals(res_amiga, res_hcg)
        self.fit_results["comparison"] = comparison

        # 5. Phase-by-phase analysis for HCGs
        print("\n3. HCG RESIDUALS BY EVOLUTIONARY PHASE")
        print("-" * 50)

        phases = np.unique(self.hcg_data["phase"])
        phases = [p for p in phases if p != ""]

        for phase in sorted(phases):
            mask = self.hcg_data["phase"] == phase
            if np.sum(mask) >= 3:
                res_phase = res_hcg[mask]
                print(f"\nPhase {phase} (n={np.sum(mask)}):")
                print(f"  Mean offset: {np.mean(res_phase):.3f} dex")
                print(f"  Median offset: {np.median(res_phase):.3f} dex")

        return self.fit_results

    # ========== PLOTTING ==========

    # ========== MULTI-SURVEY RESIDUAL ANALYSIS ==========

    def register_surveys(self, surveys=None):
        """
        Register additional survey data for cross-survey comparison.

        AMIGA and HCG are included automatically if already loaded.
        Each extra survey must provide D_HI and D_25 arrays (kpc).

        Parameters
        ----------
        surveys : dict or None
            Keys   = survey names (str)
            Values = dicts with 'D_HI' and 'D_25' (array-like, kpc).
            Optionally include 'D_HI_err' and 'D_25_err'.

            Example
            -------
            >>> analyzer.register_surveys({
            ...     'VIVA':          {'D_HI': [...], 'D_25': [...]},
            ...     'Ursa Major':    {'D_HI': [...], 'D_25': [...]},
            ...     'LITTLE THINGS': {'D_HI': [...], 'D_25': [...]},
            ... })
        """
        if not hasattr(self, "surveys") or self.surveys is None:
            self.surveys = {}

        # Always include AMIGA and HCG when available
        if self.amiga_data is not None and "AMIGA" not in self.surveys:
            self.surveys["AMIGA"] = {
                "D_HI": self.amiga_data["D_HI"].copy(),
                "D_25": self.amiga_data["D_25"].copy(),
            }
        if self.hcg_data is not None and "HCGs" not in self.surveys:
            self.surveys["HCGs"] = {
                "D_HI": self.hcg_data["D_HI"].copy(),
                "D_25": self.hcg_data["D_25"].copy(),
            }

        # Add external surveys
        if surveys:
            for name, data in surveys.items():
                d_hi = np.asarray(data["D_HI"], dtype=float)
                d_25 = np.asarray(data["D_25"], dtype=float)
                # Remove invalid entries
                mask = np.isfinite(d_hi) & np.isfinite(d_25) & (d_hi > 0) & (d_25 > 0)
                self.surveys[name] = {
                    "D_HI": d_hi[mask],
                    "D_25": d_25[mask],
                }

        print(f"Registered {len(self.surveys)} surveys:")
        for name, data in self.surveys.items():
            print(f"  {name}: {len(data['D_HI'])} galaxies")

    def load_wang_table(self, table_path):
        """
        Load the Wang et al. multi-survey table.

        Expected format (whitespace-delimited, no header):
            <D_25_kpc>  <D_HI_kpc>  <survey name>

        D_HI may be '-' for missing values (these rows are skipped).
        Survey names can contain spaces (e.g. 'LITTLE THINGS',
        'Ursa Major').

        All surveys found in the file are added to self.surveys.
        AMIGA and HCGs (loaded via load_data) are not affected.

        Parameters
        ----------
        table_path : str
            Path to the whitespace-delimited table.
        """
        import re

        if not hasattr(self, "surveys") or self.surveys is None:
            self.surveys = {}

        # Collect raw rows per survey
        raw = {}  # survey_name -> {'D_25': [], 'D_HI': []}

        with open(table_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = re.split(r"\s+", line)
                if len(tokens) < 3:
                    continue

                # First two tokens are D_25 and D_HI; rest is survey name
                d25_str, dhi_str = tokens[1], tokens[0]
                survey = " ".join(tokens[2:]).strip()

                # Skip missing D_HI
                if d25_str == "-":
                    continue
                try:
                    d25 = float(d25_str)
                    dhi = float(dhi_str)
                except ValueError:
                    continue

                if d25 <= 0 or dhi <= 0:
                    continue

                if survey not in raw:
                    raw[survey] = {"D_25": [], "D_HI": []}
                raw[survey]["D_25"].append(d25)
                raw[survey]["D_HI"].append(dhi)

        # Store as numpy arrays
        print(f"\nLoaded surveys from {table_path}:")
        for name in sorted(raw.keys()):
            self.surveys[name] = {
                "D_25": np.array(raw[name]["D_25"]),
                "D_HI": np.array(raw[name]["D_HI"]),
            }
            print(f"  {name}: {len(raw[name]['D_25'])} galaxies")

    def load_broeils_rhee(self, table_path, survey_name="B97"):
        """
        Load the Broeils & Rhee (1997) table.

        Expected format (whitespace-delimited, no header, 9 or 10 columns):
            col 0 = D_HI (kpc)
            col 1 = D_25 (kpc)
            (remaining columns are ignored)

        The survey is stored in self.surveys under `survey_name`.

        Parameters
        ----------
        table_path : str
            Path to the broelis-rhee.txt file.
        survey_name : str
            Name under which to register the survey (default 'B97').
        """
        if not hasattr(self, "surveys") or self.surveys is None:
            self.surveys = {}

        data = np.loadtxt(table_path, usecols=(0, 1))
        d_hi = data[:, 0]
        d_25 = data[:, 1]

        mask = np.isfinite(d_hi) & np.isfinite(d_25) & (d_hi > 0) & (d_25 > 0)

        self.surveys[survey_name] = {
            "D_HI": d_hi[mask],
            "D_25": d_25[mask],
        }
        print(f"Loaded {survey_name}: {np.sum(mask)} galaxies from {table_path}")

    def compute_survey_residuals(self):
        """
        Compute residuals from the AMIGA baseline for every registered
        survey.  Must call run_full_analysis() and register_surveys()
        first.

        Returns
        -------
        survey_stats : dict
            Per-survey statistics (mean, scatter, fractions, raw residuals).
        """
        fit = self.fit_results.get("amiga")
        if fit is None:
            raise RuntimeError("Run run_full_analysis() first to establish the AMIGA baseline.")
        if not hasattr(self, "surveys") or not self.surveys:
            raise RuntimeError("Call register_surveys() first to add survey data.")

        sigma = fit["scatter"]  # AMIGA intrinsic scatter
        self.survey_stats = {}

        print("\n" + "=" * 60)
        print("MULTI-SURVEY RESIDUAL ANALYSIS")
        print("=" * 60)
        print(f"AMIGA baseline scatter (σ): {sigma:.3f} dex\n")

        for name, data in self.surveys.items():
            residuals = self.compute_residuals(data["D_HI"], data["D_25"], fit)
            n = len(residuals)
            mean_r = np.mean(residuals)
            median_r = np.median(residuals)
            std_r = np.std(residuals, ddof=1)

            # Fractions (percentages)
            f_truncated = np.sum(residuals < 0) / n * 100
            f_severe_trunc = np.sum(residuals < -sigma) / n * 100
            f_extended = np.sum(residuals > sigma) / n * 100

            self.survey_stats[name] = {
                "N": n,
                "mean": mean_r,
                "median": median_r,
                "scatter": std_r,
                "f_truncated": f_truncated,
                "f_severe_trunc": f_severe_trunc,
                "f_extended": f_extended,
                "residuals": residuals,
            }

            print(
                f"{name:20s}  n={n:3d}  "
                f"<Δ>={mean_r:+.3f}  σ={std_r:.3f}  "
                f"trunc={f_truncated:5.1f}%  "
                f"sev_trunc={f_severe_trunc:5.1f}%  "
                f"ext={f_extended:5.1f}%"
            )

        return self.survey_stats

    # ---------- helper shared by the three ranked plots ----------

    def _style_axes(self, ax):
        """Ticks inward on all sides (matching existing style)."""
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2)
        ax.tick_params(which="minor", length=4, width=1)

    # ---------- Plot 1 : mean residual ± scatter ----------

    # ---------- Plot 2 : % extended ----------

    # ---------- Plot 3 : % truncated ----------

    # ---------- generic horizontal bar helper ----------

    # ========== MASS-RELATED DIAGNOSTICS (AMIGA vs HCG) ==========

    # ==================================================================
    #  FOCUSED AMIGA vs HCG COMPARISON ("better forward" analysis)
    # ==================================================================

    @staticmethod
    def _bootstrap_median(arr, n_boot=10000, ci=95):
        """Bootstrap confidence interval for the median."""
        rng = np.random.default_rng(42)
        medians = np.array(
            [np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
        )
        lo = np.percentile(medians, (100 - ci) / 2)
        hi = np.percentile(medians, 100 - (100 - ci) / 2)
        return np.median(arr), lo, hi, medians

    @staticmethod
    def _bootstrap_median_shift(arr_a, arr_b, n_boot=10000, ci=95):
        """
        Bootstrap CI for the shift in medians:
            Δmedian = median(B) - median(A)
        """
        rng = np.random.default_rng(42)
        shifts = np.array(
            [
                np.median(rng.choice(arr_b, len(arr_b), replace=True))
                - np.median(rng.choice(arr_a, len(arr_a), replace=True))
                for _ in range(n_boot)
            ]
        )
        med = np.median(arr_b) - np.median(arr_a)
        lo = np.percentile(shifts, (100 - ci) / 2)
        hi = np.percentile(shifts, 100 - (100 - ci) / 2)
        return med, lo, hi, shifts

    @staticmethod
    def _cliffs_delta(x, y):
        """
        Cliff's delta: non-parametric effect size.
        δ = (# concordant - # discordant) / (n_x * n_y)
        Ranges from -1 to +1.
        """
        nx, ny = len(x), len(y)
        more = 0
        less = 0
        for xi in x:
            more += np.sum(xi > y)
            less += np.sum(xi < y)
        delta = (more - less) / (nx * ny)
        return delta

    def _match_in_logD25(self, k=1):
        """
        Nearest-neighbour matching of HCG galaxies to AMIGA
        galaxies in log(D_25) space.

        Returns
        -------
        idx_amiga, idx_hcg : arrays of matched indices
        """
        log_d25_a = np.log10(self.amiga_data["D_25"])
        log_d25_h = np.log10(self.hcg_data["D_25"])

        idx_a = []
        idx_h = []
        used = set()

        for j, val_h in enumerate(log_d25_h):
            dists = np.abs(log_d25_a - val_h)
            # exclude already-used AMIGA galaxies (without replacement)
            order = np.argsort(dists)
            for i in order:
                if i not in used:
                    idx_a.append(i)
                    idx_h.append(j)
                    used.add(i)
                    break

        return np.array(idx_a), np.array(idx_h)

    def run_focused_comparison(self, n_boot=10000):
        """
        Focused AMIGA-vs-HCG analysis (no other surveys).

        Reports:
            1. Median Δ for AMIGA and HCG with bootstrap 95 % CI
            2. Median shift (Δ_median) with bootstrap CI
            3. Truncation index  T = 10^Δ  (median, per-phase)
            4. Cliff's delta effect size
            5. Anderson-Darling 2-sample test
            6. Size-matched comparison
            7. Binned comparison by log(D_25)

        All results are stored in self.fit_results['focused'].
        """
        res_a = self.fit_results.get("residuals_amiga")
        res_h = self.fit_results.get("residuals_hcg")
        if res_a is None or res_h is None:
            raise RuntimeError("Run run_full_analysis() first.")

        print("\n" + "=" * 60)
        print("FOCUSED AMIGA vs HCG COMPARISON")
        print("=" * 60)

        out = {}

        # ---- 1. Median residuals with bootstrap CI ----
        med_a, lo_a, hi_a, _ = self._bootstrap_median(res_a, n_boot)
        med_h, lo_h, hi_h, _ = self._bootstrap_median(res_h, n_boot)

        print(f"\n1) MEDIAN RESIDUALS (bootstrap {n_boot:,} iterations)")
        print("-" * 50)
        print(f"  AMIGA : median = {med_a:+.3f} dex  95% CI [{lo_a:+.3f}, {hi_a:+.3f}]")
        print(f"  HCG   : median = {med_h:+.3f} dex  95% CI [{lo_h:+.3f}, {hi_h:+.3f}]")
        out["median_amiga"] = (med_a, lo_a, hi_a)
        out["median_hcg"] = (med_h, lo_h, hi_h)

        # ---- 2. Median shift with bootstrap CI ----
        shift, shift_lo, shift_hi, shift_dist = self._bootstrap_median_shift(res_a, res_h, n_boot)

        print("\n2) MEDIAN SHIFT  Δmedian = median(HCG) − median(AMIGA)")
        print("-" * 50)
        print(f"  Δmedian = {shift:+.3f} dex  95% CI [{shift_lo:+.3f}, {shift_hi:+.3f}]")
        if shift_hi < 0:
            print("  → Significant negative shift (HCG disks smaller than isolated at fixed D_25)")
        out["median_shift"] = (shift, shift_lo, shift_hi)
        out["median_shift_distribution"] = shift_dist

        # ---- 3. Truncation index T = 10^Δ ----
        T_a = 10.0**res_a
        T_h = 10.0**res_h

        print("\n3) TRUNCATION INDEX  T = D_HI / D_HI,exp = 10^Δ")
        print("-" * 50)
        print(f"  AMIGA : median T = {np.median(T_a):.2f}  (mean {np.mean(T_a):.2f})")
        print(f"  HCG   : median T = {np.median(T_h):.2f}  (mean {np.mean(T_h):.2f})")
        print(
            f"  → Typical HCG galaxy has an HI disk "
            f"that is {np.median(T_h):.0%} of the expected size"
        )
        out["T_amiga_median"] = np.median(T_a)
        out["T_hcg_median"] = np.median(T_h)

        # Per phase
        phases = sorted(set(self.hcg_data["phase"]) - {""})
        out["T_by_phase"] = {}
        for ph in phases:
            pmask = self.hcg_data["phase"] == ph
            if np.sum(pmask) >= 2:
                T_ph = 10.0 ** res_h[pmask]
                med_T, lo_T, hi_T, _ = self._bootstrap_median(T_ph, n_boot)
                out["T_by_phase"][ph] = (med_T, lo_T, hi_T)
                print(
                    f"    Phase {ph:>2s}: median T = {med_T:.2f}  "
                    f"95% CI [{lo_T:.2f}, {hi_T:.2f}]  "
                    f"(n={np.sum(pmask)})"
                )

        # ---- 4. Cliff's delta ----
        cd = self._cliffs_delta(res_a, res_h)
        # Interpretation thresholds (Romano et al. 2006)
        if abs(cd) < 0.147:
            cd_label = "negligible"
        elif abs(cd) < 0.33:
            cd_label = "small"
        elif abs(cd) < 0.474:
            cd_label = "medium"
        else:
            cd_label = "large"

        print("\n4) EFFECT SIZE (Cliff's delta)")
        print("-" * 50)
        print(f"  δ = {cd:+.3f}  ({cd_label})")
        print("  Positive δ → AMIGA residuals tend to be larger than HCG residuals")
        out["cliffs_delta"] = cd
        out["cliffs_delta_label"] = cd_label

        # ---- 5. Anderson–Darling 2-sample test ----
        ad_stat, ad_crit, ad_sig = stats.anderson_ksamp([res_a, res_h])
        print("\n5) ANDERSON–DARLING 2-sample test")
        print("-" * 50)
        print(f"  Statistic = {ad_stat:.3f},  p = {ad_sig:.4f}")
        if ad_sig < 0.05:
            print("  → Distributions are significantly different (p < 0.05)")
        out["anderson_darling"] = (ad_stat, ad_sig)

        # ---- 6. Size-matched comparison ----
        print("\n6) SIZE-MATCHED COMPARISON (nearest-neighbour in log D_25)")
        print("-" * 50)
        try:
            idx_a, idx_h = self._match_in_logD25()
            res_a_m = res_a[idx_a]
            res_h_m = res_h[idx_h]
            logd_a_m = np.log10(self.amiga_data["D_25"][idx_a])
            logd_h_m = np.log10(self.hcg_data["D_25"][idx_h])

            # Quality of match
            d25_diff = np.abs(logd_a_m - logd_h_m)
            print(f"  Matched {len(idx_h)} HCG→AMIGA pairs (without replacement)")
            print(
                f"  Median |Δlog D_25| between pairs: "
                f"{np.median(d25_diff):.3f} dex  "
                f"(max {d25_diff.max():.3f})"
            )

            shift_m, shift_m_lo, shift_m_hi, _ = self._bootstrap_median_shift(
                res_a_m, res_h_m, n_boot
            )
            print(
                f"  Matched median shift: {shift_m:+.3f} dex  "
                f"95% CI [{shift_m_lo:+.3f}, {shift_m_hi:+.3f}]"
            )

            # Wilcoxon signed-rank on paired differences
            paired_diff = res_h_m - res_a_m
            w_stat, w_p = stats.wilcoxon(paired_diff)
            print(f"  Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_p:.2e}")
            out["matched_shift"] = (shift_m, shift_m_lo, shift_m_hi)
            out["matched_wilcoxon"] = (w_stat, w_p)
            out["matched_idx_a"] = idx_a
            out["matched_idx_h"] = idx_h
        except Exception as e:
            print(f"  Could not run size-matched comparison: {e}")

        # ---- 7. Binned comparison by log(D_25) ----
        print("\n7) BINNED COMPARISON BY log(D_25)")
        print("-" * 50)
        log_d25_a = np.log10(self.amiga_data["D_25"])
        log_d25_h = np.log10(self.hcg_data["D_25"])

        # Define bins that cover both samples
        all_logd = np.concatenate([log_d25_a, log_d25_h])
        bin_edges = np.linspace(all_logd.min() - 0.01, all_logd.max() + 0.01, 5)
        out["bin_edges"] = bin_edges
        out["binned"] = []

        for i in range(len(bin_edges) - 1):
            lo_b, hi_b = bin_edges[i], bin_edges[i + 1]
            ma = (log_d25_a >= lo_b) & (log_d25_a < hi_b)
            mh = (log_d25_h >= lo_b) & (log_d25_h < hi_b)
            na, nh = np.sum(ma), np.sum(mh)

            bin_info = {"lo": lo_b, "hi": hi_b, "n_amiga": na, "n_hcg": nh}

            if na >= 3 and nh >= 3:
                med_a_b = np.median(res_a[ma])
                med_h_b = np.median(res_h[mh])
                mw_stat, mw_p = stats.mannwhitneyu(res_a[ma], res_h[mh], alternative="two-sided")
                bin_info["median_amiga"] = med_a_b
                bin_info["median_hcg"] = med_h_b
                bin_info["shift"] = med_h_b - med_a_b
                bin_info["mw_p"] = mw_p
                sig = "*" if mw_p < 0.05 else ""
                print(
                    f"  [{lo_b:.2f}, {hi_b:.2f}):  "
                    f"AMIGA n={na} med={med_a_b:+.3f}  |  "
                    f"HCG n={nh} med={med_h_b:+.3f}  |  "
                    f"shift={med_h_b - med_a_b:+.3f}  "
                    f"MW p={mw_p:.3f}{sig}"
                )
            else:
                print(
                    f"  [{lo_b:.2f}, {hi_b:.2f}):  "
                    f"AMIGA n={na}  |  HCG n={nh}  "
                    f"(too few for comparison)"
                )
            out["binned"].append(bin_info)

        self.fit_results["focused"] = out

        # ---- Summary statement ----
        print(f"\n{'=' * 60}")
        print("SUMMARY STATEMENT")
        print(f"{'=' * 60}")
        print(
            f"  HCG galaxies show a median HI-disk truncation index of "
            f"T = {out['T_hcg_median']:.2f},"
        )
        print(
            f"  meaning their HI disks are {(1 - out['T_hcg_median']):.0%} "
            f"smaller than expected for isolated"
        )
        print(f"  galaxies of the same optical size.  This offset ({shift:+.3f} dex, 95% CI")
        print(
            f"  [{shift_lo:+.3f}, {shift_hi:+.3f}]) is robust to "
            f"size-matching (Δ = {shift_m:+.3f} dex,"
        )
        print(f"  Wilcoxon p = {w_p:.2e}) and corresponds to a {cd_label} effect size")
        print(f"  (Cliff's δ = {cd:+.3f}).")

        return out

    # ---------- plots for the focused comparison ----------


# ========== EXAMPLE USAGE ==========
