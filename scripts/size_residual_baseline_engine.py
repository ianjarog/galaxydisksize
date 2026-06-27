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

import argparse
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
from pathlib import Path

import figure_style  # noqa: E402
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# Optional imports
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.optimize import curve_fit  # noqa: F401

figure_style.apply()

FIGURE_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "figures")


def _figure_output_path(output_file):
    """Save all figures into the shared latex figures directory."""
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    return os.path.join(FIGURE_OUTPUT_DIR, os.path.basename(output_file))


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

    def load_data_from_arrays(self, amiga_d_hi, amiga_d_25, hcg_d_hi, hcg_d_25, hcg_phase=None):
        """
        Alternative: load data directly from arrays.

        Useful if you want to extract data from your paper tables manually.
        """
        self.amiga_data = {
            "D_HI": np.asarray(amiga_d_hi),
            "D_25": np.asarray(amiga_d_25),
        }
        self.hcg_data = {
            "D_HI": np.asarray(hcg_d_hi),
            "D_25": np.asarray(hcg_d_25),
            "phase": (
                np.asarray(hcg_phase) if hcg_phase is not None else np.array([""] * len(hcg_d_hi))
            ),
        }
        self._clean_data()

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

    def plot_correlation(self, output_file="diameter_correlation.pdf", show=True):
        """
        Main correlation plot: D_HI vs D_25 with AMIGA fit line.
        This is the key figure implementing the Perea et al. approach.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        fit = self.fit_results.get("amiga")
        if fit is None:
            print("Run run_full_analysis() first!")
            return

        # Phase colors for HCG
        phase_colors = {
            "1": "#1f77b4",  # blue
            "2": "#2ca02c",  # green
            "3": "#d62728",  # red
            "3a": "#9467bd",  # purple
            "3c": "#ff7f0e",  # orange
        }

        # Plot AMIGA galaxies
        ax.scatter(
            self.amiga_data["D_25"],
            self.amiga_data["D_HI"],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="AMIGA (isolated)",
            zorder=3,
        )

        # Plot HCG galaxies by phase
        phases_plotted = []
        for phase in ["1", "2", "3", "3a", "3c"]:
            mask = self.hcg_data["phase"] == phase
            if np.sum(mask) > 0:
                ax.scatter(
                    self.hcg_data["D_25"][mask],
                    self.hcg_data["D_HI"][mask],
                    s=100,
                    c=phase_colors.get(phase, "gray"),
                    marker="s",
                    alpha=0.7,
                    label=f"HCG Phase {phase}",
                    zorder=4,
                )
                phases_plotted.append(phase)

        # Plot the AMIGA best-fit line
        x_range = np.logspace(np.log10(5), np.log10(100), 100)
        y_fit = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range))
        ax.plot(
            x_range,
            y_fit,
            "k-",
            linewidth=2.5,
            label=f"AMIGA fit: $D_{{\\rm HI}} \\propto D_{{25}}^{{{fit['slope']:.2f}}}$",
            zorder=2,
        )

        # Plot scatter envelope (±1σ, ±2σ)
        scatter = fit["scatter"]
        y_1sig_up = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) + scatter)
        y_1sig_lo = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) - scatter)
        y_2sig_up = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) + 2 * scatter)
        y_2sig_lo = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) - 2 * scatter)

        ax.fill_between(
            x_range,
            y_1sig_lo,
            y_1sig_up,
            color="gray",
            alpha=0.2,
            label=f"±1σ ({scatter:.2f} dex)",
            zorder=1,
        )
        ax.plot(x_range, y_2sig_up, "k--", linewidth=1, alpha=0.5, zorder=1)
        ax.plot(x_range, y_2sig_lo, "k--", linewidth=1, alpha=0.5, zorder=1)

        # 1:1 line for reference
        ax.plot(x_range, x_range, "k:", linewidth=1, alpha=0.5, label="1:1")

        # Axes setup
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D_{25}$ [kpc]", fontsize=22, labelpad=15)
        ax.set_ylabel(r"$D_{\rm HI}$ [kpc]", fontsize=22, labelpad=15)

        ax.set_xlim(5, 100)
        ax.set_ylim(1, 1000)

        # Tick styling (matching your style)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2, pad=10)
        ax.tick_params(which="minor", length=4, width=1, pad=10)

        # Legend
        ax.legend(loc="lower right", fontsize=14, frameon=True, framealpha=0.9, markerfirst=False)
        # Add text with fit statistics
        textstr = (
            f"AMIGA baseline:\n$r = {fit['r_value']:.2f}$\n$\\sigma = {fit['scatter']:.2f}$ dex"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.3,
            0.85,
            textstr,
            transform=ax.transAxes,
            fontsize=15,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"\nSaved: {output_path}")

        if show:
            plt.show()

        return fig, ax

    def plot_correlation_with_all_surveys(
        self, output_file="diameter_correlation_with_all_surveys.pdf", show=True
    ):
        """
        Same as plot_correlation but with all loaded surveys overlaid,
        each with a unique marker/colour combination.
        Requires surveys to be loaded via load_wang_table /
        load_broeils_rhee / register_surveys.
        """
        fig, ax = plt.subplots(figsize=(10.5, 8))

        fit = self.fit_results.get("amiga")
        if fit is None:
            print("Run run_full_analysis() first!")
            return

        # ---- Unique symbol bank for external surveys ----
        # 14 distinct (marker, edgecolor) combos; all open-face
        _symbol_bank = [
            ("D", "#e377c2"),  # diamond, pink
            ("^", "#8c564b"),  # triangle up, brown
            ("v", "#17becf"),  # triangle down, cyan
            ("p", "#bcbd22"),  # pentagon, olive
            ("h", "#7f7f7f"),  # hexagon, grey
            (">", "#e7298a"),  # triangle right, magenta
            ("<", "#66a61e"),  # triangle left, dark green
            ("*", "#d95f02"),  # star, dark orange
            ("P", "#1b9e77"),  # plus-filled, teal
            ("X", "#984ea3"),  # x-filled, purple
            ("d", "#a65628"),  # thin diamond, sienna
            ("H", "#f781bf"),  # hexagon2, light pink
            ("8", "#377eb8"),  # octagon, blue
            ("+", "#4daf4a"),  # plus, green
        ]

        # ---- Plot external surveys (behind everything) ----
        surveys = self.surveys or {}
        # Exclude AMIGA and HCGs (they get their own treatment)
        ext_names = sorted(k for k in surveys if k not in ("AMIGA", "HCGs"))
        for idx, name in enumerate(ext_names):
            mkr, clr = _symbol_bank[idx % len(_symbol_bank)]
            d = surveys[name]
            ax.scatter(
                d["D_25"],
                d["D_HI"],
                s=80,
                marker=mkr,
                facecolors="none",
                edgecolors=clr,
                linewidths=1.1,
                alpha=1,
                label=f"{name} ({len(d['D_25'])})",
                zorder=2,
            )

        # ---- AMIGA (open black circles) ----
        ax.scatter(
            self.amiga_data["D_25"],
            self.amiga_data["D_HI"],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="AMIGA (isolated)",
            zorder=3,
        )

        # ---- HCG by phase (filled coloured squares) ----
        phase_colors = {
            "1": "#1f77b4",
            "2": "#2ca02c",
            "3": "#d62728",
            "3a": "#9467bd",
            "3c": "#ff7f0e",
        }
        for phase in ["1", "2", "3", "3a", "3c"]:
            mask = self.hcg_data["phase"] == phase
            if np.sum(mask) > 0:
                ax.scatter(
                    self.hcg_data["D_25"][mask],
                    self.hcg_data["D_HI"][mask],
                    s=100,
                    c=phase_colors.get(phase, "gray"),
                    marker="s",
                    alpha=0.7,
                    label=f"HCG Phase {phase}",
                    zorder=4,
                )

        # ---- AMIGA best-fit line + scatter envelope ----
        # Auto-range to cover all data
        np.concatenate(
            [self.amiga_data["D_25"], self.hcg_data["D_25"]] + [s["D_25"] for s in surveys.values()]
        )
        np.concatenate(
            [self.amiga_data["D_HI"], self.hcg_data["D_HI"]] + [s["D_HI"] for s in surveys.values()]
        )
        x_lo = 0.1  # max(0.1, all_d25.min() * 0.7)
        x_hi = 100  # all_d25.max() * 1.3
        y_hi = 1000  # all_dhi.max() * 1.3

        x_range = np.logspace(np.log10(x_lo), np.log10(x_hi), 100)
        y_fit = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range))
        ax.plot(
            x_range,
            y_fit,
            "k-",
            linewidth=2.5,
            label=(f"AMIGA fit: $D_{{\\rm HI}} \\propto D_{{25}}^{{{fit['slope']:.2f}}}$"),
            zorder=5,
        )

        scatter = fit["scatter"]
        y_1s_up = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) + scatter)
        y_1s_lo = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) - scatter)
        y_2s_up = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) + 2 * scatter)
        y_2s_lo = 10 ** (fit["intercept"] + fit["slope"] * np.log10(x_range) - 2 * scatter)

        ax.fill_between(
            x_range,
            y_1s_lo,
            y_1s_up,
            color="gray",
            alpha=0.15,
            label=f"±1σ ({scatter:.2f} dex)",
            zorder=1,
        )
        ax.plot(x_range, y_2s_up, "k--", lw=1, alpha=0.4, zorder=1)
        ax.plot(x_range, y_2s_lo, "k--", lw=1, alpha=0.4, zorder=1)

        # 1:1 reference
        ax.plot(x_range, x_range, "k:", lw=1, alpha=0.4, label="1:1")

        # ---- Axes ----
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D_{25}$ [kpc]", fontsize=22, labelpad=15)
        ax.set_ylabel(r"$D_{\rm HI}$ [kpc]", fontsize=22, labelpad=15)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(x_lo, y_hi)
        # ax.set_ylim(y_lo, max(y_hi, y_1s_up.max() * 1.1))
        # ax.set_xlim(1, 100)
        # ax.set_ylim(y_lo, max(y_hi, y_1s_up.max() * 1.1))

        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2, pad=10)
        ax.tick_params(which="minor", length=4, width=1, pad=10)

        # Legend: two columns to fit many entries
        # ax.legend(
        #    loc='lower right', fontsize=10, frameon=True,
        #    framealpha=0.9, markerfirst=False, ncol=2,
        #    borderpad=0.8, handletextpad=0.5, columnspacing=1.0
        # )

        divider = make_axes_locatable(ax)
        lax = divider.append_axes("right", size="5%", pad=0.2)  # adjust size/pad as needed
        lax.axis("off")  # no frame, no ticks

        handles, labels = ax.get_legend_handles_labels()

        lax.legend(
            handles,
            labels,
            loc="center left",
            fontsize=14,
            frameon=True,
            framealpha=0.9,
            markerfirst=False,
        )

        textstr = (
            f"AMIGA baseline:\n$r = {fit['r_value']:.2f}$\n$\\sigma = {fit['scatter']:.2f}$ dex"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.3,
            0.93,
            textstr,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"\nSaved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    def plot_residuals_histogram(self, output_file="diameter_residuals_hist.pdf", show=True):
        """
        Histogram of residuals for AMIGA vs HCG.
        HCG bars are stacked and coloured by evolutionary phase (with dot pattern).
        AMIGA bars are hatched and drawn on top so they remain visible.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        res_amiga = self.fit_results.get("residuals_amiga")
        res_hcg = self.fit_results.get("residuals_hcg")

        if res_amiga is None or res_hcg is None:
            print("Run run_full_analysis() first!")
            return

        # Histogram bins
        all_res = np.concatenate([res_amiga, res_hcg])
        bins = np.linspace(np.min(all_res) - 0.1, np.max(all_res) + 0.1, 20)
        bin_width = bins[1] - bins[0]

        # --- Phase setup ---
        phase_order = ["1", "2", "3a", "3c"]
        phase_colors = {
            "1": "#a6cee3",  # light blue
            "2": "#8fbc8f",  # light green
            "3a": "#c4a6d6",  # light purple
            "3c": "#f4b07c",  # light orange
        }
        phase_labels = {
            "1": "Phase 1",
            "2": "Phase 2",
            "3a": "Phase 3a",
            "3c": "Phase 3c",
        }

        # Compute per-phase histogram counts manually
        phase_counts = {}
        phase_present = []
        for phase in phase_order:
            mask = self.hcg_data["phase"] == phase
            if np.sum(mask) > 0:
                counts, _ = np.histogram(res_hcg[mask], bins=bins)
                phase_counts[phase] = counts
                phase_present.append(phase)

        # --- Draw stacked HCG bars (no internal vertical edges) ---
        bottom = np.zeros(len(bins) - 1)
        for phase in phase_present:
            counts = phase_counts[phase]
            ax.bar(
                bins[:-1],
                counts,
                width=bin_width,
                bottom=bottom,
                align="edge",
                color=phase_colors[phase],
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            bottom += counts

        # HCG total outline (step) + dot pattern via stepfilled
        ax.hist(
            res_hcg,
            bins=bins,
            histtype="stepfilled",
            facecolor="none",
            edgecolor="none",
            hatch=".",
            linewidth=0,
            zorder=3,
        )
        ax.hist(
            res_hcg,
            bins=bins,
            histtype="step",
            linewidth=2.5,
            edgecolor="blue",
            label="Galaxies in HCGs",
            zorder=4,
        )

        # --- AMIGA histogram (hatched, transparent face, visible on top) ---
        ax.hist(
            res_amiga,
            bins=bins,
            histtype="stepfilled",
            facecolor="white",
            edgecolor="none",
            hatch="/",
            linewidth=0,
            zorder=5,
            alpha=0.7,
        )
        ax.hist(
            res_amiga,
            bins=bins,
            histtype="step",
            linewidth=2.5,
            edgecolor="black",
            label="AMIGA galaxies",
            zorder=6,
        )

        # Mark means
        mean_amiga = np.mean(res_amiga)
        mean_hcg = np.mean(res_hcg)

        ax.axvline(
            mean_amiga,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"AMIGA mean: {mean_amiga:.2f}",
            zorder=7,
        )
        ax.axvline(
            mean_hcg,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"HCG mean: {mean_hcg:.2f}",
            zorder=7,
        )
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, zorder=1)

        # Labels
        ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
        ax.set_ylabel("Number of galaxies", fontsize=22, labelpad=15)

        # Tick styling
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2, pad=10)
        ax.tick_params(which="minor", length=4, width=1, pad=10)

        # --- Main legend (upper left) ---
        main_legend = ax.legend(loc="upper left", fontsize=17, frameon=True)
        hcg_handle = Patch(
            facecolor="none", edgecolor="blue", hatch=".", linewidth=1.2, label="Galaxies in HCGs"
        )

        amiga_handle = Patch(
            facecolor="white", edgecolor="black", hatch="/", linewidth=1.2, label="AMIGA galaxies"
        )

        # keep your vlines as-is (they already appear correctly in legend)
        mean_handles = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"AMIGA mean: {mean_amiga:.2f}",
            ),
            plt.Line2D(
                [0],
                [0],
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"HCG mean: {mean_hcg:.2f}",
            ),
        ]

        main_legend = ax.legend(
            handles=[hcg_handle, amiga_handle, *mean_handles],
            loc="upper left",
            fontsize=17,
            frameon=True,
        )
        ax.add_artist(main_legend)
        # ax.add_artist(main_legend)

        # --- Phase legend (right side, raised) ---
        phase_handles = [
            Patch(facecolor=phase_colors[p], edgecolor="blue", linewidth=1.2, label=phase_labels[p])
            for p in phase_present
        ]
        phase_leg = ax.legend(
            handles=phase_handles,
            title="HCG phases",
            title_fontsize=19,
            fontsize=17,
            frameon=True,
            framealpha=0.9,
            edgecolor="0.6",
            loc="center right",
            bbox_to_anchor=(0.99, 0.35),
        )
        ax.add_artist(phase_leg)
        ax.set_ylim([0, 14])
        # Add annotation
        offset = mean_hcg - mean_amiga
        ax.annotate(
            f"Offset: {offset:.2f} dex",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            fontsize=17,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")

        if show:
            plt.show()

        return fig, ax

    def plot_residuals_vs_D25(self, output_file="diameter_residuals_vs_D25.pdf", show=True):
        """
        Plot residuals vs D_25 to check for systematic trends.
        Important: if there's a correlation here, the ratio approach was hiding it.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        res_amiga = self.fit_results.get("residuals_amiga")
        res_hcg = self.fit_results.get("residuals_hcg")
        fit = self.fit_results.get("amiga")

        if res_amiga is None:
            print("Run run_full_analysis() first!")
            return

        # Phase colors
        phase_colors = {
            "1": "#1f77b4",
            "2": "#2ca02c",
            "3": "#d62728",
            "3a": "#9467bd",
            "3c": "#ff7f0e",
        }

        # Plot AMIGA
        ax.scatter(
            self.amiga_data["D_25"],
            res_amiga,
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="AMIGA",
            zorder=3,
        )

        # Plot HCG by phase
        for phase in ["1", "2", "3", "3a", "3c"]:
            mask = self.hcg_data["phase"] == phase
            if np.sum(mask) > 0:
                ax.scatter(
                    self.hcg_data["D_25"][mask],
                    res_hcg[mask],
                    s=100,
                    c=phase_colors.get(phase, "gray"),
                    marker="s",
                    alpha=0.7,
                    label=f"HCG Phase {phase}",
                    zorder=4,
                )

        # Reference lines
        ax.axhline(0, color="black", linestyle="-", linewidth=2)
        ax.axhline(fit["scatter"], color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(-fit["scatter"], color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

        # Fill the ±1σ region
        ax.get_xlim()
        ax.fill_between(
            [0, 100], -fit["scatter"], fit["scatter"], color="gray", alpha=0.1, zorder=1
        )

        ax.set_xscale("log")
        ax.set_xlabel(r"$D_{25}$ [kpc]", fontsize=22, labelpad=15)
        ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
        ax.set_xlim(5, 100)

        # Check for residual correlation with D_25
        r_amiga, p_amiga = stats.pearsonr(np.log10(self.amiga_data["D_25"]), res_amiga)
        r_hcg, p_hcg = stats.pearsonr(np.log10(self.hcg_data["D_25"]), res_hcg)

        textstr = (
            f"Residual correlation with $D_{{25}}$:\n"
            f"AMIGA: $r={r_amiga:.2f}$, $p={p_amiga:.2e}$\n"
            f"HCG: $r={r_hcg:.2f}$, $p={p_hcg:.2e}$"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.03,
            0.03,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=props,
        )

        # Tick styling
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2, pad=10)
        ax.tick_params(which="minor", length=4, width=1, pad=10)

        ax.legend(loc="upper right", fontsize=12, frameon=True)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")

        if show:
            plt.show()

        return fig, ax

    def plot_residuals_by_phase(self, output_file="diameter_residuals_by_phase.pdf", show=True):
        """
        Box plot / violin plot of residuals by HCG phase.
        Shows the progression of HI truncation with evolutionary phase.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        res_amiga = self.fit_results.get("residuals_amiga")
        res_hcg = self.fit_results.get("residuals_hcg")

        if res_amiga is None:
            print("Run run_full_analysis() first!")
            return

        # Prepare data for box plot
        phases = ["1", "2", "3a", "3c"]  # ordered by expected evolution
        data_list = [res_amiga]  # AMIGA as reference
        labels = ["AMIGA\n(isolated)"]
        colors = ["white"]

        phase_colors = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}

        for phase in phases:
            mask = self.hcg_data["phase"] == phase
            if np.sum(mask) >= 2:
                data_list.append(res_hcg[mask])
                labels.append(f"Phase {phase}\n(n={np.sum(mask)})")
                colors.append(phase_colors.get(phase, "gray"))

        # Box plot
        positions = np.arange(len(data_list))
        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True, notch=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Add individual points
        for i, (data, color) in enumerate(zip(data_list, colors)):
            x = np.random.normal(i, 0.08, size=len(data))
            ax.scatter(
                x,
                data,
                alpha=0.6,
                s=40,
                c="black" if color == "white" else color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

        # Reference line
        ax.axhline(0, color="black", linestyle="-", linewidth=2, zorder=1)
        ax.axhline(
            self.fit_results["amiga"]["scatter"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.axhline(
            -self.fit_results["amiga"]["scatter"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=19)
        ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
        ax.set_xlabel("Sample / HCG Phase", fontsize=22, labelpad=15)

        # Tick styling
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2, pad=10)
        ax.tick_params(which="minor", length=4, width=1, pad=10)
        ax.tick_params(axis="x", which="minor", bottom=False)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")

        if show:
            plt.show()

        return fig, ax

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

    def load_survey_csv(self, csv_path, survey_name):
        """
        Convenience loader: read a single survey from a CSV file
        and add it to self.surveys.

        Expected columns: hi_diameter_kpc, optical_diameter_kpc
        """
        df = pd.read_csv(csv_path)
        d_hi = df["hi_diameter_kpc"].to_numpy(dtype=float)
        d_25 = df["optical_diameter_kpc"].to_numpy(dtype=float)
        mask = np.isfinite(d_hi) & np.isfinite(d_25) & (d_hi > 0) & (d_25 > 0)

        if not hasattr(self, "surveys"):
            self.surveys = {}
        self.surveys[survey_name] = {
            "D_HI": d_hi[mask],
            "D_25": d_25[mask],
        }
        print(f"Loaded {survey_name}: {np.sum(mask)} galaxies from {csv_path}")

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

    def load_cig_stellar_masses(self, table_path):
        """
        Load CIG (AMIGA) stellar masses from a CSV file and
        cross-match with self.amiga_data['name'].

        Expected CSV columns:
            CIG       : CIG number (integer)
            logMstar  : log10(M_star / Msun)
            e_logMstar: error on logMstar  (optional)

        Galaxy names in self.amiga_data['name'] must contain a CIG
        number (e.g. 'CIG72', 'CIG_72', 'CIG 72').

        Parameters
        ----------
        table_path : str
            Path to the CSV (e.g. angmom_final.csv).
        """
        if self.amiga_data is None:
            raise RuntimeError("Load AMIGA data first via load_data().")

        import re

        df_cig = pd.read_csv(table_path)
        if "CIG" not in df_cig.columns or "logMstar" not in df_cig.columns:
            raise ValueError(
                f"CSV must contain 'CIG' and 'logMstar' columns. Found: {list(df_cig.columns)}"
            )

        # Build lookup dicts keyed by "CIG<number>"
        logm_dict = {}
        logm_err_dict = {}
        for _, row in df_cig.iterrows():
            key = f"CIG{int(row['CIG'])}"
            val = float(row["logMstar"])
            if np.isfinite(val) and val > 0:
                logm_dict[key] = val
            if "e_logMstar" in df_cig.columns:
                err = float(row["e_logMstar"])
                if np.isfinite(err):
                    logm_err_dict[key] = err

        n = len(self.amiga_data["D_HI"])
        log_mstar = np.full(n, np.nan)
        log_mstar_err = np.full(n, np.nan)

        matched = 0
        for i, raw_name in enumerate(self.amiga_data["name"]):
            m = re.match(r"[Cc][Ii][Gg][_ ]?(\d+)", str(raw_name).strip())
            if m:
                key = f"CIG{int(m.group(1))}"  # int() strips leading zeros
                if key in logm_dict:
                    log_mstar[i] = logm_dict[key]
                    matched += 1
                if key in logm_err_dict:
                    log_mstar_err[i] = logm_err_dict[key]

        self.amiga_data["log_stellar_mass"] = log_mstar
        self.amiga_data["log_stellar_mass_err"] = log_mstar_err

        n_valid = np.sum(np.isfinite(log_mstar))
        valid = log_mstar[np.isfinite(log_mstar)]
        print(f"Matched {matched} / {n} AMIGA galaxies with CIG stellar masses ({n_valid} valid)")
        if len(valid) > 0:
            print(
                f"  Matched log(M*) range: "
                f"{valid.min():.2f} – {valid.max():.2f}, "
                f"median: {np.median(valid):.2f}"
            )

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

    def plot_survey_mean_residual(
        self, rank_metric="mean", output_file="survey_mean_residual.pdf", show=True
    ):
        """
        Horizontal dot plot: mean residual ± scatter for each survey,
        ranked by mean residual.  Replaces the old peak-D_HI/D_25 plot.
        Marker size scales with the number of galaxies.
        """
        if not hasattr(self, "survey_stats"):
            raise RuntimeError("Call compute_survey_residuals() first.")

        # Build a DataFrame for easy sorting
        rows = []
        for name, s in self.survey_stats.items():
            rows.append(
                {
                    "sample": name,
                    "N": s["N"],
                    rank_metric: s[rank_metric],
                    "scatter": s["scatter"],
                }
            )
        df = pd.DataFrame(rows).sort_values(rank_metric, ascending=False).reset_index(drop=True)

        y = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, max(6, 0.45 * len(df) + 1)))

        # Error bars (scatter)
        ax.errorbar(
            df[rank_metric],
            y,
            xerr=df["scatter"],
            fmt="o",
            capsize=3,
            elinewidth=1.2,
            markersize=6,
            zorder=3,
            color="#1d5378",
        )

        # Size-scaled markers
        N = df["N"].values
        if N.max() > N.min():
            sizes = 30 + 120 * (N - N.min()) / (N.max() - N.min())
        else:
            sizes = np.full_like(N, 80, dtype=float)
        ax.scatter(df[rank_metric], y, s=sizes, zorder=4, color="#1d5378")

        # Y labels
        labels = [f"{s} (n={n})" for s, n in zip(df["sample"], df["N"])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        # Reference lines
        sigma = self.fit_results["amiga"]["scatter"]
        ax.axvline(0, ls="-", lw=1.5, color="black", alpha=0.6)
        ax.axvline(sigma, ls="--", lw=1.0, alpha=0.4, color="gray")
        ax.axvline(-sigma, ls="--", lw=1.0, alpha=0.4, color="gray")
        ax.axvspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)
        if rank_metric == "mean":
            ax.set_xlabel(
                r"Mean $\Delta\log(D_{\rm HI})$  (± scatter) [dex]", fontsize=20, labelpad=30
            )
            ax.set_title(r"Ranked by mean $\Delta\log(D_{\rm HI})$", fontsize=18, pad=10)
        else:
            ax.set_xlabel(
                r"Median $\Delta\log(D_{\rm HI})$  (± scatter) [dex]", fontsize=20, labelpad=30
            )
            ax.set_title(r"Ranked by median $\Delta\log(D_{\rm HI})$", fontsize=18, pad=10)
        ax.set_ylabel("Surveys", fontsize=20, labelpad=30)
        ax.tick_params(axis="x", pad=10)
        ax.tick_params(axis="y", pad=10)
        self._style_axes(ax)
        secaxr = ax.secondary_yaxis("right")
        secaxr.tick_params(labelright=False)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    # ---------- Plot 2 : % extended ----------

    def plot_survey_frac_extended(self, output_file="survey_frac_extended.pdf", show=True):
        r"""
        Horizontal bar chart: fraction of galaxies with
        $\Delta\log D_{\rm HI} > +\sigma_{\rm AMIGA}$ (significantly
        extended relative to the isolated-galaxy baseline), ranked
        descending.
        """
        if not hasattr(self, "survey_stats"):
            raise RuntimeError("Call compute_survey_residuals() first.")

        rows = []
        for name, s in self.survey_stats.items():
            rows.append(
                {
                    "sample": name,
                    "N": s["N"],
                    "value": s["f_extended"],
                }
            )
        df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

        self._barh_ranked(
            df,
            "value",
            title=(
                r"Ranked by $\Delta\log(D_{\rm HI})"
                r" > +\sigma_{\rm AMIGA}$"
            ),
            output_file=output_file,
            show=show,
        )

    # ---------- Plot 3 : % truncated ----------

    def plot_survey_frac_truncated(
        self, output_file="survey_frac_truncated.pdf", show=True, ensure_last="AMIGA"
    ):
        r"""
        Horizontal bar chart: fraction of galaxies with
        $\Delta\log D_{\rm HI} < -\sigma_{\rm AMIGA}$ (significantly
        truncated relative to the isolated-galaxy baseline), ranked
        descending, with AMIGA forced to the bottom.
        """
        if not hasattr(self, "survey_stats"):
            raise RuntimeError("Call compute_survey_residuals() first.")

        rows = []
        for name, s in self.survey_stats.items():
            rows.append(
                {
                    "sample": name,
                    "N": s["N"],
                    "value": s["f_severe_trunc"],
                }
            )
        df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

        self._barh_ranked(
            df,
            "value",
            title=(
                r"Ranked by $\Delta\log(D_{\rm HI})"
                r" < -\sigma_{\rm AMIGA}$"
            ),
            output_file=output_file,
            show=show,
            ensure_last_label=ensure_last,
        )

    # ---------- generic horizontal bar helper ----------

    def _barh_ranked(self, df, value_col, title, output_file, show=True, ensure_last_label=None):
        """
        Internal helper: horizontal ranked bar chart by `value_col`.
        If `ensure_last_label` is given, that survey is pinned to the
        bottom after sorting.
        """
        d = df.copy()

        # Pin a specific survey to the bottom
        if ensure_last_label is not None and (d["sample"] == ensure_last_label).any():
            tail = d[d["sample"] == ensure_last_label]
            head = d[d["sample"] != ensure_last_label]
            d = pd.concat([head, tail], axis=0).reset_index(drop=True)

        y = np.arange(len(d))

        fig, ax = plt.subplots(figsize=(9, max(6, 0.45 * len(d) + 1)))

        ax.barh(y, d[value_col], zorder=2, color="#1d5378")
        labels = [f"{s} (n={n})" for s, n in zip(d["sample"], d["N"])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        # Annotate percentages at bar tips
        for yi, val in zip(y, d[value_col]):
            ax.text(val + 0.8, yi, f"   {val:.1f}%", va="center", ha="left", fontsize=16)

        ax.invert_yaxis()
        max(50.0, d[value_col].max() * 1.15 + 2)
        # ax.set_xlim(0, min(100, xmax))
        if "extended" in output_file:
            ax.set_xlim(0, 25)
        else:
            ax.set_xlim(0, 125)
        ax.tick_params(axis="x", pad=10)
        ax.tick_params(axis="y", pad=10)
        ax.set_xlabel("Fraction of galaxies (%)", fontsize=20, labelpad=30)
        ax.set_ylabel("Surveys", fontsize=20, labelpad=30)
        ax.set_title(title, fontsize=18, pad=10)

        self._style_axes(ax)
        secax = ax.secondary_xaxis("top")
        secax.tick_params(labeltop=False)
        secaxr = ax.secondary_yaxis("right")
        secaxr.tick_params(labelright=False)
        self._style_axes(secax)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    # ========== MASS-RELATED DIAGNOSTICS (AMIGA vs HCG) ==========

    def _is_log_mass(self, arr):
        """
        Heuristic: if all positive values fall in [4, 14], the array
        is almost certainly already log10(M/Msun).  If any value
        exceeds 1e4 it is linear.
        """
        valid = arr[np.isfinite(arr) & (arr > 0)]
        if len(valid) == 0:
            return False
        return valid.max() < 20  # log10(Msun) never exceeds ~14

    def _ensure_log_mass(self, arr, label=""):
        """
        Return log10 of the mass array, auto-detecting whether the
        input is already in log form or in linear solar masses.
        """
        is_log = self._is_log_mass(arr)
        if is_log:
            print(
                f"  [{label}] hi_mass appears to be in LOG form "
                f"(max={np.nanmax(arr):.2f}); using as-is."
            )
            return arr.copy()
        else:
            print(
                f"  [{label}] hi_mass appears to be in LINEAR form "
                f"(max={np.nanmax(arr):.2e}); taking log10."
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log10(arr)

    def _get_mass_residual_arrays(self, sample, mass_key):
        """
        Internal helper: return (mass_array, residual_array) for a
        given sample ('amiga' or 'hcg'), keeping only entries where
        both the mass and the residual are finite.

        Parameters
        ----------
        sample : str   ('amiga' or 'hcg')
        mass_key : str (e.g. 'log_stellar_mass', 'hi_mass')

        Returns
        -------
        mass, residuals, phase (phase is None for AMIGA)
        """
        if sample == "amiga":
            data = self.amiga_data
            residuals = self.fit_results.get("residuals_amiga")
            phase = None
        else:
            data = self.hcg_data
            residuals = self.fit_results.get("residuals_hcg")
            phase = data.get("phase")

        if mass_key not in data:
            return None, None, None
        if residuals is None:
            raise RuntimeError("Run run_full_analysis() first.")

        mass = data[mass_key].copy()
        # For HI mass, convert to log10 (auto-detecting if already log)
        if mass_key == "hi_mass":
            mass = self._ensure_log_mass(mass, label=f"{sample}/hi_mass")

        mask = np.isfinite(mass) & np.isfinite(residuals) & (mass != 0)
        return mass[mask], residuals[mask], phase[mask] if phase is not None else None

    def plot_residuals_vs_stellar_mass(
        self, output_file="residuals_vs_stellar_mass.pdf", show=True
    ):
        r"""
        Scatter plot: $\Delta\log(D_{\rm HI})$ vs $\log(M_\star)$.
        Shows whether HI disk truncation correlates with stellar mass.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Phase colours for HCG
        phase_colors = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}

        # --- AMIGA ---
        m_a, r_a, _ = self._get_mass_residual_arrays("amiga", "log_stellar_mass")
        if m_a is not None and len(m_a) > 0:
            ax.scatter(
                m_a,
                r_a,
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="AMIGA",
                zorder=3,
            )
            # Trend line
            if len(m_a) >= 5:
                sl, ic, rv, _, _ = stats.linregress(m_a, r_a)
                xfit = np.linspace(m_a.min(), m_a.max(), 50)
                ax.plot(xfit, ic + sl * xfit, "k--", lw=1.5, alpha=0.5)

        # --- HCG by phase ---
        m_h, r_h, ph_h = self._get_mass_residual_arrays("hcg", "log_stellar_mass")
        if m_h is not None and len(m_h) > 0:
            for phase in ["1", "2", "3a", "3c"]:
                pmask = ph_h == phase
                if np.sum(pmask) > 0:
                    ax.scatter(
                        m_h[pmask],
                        r_h[pmask],
                        s=100,
                        c=phase_colors.get(phase, "gray"),
                        marker="s",
                        alpha=0.7,
                        label=f"HCG Phase {phase}",
                        zorder=4,
                    )

        sigma = self.fit_results["amiga"]["scatter"]
        ax.axhline(0, color="black", ls="-", lw=2, zorder=1)
        ax.axhline(sigma, color="gray", ls="--", lw=1.5, alpha=0.5)
        ax.axhline(-sigma, color="gray", ls="--", lw=1.5, alpha=0.5)
        ax.axhspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)

        ax.set_xlabel(r"$\log\,(M_\star\,/\,M_\odot)$", fontsize=22, labelpad=15)
        ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)

        self._style_axes(ax)
        ax.legend(loc="lower left", fontsize=12, frameon=True)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    def plot_residuals_vs_hi_mass(self, output_file="residuals_vs_hi_mass.pdf", show=True):
        r"""
        Scatter plot: $\Delta\log(D_{\rm HI})$ vs $\log(M_{\rm HI})$.
        Shows whether HI disk truncation correlates with HI content.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        phase_colors = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}

        # --- AMIGA ---
        m_a, r_a, _ = self._get_mass_residual_arrays("amiga", "hi_mass")
        if m_a is not None and len(m_a) > 0:
            ax.scatter(
                m_a,
                r_a,
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="AMIGA",
                zorder=3,
            )
            if len(m_a) >= 5:
                sl, ic, rv, _, _ = stats.linregress(m_a, r_a)
                xfit = np.linspace(m_a.min(), m_a.max(), 50)
                ax.plot(xfit, ic + sl * xfit, "k--", lw=1.5, alpha=0.5)

        # --- HCG by phase ---
        m_h, r_h, ph_h = self._get_mass_residual_arrays("hcg", "hi_mass")
        if m_h is not None and len(m_h) > 0:
            for phase in ["1", "2", "3a", "3c"]:
                pmask = ph_h == phase
                if np.sum(pmask) > 0:
                    ax.scatter(
                        m_h[pmask],
                        r_h[pmask],
                        s=100,
                        c=phase_colors.get(phase, "gray"),
                        marker="s",
                        alpha=0.7,
                        label=f"HCG Phase {phase}",
                        zorder=4,
                    )

        sigma = self.fit_results["amiga"]["scatter"]
        ax.axhline(0, color="black", ls="-", lw=2, zorder=1)
        ax.axhline(sigma, color="gray", ls="--", lw=1.5, alpha=0.5)
        ax.axhline(-sigma, color="gray", ls="--", lw=1.5, alpha=0.5)
        ax.axhspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)

        ax.set_xlabel(r"$\log\,(M_{\rm HI}\,/\,M_\odot)$", fontsize=22, labelpad=15)
        ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)

        self._style_axes(ax)
        ax.legend(loc="lower left", fontsize=12, frameon=True)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    def plot_residuals_vs_gas_fraction(
        self, output_file="residuals_vs_gas_fraction.pdf", show=True
    ):
        r"""
        Scatter plot: $\Delta\log(D_{\rm HI})$ vs
        $\log(M_{\rm HI}/M_\star)$ (gas fraction).
        Shows whether HI disk truncation correlates with the gas
        richness of a galaxy.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        phase_colors = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}

        def _gas_fraction(data, residuals, sample_label, phase=None):
            """Return log(M_HI/M_star), residuals, phase (valid only)."""
            has_hi = "hi_mass" in data
            has_ms = "log_stellar_mass" in data
            if not (has_hi and has_ms):
                return None, None, None
            log_hi = self._ensure_log_mass(data["hi_mass"], label=f"{sample_label}/hi_mass")
            log_ms = data["log_stellar_mass"]

            # Diagnostic
            valid_hi = log_hi[np.isfinite(log_hi)]
            valid_ms = log_ms[np.isfinite(log_ms)]
            if len(valid_hi) > 0:
                print(
                    f"  [{sample_label}] log(M_HI) range: "
                    f"{valid_hi.min():.2f} – {valid_hi.max():.2f}"
                )
            if len(valid_ms) > 0:
                print(
                    f"  [{sample_label}] log(M_star) range: "
                    f"{valid_ms.min():.2f} – {valid_ms.max():.2f}"
                )

            gf = log_hi - log_ms  # log(M_HI / M_star)
            mask = np.isfinite(gf) & np.isfinite(residuals)

            valid_gf = gf[mask]
            if len(valid_gf) > 0:
                print(
                    f"  [{sample_label}] gas fraction range: "
                    f"{valid_gf.min():.2f} – {valid_gf.max():.2f}"
                )

            ph = phase[mask] if phase is not None else None
            return gf[mask], residuals[mask], ph

        res_a = self.fit_results.get("residuals_amiga")
        res_h = self.fit_results.get("residuals_hcg")

        # --- AMIGA ---
        gf_a, r_a, _ = _gas_fraction(self.amiga_data, res_a, "AMIGA")
        if gf_a is not None and len(gf_a) > 0:
            ax.scatter(
                gf_a,
                r_a,
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="AMIGA",
                zorder=3,
            )
            if len(gf_a) >= 5:
                sl, ic, _, _, _ = stats.linregress(gf_a, r_a)
                xfit = np.linspace(gf_a.min(), gf_a.max(), 50)
                ax.plot(xfit, ic + sl * xfit, "k--", lw=1.5, alpha=0.5)

        # --- HCG by phase ---
        gf_h, r_h, ph_h = _gas_fraction(self.hcg_data, res_h, "HCG", self.hcg_data.get("phase"))
        if gf_h is not None and len(gf_h) > 0:
            for phase in ["1", "2", "3a", "3c"]:
                pmask = ph_h == phase
                if np.sum(pmask) > 0:
                    ax.scatter(
                        gf_h[pmask],
                        r_h[pmask],
                        s=100,
                        c=phase_colors.get(phase, "gray"),
                        marker="s",
                        alpha=0.7,
                        label=f"HCG Phase {phase}",
                        zorder=4,
                    )

        sigma = self.fit_results["amiga"]["scatter"]
        ax.axhline(0, color="black", ls="-", lw=2, zorder=1)
        ax.axhline(sigma, color="gray", ls="--", lw=1.5, alpha=0.5)
        ax.axhline(-sigma, color="gray", ls="--", lw=1.5, alpha=0.5)
        ax.axhspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)

        ax.set_xlabel(r"$\log\,(M_{\rm HI}\,/\,M_\star)$", fontsize=22, labelpad=15)
        ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)

        self._style_axes(ax)
        ax.legend(loc="lower left", fontsize=12, frameon=True)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

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

    def plot_truncation_index_by_phase(
        self, output_file="truncation_index_by_phase.pdf", show=True
    ):
        r"""
        Box-and-strip plot of the truncation index
        $T = D_{\rm HI}/D_{\rm HI,exp} = 10^{\Delta}$
        split by sample (AMIGA) and HCG phase.

        The dashed line at T = 1 marks the AMIGA expectation value.
        """
        res_a = self.fit_results.get("residuals_amiga")
        res_h = self.fit_results.get("residuals_hcg")
        if res_a is None or res_h is None:
            raise RuntimeError("Run run_full_analysis() first.")

        T_a = 10.0**res_a
        T_h = 10.0**res_h
        phases = self.hcg_data["phase"]

        phase_colors = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}
        amiga_color = "#555555"

        # Build groups: AMIGA, then each phase
        group_labels = ["AMIGA"]
        group_data = [T_a]
        group_colors = [amiga_color]
        phase_order = ["1", "2", "3a", "3c"]
        for ph in phase_order:
            pmask = phases == ph
            if np.sum(pmask) >= 2:
                group_labels.append(f"Phase {ph}")
                group_data.append(T_h[pmask])
                group_colors.append(phase_colors.get(ph, "gray"))

        fig, ax = plt.subplots(figsize=(8, 8))

        # Box plots
        bp = ax.boxplot(
            group_data,
            positions=range(len(group_data)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
        for element in ["whiskers", "caps"]:
            for line, color in zip(bp[element], np.repeat(group_colors, 2)):
                line.set_color(color)
                line.set_linewidth(1.5)
        for med_line, color in zip(bp["medians"], group_colors):
            med_line.set_color(color)
            med_line.set_linewidth(2.5)

        # Strip (jittered points)
        rng = np.random.default_rng(42)
        for i, (data, color) in enumerate(zip(group_data, group_colors)):
            jitter = rng.uniform(-0.15, 0.15, size=len(data))
            ax.scatter(
                np.full(len(data), i) + jitter,
                data,
                s=40,
                c=color,
                alpha=0.6,
                edgecolors="none",
                zorder=3,
            )

        # Reference line at T = 1
        ax.axhline(
            1.0, color="black", ls="--", lw=2, zorder=1, label=r"$T = 1$ (AMIGA expectation)"
        )
        ax.axhline(
            0.5, color="grey", ls=":", lw=1.2, zorder=1, label=r"$T = 0.5$ (50 % truncation)"
        )

        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, fontsize=16)
        ax.set_ylabel(
            r"Truncation index  "
            r"$T = D_{\rm HI}\,/\,D_{\rm HI,exp}$",
            fontsize=20,
            labelpad=15,
        )
        ax.legend(fontsize=13, loc="upper right", frameon=True)
        self._style_axes(ax)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    def plot_size_matched_comparison(self, output_file="size_matched_comparison.pdf", show=True):
        r"""
        Two-panel plot showing the size-matched AMIGA–HCG comparison.

        Left:  paired residuals (connected by grey lines) vs log D_25.
        Right: histogram of paired differences Δ_HCG − Δ_AMIGA.
        """
        focused = self.fit_results.get("focused")
        if focused is None:
            raise RuntimeError("Run run_focused_comparison() first.")

        idx_a = focused["matched_idx_a"]
        idx_h = focused["matched_idx_h"]
        res_a = self.fit_results["residuals_amiga"]
        res_h = self.fit_results["residuals_hcg"]

        res_a_m = res_a[idx_a]
        res_h_m = res_h[idx_h]
        logd_a = np.log10(self.amiga_data["D_25"][idx_a])
        logd_h = np.log10(self.hcg_data["D_25"][idx_h])
        paired_diff = res_h_m - res_a_m

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # ---- Left panel: paired residuals vs log(D_25) ----
        for la, lh, ra, rh in zip(logd_a, logd_h, res_a_m, res_h_m):
            ax1.plot([la, lh], [ra, rh], color="grey", lw=0.6, alpha=0.4, zorder=1)

        ax1.scatter(
            logd_a,
            res_a_m,
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            label="AMIGA (matched)",
            zorder=3,
        )
        ax1.scatter(
            logd_h,
            res_h_m,
            s=70,
            c="#d62728",
            marker="s",
            alpha=0.7,
            label="HCG (matched)",
            zorder=3,
        )

        sigma = self.fit_results["amiga"]["scatter"]
        ax1.axhline(0, color="black", ls="-", lw=2, zorder=1)
        ax1.axhline(sigma, color="gray", ls="--", lw=1.2, alpha=0.5)
        ax1.axhline(-sigma, color="gray", ls="--", lw=1.2, alpha=0.5)
        ax1.axhspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)

        ax1.set_xlabel(r"$\log\,(D_{25}$ / kpc)", fontsize=20, labelpad=15)
        ax1.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=20, labelpad=15)
        ax1.legend(fontsize=13, loc="lower left", frameon=True)
        self._style_axes(ax1)

        # ---- Right panel: histogram of paired differences ----
        ax2.hist(
            paired_diff,
            bins="auto",
            color="#d62728",
            alpha=0.6,
            edgecolor="#d62728",
            linewidth=1.5,
            zorder=2,
        )
        ax2.axvline(0, color="black", ls="-", lw=2, zorder=1)
        med_diff = np.median(paired_diff)
        ax2.axvline(
            med_diff,
            color="#d62728",
            ls="--",
            lw=2.5,
            zorder=3,
            label=f"median = {med_diff:+.3f} dex",
        )

        ax2.set_xlabel(
            r"Paired difference  "
            r"$\Delta_{\rm HCG} - \Delta_{\rm AMIGA}$ [dex]",
            fontsize=18,
            labelpad=15,
        )
        ax2.set_ylabel("Count", fontsize=20, labelpad=15)
        ax2.legend(fontsize=14, loc="upper left", frameon=True)
        self._style_axes(ax2)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, (ax1, ax2)

    def plot_binned_residuals_comparison(
        self, output_file="binned_residuals_comparison.pdf", show=True
    ):
        r"""
        Side-by-side comparison of AMIGA vs HCG residuals in bins of
        $\log(D_{25})$, with median and IQR shown as boxes + strips.
        """
        focused = self.fit_results.get("focused")
        if focused is None:
            raise RuntimeError("Run run_focused_comparison() first.")

        focused["bin_edges"]
        binned = focused["binned"]
        res_a = self.fit_results["residuals_amiga"]
        res_h = self.fit_results["residuals_hcg"]
        log_d25_a = np.log10(self.amiga_data["D_25"])
        log_d25_h = np.log10(self.hcg_data["D_25"])

        fig, ax = plt.subplots(figsize=(8, 8))
        sigma = self.fit_results["amiga"]["scatter"]
        ax.axhline(0, color="black", ls="-", lw=2, zorder=1)
        ax.axhspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)

        width = 0.15
        rng = np.random.default_rng(42)

        for i, b in enumerate(binned):
            xc = (b["lo"] + b["hi"]) / 2

            # AMIGA box
            ma = (log_d25_a >= b["lo"]) & (log_d25_a < b["hi"])
            if np.sum(ma) >= 2:
                data_a = res_a[ma]
                bp_a = ax.boxplot(
                    [data_a],
                    positions=[xc - width],
                    widths=width * 1.4,
                    patch_artist=True,
                    showfliers=False,
                    zorder=2,
                )
                bp_a["boxes"][0].set_facecolor("#555555")
                bp_a["boxes"][0].set_alpha(0.25)
                bp_a["boxes"][0].set_edgecolor("black")
                for el in ["whiskers", "caps", "medians"]:
                    for ln in bp_a[el]:
                        ln.set_color("black")
                        ln.set_linewidth(1.5)
                jx = rng.uniform(-width * 0.5, width * 0.5, len(data_a))
                ax.scatter(
                    xc - width + jx,
                    data_a,
                    s=25,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.5,
                    zorder=3,
                )

            # HCG box
            mh = (log_d25_h >= b["lo"]) & (log_d25_h < b["hi"])
            if np.sum(mh) >= 2:
                data_h = res_h[mh]
                bp_h = ax.boxplot(
                    [data_h],
                    positions=[xc + width],
                    widths=width * 1.4,
                    patch_artist=True,
                    showfliers=False,
                    zorder=2,
                )
                bp_h["boxes"][0].set_facecolor("#d62728")
                bp_h["boxes"][0].set_alpha(0.25)
                bp_h["boxes"][0].set_edgecolor("#d62728")
                for el in ["whiskers", "caps", "medians"]:
                    for ln in bp_h[el]:
                        ln.set_color("#d62728")
                        ln.set_linewidth(1.5)
                jx = rng.uniform(-width * 0.5, width * 0.5, len(data_h))
                ax.scatter(
                    xc + width + jx,
                    data_h,
                    s=25,
                    c="#d62728",
                    alpha=0.5,
                    edgecolors="none",
                    zorder=3,
                )

            # p-value annotation
            if "mw_p" in b:
                sig = (
                    "***"
                    if b["mw_p"] < 0.001
                    else "**"
                    if b["mw_p"] < 0.01
                    else "*"
                    if b["mw_p"] < 0.05
                    else "n.s."
                )
                ax.text(
                    xc, ax.get_ylim()[1] * 0.85, sig, ha="center", fontsize=14, fontweight="bold"
                )

        # Dummy handles for legend
        legend_elements = [
            Patch(facecolor="#555555", alpha=0.25, edgecolor="black", label="AMIGA"),
            Patch(facecolor="#d62728", alpha=0.25, edgecolor="#d62728", label="HCG"),
        ]
        ax.legend(handles=legend_elements, fontsize=14, frameon=True, loc="lower left")

        ax.set_xlabel(r"$\log\,(D_{25}$ / kpc)", fontsize=20, labelpad=15)
        ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=20, labelpad=15)
        self._style_axes(ax)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax

    def plot_bootstrap_shift_distribution(
        self, output_file="bootstrap_shift_distribution.pdf", show=True
    ):
        r"""
        Histogram of the bootstrap distribution of the median shift
        Δmedian = median(HCG) − median(AMIGA), with observed value
        and 95 % CI marked.
        """
        focused = self.fit_results.get("focused")
        if focused is None:
            raise RuntimeError("Run run_focused_comparison() first.")

        shift_dist = focused["median_shift_distribution"]
        shift, shift_lo, shift_hi = focused["median_shift"]

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.hist(
            shift_dist,
            bins=60,
            color="#1d5378",
            alpha=0.6,
            edgecolor="#1d5378",
            linewidth=0.8,
            zorder=2,
        )

        ax.axvline(
            shift, color="black", ls="-", lw=2.5, zorder=3, label=f"Observed Δmedian = {shift:+.3f}"
        )
        ax.axvline(shift_lo, color="#d62728", ls="--", lw=1.8, zorder=3)
        ax.axvline(
            shift_hi,
            color="#d62728",
            ls="--",
            lw=1.8,
            zorder=3,
            label=f"95% CI [{shift_lo:+.3f}, {shift_hi:+.3f}]",
        )
        ax.axvline(0, color="gray", ls=":", lw=1.5, zorder=1, label="No shift")

        ax.set_xlabel(
            r"$\Delta_{\rm median}$ = median(HCG) $-$ median(AMIGA) [dex]", fontsize=18, labelpad=15
        )
        ax.set_ylabel("Bootstrap count", fontsize=18, labelpad=15)
        ax.legend(fontsize=13, frameon=True)
        self._style_axes(ax)

        plt.tight_layout()
        output_path = _figure_output_path(output_file)
        plt.savefig(output_path, bbox_inches="tight", dpi=400)
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        return fig, ax


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run correlation + residuals analysis. All printed reports are saved to a file."
        )
    )
    parser.add_argument(
        "--amiga-file",
        default="isolated_galaxies_results.csv",
        help="CSV file for the AMIGA (isolated) sample.",
    )
    parser.add_argument(
        "--hcg-file",
        default="interacting_galaxies_results.csv",
        help="CSV file for the HCG (interacting) sample.",
    )
    parser.add_argument(
        "--report-file",
        default="analysis_report.txt",
        help="Path to save the full text report (captures all prints).",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Do not echo output to the terminal (save only to report file).",
    )
    parser.add_argument(
        "--font-dir",
        action="append",
        default=[d for d in [os.environ.get("GALAXYDISKSIZE_FONT_DIR")] if d],
        help=(
            "Custom font directory (can be given multiple times). "
            "Default matches the original script."
        ),
    )
    args = parser.parse_args()

    with _redirect_output(args.report_file, echo_to_console=(not args.no_console)):
        # Initialize the analysis
        analyzer = CorrelationAnalysis(font_dirs=args.font_dir)

        # ---- Step 1: Load AMIGA + HCG data and run baseline analysis ----
        analyzer.load_data(amiga_file=args.amiga_file, hcg_file=args.hcg_file)
        results = analyzer.run_full_analysis(method="bisector")

        # ---- Step 2: Generate the 4 core plots (AMIGA vs HCG) ----
        print("\n" + "=" * 60)
        print("GENERATING CORE PLOTS")
        print("=" * 60)

        analyzer.plot_correlation(output_file="diameter_correlation.pdf", show=False)
        analyzer.plot_residuals_histogram(output_file="diameter_residuals_hist.pdf", show=False)
        analyzer.plot_residuals_vs_D25(output_file="diameter_residuals_vs_D25.pdf", show=False)
        analyzer.plot_residuals_by_phase(output_file="diameter_residuals_by_phase.pdf", show=False)

        # ---- Step 3: Load CIG stellar masses for AMIGA galaxies ----
        analyzer.load_cig_stellar_masses("../angmom_final.csv")

        # ---- Step 4: Load all other surveys ----
        analyzer.load_wang_table("../wang-surveys-table.txt")
        analyzer.load_broeils_rhee("../broelis-rhee.txt", survey_name="B97")

        analyzer.register_surveys()
        analyzer.compute_survey_residuals()

        # ---- Step 5: Generate the survey-comparison plots ----
        print("\n" + "=" * 60)
        print("GENERATING SURVEY COMPARISON PLOTS")
        print("=" * 60)

        analyzer.plot_survey_mean_residual(
            rank_metric="median",
            output_file="survey_median_residual.pdf",
            show=False,
        )
        analyzer.plot_survey_frac_extended(output_file="survey_frac_extended.pdf", show=False)
        analyzer.plot_survey_frac_truncated(output_file="survey_frac_truncated.pdf", show=False)
        analyzer.plot_correlation_with_all_surveys(
            output_file="diameter_correlation_with_all_surveys.pdf",
            show=False,
        )

        # ---- Step 6: Generate mass-related diagnostic plots ----
        print("\n" + "=" * 60)
        print("GENERATING MASS-RELATED DIAGNOSTIC PLOTS")
        print("=" * 60)

        analyzer.plot_residuals_vs_stellar_mass(
            output_file="residuals_vs_stellar_mass.pdf", show=False
        )
        analyzer.plot_residuals_vs_hi_mass(output_file="residuals_vs_hi_mass.pdf", show=False)
        analyzer.plot_residuals_vs_gas_fraction(
            output_file="residuals_vs_gas_fraction.pdf", show=False
        )

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nKey results to report in your paper:")
        print(f"  - AMIGA baseline: D_HI ∝ D_25^{results['amiga']['slope']:.2f}")
        print(f"  - AMIGA scatter: {results['amiga']['scatter']:.3f} dex")
        print(f"  - HCG offset from AMIGA: {results['comparison']['offset']:.3f} dex")
        ratio = 10 ** results["comparison"]["offset"]
        print(
            "  - This means HCG galaxies have HI disks that are "
            f"{ratio:.1%} the size expected at fixed D_25"
        )

        # ---- Step 7: Focused AMIGA vs HCG comparison ----
        analyzer.run_focused_comparison()

        print("\n" + "=" * 60)
        print("GENERATING FOCUSED COMPARISON PLOTS")
        print("=" * 60)

        analyzer.plot_truncation_index_by_phase(
            output_file="truncation_index_by_phase.pdf", show=False
        )
        analyzer.plot_size_matched_comparison(output_file="size_matched_comparison.pdf", show=False)
        analyzer.plot_binned_residuals_comparison(
            output_file="binned_residuals_comparison.pdf", show=False
        )
        analyzer.plot_bootstrap_shift_distribution(
            output_file="bootstrap_shift_distribution.pdf", show=False
        )

    # Let the context manager restore stdout/stderr.
    print(f"Full text report saved to: {args.report_file}")
