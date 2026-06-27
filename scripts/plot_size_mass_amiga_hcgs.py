#!/usr/bin/env python3
"""
Standalone visualization of the HI mass-size relation using only the resolved
AMIGA sample and HCGs.

This script is meant as a visualization/audit product and fits the relation
with a Bayesian linear model in log-log space. AMIGA/HCG upper limits are
included as censored points when available.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_FIGURES_DIR = PROJECT_ROOT / "figures"
ANALYSIS_PRODUCTS_DIR = PROJECT_ROOT / "products"
ANALYSIS_LATEX_DIR = PROJECT_ROOT / "latex"

MPLCONFIGDIR = ANALYSIS_PRODUCTS_DIR / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


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
    return pd.DataFrame(
        {
            "source_group": "AMIGA",
            "source_subgroup": "resolved",
            "log_mhi": np.log10(pd.to_numeric(subset["hi_mass"], errors="coerce").to_numpy(float)),
            "log_dhi": np.log10(
                pd.to_numeric(subset["hi_diameter_kpc"], errors="coerce").to_numpy(float)
            ),
            "yerr_log_dhi": yerr,
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
    return pd.DataFrame(
        {
            "source_group": "HCG",
            "source_subgroup": subset["phase"].astype(str).str.strip().to_numpy(),
            "log_mhi": np.log10(pd.to_numeric(subset["hi_mass"], errors="coerce").to_numpy(float)),
            "log_dhi": np.log10(
                pd.to_numeric(subset["hi_diameter_kpc"], errors="coerce").to_numpy(float)
            ),
            "yerr_log_dhi": yerr,
            "is_upper_limit": subset_err.isna().to_numpy(),
        }
    )


def bayesian_fit(x_det, y_det, yerr_det, x_ul, y_ul, yerr_ul):
    def log_prior(theta):
        m, b, lnf = theta
        if -5 < m < 5 and -10 < b < 10 and -10 < lnf < 1:
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        m, b, lnf = theta
        sig2_int = np.exp(2.0 * lnf)

        mu_det = m * x_det + b
        var_det = np.clip(sig2_int + yerr_det**2, 1e-20, None)
        ll_det = -0.5 * np.sum((y_det - mu_det) ** 2 / var_det + np.log(2 * np.pi * var_det))

        if x_ul.size:
            mu_ul = m * x_ul + b
            std_ul = np.clip(np.sqrt(sig2_int + yerr_ul**2), 1e-10, None)
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

    ndim, nwalkers = 3, 100
    m0, b0 = np.polyfit(x_det, y_det, 1)
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = m0 + 1e-4 * np.random.randn(nwalkers)
    pos[:, 1] = b0 + 1e-4 * np.random.randn(nwalkers)
    pos[:, 2] = -1.0 + 1e-4 * np.random.randn(nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(pos, 4000, progress=True)
    samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    return samples


def compute_bayesian_fit(df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    det = df[~df["is_upper_limit"]].copy()
    ul = df[df["is_upper_limit"]].copy()

    x_det = det["log_mhi"].to_numpy(float)
    y_det = det["log_dhi"].to_numpy(float)
    yerr_det = det["yerr_log_dhi"].to_numpy(float)
    x_ul = ul["log_mhi"].to_numpy(float)
    y_ul = ul["log_dhi"].to_numpy(float)
    yerr_ul = ul["yerr_log_dhi"].to_numpy(float)

    samples = bayesian_fit(x_det, y_det, yerr_det, x_ul, y_ul, yerr_ul)
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
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
        "n": int(len(df)),
        "n_det": int(len(det)),
        "n_ul": int(len(ul)),
        "slope_p16": float(percentiles[0, 0]),
        "slope_p84": float(percentiles[2, 0]),
        "intercept_p16": float(percentiles[0, 1]),
        "intercept_p84": float(percentiles[2, 1]),
        "scatter_p16": float(np.exp(percentiles[0, 2])),
        "scatter_p84": float(np.exp(percentiles[2, 2])),
        "posterior_samples": samples,
        "residual_scatter_det": float(np.std(det_residuals, ddof=2)),
    }


def write_summary(summary_path: Path, fit: dict, combined_df: pd.DataFrame) -> None:
    counts = combined_df.groupby("source_group").size().to_dict()
    payload = {
        "fit_method": fit["method"],
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "scatter": fit["scatter"],
        "slope_p16": fit["slope_p16"],
        "slope_p84": fit["slope_p84"],
        "intercept_p16": fit["intercept_p16"],
        "intercept_p84": fit["intercept_p84"],
        "scatter_p16": fit["scatter_p16"],
        "scatter_p84": fit["scatter_p84"],
        "r_value": fit["r_value"],
        "p_value": fit["p_value"],
        "n_total": fit["n"],
        "n_detections": fit["n_det"],
        "n_upper_limits": fit["n_ul"],
        "n_amiga": int(counts.get("AMIGA", 0)),
        "n_hcg": int(counts.get("HCG", 0)),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_corner_plot(samples: np.ndarray, output_path: Path) -> None:
    labels = ["Slope", "Zero point", "Scatter"]
    samples_plot = np.copy(samples)
    samples_plot[:, 2] = np.exp(samples_plot[:, 2])
    truths = np.percentile(samples_plot, 50, axis=0)

    fig_corner = corner.corner(
        samples_plot,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        color="k",
        plot_datapoints=True,
        labelpad=0.12,
        plot_density=True,
        fill_contours=False,
        truths=truths,
        truth_color="red",
        label_kwargs={"fontsize": 14, "family": "tex gyre heros"},
        title_kwargs={"fontsize": 14, "family": "tex gyre heros"},
    )
    for axc in fig_corner.get_axes():
        for label in axc.get_xticklabels() + axc.get_yticklabels():
            label.set_fontsize(14)

    ndim = 3
    axes_c = np.array(fig_corner.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes_c[i, i].axvline(truths[i], color="red", lw=2)
    for yi in range(ndim):
        for xi in range(yi):
            axes_c[yi, xi].plot(truths[xi], truths[yi], "sr", markersize=8)

    fig_corner.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close(fig_corner)


def plot_mass_size(combined_df: pd.DataFrame, fit: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ul_yerr = 0.10

    hcg_styles = {
        "1": ("o", "blue", "Phase 1"),
        "2": ("o", "green", "Phase 2"),
        "3a": ("o", "red", "Phase 3a"),
        "3c": ("o", "purple", "Phase 3c"),
    }

    amiga_df = combined_df[
        (combined_df["source_group"] == "AMIGA") & (~combined_df["is_upper_limit"])
    ]
    amiga_handle = ax.scatter(
        amiga_df["log_mhi"],
        amiga_df["log_dhi"],
        marker="*",
        s=70,
        c="#fe9a01",
        label=f"AMIGA ({len(amiga_df)})",
        zorder=4,
    )
    amiga_ul = combined_df[
        (combined_df["source_group"] == "AMIGA") & (combined_df["is_upper_limit"])
    ]
    if len(amiga_ul):
        ax.errorbar(
            amiga_ul["log_mhi"],
            amiga_ul["log_dhi"],
            yerr=ul_yerr,
            uplims=True,
            fmt="none",
            ecolor="#fe9a01",
            elinewidth=1.2,
            zorder=4,
        )

    hcg_handles = {}
    hcg_df = combined_df[combined_df["source_group"] == "HCG"]
    for phase in ["1", "2", "3a", "3c"]:
        subset_det = hcg_df[(hcg_df["source_subgroup"] == phase) & (~hcg_df["is_upper_limit"])]
        subset_ul = hcg_df[(hcg_df["source_subgroup"] == phase) & (hcg_df["is_upper_limit"])]
        if len(subset_det) == 0 and len(subset_ul) == 0:
            continue
        marker, color, label = hcg_styles[phase]
        if len(subset_det):
            handle = ax.scatter(
                subset_det["log_mhi"],
                subset_det["log_dhi"],
                marker=marker,
                s=35,
                c=color,
                label=f"{label} ({len(subset_det) + len(subset_ul)})",
                zorder=5,
            )
        else:
            handle = ax.scatter(
                [], [], marker=marker, s=35, c=color, label=f"{label} ({len(subset_ul)})"
            )
        if len(subset_ul):
            ax.errorbar(
                subset_ul["log_mhi"],
                subset_ul["log_dhi"],
                yerr=ul_yerr,
                uplims=True,
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                zorder=5,
            )
        hcg_handles[label] = handle

    x_lo = float(np.floor(combined_df["log_mhi"].min() * 10.0) / 10.0) - 0.1
    x_hi = float(np.ceil(combined_df["log_mhi"].max() * 10.0) / 10.0) + 0.1
    xfit = np.linspace(x_lo, x_hi, 300)
    yfit = fit["intercept"] + fit["slope"] * xfit

    ax.plot(
        xfit,
        yfit,
        color="black",
        linewidth=2.4,
        zorder=4,
    )
    ax.fill_between(
        xfit,
        yfit - 3.0 * fit["scatter"],
        yfit + 3.0 * fit["scatter"],
        color="0.7",
        alpha=0.2,
        zorder=1,
    )

    ax.set_xlabel(r"$\log\,(M_{\rm HI} / M_\odot)$", fontsize=22, labelpad=15)
    ax.set_ylabel(r"$\log\,(D_{\rm HI} / {\rm kpc})$", fontsize=22, labelpad=15)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(
        float(np.floor(combined_df["log_dhi"].min() * 10.0) / 10.0) - 0.1,
        float(np.ceil(combined_df["log_dhi"].max() * 10.0) / 10.0) + 0.1,
    )

    (
        "Combined detection fit:\n"
        f"$N = {fit['n']}$\n"
        f"$r = {fit['r_value']:.2f}$\n"
        f"$\\sigma = {fit['scatter']:.2f}$ dex"
    )
    dict(boxstyle="round", facecolor="white", alpha=0.85)
    # ax.text(
    #    0.97,
    #    0.05,
    #    textstr,
    #    transform=ax.transAxes,
    #    fontsize=14,
    #    horizontalalignment="right",
    #    verticalalignment="bottom",
    #    bbox=props,
    # )

    style_axes(ax)
    fit_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.4,
        label=f"Bayesian fit: $\\log D_{{\\rm HI}} = {fit['slope']:.2f}\\,\\log M_{{\\rm HI}} {fit['intercept']:+.2f}$",
    )
    scatter_handle = plt.Line2D(
        [0],
        [0],
        color="0.5",
        linewidth=6,
        alpha=0.35,
        label=f"±3$\\sigma$ ({3.0 * fit['scatter']:.2f} dex)",
    )

    lgnd_fit = ax.legend(
        handles=[fit_handle, scatter_handle],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        prop={"family": "serif", "size": 12},
        frameon=False,
    )
    ax.add_artist(lgnd_fit)

    lgnd_amiga = ax.legend(
        handles=[amiga_handle],
        labels=[f"AMIGA ({len(amiga_df)})"],
        loc="lower left",
        bbox_to_anchor=(0.66, 0.29),
        prop={"family": "serif", "size": 13},
        frameon=False,
        markerfirst=False,
    )
    ax.text(
        0.685,
        0.38,
        "This work",
        transform=ax.transAxes,
        fontfamily="serif",
        fontsize=12,
        fontweight="bold",
    )
    ax.add_artist(lgnd_amiga)

    lgnd_hcg = ax.legend(
        handles=list(hcg_handles.values()),
        labels=[
            f"{label} ({len(hcg_df[hcg_df['source_subgroup'] == phase])})"
            for phase, (_, _, label) in hcg_styles.items()
            if label in hcg_handles
        ],
        loc="lower left",
        bbox_to_anchor=(0.66, 0.1),
        prop={"family": "serif", "size": 13},
        frameon=False,
        markerfirst=False,
    )
    ax.add_artist(lgnd_hcg)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the HI mass-size relation using only resolved AMIGA and HCG galaxies."
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
        "--output-file",
        default="mass_size_relation_amiga_hcgs.pdf",
        help="Output figure filename written to the figures directory.",
    )
    parser.add_argument(
        "--corner-file",
        default="mass_size_relation_amiga_hcgs_corner.pdf",
        help="Corner-plot filename written to the figures directory.",
    )
    parser.add_argument(
        "--summary-file",
        default=str(ANALYSIS_PRODUCTS_DIR / "mass_size_relation_amiga_hcgs_summary.json"),
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--report-file",
        default=str(ANALYSIS_PRODUCTS_DIR / "mass_size_relation_amiga_hcgs_report.txt"),
        help="Full text report path.",
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


def main() -> None:
    ensure_directories()
    args = parse_args()
    configure_fonts(args.font_dir)

    with _redirect_output(args.report_file, echo_to_console=(not args.no_console)):
        amiga_df = load_amiga_data(Path(args.amiga_file))
        hcg_df = load_hcg_data(Path(args.hcg_file))

        combined_df = pd.concat([amiga_df, hcg_df], ignore_index=True)
        fit = compute_bayesian_fit(combined_df)

        output_path = ANALYSIS_FIGURES_DIR / args.output_file
        corner_path = ANALYSIS_FIGURES_DIR / args.corner_file
        write_summary(Path(args.summary_file), fit, combined_df)
        plot_mass_size(combined_df, fit, output_path)
        save_corner_plot(fit["posterior_samples"], corner_path)

        print("=" * 60)
        print("AMIGA + HCG MASS-SIZE RELATION")
        print("=" * 60)
        for group in ["AMIGA", "HCG", "MIGHTEE", "Wang+16"]:
            if group in set(combined_df["source_group"]):
                print(f"{group}: {int(np.sum(combined_df['source_group'] == group))}")
        print(f"Combined sample: {len(combined_df)}")
        print(f"Detections used in likelihood: {fit['n_det']}")
        print(f"Upper limits used in likelihood: {fit['n_ul']}")
        print(f"Fit: log(D_HI) = {fit['slope']:.4f} * log(M_HI) + {fit['intercept']:.4f}")
        print(
            "Posterior slope 16/50/84: "
            f"{fit['slope_p16']:.4f}, {fit['slope']:.4f}, {fit['slope_p84']:.4f}"
        )
        print(
            "Posterior intercept 16/50/84: "
            f"{fit['intercept_p16']:.4f}, {fit['intercept']:.4f}, {fit['intercept_p84']:.4f}"
        )
        print(
            "Posterior scatter 16/50/84: "
            f"{fit['scatter_p16']:.4f}, {fit['scatter']:.4f}, {fit['scatter_p84']:.4f} dex"
        )
        print(f"Detection-only Pearson r: {fit['r_value']:.4f}")
        print(f"Figure saved to: {output_path}")
        print(f"Corner plot saved to: {corner_path}")
        print(f"Summary JSON saved to: {args.summary_file}")

    print(f"Full text report saved to: {args.report_file}")


if __name__ == "__main__":
    main()
