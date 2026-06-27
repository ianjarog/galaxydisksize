#!/usr/bin/env python3
"""
TEST FIGURES for the Figure-6 redesign: residual histogram INCLUDING the HI
beam upper limits.  Throwaway outputs (names start with "test_figure6"); nothing
in the manuscript or published figures is touched.

Upper limits are binned at their limit value (beam Bmaj) -> a conservative view:
the true HI diameters are smaller, so the true HCG distribution is shifted even
further negative.  The Kaplan-Meier HCG median is marked as the rigorous centre.

    test_figure6_stacked_counts.pdf  - AMIGA vs HCG (det by phase + limits hatched), counts
    test_figure6_density.pdf         - AMIGA vs HCG(det+lim) as densities (shapes comparable)

Run: python scripts/plot_residual_histogram.py
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# --- TeX Gyre Heros (same font as the other manuscript figures) ---
import figure_style  # noqa: E402
from survival_analysis import km_left_censored, load  # noqa: E402

figure_style.apply()

FIG = ROOT / "figures"
PROD = ROOT / "products"
AMIGA_RESID_CSV = PROD / "amiga_residuals_per_galaxy_kelley_larger_sample_dictionary.csv"

PHASE_ORDER = ["1", "2", "3c", "3a"]
PHASE_LIGHT = {"1": "#a6cee3", "2": "#8fbc8f", "3c": "#f4b07c", "3a": "#c4a6d6"}


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8, width=1.2, pad=8, labelsize=16)
    ax.tick_params(which="minor", length=4, width=1, pad=8)


def get_data():
    m, b_int, sigma, iso_delta, hcg = load()
    res_amiga = pd.read_csv(AMIGA_RESID_CSV)["resid_Bayesian"].to_numpy(float)
    res_amiga = res_amiga[np.isfinite(res_amiga)]
    hcg = hcg.copy()
    hcg["phase"] = hcg["phase"].astype(str).str.strip()
    km_all = km_left_censored(hcg["delta"].to_numpy(float), hcg["is_limit"].to_numpy(bool))
    return res_amiga, hcg, sigma, km_all


def stacked_counts(res_amiga, hcg, sigma, km_all):
    det = hcg[~hcg["is_limit"]]
    lim = hcg[hcg["is_limit"]]
    allr = np.concatenate([res_amiga, hcg["delta"].to_numpy(float)])
    bins = np.linspace(allr.min() - 0.05, allr.max() + 0.05, 26)
    bw = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(11, 8))

    # HCG detections stacked by phase
    bottom = np.zeros(len(bins) - 1)
    for ph in PHASE_ORDER:
        d = det[det["phase"] == ph]["delta"].to_numpy(float)
        if d.size:
            c, _ = np.histogram(d, bins=bins)
            ax.bar(
                bins[:-1],
                c,
                width=bw,
                bottom=bottom,
                align="edge",
                color=PHASE_LIGHT[ph],
                edgecolor="none",
                zorder=2,
                label=f"HCG Phase {ph} (det.)",
            )
            bottom += c
    # HCG upper limits stacked on top (hatched, single distinct style)
    cl, _ = np.histogram(lim["delta"].to_numpy(float), bins=bins)
    ax.bar(
        bins[:-1],
        cl,
        width=bw,
        bottom=bottom,
        align="edge",
        facecolor="white",
        edgecolor="#d2691e",
        hatch="//",
        linewidth=0.0,
        zorder=3,
        label=f"HCG upper limits (n={len(lim)})",
    )
    # HCG total outline
    ax.hist(
        hcg["delta"].to_numpy(float),
        bins=bins,
        histtype="step",
        lw=2.5,
        edgecolor="#b3590f",
        zorder=4,
    )
    # AMIGA outline (hatched, on top)
    ax.hist(
        res_amiga,
        bins=bins,
        histtype="stepfilled",
        facecolor="white",
        edgecolor="none",
        hatch="\\",
        alpha=0.6,
        zorder=5,
    )
    ax.hist(
        res_amiga,
        bins=bins,
        histtype="step",
        lw=2.5,
        edgecolor="black",
        zorder=6,
        label=f"AMIGA (n={len(res_amiga)})",
    )

    ax.axvline(
        np.median(res_amiga),
        color="black",
        ls="--",
        lw=2,
        zorder=7,
        label=f"AMIGA median ({np.median(res_amiga):+.2f})",
    )
    ax.axvline(
        np.median(det["delta"]),
        color="#1f6fb4",
        ls="--",
        lw=2,
        zorder=7,
        label=f"HCG median, det. only ({np.median(det['delta']):+.2f})",
    )
    ax.axvline(
        km_all["median"],
        color="red",
        ls="-",
        lw=2.5,
        zorder=7,
        label=f"HCG KM median ({km_all['median']:+.2f})",
    )
    ax.axvline(0, color="gray", ls=":", lw=1.5, zorder=1)

    ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_ylabel("Number of galaxies", fontsize=22, labelpad=12)
    _ticks(ax)
    ax.legend(loc="upper left", fontsize=13, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure6_stacked_counts.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure6_stacked_counts.pdf")


def density(res_amiga, hcg, sigma, km_all):
    det = hcg[~hcg["is_limit"]]
    lim = hcg[hcg["is_limit"]]
    allr = np.concatenate([res_amiga, hcg["delta"].to_numpy(float)])
    bins = np.linspace(allr.min() - 0.05, allr.max() + 0.05, 26)
    bw = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(11, 8))

    # HCG (det+lim) as density: detections solid, limits hatched on top
    nhcg = len(hcg)
    cd, _ = np.histogram(det["delta"].to_numpy(float), bins=bins)
    cl, _ = np.histogram(lim["delta"].to_numpy(float), bins=bins)
    ax.bar(
        bins[:-1],
        cd / (nhcg * bw),
        width=bw,
        align="edge",
        color="#7fb3d5",
        edgecolor="none",
        alpha=0.85,
        zorder=2,
        label="HCG detections",
    )
    ax.bar(
        bins[:-1],
        cl / (nhcg * bw),
        width=bw,
        bottom=cd / (nhcg * bw),
        align="edge",
        facecolor="white",
        edgecolor="#d2691e",
        hatch="//",
        zorder=3,
        label=f"HCG upper limits (n={len(lim)})",
    )
    ax.hist(
        res_amiga,
        bins=bins,
        density=True,
        histtype="step",
        lw=2.8,
        edgecolor="black",
        zorder=5,
        label=f"AMIGA (n={len(res_amiga)})",
    )

    ax.axvline(
        np.median(res_amiga),
        color="black",
        ls="--",
        lw=2,
        zorder=6,
        label=f"AMIGA median ({np.median(res_amiga):+.2f})",
    )
    ax.axvline(
        km_all["median"],
        color="red",
        ls="-",
        lw=2.5,
        zorder=6,
        label=f"HCG KM median ({km_all['median']:+.2f})",
    )
    ax.axvline(0, color="gray", ls=":", lw=1.5, zorder=1)

    ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_ylabel("Normalised fraction (density)", fontsize=22, labelpad=12)
    _ticks(ax)
    ax.legend(loc="upper left", fontsize=14, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure6_density.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure6_density.pdf")


def combined(res_amiga, hcg, sigma, km_all, normed, outname):
    """Detections AND upper limits MERGED into one HCG histogram (all 124),
    compared to AMIGA. Limits binned at their beam value (conservative).
    """
    rh = hcg["delta"].to_numpy(float)
    allr = np.concatenate([res_amiga, rh])
    bins = np.linspace(allr.min() - 0.05, allr.max() + 0.05, 26)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.hist(
        res_amiga,
        bins=bins,
        density=normed,
        histtype="stepfilled",
        facecolor="white",
        edgecolor="none",
        hatch="\\",
        alpha=0.6,
        zorder=2,
    )
    ax.hist(
        res_amiga,
        bins=bins,
        density=normed,
        histtype="step",
        lw=2.8,
        edgecolor="black",
        zorder=4,
        label=f"AMIGA (n={len(res_amiga)})",
    )
    ax.hist(
        rh,
        bins=bins,
        density=normed,
        histtype="stepfilled",
        facecolor="#7fb3d5",
        edgecolor="none",
        alpha=0.7,
        zorder=3,
    )
    ax.hist(
        rh,
        bins=bins,
        density=normed,
        histtype="step",
        lw=2.8,
        edgecolor="#1f6fb4",
        zorder=5,
        label=f"HCG, det.+upper limits (n={len(rh)})",
    )
    ax.axvline(
        np.median(res_amiga),
        color="black",
        ls="--",
        lw=2,
        zorder=6,
        label=f"AMIGA median ({np.median(res_amiga):+.2f})",
    )
    ax.axvline(
        km_all["median"],
        color="red",
        ls="-",
        lw=2.5,
        zorder=6,
        label=f"HCG KM median ({km_all['median']:+.2f})",
    )
    ax.axvline(0, color="gray", ls=":", lw=1.5, zorder=1)
    ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_ylabel(
        "Normalised fraction (density)" if normed else "Number of galaxies",
        fontsize=22,
        labelpad=12,
    )
    _ticks(ax)
    ax.legend(loc="upper left", fontsize=15, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / outname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] figures/{outname}")


def phases_density(res_amiga, hcg, km_all, outname):
    """Published-style Fig 6: probability density, HCG color-coded by phase with
    a dot overlay, AMIGA hatched -- but the HCG sample now INCLUDES the 70 beam
    upper limits (so phases extend into the truncated tail). Limits binned at
    their beam value (conservative). Legend drawn opaque and on top so the
    vertical mean/KM lines do not cut through it.
    """
    from matplotlib.lines import Line2D

    rh = hcg["delta"].to_numpy(float)
    ph_arr = hcg["phase"].to_numpy()
    allr = np.concatenate([res_amiga, rh])
    bins = np.linspace(allr.min() - 0.05, allr.max() + 0.05, 30)
    bw = bins[1] - bins[0]
    nhcg = len(rh)
    order = ["1", "2", "3a", "3c"]
    fig, ax = plt.subplots(figsize=(11, 8))

    # HCG stacked by phase, as probability density (sum integrates to 1)
    bottom = np.zeros(len(bins) - 1)
    for ph in order:
        c, _ = np.histogram(rh[ph_arr == ph], bins=bins)
        dens = c / (nhcg * bw)
        ax.bar(
            bins[:-1],
            dens,
            width=bw,
            bottom=bottom,
            align="edge",
            color=PHASE_LIGHT[ph],
            edgecolor="none",
            zorder=2,
        )
        bottom += dens
    # HCG dot overlay + blue outline
    ax.hist(
        rh,
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor="none",
        edgecolor="0.45",
        hatch="...",
        linewidth=0.0,
        zorder=3,
    )
    ax.hist(rh, bins=bins, density=True, histtype="step", lw=2.3, edgecolor="#3b6fb0", zorder=4)
    # AMIGA hatched + black outline
    ax.hist(
        res_amiga,
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor="none",
        edgecolor="0.4",
        hatch="//",
        linewidth=0.0,
        zorder=5,
    )
    ax.hist(
        res_amiga, bins=bins, density=True, histtype="step", lw=2.3, edgecolor="black", zorder=6
    )

    amiga_mean = float(np.mean(res_amiga))
    km_med = km_all["median"]
    if np.isfinite(km_med):
        km_label = f"HCG KM median: {km_med:+.2f}"
    else:
        # >50% non-detections -> median lies in the censored tail (unconstrained);
        # report the rigorous bound = most-truncated detection, as in
        # survival_analysis and table:survey_residuals.
        km_med = float(np.min(rh[~hcg["is_limit"].to_numpy(bool)]))
        km_label = rf"HCG KM median: $\leq{km_med:+.2f}$"
    ax.axvline(amiga_mean, color="black", ls="--", lw=2, zorder=7)
    ax.axvline(km_med, color="red", ls="--", lw=2, zorder=7)
    ax.axvline(0, color="gray", ls=":", lw=1.2, zorder=1)

    ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_ylabel(r"Probability density [dex$^{-1}$]", fontsize=22, labelpad=12)
    _ticks(ax)

    main_handles = [
        Patch(
            facecolor="none", edgecolor="#3b6fb0", hatch="...", label=f"Galaxies in HCGs (N={nhcg})"
        ),
        Patch(
            facecolor="none",
            edgecolor="black",
            hatch="//",
            label=f"AMIGA galaxies (N={len(res_amiga)})",
        ),
        Line2D([0], [0], color="black", ls="--", lw=2, label=f"AMIGA mean: {amiga_mean:+.2f}"),
        Line2D([0], [0], color="red", ls="--", lw=2, label=km_label),
    ]
    leg1 = ax.legend(
        handles=main_handles, loc="upper left", fontsize=14, frameon=True, framealpha=1.0
    )
    leg1.set_zorder(30)
    ax.add_artist(leg1)
    phase_handles = [
        Patch(facecolor=PHASE_LIGHT[p], edgecolor="none", label=f"Phase {p}") for p in order
    ]
    leg2 = ax.legend(
        handles=phase_handles,
        loc="center right",
        fontsize=13,
        frameon=True,
        framealpha=1.0,
        title="HCG phases",
        title_fontsize=13,
    )
    leg2.set_zorder(30)
    fig.tight_layout()
    fig.savefig(FIG / outname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] figures/{outname}")


def main(promote=False):
    res_amiga, hcg, sigma, km_all = get_data()
    print(
        f"AMIGA n={len(res_amiga)}  HCG det={int((~hcg['is_limit']).sum())} "
        f"lim={int(hcg['is_limit'].sum())}  HCG KM median={km_all['median']:+.3f}"
    )
    stacked_counts(res_amiga, hcg, sigma, km_all)
    density(res_amiga, hcg, sigma, km_all)
    combined(
        res_amiga, hcg, sigma, km_all, normed=True, outname="test_figure6_combined_density.pdf"
    )
    combined(
        res_amiga, hcg, sigma, km_all, normed=False, outname="test_figure6_combined_counts.pdf"
    )
    phases_density(res_amiga, hcg, km_all, "test_figure6_phases_density.pdf")
    print("Done. test_figure6_*.pdf written to figures/.")
    if promote:
        prod = FIG / "diameter_residuals_hist_kelley_larger_sample.pdf"
        bak = FIG / "diameter_residuals_hist_kelley_larger_sample_prevhist_backup.pdf"
        if prod.exists() and not bak.exists():
            shutil.copy2(prod, bak)
            print(f"[backup] {prod.name} -> {bak.name}")
        shutil.copy2(FIG / "test_figure6_phases_density.pdf", prod)
        print(f"[PROMOTED] production Figure 6 -> {prod.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--promote",
        action="store_true",
        help="write the phases_density design to the production Fig 6 file",
    )
    main(promote=ap.parse_args().promote)
