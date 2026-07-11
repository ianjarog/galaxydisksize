#!/usr/bin/env python3
"""Figure 5 (both panels): D_HI-D_25 correlation and residuals with upper limits.

Reproduces the published figure styles from the persisted pipeline products
(no MCMC re-fit, so the AMIGA baseline is untouched) and overlays the 70 HI
non-detected HCG members as beam-size upper limits (downward arrows).

Inputs:
    products/hcg_residual_statistics.json          (baseline)
    products/amiga_combined_larger_sample.csv      (AMIGA cloud)
    products/hcg_residuals_per_galaxy.csv          (HCG detections)
    data/upperlimits_bmaj_provenance.csv           (70 upper limits)

Outputs:
    figures/diameter_correlation.pdf               (Fig 5 left)
    figures/diameter_residuals_vs_D25.pdf          (Fig 5 right)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figure_style  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

figure_style.apply()

ROOT = Path(__file__).resolve().parent.parent
PROD = ROOT / "products"
DATA = ROOT / "data"
FIG = ROOT / "figures"

PHASE_COLORS = {"1": "#1f77b4", "2": "#2ca02c", "3": "#d62728", "3a": "#9467bd", "3c": "#ff7f0e"}


def load():
    b = json.load(open(PROD / "hcg_residual_statistics.json"))
    base = dict(
        slope=float(b["baseline_slope"]),
        intercept=float(b["baseline_intercept"]),
        sigma=float(b["baseline_sigma"]),
    )
    amiga = pd.read_csv(PROD / "amiga_combined_larger_sample.csv")
    amiga = amiga[(amiga["optical_diameter_kpc"] > 0) & (amiga["hi_diameter_kpc"] > 0)].copy()
    amiga["resid"] = np.log10(amiga["hi_diameter_kpc"]) - (
        base["intercept"] + base["slope"] * np.log10(amiga["optical_diameter_kpc"])
    )
    base["r"] = stats.pearsonr(
        np.log10(amiga["optical_diameter_kpc"]), np.log10(amiga["hi_diameter_kpc"])
    )[0]
    hcg = pd.read_csv(PROD / "hcg_residuals_per_galaxy.csv")
    hcg["phase"] = hcg["phase"].astype(str).str.strip()
    lim = pd.read_csv(DATA / "upperlimits_bmaj_provenance.csv")
    lim = lim[lim["optical_diameter_kpc"] > 0].copy()
    # Beam-limited members (no diameter error -> unresolved, D_HI<beam) are moved
    # from the detection set to the Bmaj upper-limit set, consistent with the
    # mass-size fit and the censored KM analysis. Their Bmaj limit values come
    # from the augmented CSV (single source of truth).
    aug = pd.read_csv(DATA / "interacting_galaxies_results_with_upperlimits_bmaj.csv")
    beamlim = aug[(aug["is_upper_limit"] == 1) & (aug["galaxy"].isin(hcg["galaxy"]))][
        ["galaxy", "group", "phase", "hi_diameter_kpc", "optical_diameter_kpc"]
    ].copy()
    hcg = hcg[~hcg["galaxy"].isin(beamlim["galaxy"])].copy()
    lim = pd.concat([lim, beamlim], ignore_index=True)
    lim = lim[lim["optical_diameter_kpc"] > 0].copy()
    lim["phase"] = lim["phase"].astype(str).str.strip()
    lim["resid"] = np.log10(lim["hi_diameter_kpc"]) - (
        base["intercept"] + base["slope"] * np.log10(lim["optical_diameter_kpc"])
    )
    return base, amiga, hcg, lim


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=17)
    ax.tick_params(which="major", length=8, width=1.2, pad=10)
    ax.tick_params(which="minor", length=4, width=1, pad=10)


def fig5_left(base, amiga, hcg, lim):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        amiga["optical_diameter_kpc"],
        amiga["hi_diameter_kpc"],
        s=80,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="AMIGA (isolated)",
        zorder=3,
    )
    for ph in ["1", "2", "3", "3a", "3c"]:
        m = hcg["phase"] == ph
        if m.any():
            ax.scatter(
                hcg.loc[m, "D_25_kpc"],
                hcg.loc[m, "D_HI_kpc"],
                s=100,
                c=PHASE_COLORS.get(ph, "gray"),
                marker="s",
                alpha=0.7,
                edgecolors="black",
                linewidths=0.4,
                label=f"HCG Phase {ph}",
                zorder=4,
            )
    # upper-limit arrows (beam Bmaj)
    ld = False
    for ph in ["1", "2", "3", "3a", "3c"]:
        m = lim["phase"] == ph
        if m.any():
            c = PHASE_COLORS.get(ph, "gray")
            ax.scatter(
                lim.loc[m, "optical_diameter_kpc"],
                lim.loc[m, "hi_diameter_kpc"],
                marker="v",
                s=70,
                facecolors="none",
                edgecolors=c,
                linewidths=1.4,
                zorder=5,
                label=None if ld else "HCG upper limits",
            )
            ld = True
            for xx, yy in zip(lim.loc[m, "optical_diameter_kpc"], lim.loc[m, "hi_diameter_kpc"]):
                ax.annotate(
                    "",
                    xy=(xx, yy * 0.78),
                    xytext=(xx, yy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1),
                    zorder=5,
                )
    x = np.logspace(np.log10(4), np.log10(100), 300)
    lx = np.log10(x)
    y = 10 ** (base["intercept"] + base["slope"] * lx)
    s = base["sigma"]
    ax.fill_between(
        x,
        10 ** (base["intercept"] + base["slope"] * lx - s),
        10 ** (base["intercept"] + base["slope"] * lx + s),
        color="gray",
        alpha=0.2,
        zorder=1,
        label=f"±1σ ({s:.2f} dex)",
    )
    ax.plot(
        x, 10 ** (base["intercept"] + base["slope"] * lx + 3 * s), "k--", lw=1, alpha=0.5, zorder=1
    )
    ax.plot(
        x, 10 ** (base["intercept"] + base["slope"] * lx - 3 * s), "k--", lw=1, alpha=0.5, zorder=1
    )
    ax.plot(
        x,
        y,
        "k-",
        lw=2.5,
        zorder=2,
        label=rf"AMIGA fit: $D_{{\rm HI}}\propto D_{{25}}^{{{base['slope']:.2f}}}$",
    )
    ax.plot(x, x, "k:", lw=1, alpha=0.5, label="1:1", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D_{25}$ [kpc]", fontsize=22, labelpad=15)
    ax.set_ylabel(r"$D_{\rm HI}$ [kpc]", fontsize=22, labelpad=15)
    ax.set_xlim(4, 100)
    ax.set_ylim(1, 1000)
    _ticks(ax)
    ax.text(
        0.30,
        0.85,
        f"AMIGA baseline:\n$r = {base['r']:.2f}$\n$\\sigma = {s:.2f}$ dex",
        transform=ax.transAxes,
        fontsize=15,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.9, markerfirst=False)
    fig.tight_layout()
    fig.savefig(FIG / "diameter_correlation.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] Fig 5 left (diameter_correlation)")


def fig5_right(base, amiga, hcg, lim):
    fig, ax = plt.subplots(figsize=(10, 8))
    s = base["sigma"]
    ax.scatter(
        amiga["optical_diameter_kpc"],
        amiga["resid"],
        s=80,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="AMIGA",
        zorder=3,
    )
    for ph in ["1", "2", "3", "3a", "3c"]:
        m = hcg["phase"] == ph
        if m.any():
            ax.scatter(
                hcg.loc[m, "D_25_kpc"],
                hcg.loc[m, "residual_dex"],
                s=100,
                c=PHASE_COLORS.get(ph, "gray"),
                marker="s",
                alpha=0.7,
                edgecolors="black",
                linewidths=0.4,
                label=f"HCG Phase {ph}",
                zorder=4,
            )
    ld = False
    for ph in ["1", "2", "3", "3a", "3c"]:
        m = lim["phase"] == ph
        if m.any():
            c = PHASE_COLORS.get(ph, "gray")
            ax.scatter(
                lim.loc[m, "optical_diameter_kpc"],
                lim.loc[m, "resid"],
                marker="v",
                s=70,
                facecolors="none",
                edgecolors=c,
                linewidths=1.4,
                zorder=5,
                label=None if ld else "HCG upper limits",
            )
            ld = True
            for xx, yy in zip(lim.loc[m, "optical_diameter_kpc"], lim.loc[m, "resid"]):
                ax.annotate(
                    "",
                    xy=(xx, yy - 0.12),
                    xytext=(xx, yy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1),
                    zorder=5,
                )
    ax.axhline(0, color="black", lw=2)
    ax.axhline(s, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(-s, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.fill_between([0, 100], -s, s, color="gray", alpha=0.1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_{25}$ [kpc]", fontsize=22, labelpad=15)
    ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
    ax.set_xlim(5, 100)
    _ticks(ax)
    ax.legend(loc="lower left", fontsize=12, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "diameter_residuals_vs_D25.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] Fig 5 right (residuals_vs_D25)")


def main():
    base, amiga, hcg, lim = load()
    print(f"baseline slope={base['slope']:.4f} r={base['r']:.3f} sigma={base['sigma']:.4f}")
    print(f"AMIGA={len(amiga)} HCG det={len(hcg)} limits={len(lim)}")
    fig5_left(base, amiga, hcg, lim)
    fig5_right(base, amiga, hcg, lim)
    print("Done. Baseline untouched.")


if __name__ == "__main__":
    main()
