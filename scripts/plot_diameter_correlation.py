#!/usr/bin/env python3
"""
Standalone upper-limit overlays for Figures 5 (both panels) and 6.
==================================================================
Reproduces the published figure styles EXACTLY from the persisted pipeline
products (no MCMC re-fit, so the AMIGA baseline is untouched) and overlays the
70 HI non-detected HCG members as beam-size upper limits (downward arrows).

Inputs (all already on disk):
    products/hcg_residual_statistics_kelley_larger_sample.json   (baseline)
    products/amiga_combined_larger_sample_kelley_larger_sample.csv (AMIGA cloud)
    products/hcg_residuals_per_galaxy_kelley_larger_sample.csv     (HCG detections)
    data/upperlimits_bmaj_provenance.csv                          (70 upper limits)

Outputs (originals backed up to *_detonly_backup.pdf once):
    figures/diameter_correlation_kelley_larger_sample.pdf          (Fig 5 left)
    figures/diameter_residuals_vs_D25_kelley_larger_sample.pdf     (Fig 5 right)
    figures/diameter_residuals_hist_kelley_larger_sample.pdf       (Fig 6)
"""

import json
import shutil
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
PHASE_LIGHT = {"1": "#a6cee3", "2": "#8fbc8f", "3a": "#c4a6d6", "3c": "#f4b07c"}


def _backup(name):
    out = FIG / name
    bak = FIG / name.replace(".pdf", "_detonly_backup.pdf")
    if out.exists() and not bak.exists():
        shutil.copy2(out, bak)
        print(f"[backup] {name} -> {bak.name}")


def load():
    b = json.load(open(PROD / "hcg_residual_statistics_kelley_larger_sample.json"))
    base = dict(
        slope=float(b["baseline_slope"]),
        intercept=float(b["baseline_intercept"]),
        sigma=float(b["baseline_sigma"]),
    )
    amiga = pd.read_csv(PROD / "amiga_combined_larger_sample_kelley_larger_sample.csv")
    amiga = amiga[(amiga["optical_diameter_kpc"] > 0) & (amiga["hi_diameter_kpc"] > 0)].copy()
    amiga["resid"] = np.log10(amiga["hi_diameter_kpc"]) - (
        base["intercept"] + base["slope"] * np.log10(amiga["optical_diameter_kpc"])
    )
    base["r"] = stats.pearsonr(
        np.log10(amiga["optical_diameter_kpc"]), np.log10(amiga["hi_diameter_kpc"])
    )[0]
    hcg = pd.read_csv(PROD / "hcg_residuals_per_galaxy_kelley_larger_sample.csv")
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
    _backup("diameter_correlation_kelley_larger_sample.pdf")
    fig.savefig(FIG / "diameter_correlation_kelley_larger_sample.pdf", bbox_inches="tight", dpi=200)
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
    _backup("diameter_residuals_vs_D25_kelley_larger_sample.pdf")
    fig.savefig(
        FIG / "diameter_residuals_vs_D25_kelley_larger_sample.pdf", bbox_inches="tight", dpi=200
    )
    plt.close(fig)
    print("[saved] Fig 5 right (residuals_vs_D25)")


def fig6_hist(base, amiga, hcg, lim):
    fig, ax = plt.subplots(figsize=(10, 8))
    ra = amiga["resid"].to_numpy()
    rh = hcg["residual_dex"].to_numpy()
    rl = lim["resid"].to_numpy()
    bins = np.linspace(min(ra.min(), rh.min(), rl.min()) - 0.1, max(ra.max(), rh.max()) + 0.1, 22)
    bw = bins[1] - bins[0]
    bottom = np.zeros(len(bins) - 1)
    for ph in ["1", "2", "3a", "3c"]:
        m = hcg["phase"] == ph
        if m.any():
            c, _ = np.histogram(rh[m.to_numpy()], bins=bins)
            ax.bar(
                bins[:-1],
                c,
                width=bw,
                bottom=bottom,
                align="edge",
                color=PHASE_LIGHT[ph],
                edgecolor="none",
                zorder=2,
            )
            bottom += c
    ax.hist(
        rh,
        bins=bins,
        histtype="step",
        lw=2.5,
        edgecolor="blue",
        label="Galaxies in HCGs (det.)",
        zorder=4,
    )
    ax.hist(
        ra,
        bins=bins,
        histtype="stepfilled",
        facecolor="white",
        edgecolor="none",
        hatch="/",
        lw=0,
        zorder=5,
        alpha=0.7,
    )
    ax.hist(
        ra, bins=bins, histtype="step", lw=2.5, edgecolor="black", label="AMIGA galaxies", zorder=6
    )
    ax.axvline(
        np.mean(ra), color="black", ls="--", lw=2, label=f"AMIGA mean: {np.mean(ra):.2f}", zorder=7
    )
    ax.axvline(
        np.mean(rh),
        color="blue",
        ls="--",
        lw=2,
        label=f"HCG mean (det.): {np.mean(rh):.2f}",
        zorder=7,
    )
    ax.axvline(0, color="gray", ls=":", lw=1.5, zorder=1)
    # upper limits: rug of downward arrows along the bottom at their residual value
    y0 = -0.045 * ax.get_ylim()[1]
    ax.scatter(
        rl,
        np.full_like(rl, y0),
        marker="v",
        s=55,
        facecolors="none",
        edgecolors="darkorange",
        linewidths=1.3,
        zorder=8,
        label=f"HCG upper limits (n={len(rl)})",
        clip_on=False,
    )
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
    ax.set_ylabel("Number of galaxies", fontsize=22, labelpad=15)
    _ticks(ax)
    ax.legend(loc="upper left", fontsize=15, frameon=True)
    fig.tight_layout()
    _backup("diameter_residuals_hist_kelley_larger_sample.pdf")
    fig.savefig(
        FIG / "diameter_residuals_hist_kelley_larger_sample.pdf", bbox_inches="tight", dpi=200
    )
    plt.close(fig)
    print("[saved] Fig 6 (residuals_hist)")


def fig8_survey(base, km_median=-0.536, km_n=124):
    """Survey median-residual forest plot + a 2nd HCG point at the KM (censored)
    median, clearly distinguished from the detection-based survey medians.
    """
    import re

    txt = (ROOT / "latex" / "autogen" / "table_survey_residuals.tex").read_text()
    rows = []
    for ln in txt.splitlines():
        mobj = re.match(
            r"\s*([A-Za-z].*?)\s*&\s*([0-9]+)\s*&\s*\$([+-][0-9.]+)\$\s*&\s*([0-9.]+)", ln
        )
        if mobj:
            rows.append(
                (
                    mobj.group(1).strip(),
                    int(mobj.group(2)),
                    float(mobj.group(3)),
                    float(mobj.group(4)),
                )
            )
    df = pd.DataFrame(rows, columns=["sample", "N", "median", "scatter"])
    # add the censored HCG point
    km_row = pd.DataFrame(
        [["HCGs (incl. limits, KM)", km_n, km_median, np.nan]], columns=df.columns
    )
    df = (
        pd.concat([df, km_row], ignore_index=True)
        .sort_values("median", ascending=False)
        .reset_index(drop=True)
    )
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, max(6, 0.45 * len(df) + 1)))
    is_km = df["sample"].str.contains("KM")
    det = df[~is_km]
    ax.errorbar(
        det["median"],
        y[~is_km.to_numpy()],
        xerr=det["scatter"],
        fmt="o",
        capsize=3,
        elinewidth=1.2,
        markersize=6,
        zorder=3,
        color="#1d5378",
    )
    N = det["N"].values
    sizes = (
        30 + 120 * (N - N.min()) / (N.max() - N.min())
        if N.max() > N.min()
        else np.full_like(N, 80, float)
    )
    ax.scatter(det["median"], y[~is_km.to_numpy()], s=sizes, zorder=4, color="#1d5378")
    # KM HCG point: red star, no symmetric error bar (censored, asymmetric)
    ky = y[is_km.to_numpy()]
    ax.scatter(
        df.loc[is_km, "median"],
        ky,
        marker="*",
        s=420,
        zorder=6,
        color="#d62728",
        edgecolors="black",
        linewidths=0.8,
        label="HCGs incl. upper limits (Kaplan--Meier)",
    )
    sigma = base["sigma"]
    ax.axvline(0, ls="-", lw=1.5, color="black", alpha=0.6)
    ax.axvline(sigma, ls="--", lw=1.0, alpha=0.4, color="gray")
    ax.axvline(-sigma, ls="--", lw=1.0, alpha=0.4, color="gray")
    ax.axvspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(df["sample"], df["N"])])
    ax.set_xlabel(r"Median $\Delta\log(D_{\rm HI})$  (± scatter) [dex]", fontsize=20, labelpad=30)
    ax.set_title(r"Ranked by median $\Delta\log(D_{\rm HI})$", fontsize=18, pad=10)
    # descending sort + default y-up => most-truncated (most negative) at TOP
    _ticks(ax)
    ax.annotate(
        "red star: HCGs with upper limits\n(Kaplan--Meier median)",
        xy=(0.97, 0.07),
        xycoords="axes fraction",
        ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    name = "survey_median_residual_kelley_larger_well_defined_sample_hydra_split.pdf"
    _backup(name)
    fig.savefig(FIG / name, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] Fig 8 (survey_median + KM HCG point)")


def main():
    base, amiga, hcg, lim = load()
    print(f"baseline slope={base['slope']:.4f} r={base['r']:.3f} sigma={base['sigma']:.4f}")
    print(f"AMIGA={len(amiga)} HCG det={len(hcg)} limits={len(lim)}")
    fig5_left(base, amiga, hcg, lim)
    fig5_right(base, amiga, hcg, lim)
    # NOTE: this script only owns the Figure 5 panels (its declared Snakemake
    # outputs). The residual histogram (Figure 6) and the survey forest plot
    # (Figure 8 top) are owned by plot_residual_histogram.py and
    # plot_survey_residual_forest.py respectively; writing them here would
    # overwrite those rules' outputs, so fig6_hist()/fig8_survey() are not called.
    print("Done. Baseline untouched; originals backed up to *_detonly_backup.pdf.")


if __name__ == "__main__":
    main()
