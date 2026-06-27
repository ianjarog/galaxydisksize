#!/usr/bin/env python3
"""
TEST FIGURES for the Figure-7 redesign (HCG truncation by evolutionary phase).
==============================================================================
Produces FOUR candidate presentations (all statistically correct, Kaplan-Meier)
so the best one can be chosen.  These are throwaway figures: every output name
starts with "test_figure7" and NOTHING in the manuscript or the published
figures is touched.

    test_figure7_km_box.pdf       - KM "survival" box-and-whisker (open-ended where censored)
    test_figure7_km_cdf.pdf       - KM cumulative-distribution curves per phase
    test_figure7_strip_shaded.pdf - strip (dots+arrows) + bold KM bar + shaded truncated zone
    test_figure7_pct_bars.pdf     - KM fraction-below-baseline bars per phase

Run:  python scripts/plot_residuals_by_phase.py
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# --- TeX Gyre Heros (same font as the other manuscript figures) ---
import figure_style  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

figure_style.apply()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from survival_analysis import km_left_censored, load  # noqa: E402

FIG = ROOT / "figures"
PROD = ROOT / "products"
AMIGA_RESID_CSV = PROD / "amiga_residuals_per_galaxy_kelley_larger_sample_dictionary.csv"

PHASES = ["1", "2", "3c", "3a"]
PCOL = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}


# --------------------------------------------------------------------------
def km_quantile(km, q, grid):
    """Invert the KM CDF F(x)=frac_below(x): smallest x with F(x) >= q.

    Returns (value, defined). The q-quantile is only 'defined' when the KM CDF
    actually CROSSES q inside the grid, i.e. F(grid[0]) < q <= F(grid[-1]).
    If F(grid[0]) >= q already, the quantile lies below the resolved range
    (heavy left-censoring) -> unconstrained, defined=False.
    """
    F = np.array([km["frac_below"](x) for x in grid])
    if F[0] >= q:  # crossing is below the grid -> unconstrained
        return grid[0], False
    ok = np.where(F >= q)[0]
    if ok.size == 0:  # never reaches q (above grid) -> unconstrained
        return grid[-1], False
    return grid[ok[0]], True


def phase_blocks(hcg):
    """Yield dicts with per-phase residuals, limit mask, KM object, n, n_lim."""
    out = []
    for ph in PHASES:
        sub = hcg[hcg["phase"].astype(str) == ph]
        if len(sub) < 2:
            continue
        d = sub["delta"].to_numpy(float)
        lim = sub["is_limit"].to_numpy(bool)
        out.append(
            dict(
                phase=ph,
                d=d,
                lim=lim,
                n=len(sub),
                n_lim=int(lim.sum()),
                km=km_left_censored(d, lim),
            )
        )
    return out


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8, width=1.2, pad=8)
    ax.tick_params(which="minor", length=4, width=1, pad=8)


# --------------------------------------------------------------------------
def variant_km_box(res_amiga, blocks, sigma):
    grid = np.linspace(-1.5, 0.6, 1600)
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = ["AMIGA\n(isolated)"]
    # AMIGA ordinary box
    bp = ax.boxplot(
        [res_amiga], positions=[0], widths=0.55, patch_artist=True, whis=(5, 95), showfliers=False
    )
    bp["boxes"][0].set_facecolor("white")
    bp["medians"][0].set_color("black")
    floor = -0.95
    for i, b in enumerate(blocks, start=1):
        km = b["km"]
        q1, d1 = km_quantile(km, 0.25, grid)
        med, dm = km_quantile(km, 0.50, grid)
        q3, d3 = km_quantile(km, 0.75, grid)
        lo, dlo = km_quantile(km, 0.05, grid)
        hi, dhi = km_quantile(km, 0.95, grid)
        c = PCOL[b["phase"]]
        top = q3 if d3 else (hi if dhi else 0.0)
        bottom = q1 if d1 else floor  # open-ended box when Q1 is censored
        ax.add_patch(
            plt.Rectangle(
                (i - 0.28, bottom),
                0.56,
                top - bottom,
                facecolor=c,
                alpha=0.45 if d1 else 0.30,
                edgecolor="black" if d1 else c,
                ls="-" if d1 else "--",
                zorder=3,
            )
        )
        if d1 and dlo:
            ax.plot([i, i], [bottom, lo], color="black", lw=1.2, zorder=2)
        if d3 and dhi:
            ax.plot([i, i], [top, hi], color="black", lw=1.2, zorder=2)
        if dm:  # median is a real number
            ax.plot([i - 0.28, i + 0.28], [med, med], color="black", lw=2.5, zorder=5)
        else:  # median unconstrained -> red upper bound + arrow
            mbound = float(np.median(b["d"]))
            ax.plot([i - 0.28, i + 0.28], [mbound, mbound], color="red", lw=2.5, zorder=5)
            ax.annotate(
                "",
                xy=(i, floor - 0.12),
                xytext=(i, floor),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2),
                zorder=5,
            )
        labels.append(f"Phase {b['phase']}\n(n={b['n']}, {b['n_lim']} lim)")
    ax.axhline(0, color="black", lw=2)
    ax.axhline(sigma, color="gray", ls="--", lw=1.3, alpha=0.7)
    ax.axhline(-sigma, color="gray", ls="--", lw=1.3, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_xlabel("Sample / HCG Phase", fontsize=22, labelpad=12)
    ax.set_ylim(-1.15, 0.75)
    _ticks(ax)
    ax.plot([], [], color="black", lw=2.5, label="KM median (defined)")
    ax.plot([], [], color="red", lw=2.5, label="KM median upper bound")
    ax.legend(loc="upper right", fontsize=13, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure7_km_box.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure7_km_box.pdf")


def variant_km_cdf(res_amiga, blocks):
    grid = np.linspace(-1.3, 0.7, 600)
    fig, ax = plt.subplots(figsize=(10, 8))
    xa = np.sort(res_amiga)
    ya = np.arange(1, len(xa) + 1) / len(xa)
    ax.step(xa, ya, where="post", color="black", lw=2.5, label=f"AMIGA (n={len(xa)})")
    for b in blocks:
        F = np.array([b["km"]["frac_below"](x) for x in grid])
        ax.plot(
            grid,
            F,
            lw=2.6,
            color=PCOL[b["phase"]],
            label=f"Phase {b['phase']} (n={b['n']}, {b['n_lim']} lim)",
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1.5)
    ax.axvline(0, color="black", ls="-", lw=1.5, alpha=0.6)
    ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_ylabel(r"Cumulative fraction  $P(\Delta < x)$", fontsize=22, labelpad=12)
    ax.set_xlim(-1.3, 0.7)
    ax.set_ylim(0, 1.02)
    _ticks(ax)
    ax.text(
        0.03,
        0.93,
        "further left = more truncated",
        transform=ax.transAxes,
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="lower right", fontsize=14, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure7_km_cdf.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure7_km_cdf.pdf")


def variant_strip_shaded(res_amiga, blocks, sigma):
    grid = np.linspace(-1.5, 0.6, 1600)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    # shaded truncated zone (below -sigma)
    ax.axhspan(-1.15, -sigma, color="red", alpha=0.06, zorder=0)
    # AMIGA strip
    xa = rng.normal(0, 0.08, size=len(res_amiga))
    ax.scatter(xa, res_amiga, s=20, c="0.5", alpha=0.5, zorder=2)
    ax.plot([-0.3, 0.3], [np.median(res_amiga)] * 2, color="black", lw=3, zorder=4)
    labels = ["AMIGA\n(isolated)"]
    for i, b in enumerate(blocks, start=1):
        c = PCOL[b["phase"]]
        d, lim = b["d"], b["lim"]
        x = rng.normal(i, 0.09, size=len(d))
        ax.scatter(x[~lim], d[~lim], s=42, c=c, edgecolors="black", linewidths=0.4, zorder=3)
        ax.scatter(
            x[lim],
            d[lim],
            marker="v",
            s=70,
            facecolors="none",
            edgecolors=c,
            linewidths=1.4,
            zorder=3,
        )
        for xi, yi in zip(x[lim], d[lim]):
            ax.annotate(
                "",
                xy=(xi, yi - 0.08),
                xytext=(xi, yi),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0),
                zorder=3,
            )
        med, dm = km_quantile(b["km"], 0.50, grid)
        if dm:
            ax.plot([i - 0.32, i + 0.32], [med, med], color="black", lw=4, zorder=6)
        else:
            mbound = float(np.median(d))
            ax.plot([i - 0.32, i + 0.32], [mbound, mbound], color="red", lw=4, zorder=6)
            ax.annotate(
                "",
                xy=(i, mbound - 0.12),
                xytext=(i, mbound),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.4),
                zorder=6,
            )
        labels.append(f"Phase {b['phase']}\n(n={b['n']}, {b['n_lim']} lim)")
    ax.axhline(0, color="black", lw=2)
    ax.axhline(-sigma, color="gray", ls="--", lw=1.3, alpha=0.7)
    ax.axhline(sigma, color="gray", ls="--", lw=1.3, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_xlabel("Sample / HCG Phase", fontsize=22, labelpad=12)
    ax.set_ylim(-1.15, 0.75)
    _ticks(ax)
    ax.plot([], [], color="black", lw=4, label="KM median (defined)")
    ax.plot([], [], color="red", lw=4, label="KM median upper bound")
    ax.text(
        0.02,
        0.04,
        "shaded: truncated zone ($\\Delta < -\\sigma_{\\rm AMIGA}$)",
        transform=ax.transAxes,
        fontsize=12,
        color="darkred",
    )
    ax.legend(loc="lower right", fontsize=13, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure7_strip_shaded.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure7_strip_shaded.pdf")


def variant_pct_bars(blocks, grid=None):
    grid = np.linspace(-1.5, 0.6, 1600)
    fig, ax = plt.subplots(figsize=(10, 7))
    ys = np.arange(len(blocks))[::-1]
    for y, b in zip(ys, blocks):
        pct = 100 * b["km"]["frac_below"](0.0)
        med, dm = km_quantile(b["km"], 0.50, grid)
        ax.barh(y, pct, color=PCOL[b["phase"]], alpha=0.8, edgecolor="black", height=0.6)
        mtxt = f"med {med:+.2f}" if dm else f"med $\\leq{np.median(b['d']):+.2f}$"
        ax.text(pct + 1.2, y, f"{pct:.0f}%   {mtxt}", va="center", fontsize=15)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"Phase {b['phase']}\n(n={b['n']}, {b['n_lim']} lim)" for b in blocks], fontsize=15
    )
    ax.set_xlabel("KM fraction below AMIGA baseline [%]", fontsize=20, labelpad=12)
    ax.set_xlim(0, 118)
    ax.axvline(50, color="gray", ls=":", lw=1.5)
    _ticks(ax)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure7_pct_bars.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure7_pct_bars.pdf")


def variant_box_kmmedian(res_amiga, blocks, sigma):
    """'Before'-style notched box plot (boxes from ALL points, limits at their
    limit value -> a conservative/upper-bound view) + detection dots + upper-limit
    arrows, with the strip_shaded KM-median bars overlaid as the median indicator
    (box's own median suppressed). black bar = KM median (defined); red bar+arrow
    = KM upper bound (Phase 3c/3a, >50% limits).
    """
    grid = np.linspace(-1.5, 0.6, 1600)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    data_list = [res_amiga] + [b["d"] for b in blocks]
    box_colors = ["white"] + [PCOL[b["phase"]] for b in blocks]
    positions = np.arange(len(data_list))
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        notch=True,
        showfliers=False,
        medianprops=dict(linewidth=0),  # suppress box median
        whiskerprops=dict(color="0.3"),
        capprops=dict(color="0.3"),
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45 if c != "white" else 1.0)
        patch.set_edgecolor("black")

    # AMIGA points
    xa = rng.normal(0, 0.07, size=len(res_amiga))
    ax.scatter(xa, res_amiga, s=14, c="0.25", alpha=0.45, zorder=3)
    # per-phase points (detections filled, limits open triangle + arrow) + KM bar
    det_done = lim_done = kmd_done = kmb_done = False
    for i, b in enumerate(blocks, start=1):
        c = PCOL[b["phase"]]
        d, lim = b["d"], b["lim"]
        x = rng.normal(i, 0.085, size=len(d))
        ax.scatter(
            x[~lim],
            d[~lim],
            s=42,
            c=c,
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
            label=None if det_done else "Detections",
        )
        det_done = True
        ax.scatter(
            x[lim],
            d[lim],
            marker="v",
            s=72,
            facecolors="none",
            edgecolors=c,
            linewidths=1.5,
            zorder=4,
            label=None if lim_done else "Upper limits",
        )
        lim_done = True
        for xi, yi in zip(x[lim], d[lim]):
            ax.annotate(
                "",
                xy=(xi, yi - 0.08),
                xytext=(xi, yi),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0),
                zorder=4,
            )
        med, dm = km_quantile(b["km"], 0.50, grid)
        if dm:
            ax.plot(
                [i - 0.34, i + 0.34],
                [med, med],
                color="black",
                lw=4.5,
                zorder=7,
                solid_capstyle="butt",
                label=None if kmd_done else "KM median",
            )
            kmd_done = True
        else:
            mb = float(np.median(d))
            ax.plot(
                [i - 0.34, i + 0.34],
                [mb, mb],
                color="red",
                lw=4.5,
                zorder=7,
                solid_capstyle="butt",
                label=None if kmb_done else "KM median (upper bound)",
            )
            kmb_done = True
            ax.annotate(
                "",
                xy=(i, mb - 0.14),
                xytext=(i, mb),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.6),
                zorder=7,
            )
    # AMIGA KM-style median bar (ordinary median; all detections)
    ax.plot(
        [-0.34, 0.34],
        [np.median(res_amiga)] * 2,
        color="black",
        lw=4.5,
        zorder=7,
        solid_capstyle="butt",
    )

    ax.axhline(0, color="black", lw=2, zorder=1)
    ax.axhline(sigma, color="gray", ls="--", lw=1.3, alpha=0.7, zorder=1)
    ax.axhline(-sigma, color="gray", ls="--", lw=1.3, alpha=0.7, zorder=1)
    labels = ["AMIGA\n(isolated)"] + [f"Phase {b['phase']}\n(n={b['n']})" for b in blocks]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=19)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=12)
    ax.set_xlabel("Sample / HCG Phase", fontsize=22, labelpad=12)
    ax.set_ylim(-1.2, 0.78)
    _ticks(ax)
    ax.legend(loc="lower left", fontsize=14, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "test_figure7_box_kmmedian.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/test_figure7_box_kmmedian.pdf")


def main(promote=False):
    m, b_int, sigma, iso_delta, hcg = load()
    res_amiga = pd.read_csv(AMIGA_RESID_CSV)["resid_Bayesian"].to_numpy(float)
    res_amiga = res_amiga[np.isfinite(res_amiga)]
    blocks = phase_blocks(hcg)
    print(f"AMIGA n={len(res_amiga)}  sigma={sigma:.4f}")
    grid = np.linspace(-1.5, 0.6, 1600)
    print("Per-phase KM summary (for cross-check):")
    for blk in blocks:
        med, dm = km_quantile(blk["km"], 0.50, grid)
        ms = f"{med:+.3f}" if dm else f"<= {np.median(blk['d']):+.2f} (bound)"
        print(
            f"  Phase {blk['phase']:2}: n={blk['n']:2} ({blk['n_lim']} lim)  "
            f"KM median={ms}  %below={100 * blk['km']['frac_below'](0.0):.0f}%"
        )
    variant_km_box(res_amiga, blocks, sigma)
    variant_km_cdf(res_amiga, blocks)
    variant_strip_shaded(res_amiga, blocks, sigma)
    variant_pct_bars(blocks)
    variant_box_kmmedian(res_amiga, blocks, sigma)
    print("\nDone. 5 test_figure7_*.pdf written to figures/. Nothing else touched.")

    if promote:
        prod = FIG / "diameter_residuals_by_phase_kelley_larger_sample.pdf"
        bak = FIG / "diameter_residuals_by_phase_kelley_larger_sample_kmdiamond_backup.pdf"
        if prod.exists() and not bak.exists():
            shutil.copy2(prod, bak)
            print(f"[backup] {prod.name} -> {bak.name}")
        shutil.copy2(FIG / "test_figure7_box_kmmedian.pdf", prod)
        print(f"[PROMOTED] production Figure 7 -> {prod.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--promote",
        action="store_true",
        help="also write the box_kmmedian design to the production Fig 7 file",
    )
    main(promote=ap.parse_args().promote)
