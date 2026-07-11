#!/usr/bin/env python3
"""Figure 7: HCG truncation by evolutionary phase with Kaplan-Meier medians.

Notched box plot per phase (boxes from ALL points, upper limits at their limit
value -> a conservative view) with detection dots, upper-limit arrows, and the
Kaplan-Meier median bar overlaid: black bar = KM median (defined); red bar +
arrow = KM upper bound (phases with >50% upper limits).

Output: figures/diameter_residuals_by_phase.pdf
Run:    python scripts/plot_residuals_by_phase.py
"""

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
AMIGA_RESID_CSV = PROD / "amiga_residuals_per_galaxy.csv"

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


def box_kmmedian(res_amiga, blocks, sigma):
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
    fig.savefig(FIG / "diameter_residuals_by_phase.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[saved] figures/diameter_residuals_by_phase.pdf")


def main():
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
    box_kmmedian(res_amiga, blocks, sigma)


if __name__ == "__main__":
    main()
