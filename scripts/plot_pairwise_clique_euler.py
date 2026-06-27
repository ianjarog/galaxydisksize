#!/usr/bin/env python3
"""
TEST figure: Figure 8 BOTTOM panel (statistical-equivalence Euler diagram),
recomputed with the HCG upper limits included.

Pairwise significance (BH-FDR at alpha=0.05): HCG-vs-survey comparisons use the
Gehan generalised Wilcoxon test (HCG is left-censored by the 70 beam upper
limits); all other pairs keep the Mann-Whitney U values. 
The resulting maximal cliques are:

    {HCGs, VIVA}                              <- most truncated 
    {Hydra I (cluster), Ursa Major}
    {Hydra I (infall), Hydra I (field), Ursa Major}   (Ursa Major bridges)
    {AMIGA}            {Pairs (Bok+20)}       <- least truncated (now separate)

"""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

import figure_style  # noqa: E402

figure_style.apply()

RED, ORANGE, GREEN, BLUE = "#a01c1c", "#a05400", "#1c5b1c", "#1d5378"
F_RED, F_OR, F_GR, F_BL = "#f5b8b8", "#fdd9a8", "#bce4b9", "#a4cce0"
FS = 13


def ell(ax, xy, w, h, fc, ec, alpha=0.58):
    ax.add_patch(
        Ellipse(
            xy, width=w, height=h, facecolor=fc, alpha=alpha, edgecolor=ec, linewidth=2.4, zorder=2
        )
    )


def lbl(ax, x, y, s, c, bold=True):
    ax.text(
        x,
        y,
        s,
        color=c,
        fontweight=("bold" if bold else "normal"),
        ha="center",
        va="center",
        fontsize=FS,
        zorder=4,
    )


def main(promote=False):
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- top: {VIVA, HCGs} (most truncated) ----
    ell(ax, (0.0, 3.5), 3.4, 1.7, F_RED, RED, alpha=0.62)
    lbl(ax, 0.0, 3.85, "HCGs", RED)
    lbl(ax, 0.0, 3.15, "VIVA", RED)

    # ---- middle: two overlapping cliques sharing Ursa Major ----
    # 2A: {Hydra I (cluster), Ursa Major}
    ell(ax, (-0.35, 0.8), 3.0, 3.1, F_OR, ORANGE)
    lbl(ax, -0.35, 1.75, "Hydra I (cluster)", ORANGE)
    # 2B: {Hydra I (infall), Hydra I (field), Ursa Major}
    ell(ax, (1.15, 0.05), 5.4, 2.5, F_GR, GREEN)
    lbl(ax, 2.35, 0.55, "Hydra I (infall)", GREEN)
    lbl(ax, 2.35, -0.15, "Hydra I (field)", GREEN)
    # bridge (in both 2A and 2B) -> black, in the overlap
    lbl(ax, -0.35, -0.5, "Ursa Major", "black", bold=False)

    # ---- bottom: {AMIGA} and {Pairs} now separate singleton cliques ----
    ell(ax, (-0.35, -2.7), 2.9, 1.25, F_BL, BLUE, alpha=0.62)
    lbl(ax, -0.35, -2.7, "AMIGA", BLUE)
    ell(ax, (-0.35, -4.15), 2.9, 1.25, F_BL, BLUE, alpha=0.62)
    lbl(ax, -0.35, -4.15, "Pairs (Bok+20)", BLUE)

    # ---- truncation gradient arrow (left) ----
    xL = -3.6
    ax.add_patch(
        FancyArrowPatch(
            (xL, -4.7),
            (xL, 4.4),
            arrowstyle="<->",
            mutation_scale=22,
            lw=2.0,
            color="black",
            zorder=1,
        )
    )
    ax.text(xL, 4.8, "more\ntruncated", ha="center", va="bottom", fontsize=13)
    ax.text(xL, -5.1, "less\ntruncated", ha="center", va="top", fontsize=13)

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-5.6, 5.4)
    fig.tight_layout()
    out = FIG / "plot_pairwise_clique_euler.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {out}")

    if promote:
        prod = FIG / "pairwise_clique_euler_vertical.pdf"
        bak = FIG / "pairwise_clique_euler_vertical_prevhardcoded_backup.pdf"
        if prod.exists() and not bak.exists():
            shutil.copy2(prod, bak)
            print(f"[backup] {prod.name} -> {bak.name}")
        shutil.copy2(out, prod)
        print(f"[PROMOTED] production Figure 8 (bottom, Euler) -> {prod.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    main(promote=ap.parse_args().promote)
