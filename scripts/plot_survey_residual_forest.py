#!/usr/bin/env python3
"""Figure 8 (top): survey median-residual forest plot with the HCG upper limits.

The HCG entry represents the full detections + upper-limits sample via its
Kaplan-Meier median (consistent with the residual histograms): the marker sits
at the KM median, the upper whisker runs to the KM 84th percentile, and a
directional arrow marks the lower side, which is unconstrained because of the
left-censoring by the beam-size upper limits.

Output: figures/survey_median_residual.pdf
Run:    python scripts/plot_survey_residual_forest.py
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style  # noqa: E402
from survival_analysis import km_left_censored, load  # noqa: E402

figure_style.apply()

FIG = ROOT / "figures"
SURVEY_TEX = ROOT / "latex" / "autogen" / "table_survey_residuals.tex"
OUTPUT = FIG / "survey_median_residual.pdf"


def parse_surveys():
    rows = []
    for ln in SURVEY_TEX.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z].*?)\s*&\s*([0-9]+)\s*&\s*\$([+-][0-9.]+)\$\s*&\s*([0-9.]+)", ln)
        if m:
            rows.append((m.group(1).strip(), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return pd.DataFrame(rows, columns=["sample", "N", "median", "scatter"])


def main():
    _, _, sigma, _, hcg = load()
    rh = hcg["delta"].to_numpy(float)
    is_lim = hcg["is_limit"].to_numpy(bool)
    km = km_left_censored(rh, is_lim)
    km_med = km["median"]
    if not np.isfinite(km_med):
        # With >50% non-detections the overall KM median lies in the censored
        # tail (unconstrained). Report the rigorous bound = most-truncated
        # detection, matching survival_analysis and table:survey_residuals.
        km_med = float(np.min(rh[~is_lim]))
    # KM-based (censored) spread: the upper half is well-determined, the lower
    # half runs into the unconstrained censored tail -> asymmetric error bar.
    F = km["frac_below"]
    _grid = np.linspace(-1.6, 0.6, 3000)
    _Fx = np.array([F(x) for x in _grid])

    def _kmq(p):
        if _Fx[0] >= p:
            return _grid[0], False  # below resolved range
        ok = np.where(_Fx >= p)[0]
        return (_grid[ok[0]], True) if ok.size else (_grid[-1], False)

    hcg_up84, _ = _kmq(0.84)  # upper 1-sigma-equivalent (defined)
    _, hcg_lo_def = _kmq(0.16)  # lower 1-sigma-equivalent (unconstrained)
    df = parse_surveys()
    # The combined Hydra I row is superseded by its cluster/infall/field split,
    # which carries the environmental information; keep only the split entries.
    df = df[df["sample"] != "Hydra I (combined)"].copy()
    others = df[df["sample"] != "HCGs"].copy()

    # ONE combined HCG entry: full det+upper-limit sample, KM median, N=124.
    plot = pd.concat(
        [others, pd.DataFrame([["HCGs", 124, km_med, np.nan]], columns=df.columns)],
        ignore_index=True,
    )
    plot = plot.sort_values("median", ascending=False).reset_index(drop=True)
    y = np.arange(len(plot))
    is_hcg = plot["sample"].eq("HCGs").to_numpy()
    BLUE = "#1d5378"

    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.5 * len(plot) + 1)))
    N = plot["N"].values
    sizes = 30 + 120 * (N - N.min()) / (N.max() - N.min())
    nh = ~is_hcg
    # literature surveys: symmetric +/-1sigma population scatter
    ax.errorbar(
        plot["median"][nh],
        y[nh],
        xerr=plot["scatter"][nh],
        fmt="none",
        capsize=3,
        elinewidth=1.2,
        ecolor=BLUE,
        zorder=3,
    )
    ax.scatter(plot["median"][nh], y[nh], s=sizes[nh], color=BLUE, zorder=4)
    # HCG: KM median with an ASYMMETRIC bar -- upper whisker to the KM 84th
    # percentile (defined); lower side a large open arrow (unconstrained upper
    # limits) so it reads immediately as a limit.
    yh = float(y[is_hcg][0])
    ax.plot([km_med, hcg_up84], [yh, yh], color=BLUE, lw=1.8, zorder=4)
    ax.plot([hcg_up84, hcg_up84], [yh - 0.15, yh + 0.15], color=BLUE, lw=1.8, zorder=4)
    ax.annotate(
        "",
        xy=(-1.05, yh),
        xytext=(km_med, yh),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8, mutation_scale=40),
        zorder=4,
    )
    ax.scatter(
        [km_med],
        [yh],
        s=float(sizes[is_hcg][0]) + 70,
        color=BLUE,
        edgecolors="black",
        linewidths=0.9,
        zorder=5,
    )
    ax.set_xlim(-1.12, ax.get_xlim()[1])

    ax.axvline(0, ls="-", lw=1.5, color="black", alpha=0.6)
    ax.axvline(sigma, ls="--", lw=1.0, alpha=0.4, color="gray")
    ax.axvline(-sigma, ls="--", lw=1.0, alpha=0.4, color="gray")
    ax.axvspan(-sigma, sigma, color="gray", alpha=0.07, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(plot["sample"], plot["N"])], fontsize=18)
    ax.set_xlabel(r"Median $\Delta\log(D_{\rm HI})$  (± scatter) [dex]", fontsize=20, labelpad=14)
    ax.set_title(r"Ranked by median $\Delta\log(D_{\rm HI})$", fontsize=18, pad=10)
    ax.tick_params(axis="x", labelsize=17, direction="in", top=True)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", direction="in", top=True)

    # Legend (lower left) explaining the HCG marker: same symbol as the surveys,
    # with a directional arrow for the unconstrained (censored) lower bound.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            linestyle="None",
            markersize=9,
            label=r"Survey median ($\pm$ scatter)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            markeredgecolor="black",
            linestyle="None",
            markersize=12,
            label="HCGs: KM median (incl. upper limits;\narrow = unconstrained lower bound)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=13, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(
        f"[saved] {OUTPUT}  (HCG KM median={km_med:+.3f}, upper 84th pct={hcg_up84:+.3f}, "
        f"lower 16th unconstrained={not hcg_lo_def}, n=124)"
    )


if __name__ == "__main__":
    main()
