#!/usr/bin/env python3
"""
SURVIVAL ANALYSIS of HCG HI-diameter residuals with beam-size upper limits
==========================================================================
The 70 HI non-detected HCG members carry an *upper limit* on D_HI (beam Bmaj),
hence an upper limit on the residual

    Delta = log10(D_HI) - [ slope*log10(D_25) + intercept ]      (left-censored).

We treat these as left-censored data and use the Kaplan-Meier (KM) estimator
(Feigelson & Nelson 1985, ApJ 293, 192) for the residual distribution, and the
Gehan generalised Wilcoxon two-sample test (Gehan 1965) to compare the AMIGA
(uncensored) and HCG (det + upper-limit) residual distributions.

Left-censored -> right-censored trick: work with U = -Delta. An upper limit
Delta <= Dlim becomes U >= -Dlim (right-censored). Standard right-censored KM
on U gives S_U(t)=P(U>t); medians and tail fractions map back via Delta=-U.

This script ONLY READS existing data + the persisted AMIGA baseline and WRITES
new autogen files (it never re-fits and never overwrites the detection-only
pipeline outputs):
    autogen/table_phase_stats_censored.tex
    autogen/table_stat_tests_censored.tex
    autogen/macros_hcg_censored.tex
    autogen/table_upperlimits.tex
Run with --emit to write the files; default just prints the numbers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROD = ROOT / "products"
AUTOGEN = ROOT / "latex" / "autogen"

ISO_CSV = DATA / "isolated_galaxies_results.csv"
# Full combined AMIGA reference (407 = 35 resolved + 372 single-dish inferred),
# i.e. the same isolated-galaxy sample used everywhere else in the paper.
AMIGA_RESID_CSV = PROD / "amiga_residuals_per_galaxy.csv"
HCG_AUG_CSV = DATA / "interacting_galaxies_results_with_upperlimits_bmaj.csv"
PROV_CSV = DATA / "upperlimits_bmaj_provenance.csv"
BASELINE_JSON = PROD / "hcg_residual_statistics.json"


# --------------------------------------------------------------------------
# Kaplan-Meier for left-censored (upper-limit) data
# --------------------------------------------------------------------------
def km_left_censored(delta, is_limit):
    """KM survival function of Delta given left-censoring (upper limits).

    Returns callables/arrays describing F(x)=P(Delta<=x) and helpers.
    Internally works on U=-Delta (right-censored). Detections are 'events'.
    """
    delta = np.asarray(delta, float)
    is_limit = np.asarray(is_limit, bool)
    U = -delta  # right-censored at U for limits
    # increasing event times in U
    uniq = np.unique(U)
    S = 1.0
    ts, Ss = [], []
    for t in uniq:
        at = U == t
        d = int(np.sum(at & ~is_limit))  # detections (events) at t
        n_at = int(np.sum(U >= t))  # at risk (U>=t)
        if d > 0 and n_at > 0:
            S *= 1.0 - d / n_at
        ts.append(t)
        Ss.append(S)
    ts = np.asarray(ts)
    Ss = np.asarray(Ss)  # S_U(t)=P(U>t) (right-continuous step)

    def median_delta():
        # median of U: smallest t with S_U(t) <= 0.5 ; Delta_med = -that
        below = np.where(Ss <= 0.5)[0]
        if below.size == 0:
            return np.nan  # not reached (heavy censoring)
        return -ts[below[0]]

    def frac_delta_below(x):
        # P(Delta < x) = P(U > -x) = S_U(-x)
        t = -x
        idx = np.where(ts <= t)[0]
        return float(Ss[idx[-1]]) if idx.size else 1.0

    def km_mean_delta():
        # KM mean of U via area under S_U, then negate. Restricted to support.
        # E[U] = integral_0..inf S(t) dt  (for U>=min). Use full support shift.
        tmin = ts[0]
        # prepend the starting point
        tt = np.concatenate(([tmin], ts))
        ss = np.concatenate(([1.0], Ss))
        area = np.sum((tt[1:] - tt[:-1]) * ss[:-1])
        return -(tmin + area)

    return dict(
        ts=ts, Ss=Ss, median=median_delta(), frac_below=frac_delta_below, mean=km_mean_delta()
    )


# --------------------------------------------------------------------------
# Gehan generalised Wilcoxon two-sample test (with right-censored machinery)
# --------------------------------------------------------------------------
def gehan_test(delta_a, lim_a, delta_b, lim_b):
    """Gehan generalised Wilcoxon test comparing samples A and B.

    Upper limits (left-censored) handled via U=-Delta (right-censored).
    Mantel form: score each subject by U_i = (#definitely-less) - (#definitely-greater)
    over the pooled sample, where censoring makes some comparisons ties.
    Statistic W = sum of scores over sample A; variance from permutation form.
    Reduces to the Wilcoxon rank-sum test when there is no censoring.
    """
    U = np.concatenate([-np.asarray(delta_a, float), -np.asarray(delta_b, float)])
    cens = np.concatenate([np.asarray(lim_a, bool), np.asarray(lim_b, bool)])
    grp = np.concatenate([np.zeros(len(delta_a), int), np.ones(len(delta_b), int)])
    N = len(U)
    # Gehan score h_ij for ordered pair: compare i vs j (right-censored data)
    # value x_i with censor c_i (c=1 => x_i is a lower bound, true >= x_i)
    # +1 if x_i definitely > x_j, -1 if definitely <, 0 otherwise.
    score = np.zeros(N)
    for i in range(N):
        s = 0
        xi, ci = U[i], cens[i]
        for j in range(N):
            if i == j:
                continue
            xj, cj = U[j], cens[j]
            # definitely greater: xi>xj and (xi not censored-up beyond? )
            # right-censored: ci => true_i>=xi ; cj => true_j>=xj
            if not ci and not cj:
                if xi > xj:
                    s += 1
                elif xi < xj:
                    s -= 1
            elif ci and not cj:
                # true_i >= xi ; if xi >= xj -> definitely >=, count +1 (i>j)
                if xi >= xj:
                    s += 1
                # else indeterminate
            elif not ci and cj:
                if xj >= xi:
                    s -= 1
                # else indeterminate
            else:
                pass  # both censored -> indeterminate
        score[i] = s
    W = float(np.sum(score[grp == 0]))
    n1 = int(np.sum(grp == 0))
    # permutation variance (Mantel): Var = n1*n2/(N*(N-1)) * sum(score^2)
    n2 = N - n1
    var = n1 * n2 / (N * (N - 1)) * np.sum(score**2)
    z = W / np.sqrt(var) if var > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z))
    return dict(W=W, z=z, p=p, var=var)


# --------------------------------------------------------------------------
def load():
    base = json.load(open(BASELINE_JSON))
    m, b = base["baseline_slope"], base["baseline_intercept"]
    sig = base["baseline_sigma"]

    # AMIGA reference = full combined sample (407), recomputed from the SAME
    # persisted baseline (m, b) used for the HCG side, so both arms of the
    # two-sample comparison share one baseline. (Previously this used only the
    # 35 resolved baseline-fitting galaxies, which sit ~0 by construction.)
    iso = pd.read_csv(AMIGA_RESID_CSV).dropna(subset=["log_D25", "log_DHI"])
    iso_delta = iso["log_DHI"].to_numpy() - (m * iso["log_D25"].to_numpy() + b)

    hcg = pd.read_csv(HCG_AUG_CSV)
    hcg = hcg[hcg["optical_diameter_kpc"] > 0].copy()
    hcg["delta"] = np.log10(hcg["hi_diameter_kpc"]) - (
        m * np.log10(hcg["optical_diameter_kpc"]) + b
    )
    hcg["is_limit"] = hcg["is_upper_limit"].astype(bool)
    return m, b, sig, np.asarray(iso_delta), hcg


PHASE_ORDER = ["1", "2", "3c", "3a"]


def phase_stats(hcg):
    """Per-phase KM summary: median (or bound), %below baseline, counts."""
    rows = []
    for ph in PHASE_ORDER:
        sub = hcg[hcg["phase"].astype(str) == ph]
        if len(sub) < 2:
            continue
        d = sub["delta"].values
        lim = sub["is_limit"].values
        km = km_left_censored(d, lim)
        med = km["median"]
        defined = np.isfinite(med)
        # Rigorous upper bound on the true median when KM is unconstrained:
        # every limit galaxy has Delta <= Delta_lim, so replacing limits by their
        # limit value gives an element-wise upper bound; its median bounds the
        # true median (true median <= this), with no KM extrapolation.
        bound = float(np.median(d))
        rows.append(
            dict(
                phase=ph,
                n=len(sub),
                n_lim=int(lim.sum()),
                km_median=med,
                defined=defined,
                bound=bound,
                pct_below=100 * km["frac_below"](0.0),
                deficit=(100 * (1 - 10**med)) if defined else np.nan,
                det_median=float(np.median(d[~lim])) if np.any(~lim) else np.nan,
            )
        )
    return rows


def fmt_med(r):
    if r["defined"]:
        return f"${r['km_median']:+.3f}$"
    return f"$\\leq {r['bound']:+.2f}$"


def emit_phase_table(rows, path):
    L = [
        r"% Auto-generated by scripts/survival_analysis.py (Kaplan-Meier).",
        r"% Censored (detection + beam upper-limit) residual statistics by HCG phase.",
        r"\begin{tabular}{lccccc}",
        r"\toprule \toprule",
        r"Phase & $N$ & $N_{\rm lim}$ & KM median $\Delta$ & \% below & KM deficit \\",
        r"      &      &              & [dex]               & baseline & [\%] \\",
        r"\midrule",
    ]
    for r in rows:
        if r["defined"]:
            defi = f"{r['deficit']:.0f}"
        else:
            defi = f"$\\geq {100 * (1 - 10 ** r['bound']):.0f}$"
        L.append(
            f"{r['phase']} & {r['n']} & {r['n_lim']} & {fmt_med(r)} & "
            f"{r['pct_below']:.0f}\\% & {defi} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def _latex_sci(p, sig=1):
    """Format a small p-value as LaTeX scientific notation, e.g. 4.3\\times10^{-12}."""
    if p == 0 or not np.isfinite(p):
        return f"{p}"
    exp = int(np.floor(np.log10(abs(p))))
    mant = p / 10**exp
    return f"{mant:.{sig}f}\\times10^{{{exp}}}"


def emit_stat_tests(iso_delta, hcg, km_overall, gehan, path):
    hcg["delta"].values
    L = [
        r"% Auto-generated by scripts/survival_analysis.py.",
        r"% Two-sample comparison AMIGA vs HCG (HCG includes beam upper limits).",
        r"\begin{tabular}{lcc}",
        r"\toprule \toprule",
        r"Statistic & AMIGA & HCG (det+lim) \\",
        r"\midrule",
        f"$N$ & {iso_delta.size} & {len(hcg)} ({int(hcg['is_limit'].sum())} lim) \\\\",
        f"KM median $\\Delta$ [dex] & ${np.median(iso_delta):+.3f}$ & "
        f"${km_overall['median']:+.3f}$ \\\\",
        f"\\% below baseline & {100 * np.mean(iso_delta < 0):.0f}\\% & "
        f"{100 * km_overall['frac_below'](0.0):.0f}\\% \\\\",
        r"\midrule",
        r"\multicolumn{3}{l}{Gehan generalised Wilcoxon (censored):} \\",
        f"\\multicolumn{{3}}{{l}}{{\\quad $z={gehan['z']:.2f}$, "
        f"$p={_latex_sci(gehan['p'])}$}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(L) + "\n")


def emit_macros(iso_delta, hcg, km_overall, gehan, rows, path):
    hcg["delta"].values
    pct_below = 100 * km_overall["frac_below"](0.0)
    # Overall KM median. With >50% non-detections the KM survival floor can sit
    # just above 0.5, so the median is formally unconstrained (lies in the
    # censored tail below the most-truncated detection). In that case we report
    # a rigorous upper bound: median <= min(detection Delta), i.e. the deficit is
    # AT LEAST this value. \HCGKMMedianBound flags this so the prose can use a
    # "<=" / "at least" qualifier.
    km_med = km_overall["median"]
    if np.isfinite(km_med):
        median_bound = 0
    else:
        km_med = float(np.min(hcg.loc[~hcg["is_limit"], "delta"]))  # most-truncated detection
        median_bound = 1
    deficit = 100 * (1 - 10**km_med)
    size_fraction = 100 * 10**km_med  # D_HI as % of expected size
    n_tot = len(hcg)
    n_lim = int(hcg["is_limit"].sum())
    # Censored Cliff's delta, AMIGA vs HCG (HCG left-censored by the upper limits):
    # A>H is certain when A exceeds an HCG detection, or when A >= an HCG upper
    # limit (since the true value lies below the limit); A<limit is indeterminate.
    _A = np.asarray(iso_delta, float)[:, None]
    _h = hcg["delta"].to_numpy(float)
    _lim = hcg["is_limit"].to_numpy(bool)
    _hd, _hu = _h[~_lim][None, :], _h[_lim][None, :]
    _g = int((_A > _hd).sum() + (_A >= _hu).sum())
    _l = int((_A < _hd).sum())
    _npair = _A.shape[0] * len(_h)
    cliff_prob = 100 * _g / _npair
    cliff_delta = (_g - _l) / _npair
    L = [
        r"% Auto-generated by scripts/survival_analysis.py.",
        r"% Censored (Kaplan-Meier) overrides for the HCG-vs-AMIGA comparison.",
        r"% \input AFTER macros_hcg_comparison.tex so these take precedence.",
        f"\\providecommand{{\\HCGSampleSizeCensored}}{{{n_tot}}}",
        f"\\providecommand{{\\HCGUpperLimitCount}}{{{n_lim}}}",
        f"\\providecommand{{\\HCGKMMedianDex}}{{{km_med:+.3f}}}",
        f"\\providecommand{{\\HCGKMMedianBound}}{{{median_bound}}}",
        f"\\providecommand{{\\HCGKMDeficitPercent}}{{{deficit:.0f}}}",
        f"\\providecommand{{\\HCGKMSizeFractionPercent}}{{{size_fraction:.0f}}}",
        f"\\providecommand{{\\HCGKMBelowBaselinePercent}}{{{pct_below:.0f}}}",
        f"\\providecommand{{\\CliffProbAmigaGreaterHCGCensoredPercent}}{{{cliff_prob:.0f}}}",
        f"\\providecommand{{\\CliffDeltaCensored}}{{{cliff_delta:.2f}}}",
        f"\\providecommand{{\\GehanZ}}{{{gehan['z']:.1f}}}",
        f"\\providecommand{{\\GehanP}}{{{_latex_sci(gehan['p'])}}}",
    ]
    for r in rows:
        tag = {"1": "PhaseOne", "2": "PhaseTwo", "3c": "PhaseThreeC", "3a": "PhaseThreeA"}[
            r["phase"]
        ]
        L.append(f"\\providecommand{{\\{tag}NCensored}}{{{r['n']}}}")
        L.append(f"\\providecommand{{\\{tag}NLim}}{{{r['n_lim']}}}")
        L.append(f"\\providecommand{{\\{tag}PctBelow}}{{{r['pct_below']:.0f}}}")
    path.write_text("\n".join(L) + "\n")


def emit_upperlimit_table(path):
    prov = pd.read_csv(PROV_CSV).sort_values(["group", "galaxy"])
    prov["beamlim"] = False
    # Beam-limited members: detected in HI but spatially unresolved (D_HI < beam),
    # entering as Bmaj upper limits exactly like the non-detections. They live in
    # the augmented CSV (is_upper_limit==1) but not in the non-detection
    # provenance. Reconstruct B_maj(arcsec) by inverting arcsec->kpc so the table
    # lists ALL upper limits (\HCGUpperLimitCount), marked with a footnote.
    aug = pd.read_csv(HCG_AUG_CSV)
    extra = aug[(aug["is_upper_limit"] == 1) & (~aug["galaxy"].isin(prov["galaxy"]))].copy()
    extra["group"] = extra["group"].astype(str)
    extra["bmaj_arcsec"] = extra["hi_diameter_kpc"] * 1000.0 / (4.848 * extra["distance_mpc"])
    extra["beamlim"] = True
    cols = [
        "group",
        "galaxy",
        "phase",
        "bmaj_arcsec",
        "distance_mpc",
        "optical_diameter_kpc",
        "hi_diameter_kpc",
        "beamlim",
    ]
    allul = pd.concat([prov[cols], extra[cols]], ignore_index=True)
    # Order rows by the numeric HCG index, then member: a plain string sort puts
    # "HCG 7" after "HCG 10/15/...". Sorting whole rows keeps each member's data
    # together, so the numbers are not mixed up.
    allul["_gnum"] = allul["group"].astype(str).str.extract(r"(\d+)").astype(int)
    allul = allul.sort_values(["_gnum", "galaxy"]).drop(columns="_gnum").reset_index(drop=True)
    L = [
        r"% Auto-generated by scripts/survival_analysis.py.",
        r"% Appendix table: all HCG members entering as beam-size upper limits",
        r"% (HI non-detections + beam-limited/unresolved members, marked $^{a}$).",
        r"\begin{longtable}{llccccc}",
        r"\caption{\label{table:upperlimits}HCG members entering the residual "
        r"statistics as left-censored (beam-size) upper limits: the HI "
        r"non-detections plus the members marked $^{a}$, which are detected in "
        r"\HI\ but spatially unresolved ($D_{\rm HI}<$ beam). $D_{\rm HI}$ is the "
        r"major-axis beam ($B_{\rm maj}$) converted to kpc at the group distance; "
        r"$D_{25}$ is from HyperLeda.}\\",
        r"\toprule \toprule",
        r"HCG & member & phase & $B_{\rm maj}$ & $D$ & $D_{25}$ & $D_{\rm HI}$ \\",
        r"    &        &       & [\arcsec]      & [Mpc] & [kpc] & [kpc] \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule HCG & member & phase & $B_{\rm maj}$ & $D$ & $D_{25}$ & "
        r"$D_{\rm HI}$ \\ \midrule \endhead",
        r"\bottomrule \endfoot",
    ]
    for _, x in allul.iterrows():
        grp = str(x["group"]).replace("HCG", "")
        mark = "$^{a}$" if x["beamlim"] else ""
        L.append(
            f"{grp} & {x['galaxy']}{mark} & {x['phase']} & {x['bmaj_arcsec']:.1f} & "
            f"{x['distance_mpc']:.0f} & {x['optical_diameter_kpc']:.1f} & "
            f"$<{x['hi_diameter_kpc']:.1f}$ \\\\"
        )
    L += [r"\end{longtable}"]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    m, b, sig, iso_delta, hcg = load()
    det = hcg[~hcg["is_limit"]]
    print(f"baseline: slope={m:.4f} intercept={b:.4f} sigma={sig:.4f}")
    print(
        f"AMIGA: n={iso_delta.size}  HCG det={len(det)}  HCG lim={int(hcg['is_limit'].sum())}"
        f"  total HCG={len(hcg)}\n"
    )

    # ---- validation: detection-only HCG ----
    d_det = det["delta"].values
    print("=== DETECTION-ONLY (validation vs current macros) ===")
    print(
        f"  HCG det median Delta = {np.median(d_det):+.3f}  (macro -0.118? mean={np.mean(d_det):+.3f})"
    )
    print(
        f"  below baseline: {100 * np.mean(d_det < 0):.0f}%  ({int(np.sum(d_det < 0))}/{len(d_det)})"
    )
    print(
        f"  size fraction 10^median = {100 * 10 ** np.median(d_det):.0f}%  (deficit {100 * (1 - 10 ** np.median(d_det)):.0f}%)"
    )

    # ---- KM censored: overall ----
    alld = hcg["delta"].values
    alll = hcg["is_limit"].values
    km = km_left_censored(alld, alll)
    print("\n=== KAPLAN-MEIER (det + upper limits, censored) ===")
    print(f"  HCG KM median Delta = {km['median']:+.3f}  (KM mean={km['mean']:+.3f})")
    print(f"  KM P(Delta<0) = {100 * km['frac_below'](0.0):.0f}%")
    print(
        f"  KM size fraction 10^median = {100 * 10 ** km['median']:.0f}%  (deficit {100 * (1 - 10 ** km['median']):.0f}%)"
    )
    naive_med = np.median(alld)
    print(
        f"  [naive det+lim-as-points median = {naive_med:+.3f} -> {100 * (1 - 10**naive_med):.0f}% deficit]"
    )

    # ---- per phase ----
    print("\n=== PER PHASE (KM median or bound, det+lim) ===")
    rows = phase_stats(hcg)
    for r in rows:
        med = f"{r['km_median']:+.3f}" if r["defined"] else f"< {r['bound']:+.2f} (unconstr.)"
        print(
            f"  Phase {r['phase']:2}: n={r['n']:2} ({r['n_lim']} lim)  KM med={med}"
            f"  %below={r['pct_below']:.0f}%  det-med={r['det_median']:+.3f}"
        )

    # ---- Gehan test AMIGA vs HCG ----
    print("\n=== GEHAN generalised Wilcoxon: AMIGA vs HCG ===")
    g = gehan_test(iso_delta, np.zeros(iso_delta.size, bool), alld, alll)
    print(f"  W={g['W']:.1f}  z={g['z']:.2f}  p={g['p']:.2e}")
    g0 = gehan_test(iso_delta, np.zeros(iso_delta.size, bool), d_det, np.zeros(d_det.size, bool))
    mw = stats.mannwhitneyu(iso_delta, d_det, alternative="two-sided")
    print(f"  [validation no-censor: Gehan p={g0['p']:.2e}  vs Mann-Whitney p={mw.pvalue:.2e}]")

    if args.emit:
        AUTOGEN.mkdir(exist_ok=True)
        emit_phase_table(rows, AUTOGEN / "table_phase_stats_censored.tex")
        emit_stat_tests(iso_delta, hcg, km, g, AUTOGEN / "table_stat_tests_censored.tex")
        emit_macros(iso_delta, hcg, km, g, rows, AUTOGEN / "macros_hcg_censored.tex")
        emit_upperlimit_table(AUTOGEN / "table_upperlimits.tex")
        print(
            "\n[emit] wrote 4 autogen files: table_phase_stats_censored, "
            "table_stat_tests_censored, macros_hcg_censored, table_upperlimits"
        )


if __name__ == "__main__":
    main()
