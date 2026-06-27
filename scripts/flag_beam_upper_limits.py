#!/usr/bin/env python3
"""
INVESTIGATION SCRIPT -- Bmaj upper limits for ALL HI-undetected HCG members
===========================================================================

What changed vs the previous upper-limit script
------------------------------------------------
1. UPPER-LIMIT DEFINITION: now the beam MAJOR AXIS (Bmaj) instead of the
   geometric mean sqrt(Bmaj*Bmin).  Applied to EVERY upper limit, including
   the Phase-3a/3c non-detections done previously.
2. SAMPLE: the original 22 (HCG30/37/62/92/97) PLUS 48 additional clean
   member non-detections vetted against Jones et al. (2023) Sect. 3 +
   the project yaml notes (false groups / unreliable cubes / non-members
   removed).  See the cleaned list below.

No fitting is performed: every target was verified to have NO SoFiA-separated
HI mask (its group's separated_features/ contains only detected galaxies), i.e.
they are genuine non-detections -> they cannot have a measured 1 Msun/pc^2
diameter and so receive a beam (Bmaj) upper limit.

Non-destructive
---------------
* Reads (never writes) interacting_galaxies_results.csv  (the 56 fitted rows).
* Reuses the previous provenance (distance + logD25) for the original 22 so
  they are NOT re-queried.
* Writes ONLY new files:
    data/upperlimits_bmaj_provenance.csv
    data/interacting_galaxies_results_with_upperlimits_bmaj.csv
* Figure: backs up the existing PDFs before overwriting.

Run
---
    python flag_beam_upper_limits.py            # query CF3/HyperLEDA for the 48 new
    python flag_beam_upper_limits.py --no-network   # offline (needs cache)
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import project_config  # noqa: E402  (sibling module; scripts/ is on sys.path)
import requests
import yaml
from astropy.io import fits

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
PRODUCTS_DIR = PROJECT_ROOT / "products"
# HCG SoFiA masks (external; provides the major-axis beam for beam-limited members).
HCG_BASE = project_config.external_dir("GALAXYDISKSIZE_HCG_MASKS", "SoFiA_masks")

ORIG_HCG_CSV = DATA_DIR / "interacting_galaxies_results.csv"  # READ-ONLY
POSITIONS_YAML = DATA_DIR / "config_hcg_positions.yaml"
GALAXIES_YAML = DATA_DIR / "config_hcg_galaxies.yaml"
OLD_PROVENANCE = DATA_DIR / "phase3_upperlimits_provenance.csv"  # reuse cache

NEW_PROVENANCE = DATA_DIR / "upperlimits_bmaj_provenance.csv"
NEW_AUG_CSV = DATA_DIR / "interacting_galaxies_results_with_upperlimits_bmaj.csv"

# persisted canonical products (for faithful figure reproduction)
AMIGA_RESID_CSV = PRODUCTS_DIR / "amiga_residuals_per_galaxy_kelley_larger_sample_dictionary.csv"
HCG_RESID_CSV = PRODUCTS_DIR / "hcg_residuals_per_galaxy_kelley_larger_sample.csv"
BASELINE_JSON = PRODUCTS_DIR / "hcg_residual_statistics_kelley_larger_sample.json"

FIGURE_NAME = "diameter_residuals_by_phase_kelley_larger_sample.pdf"
VARIANT_NAME = "diameter_residuals_by_phase_kelley_larger_sample_with_upperlimits.pdf"
HYPERLEDA_CATALOG = "VII/237"

# ---------------------------------------------------------------------------
# Upper-limit targets: group -> list of member galaxies (all HI non-detections)
#   * first block: original 22 (Phase 3a / 3c)
#   * second block: 48 cleaned additions (Phase 1 / 2 / 3c)
# ---------------------------------------------------------------------------
TARGETS = {
    # --- original 22 ---
    "HCG30": ["HCG30a", "HCG30b", "HCG30c", "HCG30d"],
    "HCG37": ["HCG37a", "HCG37b", "HCG37c", "HCG37d", "HCG37e"],
    "HCG62": ["HCG62a", "HCG62b", "HCG62c", "HCG62d"],
    "HCG92": ["HCG92b", "HCG92c", "HCG92d", "HCG92e"],
    "HCG97": ["HCG97a", "HCG97b", "HCG97c", "HCG97d", "HCG97e"],
    # --- 48 cleaned additions ---
    # Phase 1
    "HCG7": ["HCG7b"],
    "HCG10": ["HCG10b", "HCG10c"],
    "HCG19": ["HCG19a"],
    "HCG23": ["HCG23c"],
    "HCG25": ["HCG25d", "HCG25f"],
    "HCG26": ["HCG26b", "HCG26c", "HCG26d", "HCG26f", "HCG26g"],
    # Phase 2
    "HCG40": ["HCG40a", "HCG40b", "HCG40e"],
    "HCG71": ["HCG71b"],
    "HCG79": ["HCG79a", "HCG79b", "HCG79c"],
    "HCG91": ["HCG91d"],
    "HCG96": ["HCG96b", "HCG96c", "HCG96d"],
    # Phase 3c
    "HCG15": ["HCG15a", "HCG15b", "HCG15c", "HCG15d", "HCG15e"],
    "HCG22": ["NGC1188", "HCG22a", "HCG22b"],
    "HCG33": ["HCG33a", "HCG33b", "HCG33d"],
    "HCG56": ["HCG56b", "HCG56c", "HCG56d", "HCG56e"],
    "HCG68": ["HCG68a", "HCG68b", "HCG68d", "HCG68e"],
    "HCG90": ["HCG90b", "HCG90c", "HCG90d"],
    "HCG93": ["HCG93a", "HCG93c", "HCG93d"],
}


# ---------------------------------------------------------------------------
# Helpers (same conventions as the production pipeline)
# ---------------------------------------------------------------------------
def arcsec_to_kpc(arcsec, distance_mpc):
    return arcsec * 4.848 * distance_mpc / 1000.0


def logd25_to_arcmin(logd25):
    return 0.1 * (10.0**logd25)


def get_distance_from_velocity(ra, dec, cz, calculator="CF3"):
    q = {
        "coordinate": [float(ra), float(dec)],
        "system": "equatorial",
        "parameter": "velocity",
        "value": float(cz),
    }
    url = f"http://edd.ifa.hawaii.edu/{calculator}calculator/api.php"
    r = requests.get(
        url, data=json.dumps(q), headers={"Content-type": "application/json"}, timeout=30
    )
    return int(np.round(json.loads(r.text)["observed"]["distance"][0]))


def bmaj_arcsec_for_group(group, _cache={}):
    if group not in _cache:
        hdr = fits.getheader(HCG_BASE / group / f"{group}_HI.pbcor.fits")
        _cache[group] = (float(hdr["BMAJ"]) * 3600.0, float(hdr["BMIN"]) * 3600.0)
    return _cache[group]


def live_hyperleda_logd25(name):
    """Query the LIVE HyperLEDA (ledacat) by object designation.

    The Vizier VII/237 catalogue is a frozen, older HyperLEDA snapshot whose
    logD25 can differ substantially from the current database (e.g. NGC7173:
    1.67 in VII/237 vs 1.27 live).  We therefore use the live ledacat values,
    matched by designation (which also avoids cone-search mis-matches).
    Returns (logd25, resolved_name, source).
    """
    import html as _html
    import re as _re

    url = f"http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?o={name}"
    t = requests.get(url, timeout=30).text
    if "single object" not in t:
        return math.nan, "", "no-single-match"
    txt = _html.unescape(_re.sub(r"<[^>]+>", " ", t))
    txt = _re.sub(r"\s+", " ", txt)
    m = _re.search(r"logd25\s+([0-9.]+)\s*±?\s*([0-9.]+)?\s*log\(0\.1 arcmin\)", txt)
    res = _re.search(r"(NGC\d+|UGC\w+|IC\d+|PGC\d+|MCG[\d.+-]+|ESO[\w-]+)", txt)
    resolved = res.group(1) if res else ""
    if not m:
        return math.nan, resolved, "no-logd25"
    return float(m.group(1)), resolved, "live"


# ---------------------------------------------------------------------------
# Step 1: provenance (Bmaj upper limit, D25, distance)
# ---------------------------------------------------------------------------
def build_provenance(use_network):
    positions = yaml.safe_load(open(POSITIONS_YAML))["galaxies"]
    phases = yaml.safe_load(open(GALAXIES_YAML))["phases"]

    # Distance (CF3) is deterministic and source-independent -> cache from any
    # previous provenance.  logD25 must come from LIVE HyperLEDA, so we only
    # reuse cached logD25 rows explicitly marked logd25_source == 'live'.
    dist_cache, logd_cache = {}, {}
    for src in (OLD_PROVENANCE, NEW_PROVENANCE):
        if src.exists():
            df_ = pd.read_csv(src)
            for _, r in df_.iterrows():
                if np.isfinite(r.get("distance_mpc", np.nan)):
                    dist_cache[r["galaxy"]] = int(r["distance_mpc"])
                if str(r.get("logd25_source", "")) == "live" and np.isfinite(
                    r.get("logD25", np.nan)
                ):
                    logd_cache[r["galaxy"]] = float(r["logD25"])

    rows = []
    for group, members in TARGETS.items():
        bmaj_as, bmin_as = bmaj_arcsec_for_group(group)
        phase = str(phases.get(group, "?"))
        for gal in members:
            pos = positions[gal]
            ra, dec, cz = pos["ra"], pos["dec"], pos["cz"]

            dist = dist_cache.get(gal)
            if dist is None:
                if not use_network:
                    raise SystemExit(f"--no-network but no cached distance for {gal}")
                dist = get_distance_from_velocity(ra, dec, cz)

            if gal in logd_cache:
                logd25, anames, src = logd_cache[gal], "(cached-live)", "live"
            elif use_network:
                logd25, anames, src = live_hyperleda_logd25(gal)
            else:
                raise SystemExit(f"--no-network but no cached live logD25 for {gal}")

            d_hi_kpc = arcsec_to_kpc(bmaj_as, dist)  # *** Bmaj upper limit ***
            d25_kpc = arcsec_to_kpc(logd25_to_arcmin(logd25) * 60.0, dist)

            # Safety net: a logD25 > 1.8 (D25 > ~6.3') for these galaxies would
            # indicate a group/multiple-system entry, not the galaxy.  (Live
            # HyperLEDA values are clean, so this should not trigger.)
            if np.isfinite(logd25) and logd25 > 1.8:
                print(f"  !! {gal}: logD25={logd25:.2f} looks group-contaminated -> D25=NaN")
                d25_kpc = np.nan

            rows.append(
                {
                    "galaxy": gal,
                    "group": group,
                    "phase": phase,
                    "ra": ra,
                    "dec": dec,
                    "cz": cz,
                    "distance_mpc": dist,
                    "bmaj_arcsec": bmaj_as,
                    "bmin_arcsec": bmin_as,
                    "logD25": logd25,
                    "hyperleda_resolved": anames,
                    "logd25_source": "live",
                    "hi_diameter_kpc": d_hi_kpc,
                    "optical_diameter_kpc": d25_kpc,
                }
            )
            print(
                f"  {gal:10} {group:6} ph{phase:3} D={dist:4d}Mpc "
                f'Bmaj={bmaj_as:5.1f}" -> D_HI<{d_hi_kpc:6.2f}kpc | '
                f"logD25(live)={logd25:.2f}->D25={d25_kpc:6.2f}kpc [{src}]"
            )

    df = pd.DataFrame(rows)
    df.to_csv(NEW_PROVENANCE, index=False)
    print(f"\n[saved] provenance -> {NEW_PROVENANCE}  ({len(df)} upper limits)")
    return df


# ---------------------------------------------------------------------------
# Step 2: new augmented CSV (original 56 untouched + 70 upper-limit rows)
# ---------------------------------------------------------------------------
def write_augmented_csv(prov):
    orig = pd.read_csv(ORIG_HCG_CSV).copy()
    orig["is_upper_limit"] = 0
    # Beam-limited "detections": members whose ellipse-fit D_HI carries no
    # diameter error (hi_diameter_err_kpc is NaN) are unresolved (D_HI < beam),
    # exactly the same situation as the formally non-detected members. They are
    # excluded from the mass-size fit (detection-only), so for consistency they
    # must also enter the residual/KM analysis as left-censored Bmaj upper
    # limits rather than as detections. Reclassify them here.
    beamlim = orig["hi_diameter_err_kpc"].isna()
    for idx in orig.index[beamlim]:
        grp = orig.at[idx, "group"]
        dist = float(orig.at[idx, "distance_mpc"])
        bmaj_as, _ = bmaj_arcsec_for_group(grp)
        orig.at[idx, "hi_diameter_kpc"] = arcsec_to_kpc(bmaj_as, dist)  # Bmaj limit
        orig.at[idx, "is_upper_limit"] = 1
        print(
            f"  [beam-limited -> upper limit] {orig.at[idx, 'galaxy']:8} {grp:6} "
            f'Bmaj={bmaj_as:5.1f}" D={dist:4.0f}Mpc -> D_HI<{orig.at[idx, "hi_diameter_kpc"]:6.2f}kpc'
        )
    new_rows = []
    for _, r in prov.iterrows():
        row = {c: np.nan for c in orig.columns}
        row["galaxy"] = r["galaxy"]
        row["group"] = r["group"]
        row["phase"] = r["phase"]
        row["hi_diameter_kpc"] = r["hi_diameter_kpc"]
        row["hi_diameter_err_kpc"] = np.nan
        row["optical_diameter_kpc"] = r["optical_diameter_kpc"]
        row["distance_mpc"] = r["distance_mpc"]
        row["is_upper_limit"] = 1
        new_rows.append(row)
    aug = pd.concat([orig, pd.DataFrame(new_rows)], ignore_index=True)
    aug.to_csv(NEW_AUG_CSV, index=False)
    n_det = int((aug["is_upper_limit"] == 0).sum())
    n_lim = int((aug["is_upper_limit"] == 1).sum())
    print(
        f"[saved] augmented CSV -> {NEW_AUG_CSV} "
        f"({n_det} detections + {n_lim} upper limits = {len(aug)})"
    )
    print(f"        {ORIG_HCG_CSV.name} left UNMODIFIED.")
    return aug


# ---------------------------------------------------------------------------
# Step 3: figure (persisted baseline + residuals; new Bmaj upper limits)
# ---------------------------------------------------------------------------
def replot(prov):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = json.load(open(BASELINE_JSON))
    b_slope = float(baseline["baseline_slope"])
    b_int = float(baseline["baseline_intercept"])
    scatter = float(baseline["baseline_sigma"])
    print(f"[baseline] slope={b_slope:.4f} intercept={b_int:.4f} sigma={scatter:.4f}")

    res_amiga = pd.read_csv(AMIGA_RESID_CSV)["resid_Bayesian"].to_numpy(float)
    res_amiga = res_amiga[np.isfinite(res_amiga)]

    # Single source of truth: the augmented CSV (53 detections + 73 Bmaj upper
    # limits, the 3 beam-limited members already reclassified). Residuals are
    # recomputed from the SAME persisted baseline used by survival_analysis, so
    # this figure cannot diverge from the censored macros/tables.
    aug = pd.read_csv(NEW_AUG_CSV)
    aug = aug[aug["optical_diameter_kpc"] > 0].copy()
    phase_all = aug["phase"].astype(str).str.strip().to_numpy()
    resid_all = np.log10(aug["hi_diameter_kpc"].to_numpy(float)) - (
        b_int + b_slope * np.log10(aug["optical_diameter_kpc"].to_numpy(float))
    )
    is_ul_all = aug["is_upper_limit"].astype(bool).to_numpy()
    # Drop points with no finite residual (e.g. HCG7b, optical size unreliable).
    finite = np.isfinite(resid_all)
    if (~finite).sum():
        print(
            f"[drop] {int((~finite).sum())} upper limit(s) with no reliable optical "
            f"diameter excluded from figure"
        )
    phase_all, resid_all, is_ul_all = phase_all[finite], resid_all[finite], is_ul_all[finite]
    print(
        f"[check] HCG points: {resid_all.size} "
        f"({int(is_ul_all.sum())} upper limits + {int((~is_ul_all).sum())} detections)"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    phases = ["1", "2", "3c", "3a"]
    pcol = {"1": "#1f77b4", "2": "#2ca02c", "3a": "#9467bd", "3c": "#ff7f0e"}

    # Kaplan-Meier median per phase (left-censored upper limits); boxes are drawn
    # from DETECTIONS ONLY so the quartiles are not biased by treating a limit
    # "<x" as a measurement "=x". KM medians (or upper bounds) are overlaid.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from survival_analysis import km_left_censored

    # box data = detections only; groups keep (all resid, ul mask) for scatter/KM
    data_list, labels, colors = [res_amiga], ["AMIGA\n(isolated)"], ["white"]
    groups = [(res_amiga, np.zeros(res_amiga.size, bool))]
    km_markers = [None]  # (median, defined) per box position, None => skip
    for ph in phases:
        m = phase_all == ph
        if np.sum(m) >= 2:
            det = resid_all[m][~is_ul_all[m]]
            data_list.append(det if det.size else resid_all[m])
            n_lim = int(is_ul_all[m].sum())
            labels.append(f"Phase {ph}\n(n={int(m.sum())}" + (f", {n_lim} lim)" if n_lim else ")"))
            colors.append(pcol.get(ph, "gray"))
            groups.append((resid_all[m], is_ul_all[m]))
            km = km_left_censored(resid_all[m], is_ul_all[m])
            if np.isfinite(km["median"]):
                km_markers.append((km["median"], True))
            else:
                km_markers.append((float(np.median(resid_all[m])), False))  # upper bound

    positions = np.arange(len(data_list))
    bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True, notch=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    # KM medians (filled red diamond) / upper bounds (open diamond + down-arrow)
    km_done = bound_done = False
    for i, mk in enumerate(km_markers):
        if mk is None:
            continue
        val, defined = mk
        if defined:
            ax.scatter(
                i,
                val,
                marker="D",
                s=110,
                c="red",
                edgecolors="black",
                zorder=6,
                label=None if km_done else "KM median",
            )
            km_done = True
        else:
            ax.scatter(
                i,
                val,
                marker="D",
                s=110,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=6,
                label=None if bound_done else "KM median (upper bound)",
            )
            bound_done = True
            ax.annotate(
                "",
                xy=(i, val - 0.12),
                xytext=(i, val),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2),
                zorder=6,
            )

    rng = np.random.default_rng(0)
    det_done = lim_done = False
    for i, (data, ul) in enumerate(groups):
        x = rng.normal(i, 0.08, size=len(data))
        base_c = "black" if colors[i] == "white" else colors[i]
        det = ~ul
        if np.any(det):
            ax.scatter(
                x[det],
                data[det],
                alpha=0.6,
                s=40,
                c=base_c,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
                label=None if det_done else "Detections",
            )
            det_done = True
        if np.any(ul):
            ax.scatter(
                x[ul],
                data[ul],
                marker="v",
                s=90,
                facecolors="none",
                edgecolors=base_c,
                linewidths=1.6,
                zorder=4,
                label=None if lim_done else "Upper limits",
            )
            lim_done = True
            for xi, yi in zip(x[ul], data[ul]):
                ax.annotate(
                    "",
                    xy=(xi, yi - 0.10),
                    xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="-|>", color=base_c, lw=1.4),
                    zorder=4,
                )

    ax.axhline(0, color="black", lw=2, zorder=1)
    ax.axhline(scatter, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(-scatter, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
    ax.set_xlabel("Sample / HCG Phase", fontsize=22, labelpad=15)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8, width=1.2, pad=10)
    ax.tick_params(which="minor", length=4, width=1, pad=10)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.legend(loc="lower left", fontsize=14, frameon=True, framealpha=0.9)
    fig.tight_layout()

    for name in (FIGURE_NAME, VARIANT_NAME):
        out = FIGURES_DIR / name
        if out.exists():
            bak = FIGURES_DIR / name.replace(".pdf", "_sqrtbeam_backup.pdf")
            if not bak.exists():
                shutil.copy2(out, bak)
                print(f"[backup] {out.name} -> {bak.name}")
        fig.savefig(out, bbox_inches="tight", dpi=400)
        print(f"[saved] {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()
    print("=" * 72)
    print("Bmaj UPPER LIMITS for all HI-undetected HCG members + residual replot")
    print("=" * 72)
    prov = build_provenance(use_network=not args.no_network)
    write_augmented_csv(prov)
    replot(prov)
    print("\nDone. No fitted value and no existing data file was overwritten.")


if __name__ == "__main__":
    main()
