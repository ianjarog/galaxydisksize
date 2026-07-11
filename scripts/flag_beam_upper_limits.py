#!/usr/bin/env python3
"""Beam-size (Bmaj) upper limits for all HI-undetected HCG members.

Upper-limit definition
----------------------
1. The upper limit is the beam MAJOR AXIS (Bmaj), not the geometric mean
   sqrt(Bmaj*Bmin).  Applied to EVERY upper limit, including the Phase-3a/3c
   non-detections.
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

Run
---
    python flag_beam_upper_limits.py            # query CF3/HyperLEDA for the 48 new
    python flag_beam_upper_limits.py --no-network   # offline (needs cache)
"""

import argparse
import json
import math
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()
    print("=" * 72)
    print("Bmaj UPPER LIMITS for all HI-undetected HCG members")
    print("=" * 72)
    prov = build_provenance(use_network=not args.no_network)
    write_augmented_csv(prov)
    print("\nDone. No fitted value and no existing data file was overwritten.")


if __name__ == "__main__":
    main()
