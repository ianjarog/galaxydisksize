"""Cross-match the Bok+2020 ALFALFA pair sample against HyperLEDA for D_25.

Reads ``data/JAMIE-BOK/Mike_ALFALFApairs.txt``, dedupes by catID, and uses
the CDS Vizier ``VII/237`` (HyperLEDA-A1) catalogue to recover an optical
diameter for each pair member. ``catID`` strings are mixed (AGC, UGC, NGC,
IC, 2MASX, VGS, ...) so we try a small set of name normalisations and a
direct coordinate parse for 2MASXJ entries. Output CSVs follow the same
schema used for the Reynolds Hydra I cross-match.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

warnings.simplefilter("ignore")

CATALOG = "VII/237"
ALFALFA_CATALOG = "J/ApJ/861/49"
# Repository-relative defaults; the BOK pair list is a catalogue-tier input that
# the user supplies under data/JAMIE-BOK (override with --input).
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = str(_DATA_DIR / "JAMIE-BOK" / "Mike_ALFALFApairs.txt")
DEFAULT_OUT_DIR = str(_DATA_DIR)


@dataclass(frozen=True)
class PairRow:
    catID: str
    distance_mpc: float
    log_smass: float
    esmass: float
    log_himass: float


def parse_pair_table(path: Path) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 5:
                continue
            try:
                rows.append(
                    {
                        "catID": tokens[0],
                        "distance_mpc": float(tokens[1]),
                        "log_smass": float(tokens[2]),
                        "esmass": float(tokens[3]),
                        "log_himass": float(tokens[4]),
                    }
                )
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    n_raw = len(df)
    df_dedup = df.drop_duplicates(subset=["catID"], keep="first").reset_index(drop=True)
    n_dedup = len(df_dedup)
    return df_dedup, n_raw, n_dedup


def parse_2masx_coords(catID: str) -> tuple[float, float] | None:
    m = re.match(
        r"^2MASXJ(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})(\d?)$",
        catID,
    )
    if not m:
        return None
    hh, mm, ss, ss_dec, sign, dd, dm, ds, ds_dec = m.groups()
    ra_h = int(hh) + int(mm) / 60.0 + (int(ss) + int(ss_dec) / 100.0) / 3600.0
    ra_deg = 15.0 * ra_h
    dec_mag = int(dd) + int(dm) / 60.0 + (int(ds) + (int(ds_dec) / 10.0 if ds_dec else 0)) / 3600.0
    dec_deg = dec_mag if sign == "+" else -dec_mag
    return ra_deg, dec_deg


def name_candidates(catID: str) -> list[str]:
    cands: list[str] = [catID]
    # Strip leading underscore variants and zero-pad: "AGC005751" -> "AGC 5751"
    m = re.match(r"^([A-Za-z]+)[_\s]*0*(\d+)(.*)$", catID)
    if m:
        prefix, number, suffix = m.groups()
        cands.append(f"{prefix} {number}{suffix}")
    # NED01 / NED02 suffixes -> drop suffix when querying HyperLEDA
    if "_NED" in catID:
        cands.append(catID.split("_NED")[0])
    # Underscore -> space (UGC_01761 -> UGC 01761 -> UGC 1761 already covered)
    if "_" in catID and catID.replace("_", " ") not in cands:
        cands.append(catID.replace("_", " "))
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_vizier() -> Vizier:
    return Vizier(
        columns=[
            "_RAJ2000",
            "_DEJ2000",
            "PGC",
            "objname",
            "logD25",
            "logR25",
            "ANames",
            "OType",
            "MType",
        ],
        row_limit=20,
    )


def best_match_from_table(table, target: SkyCoord) -> dict | None:
    if table is None or len(table) == 0:
        return None
    match_coords = SkyCoord(table["_RAJ2000"], table["_DEJ2000"], unit="deg", frame="icrs")
    seps = target.separation(match_coords).arcsec
    idx = int(seps.argmin())
    row = table[idx]

    def _row_value(*names):
        for n in names:
            if n in row.colnames:
                return row[n]
        return None

    pgc_val = _row_value("PGC")
    pgc = int(pgc_val) if pgc_val is not None else None

    return {
        "match_count": int(len(table)),
        "match_index": idx,
        "separation_arcsec": float(seps[idx]),
        "pgc": pgc,
        "hyperleda_objname": str(_row_value("objname", "Objname", "ANames") or "").strip(),
        "hyperleda_anames": str(_row_value("ANames") or "").strip(),
        "hyperleda_otype": str(_row_value("OType") or "").strip(),
        "hyperleda_mtype": str(_row_value("MType") or "").strip(),
        "hyperleda_logd25": float(_row_value("logD25"))
        if _row_value("logD25") is not None
        else math.nan,
        "hyperleda_logr25": float(_row_value("logR25"))
        if _row_value("logR25") is not None
        else math.nan,
        "hyperleda_ra_deg": float(_row_value("_RAJ2000")),
        "hyperleda_dec_deg": float(_row_value("_DEJ2000")),
    }


def query_by_name(name: str, vizier: Vizier, radius_arcsec: float) -> dict | None:
    try:
        result = vizier.query_object(name, catalog=CATALOG, radius=radius_arcsec * u.arcsec)
    except Exception:
        return None
    if not result:
        return None
    table = result[0]
    if len(table) == 0:
        return None
    target_centre = SkyCoord(table["_RAJ2000"][0], table["_DEJ2000"][0], unit="deg", frame="icrs")
    return best_match_from_table(table, target_centre)


def query_by_coords(coords: tuple[float, float], vizier: Vizier, radii: list[float]) -> dict | None:
    target = SkyCoord(coords[0], coords[1], unit="deg", frame="icrs")
    for radius in radii:
        try:
            result = vizier.query_region(target, radius=radius * u.arcsec, catalog=CATALOG)
        except Exception:
            return None
        if not result:
            continue
        match = best_match_from_table(result[0], target)
        if match is None:
            continue
        match["query_radius_arcsec"] = float(radius)
        return match
    return None


def resolve_agc_via_alfalfa(catID: str, alfalfa_vizier: Vizier) -> tuple[float, float] | None:
    """Resolve an ``AGC######`` catID to (RA, Dec) using Haynes+2018 α.100."""
    m = re.match(r"^AGC0*(\d+)$", catID)
    if not m:
        return None
    agc_number = m.group(1)
    try:
        result = alfalfa_vizier.query_constraints(catalog=ALFALFA_CATALOG, AGC=agc_number)
    except Exception:
        return None
    if not result:
        return None
    table = result[0]
    if len(table) == 0:
        return None
    ra_str = str(table["RAJ2000"][0]).strip()
    dec_str = str(table["DEJ2000"][0]).strip()
    try:
        sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    except Exception:
        return None
    return float(sc.ra.deg), float(sc.dec.deg)


def resolve_one(catID: str, vizier: Vizier, alfalfa_vizier: Vizier, radii: list[float]) -> dict:
    # Strategy 1: parse 2MASX coordinates directly, then position-query HyperLEDA.
    coords = parse_2masx_coords(catID)
    if coords is not None:
        match = query_by_coords(coords, vizier, radii)
        if match is not None:
            match.setdefault("query_radius_arcsec", float(radii[0]))
            match["match_status"] = "matched"
            match["resolved_via"] = "2MASX_name"
            match["resolved_ra_deg"] = coords[0]
            match["resolved_dec_deg"] = coords[1]
            return match

    # Strategy 2: AGC -> RA/Dec via ALFALFA α.100 -> position-query HyperLEDA.
    agc_coords = resolve_agc_via_alfalfa(catID, alfalfa_vizier)
    if agc_coords is not None:
        match = query_by_coords(agc_coords, vizier, radii)
        if match is not None:
            match.setdefault("query_radius_arcsec", float(radii[0]))
            match["match_status"] = "matched"
            match["resolved_via"] = "AGC_alfalfa"
            match["resolved_ra_deg"] = agc_coords[0]
            match["resolved_dec_deg"] = agc_coords[1]
            return match

    # Strategy 3: SIMBAD name resolution via Vizier's query_object.
    for name in name_candidates(catID):
        match = query_by_name(name, vizier, radii[-1])
        if match is not None:
            match["query_radius_arcsec"] = float(radii[-1])
            match["match_status"] = "matched"
            match["resolved_via"] = f"name:{name}"
            match["resolved_ra_deg"] = agc_coords[0] if agc_coords else math.nan
            match["resolved_dec_deg"] = agc_coords[1] if agc_coords else math.nan
            return match

    return {
        "match_status": "unmatched",
        "query_radius_arcsec": float(radii[-1]),
        "match_count": 0,
        "match_index": None,
        "separation_arcsec": math.nan,
        "pgc": None,
        "hyperleda_objname": "",
        "hyperleda_anames": "",
        "hyperleda_otype": "",
        "hyperleda_mtype": "",
        "hyperleda_logd25": math.nan,
        "hyperleda_logr25": math.nan,
        "hyperleda_ra_deg": math.nan,
        "hyperleda_dec_deg": math.nan,
        "resolved_via": "none",
        "resolved_ra_deg": coords[0] if coords else math.nan,
        "resolved_dec_deg": coords[1] if coords else math.nan,
    }


def logd25_to_arcmin(logd25: float) -> float:
    return 0.1 * (10.0**logd25)


def arcmin_to_kpc(arcmin: float, distance_mpc: float) -> float:
    return distance_mpc * 1000.0 * math.radians(arcmin * 60.0 / 3600.0)


def build_row(pair_row: dict, match: dict) -> dict:
    distance_mpc = pair_row["distance_mpc"]
    logd25 = match.get("hyperleda_logd25", math.nan)
    if logd25 is not None and math.isfinite(logd25):
        d25_arcmin = logd25_to_arcmin(logd25)
        d25_kpc = arcmin_to_kpc(d25_arcmin, distance_mpc)
    else:
        d25_arcmin = math.nan
        d25_kpc = math.nan

    return {
        "catID": pair_row["catID"],
        "distance_mpc": distance_mpc,
        "log_smass": pair_row["log_smass"],
        "esmass": pair_row["esmass"],
        "log_himass": pair_row["log_himass"],
        **match,
        "hyperleda_d25_arcmin": d25_arcmin,
        "hyperleda_d25_kpc": d25_kpc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Mike_ALFALFApairs.txt path.")
    parser.add_argument(
        "--output-matches",
        default=str(Path(DEFAULT_OUT_DIR) / "bok_hyperleda_pair_matches.csv"),
    )
    parser.add_argument(
        "--output-unmatched",
        default=str(Path(DEFAULT_OUT_DIR) / "bok_hyperleda_pair_unmatched.csv"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N (0 = all).")
    parser.add_argument("--sleep-s", type=float, default=0.05, help="Sleep between Vizier queries.")
    parser.add_argument(
        "--radii-arcsec",
        type=str,
        default="15,30,60",
        help="Comma-separated search radii in arcsec.",
    )
    args = parser.parse_args()

    radii = [float(x) for x in args.radii_arcsec.split(",") if x]
    df, n_raw, n_dedup = parse_pair_table(Path(args.input))
    print(
        f"Read {n_raw} rows from {args.input}; "
        f"deduped to {n_dedup} unique catIDs (dropped {n_raw - n_dedup}).",
        flush=True,
    )

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
        print(f"  --limit {args.limit}: processing first {len(df)} only.")

    vizier = make_vizier()
    alfalfa_vizier = Vizier(
        catalog=ALFALFA_CATALOG,
        columns=["AGC", "RAJ2000", "DEJ2000"],
        row_limit=5,
    )
    rows: list[dict] = []
    for i, r in df.iterrows():
        match = resolve_one(r["catID"], vizier, alfalfa_vizier, radii)
        rows.append(build_row(r.to_dict(), match))
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
        if (i + 1) % 25 == 0:
            n_matched = sum(1 for x in rows if x["match_status"] == "matched")
            print(
                f"  Processed {i + 1}/{len(df)}  matched so far: {n_matched}",
                file=sys.stderr,
                flush=True,
            )

    out_df = pd.DataFrame(rows)
    matched = out_df[out_df["match_status"] == "matched"].copy()
    unmatched = out_df[out_df["match_status"] != "matched"].copy()
    matched_path = Path(args.output_matches)
    unmatched_path = Path(args.output_unmatched)
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    matched.sort_values("catID").to_csv(matched_path, index=False)
    unmatched.sort_values("catID").to_csv(unmatched_path, index=False)

    print()
    print(f"Total processed:  {len(out_df)}")
    print(f"  matched:        {len(matched)}")
    print(f"  unmatched:      {len(unmatched)}")
    print(f"Wrote: {matched_path}")
    print(f"Wrote: {unmatched_path}")


if __name__ == "__main__":
    main()
