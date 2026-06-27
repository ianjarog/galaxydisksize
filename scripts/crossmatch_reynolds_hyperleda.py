from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier

# Repository-relative data directory (the Reynolds catalogue ships under data/).
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


CATALOG = "VII/237"


@dataclass(frozen=True)
class ParsedObject:
    object_id: str
    ra_deg: float
    dec_deg: float


def parse_reynolds_object_name(object_id: str) -> ParsedObject:
    name = object_id.strip()

    match_j = re.fullmatch(r"J(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})", name)
    if match_j:
        hh, mm, ss, sign, dd, dm, ds = match_j.groups()
        ra_deg = 15.0 * (int(hh) + int(mm) / 60.0 + int(ss) / 3600.0)
        dec_mag = int(dd) + int(dm) / 60.0 + int(ds) / 3600.0
        dec_deg = dec_mag if sign == "+" else -dec_mag
        return ParsedObject(name, ra_deg, dec_deg)

    match_g = re.fullmatch(r"g(\d{2})(\d{2})(\d{2})(\d)([+-])(\d{2})(\d{2})(\d{2})", name)
    if match_g:
        hh, mm, ss, tenth, sign, dd, dm, ds = match_g.groups()
        sec = int(ss) + int(tenth) / 10.0
        ra_deg = 15.0 * (int(hh) + int(mm) / 60.0 + sec / 3600.0)
        dec_mag = int(dd) + int(dm) / 60.0 + int(ds) / 3600.0
        dec_deg = dec_mag if sign == "+" else -dec_mag
        return ParsedObject(name, ra_deg, dec_deg)

    raise ValueError(f"Unsupported Reynolds object format: {object_id!r}")


def logd25_to_arcmin(logd25: float) -> float:
    return 0.1 * (10.0**logd25)


def arcmin_to_kpc(arcmin: float, distance_mpc: float) -> float:
    arcsec = arcmin * 60.0
    return distance_mpc * 1000.0 * math.radians(arcsec / 3600.0)


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
        row_limit=50,
    )


def row_value(row, *names: str) -> object:
    for name in names:
        if name in row.colnames:
            return row[name]
    return None


def choose_best_match(table, target: SkyCoord) -> dict[str, object] | None:
    if len(table) == 0:
        return None

    match_coords = SkyCoord(table["_RAJ2000"], table["_DEJ2000"], unit="deg", frame="icrs")
    separations = target.separation(match_coords).arcsec
    nearest_index = int(separations.argmin())
    row = table[nearest_index]

    return {
        "match_count": int(len(table)),
        "match_index": nearest_index,
        "separation_arcsec": float(separations[nearest_index]),
        "pgc": int(row_value(row, "PGC")) if row_value(row, "PGC") is not None else None,
        "hyperleda_objname": str(row_value(row, "objname", "Objname", "ANames") or "").strip(),
        "hyperleda_anames": str(row_value(row, "ANames") or "").strip(),
        "hyperleda_otype": str(row_value(row, "OType") or "").strip(),
        "hyperleda_mtype": str(row_value(row, "MType") or "").strip(),
        "hyperleda_logd25": float(row_value(row, "logD25"))
        if row_value(row, "logD25") is not None
        else math.nan,
        "hyperleda_logr25": float(row_value(row, "logR25"))
        if row_value(row, "logR25") is not None
        else math.nan,
        "hyperleda_ra_deg": float(row_value(row, "_RAJ2000")),
        "hyperleda_dec_deg": float(row_value(row, "_DEJ2000")),
    }


def query_hyperleda(
    parsed: ParsedObject, vizier: Vizier, radii_arcsec: list[float], sleep_s: float
) -> dict[str, object]:
    target = SkyCoord(parsed.ra_deg, parsed.dec_deg, unit="deg", frame="icrs")

    for radius_arcsec in radii_arcsec:
        result = vizier.query_region(target, radius=radius_arcsec * u.arcsec, catalog=CATALOG)
        if result:
            best = choose_best_match(result[0], target)
            if best is not None:
                best["query_radius_arcsec"] = float(radius_arcsec)
                best["match_status"] = "matched"
                return best
        if sleep_s > 0:
            time.sleep(sleep_s)

    return {
        "match_status": "unmatched",
        "query_radius_arcsec": float(radii_arcsec[-1]),
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
    }


def arcsec_to_kpc(arcsec: float, distance_mpc: float) -> float:
    return distance_mpc * 1000.0 * math.radians(arcsec / 3600.0)


def build_row(row, parsed: ParsedObject, match: dict[str, object]) -> dict[str, object]:
    distance_mpc = float(row["DISTANCE"])
    reynolds_log_mhi = float(row["lgMHI"]) if pd.notna(row["lgMHI"]) else math.nan
    # DIAMETER_HI / DIAMETER_R / DIAMETER_NUV in the source FITS are in arcsec
    # (Reynolds+22 Table A1). Keep both the raw arcsec values and the
    # distance-converted kpc values so downstream code can use either.
    reynolds_dhi_arcsec = float(row["DIAMETER_HI"]) if pd.notna(row["DIAMETER_HI"]) else math.nan
    reynolds_dr_arcsec = float(row["DIAMETER_R"]) if pd.notna(row["DIAMETER_R"]) else math.nan
    reynolds_dhi_kpc = (
        arcsec_to_kpc(reynolds_dhi_arcsec, distance_mpc)
        if pd.notna(reynolds_dhi_arcsec)
        else math.nan
    )
    reynolds_dr_kpc = (
        arcsec_to_kpc(reynolds_dr_arcsec, distance_mpc)
        if pd.notna(reynolds_dr_arcsec)
        else math.nan
    )

    hyperleda_logd25 = match["hyperleda_logd25"]
    hyperleda_d25_arcmin = (
        logd25_to_arcmin(hyperleda_logd25) if pd.notna(hyperleda_logd25) else math.nan
    )
    hyperleda_d25_kpc = (
        arcmin_to_kpc(hyperleda_d25_arcmin, distance_mpc)
        if pd.notna(hyperleda_d25_arcmin)
        else math.nan
    )

    return {
        "reynolds_object": parsed.object_id,
        "environment": str(row["ENVIRONMENT"]).strip(),
        "distance_mpc": distance_mpc,
        "reynolds_lgMHI": reynolds_log_mhi,
        "reynolds_diameter_hi_arcsec": reynolds_dhi_arcsec,
        "reynolds_diameter_r_arcsec": reynolds_dr_arcsec,
        "reynolds_diameter_hi_kpc": reynolds_dhi_kpc,
        "reynolds_diameter_r_kpc": reynolds_dr_kpc,
        "reynolds_ra_deg": parsed.ra_deg,
        "reynolds_dec_deg": parsed.dec_deg,
        **match,
        "hyperleda_d25_arcmin": hyperleda_d25_arcmin,
        "hyperleda_d25_kpc": hyperleda_d25_kpc,
    }


def crossmatch_table(
    input_path: Path,
    output_matches: Path,
    output_unmatched: Path,
    radii_arcsec: list[float],
    sleep_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = Table.read(input_path)
    vizier = make_vizier()

    rows: list[dict[str, object]] = []
    for idx, source_row in enumerate(table, start=1):
        parsed = parse_reynolds_object_name(str(source_row["OBJECTS"]))
        match = query_hyperleda(parsed, vizier, radii_arcsec=radii_arcsec, sleep_s=sleep_s)
        rows.append(build_row(source_row, parsed, match))
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(table)} Reynolds objects", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    matched = df[df["match_status"] == "matched"].copy()
    unmatched = df[df["match_status"] != "matched"].copy()

    matched.sort_values(["environment", "reynolds_object"]).to_csv(output_matches, index=False)
    unmatched.sort_values(["environment", "reynolds_object"]).to_csv(output_unmatched, index=False)
    return matched, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-match Reynolds WALLABY objects to HyperLEDA/VizieR and recover D25."
    )
    parser.add_argument(
        "--input",
        default=str(_DATA_DIR / "hi_detection_catalogue_reynolds.fits"),
        help="Input Reynolds FITS table.",
    )
    parser.add_argument(
        "--output-matches",
        default=str(_DATA_DIR / "reynolds_hyperleda_detection_matches.csv"),
        help="CSV for matched objects.",
    )
    parser.add_argument(
        "--output-unmatched",
        default=str(_DATA_DIR / "reynolds_hyperleda_detection_unmatched.csv"),
        help="CSV for unmatched objects.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.05,
        help="Pause between remote queries to be gentle on VizieR.",
    )
    args = parser.parse_args()

    radii_arcsec = [15.0, 30.0, 60.0]
    matches_path = Path(args.output_matches)
    unmatched_path = Path(args.output_unmatched)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)

    matched, unmatched = crossmatch_table(
        input_path=Path(args.input),
        output_matches=matches_path,
        output_unmatched=unmatched_path,
        radii_arcsec=radii_arcsec,
        sleep_s=args.sleep_s,
    )

    env_counts = matched["environment"].value_counts().to_dict()
    print(f"Matched {len(matched)} / {len(matched) + len(unmatched)} Reynolds objects")
    print(f"Environment counts among matches: {env_counts}")
    print(f"Wrote matches to: {matches_path}")
    print(f"Wrote unmatched to: {unmatched_path}")


if __name__ == "__main__":
    main()
