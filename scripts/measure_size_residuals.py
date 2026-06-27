#!/usr/bin/env python3
"""
Extending AMIGA sample.

This script does the followiing:
1. Loads the Bayesian HI mass-size calibration fit on the combined
   AMIGA resolved + HCGs + MIGHTEE + Wang+16 sample, produced by
   scripts/plot_size_mass_all_surveys.py.
2. Uses Jones et al. (https://www.aanda.org/articles/aa/full_html/2018/01/aa31448-17/aa31448-17.html)
   AMIGA single-dish detections (science sample: isolated, complete,
   detection + quality flags OK, 399 galaxies) to infer additional D_HI
   values from that combined-sample calibration.
3. For the 16 Jones galaxies that overlap with the resolved AMIGA sample,
   the resolved (moment-map) HI diameter is used directly; only the
   remaining Jones-only galaxies are inferred from M_HI.
4. Combines the inferred Jones-only AMIGA galaxies with the resolved AMIGA sample.
5. Re-runs the residual analysis using the larger AMIGA baseline and OLS(Y|X).
6. Writes all figures and products to the dedicated analysis directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from scipy import stats
from scipy.odr import ODR, Model, RealData

try:
    from bces.bces import bces as bces_regression

    HAS_BCES = True
except Exception:
    HAS_BCES = False

try:
    import linmix

    HAS_LINMIX = True
except Exception:
    HAS_LINMIX = False

try:
    from hyperfit.linfit import LinFit

    HAS_HYPERFIT = True
except Exception:
    HAS_HYPERFIT = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_FIGURES_DIR = PROJECT_ROOT / "figures"
ANALYSIS_PRODUCTS_DIR = PROJECT_ROOT / "products"
ANALYSIS_LATEX_DIR = PROJECT_ROOT / "latex"

ORIGINAL_SCRIPT = SCRIPTS_DIR / "size_residual_baseline_engine.py"
PIPELINE_SCRIPT = SCRIPTS_DIR / "measure_hi_disk_sizes.py"

# CIGs flagged for exclusion from the Jones single-dish detections. These four
# galaxies appear as >3 sigma outliers below the Bayesian D_HI-D_25 baseline
# because their inferred D_HI (from single-dish M_HI) is more than an order of
# magnitude below the size-mass relation. They were dropped from the Jones
# science sample and we drop them here as well so that the enlarged AMIGA
# sample used for the D_HI-D_25 fit matches the upstream sample definition.
JONES_EXCLUDED_CIGS = frozenset({68, 402, 609, 1042})

MPLCONFIGDIR = ANALYSIS_PRODUCTS_DIR / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_directories() -> None:
    for directory in (
        ANALYSIS_FIGURES_DIR,
        ANALYSIS_PRODUCTS_DIR,
        ANALYSIS_LATEX_DIR,
        MPLCONFIGDIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def read_jones_whitespace_table(path: Path) -> pd.DataFrame:
    """Read Jones tables as whitespace-delimited text without FWF inference."""
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")


def parse_jones_mass_rows(path: Path) -> pd.DataFrame:
    """
    Parse the Jones mass table row-by-row.

    Some rows have missing interior fields, which makes generic whitespace or
    fixed-width parsing shift columns. Keep only rows with the full expected
    token count and plausible mass/error values.
    """
    rows: list[dict[str, float | int | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            raw_jones_key = int(parts[0])
            log_hi_mass = float(parts[5])
            log_hi_mass_err = float(parts[6])
            legacy_detection_flag = int(parts[9])
        except ValueError:
            continue
        if not (6.0 <= log_hi_mass <= 12.0):
            continue
        if not (0.0 <= log_hi_mass_err <= 1.0):
            continue
        rows.append(
            {
                "raw_jones_key": raw_jones_key,
                "log_hi_mass_jones": log_hi_mass,
                "log_hi_mass_err_jones": log_hi_mass_err,
                "legacy_detection_flag_jones": legacy_detection_flag,
            }
        )
    return pd.DataFrame(rows)


def parse_jones_d25_rows(path: Path, include_d25: bool) -> pd.DataFrame:
    """
    Parse the Jones D25 table row-by-row using only complete rows.

    Complete rows have 22 tokens in this file. Shorter rows indicate missing
    interior columns and should not be used for positional extraction.
    """
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 22:
            continue
        try:
            raw_jones_key = int(parts[0])
            distance_mpc = float(parts[4])
        except ValueError:
            continue
        if distance_mpc <= 0:
            continue

        row: dict[str, float | int] = {
            "raw_jones_key": raw_jones_key,
            "distance_mpc_jones": distance_mpc,
        }
        if include_d25:
            try:
                d25_raw = float(parts[16])
            except ValueError:
                continue
            if d25_raw <= 0:
                continue
            row["d25_raw_jones"] = d25_raw
        rows.append(row)
    return pd.DataFrame(rows)


def build_jones_cig_lookup(
    raw_indices: pd.Series | np.ndarray | list[int],
) -> tuple[dict[int, int], str]:
    """
    Build an explicit lookup from Jones table keys to public-facing CIG labels.

    For the current fixed-width Jones tables this will usually be the identity
    mapping. If a true zero-based sequence is encountered, shift it onto the
    standard one-based CIG numbering.
    """
    unique = sorted({int(value) for value in raw_indices if pd.notna(value)})
    if not unique:
        return {}, "empty"

    if unique[0] == 0:
        return {raw: raw + 1 for raw in unique}, "zero_based_plus_one"
    return {raw: raw for raw in unique}, "identity"


def load_jones_mass_table(path: Path) -> pd.DataFrame:
    """Read the Jones detections table with row-level validation."""
    subset = parse_jones_mass_rows(path)
    subset = subset.dropna(subset=["raw_jones_key", "log_hi_mass_jones"])
    subset["raw_jones_key"] = subset["raw_jones_key"].astype(int)
    cig_lookup, lookup_mode = build_jones_cig_lookup(subset["raw_jones_key"])
    subset["cig_index"] = subset["raw_jones_key"].map(cig_lookup)
    subset["jones_lookup_mode"] = lookup_mode
    # In the legacy extracted Jones mass file the trailing flag distinguishes
    # clean detections (0) from marginal detections (1). Keep only detections.
    if "legacy_detection_flag_jones" in subset.columns:
        subset = subset[subset["legacy_detection_flag_jones"] == 0].copy()
    return subset.drop_duplicates(subset="cig_index")


def parse_cds_tableb1_flags(path: Path) -> pd.DataFrame:
    """Parse the CDS tableb1 fixed-width file for detection and quality flags."""
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cig_text = line[0:4].strip()
        det_text = line[112:113].strip()
        qual_text = line[114:115].strip()
        if not cig_text:
            continue
        try:
            record = {"cig_index": int(cig_text)}
            if det_text:
                record["cds_det_flag"] = int(det_text)
            if qual_text:
                record["cds_qual_flag"] = int(qual_text)
        except ValueError:
            continue
        rows.append(record)
    return pd.DataFrame(rows)


def parse_cds_tableb3_flags(path: Path) -> pd.DataFrame:
    """Parse the CDS tableb3 fixed-width file for completeness/isolation flags."""
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cig_text = line[1:5].strip()
        iso_text = line[7:8].strip()
        comp_text = line[10:11].strip()
        if not cig_text:
            continue
        try:
            record = {"cig_index": int(cig_text)}
            if iso_text:
                record["cds_iso_flag"] = int(iso_text)
            if comp_text:
                record["cds_comp_flag"] = int(comp_text)
        except ValueError:
            continue
        rows.append(record)
    return pd.DataFrame(rows)


def apply_jones_sample_filters(
    jones_mass: pd.DataFrame,
    cds_tableb1_path: Path | None,
    cds_tableb3_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """
    Filter the Jones mass table to the intended science subset.

    If CDS tables are available, reproduce the Jones science-sample detection cut:
    detections only, isolated, complete, and quality-OK spectra.
    Otherwise fall back to the legacy extracted detections-only subset.
    """
    metadata: dict[str, int | str] = {
        "jones_mass_input_n": int(len(jones_mass)),
        "jones_filter_mode": "legacy_detections_only",
    }

    if cds_tableb1_path is None or cds_tableb3_path is None:
        metadata["jones_mass_filtered_n"] = int(len(jones_mass))
        return jones_mass.copy(), metadata
    if not cds_tableb1_path.exists() or not cds_tableb3_path.exists():
        metadata["jones_filter_mode"] = "legacy_detections_only_missing_cds_tables"
        metadata["jones_mass_filtered_n"] = int(len(jones_mass))
        return jones_mass.copy(), metadata

    cds_b1 = parse_cds_tableb1_flags(cds_tableb1_path)
    cds_b3 = parse_cds_tableb3_flags(cds_tableb3_path)
    cds = cds_b1.merge(cds_b3, on="cig_index", how="inner")

    filtered = jones_mass.merge(cds, on="cig_index", how="inner")
    metadata["jones_cds_overlap_n"] = int(len(filtered))

    filtered = filtered[
        (filtered["cds_det_flag"] == 0)
        & (filtered["cds_qual_flag"] == 0)
        & (filtered["cds_iso_flag"] == 0)
        & (filtered["cds_comp_flag"] == 0)
    ].copy()
    metadata["jones_filter_mode"] = "cds_science_sample_detection_only"
    metadata["jones_mass_filtered_n"] = int(len(filtered))
    return filtered, metadata


def load_jones_d25_table(path: Path, hi_module) -> tuple[pd.DataFrame, str]:
    """
    Read the Jones D25 table robustly as fixed-width text.

    The user pointed to column 17 (0-based 16), but the values in this file are not
    consistently encoded as RC2 logD25. When that column contains values > 3, they are
    physically consistent with major-axis arcminutes, not logD25. We therefore detect the
    representation from the data itself and record which interpretation was used.
    """
    if path.suffix.lower() == ".csv":
        csv_table = pd.read_csv(path, comment="#")
        required = {"CIG", "logd25", "E_logd25"}
        missing = required.difference(csv_table.columns)
        if missing:
            raise ValueError(f"CSV D25 file is missing required columns: {sorted(missing)}")

        subset = csv_table[["CIG", "logd25", "E_logd25"]].copy()
        subset.columns = ["cig_index", "logd25_jones", "e_logd25_jones"]
        subset = subset.dropna(subset=["cig_index", "logd25_jones"])
        subset["cig_index"] = subset["cig_index"].astype(int)
        subset = subset.drop_duplicates(subset="cig_index")
        subset = subset[subset["logd25_jones"] > 0].copy()

        distance_subset = parse_jones_d25_rows(
            DATA_DIR / "jones-detectionsd25.txt",
            include_d25=False,
        )
        distance_subset = distance_subset.dropna(subset=["raw_jones_key", "distance_mpc_jones"])
        distance_subset["raw_jones_key"] = distance_subset["raw_jones_key"].astype(int)
        cig_lookup, lookup_mode = build_jones_cig_lookup(distance_subset["raw_jones_key"])
        distance_subset["cig_index"] = distance_subset["raw_jones_key"].map(cig_lookup)
        distance_subset["jones_lookup_mode"] = lookup_mode
        distance_subset = distance_subset.drop_duplicates(subset="cig_index")
        distance_subset = distance_subset[distance_subset["distance_mpc_jones"] > 0].copy()

        subset = subset.merge(distance_subset, on="cig_index", how="inner")
        subset["optical_diameter_arcmin_jones"] = subset["logd25_jones"].apply(
            hi_module.logd25_to_arcmin
        )
        subset["optical_diameter_arcmin_err_jones"] = (
            np.log(10.0)
            * subset["optical_diameter_arcmin_jones"]
            * subset["e_logd25_jones"].fillna(0.0)
        )
        subset["optical_diameter_kpc_jones"] = subset.apply(
            lambda row: hi_module.arcsec_to_kpc(
                row["optical_diameter_arcmin_jones"] * 60.0,
                row["distance_mpc_jones"],
            ),
            axis=1,
        )
        subset["optical_diameter_err_kpc_jones"] = subset.apply(
            lambda row: hi_module.arcsec_to_kpc(
                row["optical_diameter_arcmin_err_jones"] * 60.0,
                row["distance_mpc_jones"],
            ),
            axis=1,
        )
        return subset, f"csv_logd25_with_errors_dictionary_{lookup_mode}"

    subset = parse_jones_d25_rows(path, include_d25=True)
    subset = subset.dropna(subset=["raw_jones_key", "distance_mpc_jones", "d25_raw_jones"])
    subset["raw_jones_key"] = subset["raw_jones_key"].astype(int)
    cig_lookup, lookup_mode = build_jones_cig_lookup(subset["raw_jones_key"])
    subset["cig_index"] = subset["raw_jones_key"].map(cig_lookup)
    subset["jones_lookup_mode"] = lookup_mode
    subset = subset.drop_duplicates(subset="cig_index")
    subset = subset[(subset["distance_mpc_jones"] > 0) & (subset["d25_raw_jones"] > 0)].copy()

    max_raw = float(subset["d25_raw_jones"].max())
    if max_raw > 3.0:
        subset["optical_diameter_arcmin_jones"] = subset["d25_raw_jones"].astype(float)
        d25_mode = "arcmin_major_axis"
    else:
        subset["optical_diameter_arcmin_jones"] = subset["d25_raw_jones"].apply(
            hi_module.logd25_to_arcmin
        )
        d25_mode = "logd25"

    subset["optical_diameter_kpc_jones"] = subset.apply(
        lambda row: hi_module.arcsec_to_kpc(
            row["optical_diameter_arcmin_jones"] * 60.0,
            row["distance_mpc_jones"],
        ),
        axis=1,
    )
    subset["optical_diameter_err_kpc_jones"] = np.nan
    return subset, f"{d25_mode}_dictionary_{lookup_mode}"


def load_bayesian_mass_size_fit(summary_path: Path) -> dict[str, float | int | str]:
    """Load the Bayesian HI mass-size calibration from the all-surveys fit.

    The fit is produced by scripts/plot_size_mass_all_surveys.py on the
    combined AMIGA resolved + HCGs + MIGHTEE + Wang+16 sample. It is used as
    the calibration to infer D_HI from M_HI for the Jones-only AMIGA galaxies.
    """
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Bayesian mass-size fit summary not found at {summary_path}. "
            "Run scripts/plot_size_mass_all_surveys.py first to produce it."
        )
    with open(summary_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "method": "Bayesian (AMIGA resolved + HCGs + MIGHTEE + Wang+16)",
        "source_label": data.get("dataset_label", "All surveys combined"),
        "slope": float(data["slope"]),
        "intercept": float(data["intercept"]),
        "scatter": float(data["scatter"]),
        "slope_p16": float(data["slope_p16"]),
        "slope_p84": float(data["slope_p84"]),
        "intercept_p16": float(data["intercept_p16"]),
        "intercept_p84": float(data["intercept_p84"]),
        "scatter_p16": float(data["scatter_p16"]),
        "scatter_p84": float(data["scatter_p84"]),
        "r_value": float(data["r_value"]),
        "r_squared": float(data["r_value"]) ** 2,
        "p_value": float(data["p_value"]),
        "n_total": int(data["n_total"]),
        "n_detections": int(data["n_detections"]),
        "n_upper_limits": int(data["n_upper_limits"]),
        "n_amiga": int(data.get("n_amiga", 0)),
        "n_hcg": int(data.get("n_hcg", 0)),
        "n_mightee": int(data.get("n_mightee", 0)),
        "n_wang16": int(data.get("n_wang16", 0)),
    }


def build_larger_amiga_sample(
    analyzer,
    jones_mass_path: Path,
    jones_d25_path: Path,
    cds_tableb1_path: Path | None,
    cds_tableb3_path: Path | None,
    hi_module,
    bayesian_fit_summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, float | int | str], str, dict[str, int | str]]:
    resolved = pd.read_csv(DATA_DIR / "isolated_galaxies_results.csv").copy()
    resolved["cig_index"] = resolved["cig_index"].astype(int)
    resolved["galaxy"] = resolved["galaxy"].astype(str)
    resolved["sample_origin"] = "resolved"
    if "log_stellar_mass" not in resolved.columns:
        resolved["log_stellar_mass"] = np.nan

    mass_size_fit = load_bayesian_mass_size_fit(bayesian_fit_summary_path)

    jones_mass = load_jones_mass_table(jones_mass_path)
    jones_mass, sample_metadata = apply_jones_sample_filters(
        jones_mass=jones_mass,
        cds_tableb1_path=cds_tableb1_path,
        cds_tableb3_path=cds_tableb3_path,
    )
    jones_d25, d25_mode = load_jones_d25_table(jones_d25_path, hi_module)
    jones = jones_mass.merge(jones_d25, on="cig_index", how="inner")
    jones = jones.dropna(
        subset=["log_hi_mass_jones", "optical_diameter_kpc_jones", "distance_mpc_jones"]
    )
    sample_metadata["jones_after_d25_merge_n"] = int(len(jones))

    resolved_cigs = set(resolved["cig_index"].tolist())
    jones_only = jones[~jones["cig_index"].isin(resolved_cigs)].copy()
    sample_metadata["jones_overlap_with_resolved_n"] = int(len(jones) - len(jones_only))

    jones_before_exclusion = len(jones_only)
    jones_only = jones_only[~jones_only["cig_index"].isin(JONES_EXCLUDED_CIGS)].copy()
    sample_metadata["jones_excluded_cigs"] = sorted(JONES_EXCLUDED_CIGS)
    sample_metadata["jones_excluded_n"] = int(jones_before_exclusion - len(jones_only))
    sample_metadata["jones_only_inferred_n"] = int(len(jones_only))

    predicted_log_dhi = mass_size_fit["intercept"] + mass_size_fit["slope"] * jones_only[
        "log_hi_mass_jones"
    ].to_numpy(float)
    predicted_sigma = np.sqrt(
        (mass_size_fit["slope"] * jones_only["log_hi_mass_err_jones"].fillna(0.0)) ** 2
        + mass_size_fit["scatter"] ** 2
    )
    predicted_dhi = 10**predicted_log_dhi
    predicted_dhi_err = np.log(10.0) * predicted_dhi * predicted_sigma
    predicted_hi_mass = 10 ** jones_only["log_hi_mass_jones"].to_numpy(float)

    inferred = pd.DataFrame(
        {
            "galaxy": [f"CIG{cig}" for cig in jones_only["cig_index"].astype(int)],
            "cig_index": jones_only["cig_index"].astype(int),
            "hi_diameter_kpc": predicted_dhi,
            "hi_diameter_err_kpc": predicted_dhi_err,
            "optical_diameter_kpc": jones_only["optical_diameter_kpc_jones"].to_numpy(float),
            "optical_diameter_err_kpc": jones_only["optical_diameter_err_kpc_jones"].to_numpy(
                float
            ),
            "hi_mass": predicted_hi_mass,
            "hi_mass_err": np.nan,
            "distance_mpc": jones_only["distance_mpc_jones"].to_numpy(float),
            "log_stellar_mass": np.full(len(jones_only), np.nan),
            "sample_origin": "inferred_jones",
        }
    )

    keep_cols = [
        "galaxy",
        "cig_index",
        "hi_diameter_kpc",
        "hi_diameter_err_kpc",
        "optical_diameter_kpc",
        "optical_diameter_err_kpc",
        "hi_mass",
        "hi_mass_err",
        "distance_mpc",
        "log_stellar_mass",
        "sample_origin",
    ]
    combined = pd.concat(
        [resolved[keep_cols], inferred[keep_cols]],
        ignore_index=True,
    )

    analyzer.amiga_data = {
        "D_HI": combined["hi_diameter_kpc"].to_numpy(float),
        "D_25": combined["optical_diameter_kpc"].to_numpy(float),
        "name": combined["galaxy"].to_numpy(dtype=str),
        "D_HI_err": combined["hi_diameter_err_kpc"].to_numpy(float),
        "hi_mass": combined["hi_mass"].to_numpy(float),
        "log_stellar_mass": combined["log_stellar_mass"].to_numpy(float),
        "sample_origin": combined["sample_origin"].to_numpy(dtype=str),
        "cig_index": combined["cig_index"].to_numpy(int),
    }
    analyzer.amiga_data["D_25_err"] = combined["optical_diameter_err_kpc"].to_numpy(float)

    return combined, mass_size_fit, d25_mode, sample_metadata


def plot_mass_size_calibration(
    original_module,
    analyzer,
    combined_df: pd.DataFrame,
    mass_size_fit: dict,
    output_file: str,
) -> None:
    fig, ax = original_module.plt.subplots(figsize=(8.5, 8.0))

    resolved_mask = combined_df["sample_origin"] == "resolved"
    inferred_mask = combined_df["sample_origin"] == "inferred_jones"

    resolved_log_mhi = np.log10(combined_df.loc[resolved_mask, "hi_mass"].to_numpy(float))
    resolved_log_dhi = np.log10(combined_df.loc[resolved_mask, "hi_diameter_kpc"].to_numpy(float))

    ax.scatter(
        resolved_log_mhi,
        resolved_log_dhi,
        s=75,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        label="Resolved AMIGA",
        zorder=3,
    )

    if inferred_mask.any():
        inferred_log_mhi = np.log10(combined_df.loc[inferred_mask, "hi_mass"].to_numpy(float))
        inferred_log_dhi = np.log10(
            combined_df.loc[inferred_mask, "hi_diameter_kpc"].to_numpy(float)
        )
        ax.scatter(
            inferred_log_mhi,
            inferred_log_dhi,
            s=45,
            c="#c9a227",
            marker="o",
            alpha=0.45,
            label="Jones-inferred AMIGA",
            zorder=2,
        )

    combined_log_mhi = np.log10(combined_df["hi_mass"].to_numpy(float))
    xfit = np.linspace(resolved_log_mhi.min(), combined_log_mhi.max(), 200)
    yfit = mass_size_fit["intercept"] + mass_size_fit["slope"] * xfit
    ax.plot(
        xfit,
        yfit,
        color="#1d5378",
        lw=2.6,
        label=(
            f"Bayesian fit (AMIGA+HCG+MIGHTEE+Wang+16): "
            f"slope={mass_size_fit['slope']:.3f}, "
            f"intercept={mass_size_fit['intercept']:.3f}"
        ),
        zorder=4,
    )
    ax.fill_between(
        xfit,
        yfit - mass_size_fit["scatter"],
        yfit + mass_size_fit["scatter"],
        color="0.7",
        alpha=0.2,
        zorder=1,
        label=f"±1σ intrinsic scatter ({mass_size_fit['scatter']:.3f} dex)",
    )

    ax.set_xlabel(r"$\log\,(M_{\rm HI} / M_\odot)$", fontsize=22, labelpad=15)
    ax.set_ylabel(r"$\log\,(D_{\rm HI} / {\rm kpc})$", fontsize=22, labelpad=15)
    analyzer._style_axes(ax)
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    original_module.plt.tight_layout()
    output_path = original_module._figure_output_path(output_file)
    original_module.plt.savefig(output_path, bbox_inches="tight", dpi=400)
    original_module.plt.close(fig)
    print(f"Saved: {output_path}")


def plot_larger_sample_baseline(
    original_module,
    analyzer,
    combined_df: pd.DataFrame,
    output_file: str,
) -> None:
    fig, ax = original_module.plt.subplots(figsize=(9.2, 8.2))
    resolved_mask = combined_df["sample_origin"] == "resolved"
    inferred_mask = combined_df["sample_origin"] == "inferred_jones"

    ax.scatter(
        combined_df.loc[inferred_mask, "optical_diameter_kpc"],
        combined_df.loc[inferred_mask, "hi_diameter_kpc"],
        s=40,
        c="#c9a227",
        alpha=0.35,
        label="Jones-inferred AMIGA only",
        zorder=1,
    )
    ax.scatter(
        combined_df.loc[resolved_mask, "optical_diameter_kpc"],
        combined_df.loc[resolved_mask, "hi_diameter_kpc"],
        s=75,
        facecolors="white",
        edgecolors="black",
        linewidths=1.4,
        label="Resolved AMIGA",
        zorder=3,
    )
    ax.scatter(
        analyzer.hcg_data["D_25"],
        analyzer.hcg_data["D_HI"],
        s=70,
        marker="s",
        facecolors="#4c78a8",
        edgecolors="black",
        alpha=0.55,
        label="HCG",
        zorder=2,
    )

    fit = analyzer.fit_results["amiga"]
    x_grid = np.geomspace(
        0.9 * min(combined_df["optical_diameter_kpc"].min(), analyzer.hcg_data["D_25"].min()),
        1.1 * max(combined_df["optical_diameter_kpc"].max(), analyzer.hcg_data["D_25"].max()),
        300,
    )
    log_grid = np.log10(x_grid)
    y_grid = 10 ** (fit["intercept"] + fit["slope"] * log_grid)
    sigma = fit["scatter"]

    ax.plot(x_grid, y_grid, color="#d95f02", lw=2.8, label="Larger-sample OLS baseline", zorder=4)
    ax.plot(
        x_grid,
        10 ** (fit["intercept"] + (fit["slope"] * log_grid) + sigma),
        color="0.5",
        lw=1.6,
        ls="--",
        zorder=1,
    )
    ax.plot(
        x_grid,
        10 ** (fit["intercept"] + (fit["slope"] * log_grid) - sigma),
        color="0.5",
        lw=1.6,
        ls="--",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D_{25}$ [kpc]", fontsize=22, labelpad=15)
    ax.set_ylabel(r"$D_{\rm HI}$ [kpc]", fontsize=22, labelpad=15)
    analyzer._style_axes(ax)
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    original_module.plt.tight_layout()
    output_path = original_module._figure_output_path(output_file)
    original_module.plt.savefig(output_path, bbox_inches="tight", dpi=400)
    original_module.plt.close(fig)
    print(f"Saved: {output_path}")


def write_summary(
    summary_path: Path,
    results: dict,
    combined_df: pd.DataFrame,
    mass_size_fit: dict,
    d25_mode: str,
    sample_metadata: dict[str, int | str],
) -> None:
    focused = results.get("focused", {})
    payload = {
        "fit_method": results["amiga"]["method"],
        "mass_size_calibration_method": mass_size_fit.get("method"),
        "mass_size_calibration_source": mass_size_fit.get("source_label"),
        "resolved_amiga_n": int(np.sum(combined_df["sample_origin"] == "resolved")),
        "jones_inferred_n": int(np.sum(combined_df["sample_origin"] == "inferred_jones")),
        "combined_amiga_n": int(len(combined_df)),
        "jones_d25_interpretation": d25_mode,
        "jones_filter_mode": sample_metadata.get("jones_filter_mode"),
        "jones_mass_input_n": sample_metadata.get("jones_mass_input_n"),
        "jones_mass_filtered_n": sample_metadata.get("jones_mass_filtered_n"),
        "jones_cds_overlap_n": sample_metadata.get("jones_cds_overlap_n"),
        "jones_after_d25_merge_n": sample_metadata.get("jones_after_d25_merge_n"),
        "jones_overlap_with_resolved_n": sample_metadata.get("jones_overlap_with_resolved_n"),
        "mass_size_fit": mass_size_fit,
        "larger_sample_baseline": {
            "slope": results["amiga"]["slope"],
            "intercept": results["amiga"]["intercept"],
            "scatter": results["amiga"]["scatter"],
        },
        "hcg_offset_dex": results["comparison"]["offset"],
        "hcg_size_fraction": 10 ** results["comparison"]["offset"],
        "mannwhitney_p": results["comparison"]["mw_p"],
        "ks_p": results["comparison"]["ks_p"],
        "focused_median_shift_dex": (
            focused.get("median_shift", [None])[0] if focused.get("median_shift") else None
        ),
        "focused_hcg_median_truncation_index": focused.get("T_hcg_median"),
        "cliffs_delta": focused.get("cliffs_delta"),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


_RESIDUAL_TREND_METHOD_ORDER = (
    "OLS(Y|X)",
    "Bayesian",
    "Theil-Sen",
    "Bisector",
    "ODR",
    "BCES(Y|X)",
    "York",
    "linmix (Kelly 2007)",
    "HYPER-FIT",
)

_RESIDUAL_TREND_METHOD_COLORS = {
    "OLS(Y|X)": "#1b9e77",
    "Bayesian": "#66a61e",
    "Theil-Sen": "#e7298a",
    "Bisector": "#d95f02",
    "ODR": "#7570b3",
    "BCES(Y|X)": "#1f78b4",
    "York": "#a6761d",
    "linmix (Kelly 2007)": "#e6ab02",
    "HYPER-FIT": "#666666",
}


def _fractional_to_log_error(values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    log_err = errors / (values * np.log(10.0))
    return np.where(np.isfinite(log_err), log_err, np.nan)


def _fit_ols(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float]:
    slope, intercept, _, _, _ = stats.linregress(log_x, log_y)
    return float(slope), float(intercept)


def _fit_theil_sen(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float]:
    slope, intercept, _, _ = stats.theilslopes(log_y, log_x, 0.95)
    return float(slope), float(intercept)


def _fit_bisector(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float]:
    slope_yx, _, _, _, _ = stats.linregress(log_x, log_y)
    slope_xy_inv, _, _, _, _ = stats.linregress(log_y, log_x)
    slope_xy = 1.0 / slope_xy_inv
    slope_bis = (
        (slope_yx * slope_xy - 1.0) + np.sqrt((1.0 + slope_yx**2) * (1.0 + slope_xy**2))
    ) / (slope_yx + slope_xy)
    intercept_bis = float(np.mean(log_y) - slope_bis * np.mean(log_x))
    return float(slope_bis), intercept_bis


def _fit_odr(
    log_x: np.ndarray, log_y: np.ndarray, log_x_err: np.ndarray, log_y_err: np.ndarray
) -> tuple[float, float]:
    x_err = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 1e-3), 0.05)
    y_err = np.where(np.isfinite(log_y_err), np.maximum(log_y_err, 1e-3), 0.05)

    def linear(beta, x_values):
        return beta[0] * x_values + beta[1]

    slope0, intercept0 = _fit_ols(log_x, log_y)
    odr = ODR(RealData(log_x, log_y, sx=x_err, sy=y_err), Model(linear), beta0=[slope0, intercept0])
    output = odr.run()
    return float(output.beta[0]), float(output.beta[1])


def _fit_bces(
    log_x: np.ndarray, log_y: np.ndarray, log_x_err: np.ndarray, log_y_err: np.ndarray
) -> tuple[float, float]:
    x_err = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 1e-6), 0.05)
    y_err = np.where(np.isfinite(log_y_err), np.maximum(log_y_err, 1e-6), 0.05)
    slopes, intercepts, _, _, _ = bces_regression(log_x, x_err, log_y, y_err, np.zeros_like(log_x))
    return float(slopes[0]), float(intercepts[0])


def _fit_york(
    log_x: np.ndarray,
    log_y: np.ndarray,
    log_x_err: np.ndarray,
    log_y_err: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, float]:
    sx = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 1e-6), 0.05)
    sy = np.where(np.isfinite(log_y_err), np.maximum(log_y_err, 1e-6), 0.05)
    r = np.zeros_like(log_x)
    slope, _ = _fit_ols(log_x, log_y)
    wx = 1.0 / sx**2
    wy = 1.0 / sy**2
    alpha = np.sqrt(wx * wy)
    x_bar = float(np.mean(log_x))
    y_bar = float(np.mean(log_y))
    for _ in range(max_iter):
        weights = wx * wy / (wx + slope**2 * wy - 2.0 * slope * r * alpha)
        x_bar = float(np.sum(weights * log_x) / np.sum(weights))
        y_bar = float(np.sum(weights * log_y) / np.sum(weights))
        u = log_x - x_bar
        v = log_y - y_bar
        beta = weights * ((u / wy) + slope * (v / wx) - ((slope * u + v) * r / alpha))
        new_slope = float(np.sum(weights * beta * v) / np.sum(weights * beta * u))
        if not np.isfinite(new_slope):
            break
        if abs(new_slope - slope) < tol:
            slope = new_slope
            break
        slope = new_slope
    intercept = float(y_bar - slope * x_bar)
    return float(slope), intercept


def _fit_linmix(
    log_x: np.ndarray,
    log_y: np.ndarray,
    log_x_err: np.ndarray,
    log_y_err: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    xsig = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 1e-6), 0.05)
    ysig = np.where(np.isfinite(log_y_err), np.maximum(log_y_err, 1e-6), 0.05)
    lm = linmix.LinMix(log_x, log_y, xsig=xsig, ysig=ysig, parallelize=False, seed=seed)
    lm.run_mcmc(miniter=800, maxiter=1600, silent=True)
    slope_med = float(np.median(lm.chain["beta"]))
    intercept_med = float(np.median(lm.chain["alpha"]))
    return slope_med, intercept_med


def _fit_hyperfit(
    log_x: np.ndarray,
    log_y: np.ndarray,
    log_x_err: np.ndarray,
    log_y_err: np.ndarray,
) -> tuple[float, float]:
    data = np.vstack([log_x, log_y])
    cov = np.zeros((2, 2, len(log_x)))
    cov[0, 0, :] = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 1e-6), 0.05) ** 2
    cov[1, 1, :] = np.where(np.isfinite(log_y_err), np.maximum(log_y_err, 1e-6), 0.05) ** 2
    fit = LinFit(data, cov, vertaxis=1)
    bounds = (
        (float(log_x.min() - 1.0), float(log_x.max() + 1.0)),
        (float(log_y.min() - 1.0), float(log_y.max() + 1.0)),
        (1e-6, 1.0),
    )
    coords, _, _ = fit.optimize(bounds=bounds, tol=1e-6, verbose=False)
    return float(coords[0]), float(coords[1])


def _fit_bayesian_emcee(
    log_x: np.ndarray,
    log_y: np.ndarray,
    log_y_err: np.ndarray,
    seed: int = 42,
    log_x_err: np.ndarray = None,
) -> dict[str, float | np.ndarray]:
    """Bayesian linear fit of log_y vs log_x with intrinsic scatter.

    Mirrors the prescription in scripts/plot_size_mass_all_surveys.py
    (emcee, uniform priors on slope, intercept, ln(intrinsic scatter)).

    Implements the Jones et al. (2018) Eq. 14 errors-in-variables likelihood in
    its marginalized (effective-variance) form: the x-uncertainty is projected
    onto y through the slope, var_eff = sigma_int^2 + sigma_y^2 + m^2 sigma_x^2.
    log_x_err=None recovers the previous y-only likelihood.
    """
    yerr = np.where(np.isfinite(log_y_err), np.maximum(log_y_err, 1e-6), 0.05)
    if log_x_err is None:
        xerr = np.zeros_like(np.asarray(log_x, float))
    else:
        xerr = np.where(np.isfinite(log_x_err), np.maximum(log_x_err, 0.0), 0.0)

    def log_prior(theta):
        m, b, lnf = theta
        if -5 < m < 5 and -10 < b < 10 and -10 < lnf < 1:
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        m, b, lnf = theta
        sig2 = np.exp(2.0 * lnf)
        mu = m * log_x + b
        var = np.clip(sig2 + yerr**2 + (m**2) * xerr**2, 1e-20, None)
        return -0.5 * np.sum((log_y - mu) ** 2 / var + np.log(2 * np.pi * var))

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim, nwalkers = 3, 50
    m0, b0 = np.polyfit(log_x, log_y, 1)
    rng = np.random.default_rng(seed)
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = m0 + 1e-4 * rng.standard_normal(nwalkers)
    pos[:, 1] = b0 + 1e-4 * rng.standard_normal(nwalkers)
    pos[:, 2] = -1.0 + 1e-4 * rng.standard_normal(nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(pos, 3000, progress=False)
    samples = sampler.get_chain(discard=800, thin=10, flat=True)
    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)
    return {
        "slope": float(p50[0]),
        "intercept": float(p50[1]),
        "sigma_int": float(np.exp(p50[2])),
        "slope_p16": float(p16[0]),
        "slope_p84": float(p84[0]),
        "intercept_p16": float(p16[1]),
        "intercept_p84": float(p84[1]),
        "sigma_int_p16": float(np.exp(p16[2])),
        "sigma_int_p84": float(np.exp(p84[2])),
        "samples": samples,
    }


def _run_single_method(
    method: str,
    log_x: np.ndarray,
    log_y: np.ndarray,
    log_x_err: np.ndarray,
    log_y_err: np.ndarray,
    seed: int,
) -> dict[str, float | np.ndarray | None]:
    extras: dict[str, float | np.ndarray | None] = {}
    if method == "OLS(Y|X)":
        slope, intercept = _fit_ols(log_x, log_y)
    elif method == "Bayesian":
        bayes = _fit_bayesian_emcee(log_x, log_y, log_y_err, seed=seed, log_x_err=log_x_err)
        slope, intercept = bayes["slope"], bayes["intercept"]
        extras.update(
            {
                "sigma_int": bayes["sigma_int"],
                "slope_p16": bayes["slope_p16"],
                "slope_p84": bayes["slope_p84"],
                "intercept_p16": bayes["intercept_p16"],
                "intercept_p84": bayes["intercept_p84"],
                "sigma_int_p16": bayes["sigma_int_p16"],
                "sigma_int_p84": bayes["sigma_int_p84"],
                "posterior_samples": bayes["samples"],
            }
        )
    elif method == "Theil-Sen":
        slope, intercept = _fit_theil_sen(log_x, log_y)
    elif method == "Bisector":
        slope, intercept = _fit_bisector(log_x, log_y)
    elif method == "ODR":
        slope, intercept = _fit_odr(log_x, log_y, log_x_err, log_y_err)
    elif method == "BCES(Y|X)":
        slope, intercept = _fit_bces(log_x, log_y, log_x_err, log_y_err)
    elif method == "York":
        slope, intercept = _fit_york(log_x, log_y, log_x_err, log_y_err)
    elif method == "linmix (Kelly 2007)":
        slope, intercept = _fit_linmix(log_x, log_y, log_x_err, log_y_err, seed=seed)
    elif method == "HYPER-FIT":
        slope, intercept = _fit_hyperfit(log_x, log_y, log_x_err, log_y_err)
    else:
        raise ValueError(f"Unknown fit method: {method}")

    residuals = log_y - (intercept + slope * log_x)
    scatter = float(np.std(residuals, ddof=2))
    resid_pearson = stats.pearsonr(log_x, residuals)
    resid_spearman = stats.spearmanr(log_x, residuals)
    resid_trend = stats.linregress(log_x, residuals)
    low_mask = log_x <= np.median(log_x)
    result = {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "scatter": scatter,
        "residual_pearson_r": float(resid_pearson.statistic),
        "residual_pearson_p": float(resid_pearson.pvalue),
        "residual_spearman_rho": float(resid_spearman.statistic),
        "residual_spearman_p": float(resid_spearman.pvalue),
        "residual_trend_slope": float(resid_trend.slope),
        "residual_trend_intercept": float(resid_trend.intercept),
        "low_bin_mean": float(np.mean(residuals[low_mask])),
        "high_bin_mean": float(np.mean(residuals[~low_mask])),
        "n": int(len(log_x)),
    }
    result.update(extras)
    return result


def _available_trend_methods() -> tuple[str, ...]:
    methods = []
    for method in _RESIDUAL_TREND_METHOD_ORDER:
        if method == "BCES(Y|X)" and not HAS_BCES:
            continue
        if method == "linmix (Kelly 2007)" and not HAS_LINMIX:
            continue
        if method == "HYPER-FIT" and not HAS_HYPERFIT:
            continue
        methods.append(method)
    return tuple(methods)


def _plot_residual_trends(
    plt_module,
    style_fn,
    combined_df: pd.DataFrame,
    method_results: dict[str, dict],
    output_path: Path,
) -> None:
    methods = list(method_results.keys())
    ncols = min(3, len(methods))
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt_module.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.6 * nrows), sharey=True)

    log_x = np.log10(combined_df["optical_diameter_kpc"].to_numpy(float))
    log_y = np.log10(combined_df["hi_diameter_kpc"].to_numpy(float))
    order = np.argsort(log_x)
    x_sorted = log_x[order]

    axes_arr = np.atleast_1d(axes).ravel()
    for ax, method in zip(axes_arr, methods):
        result = method_results[method]
        color = _RESIDUAL_TREND_METHOD_COLORS.get(method, "#555555")
        residuals = log_y - (result["intercept"] + result["slope"] * log_x)
        residuals_sorted = residuals[order]

        ax.scatter(
            log_x,
            residuals,
            s=70,
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            zorder=3,
        )
        ax.axhline(0.0, color="0.25", linewidth=2.0, linestyle="--", zorder=1)
        ax.axhline(result["scatter"], color="0.6", linewidth=1.6, linestyle=":", zorder=1)
        ax.axhline(-result["scatter"], color="0.6", linewidth=1.6, linestyle=":", zorder=1)

        trend_y = result["residual_trend_intercept"] + result["residual_trend_slope"] * x_sorted
        ax.plot(x_sorted, trend_y, color=color, linewidth=2.4, zorder=4)

        if len(x_sorted) >= 5:
            bin_edges = np.quantile(x_sorted, np.linspace(0, 1, 6))
            bin_edges[0] -= 1e-6
            bin_edges[-1] += 1e-6
            bin_ids = np.digitize(x_sorted, bin_edges[1:-1], right=False)
            x_med, y_med = [], []
            for bin_id in np.unique(bin_ids):
                mask = bin_ids == bin_id
                x_med.append(float(np.median(x_sorted[mask])))
                y_med.append(float(np.median(residuals_sorted[mask])))
            ax.plot(x_med, y_med, marker="o", markersize=6, color="black", linewidth=1.6, zorder=5)

        annotation = "\n".join(
            [
                method,
                rf"$\rho_{{\rm S}} = {result['residual_spearman_rho']:+.3f}$",
                rf"$p = {result['residual_spearman_p']:.3g}$",
                rf"$\langle \Delta \rangle_{{\rm low}} = {result['low_bin_mean']:+.3f}$",
                rf"$\langle \Delta \rangle_{{\rm high}} = {result['high_bin_mean']:+.3f}$",
            ]
        )
        ax.text(
            0.04,
            0.04,
            annotation,
            transform=ax.transAxes,
            fontsize=14,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        ax.set_xlabel(r"$\log(D_{25} / {\rm kpc})$", fontsize=20, labelpad=15)
        style_fn(ax)

    for ax in axes_arr[len(methods) :]:
        ax.set_visible(False)

    ylabel = r"$\Delta \log(D_{\rm HI})$ [dex]"
    for left_idx in (0, ncols, 2 * ncols):
        if left_idx < len(axes_arr):
            axes_arr[left_idx].set_ylabel(ylabel, fontsize=22, labelpad=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt_module.close(fig)


def run_residual_trend_audit(
    combined_df: pd.DataFrame,
    plt_module,
    style_fn,
    figure_output_path: Path,
    products_dir: Path,
    suffix: str = "_kelley_larger_sample_dictionary",
    random_seed: int = 42,
) -> dict[str, dict]:
    """Run the 9-method residual-trend audit on the larger AMIGA sample.

    The "Sigma-clipped OLS(Y|X)" slot of the legacy audit is replaced by a
    Bayesian emcee fit (same prescription as plot_size_mass_all_surveys.py).
    The combined resolved + Jones-inferred AMIGA sample is treated as a single
    population and plotted with one symbol per panel.
    """
    missing = [
        col
        for col in (
            "hi_diameter_kpc",
            "hi_diameter_err_kpc",
            "optical_diameter_kpc",
            "optical_diameter_err_kpc",
        )
        if col not in combined_df.columns
    ]
    if missing:
        raise ValueError(f"combined_df is missing required columns for audit: {missing}")

    dhi = combined_df["hi_diameter_kpc"].to_numpy(float)
    d25 = combined_df["optical_diameter_kpc"].to_numpy(float)
    dhi_err = combined_df["hi_diameter_err_kpc"].to_numpy(float)
    d25_err = combined_df["optical_diameter_err_kpc"].to_numpy(float)

    finite = np.isfinite(dhi) & np.isfinite(d25) & (dhi > 0) & (d25 > 0)
    dhi, d25, dhi_err, d25_err = dhi[finite], d25[finite], dhi_err[finite], d25_err[finite]
    combined_finite = combined_df.loc[finite].reset_index(drop=True).copy()

    log_x = np.log10(d25)
    log_y = np.log10(dhi)
    log_x_err = _fractional_to_log_error(d25, d25_err)
    log_y_err = _fractional_to_log_error(dhi, dhi_err)

    sample_pearson = stats.pearsonr(log_x, log_y)
    sample_spearman = stats.spearmanr(log_x, log_y)
    sample_stats = {
        "n": int(len(log_x)),
        "log_d25_min": float(np.min(log_x)),
        "log_d25_max": float(np.max(log_x)),
        "log_d25_median": float(np.median(log_x)),
        "log_dhi_min": float(np.min(log_y)),
        "log_dhi_max": float(np.max(log_y)),
        "log_dhi_median": float(np.median(log_y)),
        "pearson_r_logD25_logDHI": float(sample_pearson.statistic),
        "pearson_r_squared_logD25_logDHI": float(sample_pearson.statistic) ** 2,
        "pearson_p_logD25_logDHI": float(sample_pearson.pvalue),
        "spearman_rho_logD25_logDHI": float(sample_spearman.statistic),
        "spearman_p_logD25_logDHI": float(sample_spearman.pvalue),
    }

    method_results: dict[str, dict] = {}
    for method in _available_trend_methods():
        print(f"  Fitting method: {method}")
        payload = _run_single_method(method, log_x, log_y, log_x_err, log_y_err, seed=random_seed)
        if "slope_p16" in payload and "slope_p84" in payload:
            payload["slope_err_1sigma"] = float(0.5 * (payload["slope_p84"] - payload["slope_p16"]))
        if "intercept_p16" in payload and "intercept_p84" in payload:
            payload["intercept_err_1sigma"] = float(
                0.5 * (payload["intercept_p84"] - payload["intercept_p16"])
            )
        if "sigma_int_p16" in payload and "sigma_int_p84" in payload:
            payload["sigma_int_err_1sigma"] = float(
                0.5 * (payload["sigma_int_p84"] - payload["sigma_int_p16"])
            )
        method_results[method] = payload

    _plot_residual_trends(plt_module, style_fn, combined_finite, method_results, figure_output_path)

    serialisable: dict[str, dict] = {}
    for method, payload in method_results.items():
        serialisable[method] = {
            k: v
            for k, v in payload.items()
            if k != "posterior_samples" and not isinstance(v, np.ndarray)
        }
    products_dir.mkdir(parents=True, exist_ok=True)
    summary_path = products_dir / f"amiga_residual_trend_audit{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_n": int(len(combined_finite)),
                "resolved_n": int(np.sum(combined_finite["sample_origin"] == "resolved"))
                if "sample_origin" in combined_finite.columns
                else None,
                "inferred_n": int(np.sum(combined_finite["sample_origin"] == "inferred_jones"))
                if "sample_origin" in combined_finite.columns
                else None,
                "sample_statistics": sample_stats,
                "methods_available": list(method_results.keys()),
                "methods": serialisable,
            },
            handle,
            indent=2,
        )
    csv_path = products_dir / f"amiga_residual_trend_audit{suffix}.csv"
    pd.DataFrame(list(serialisable.values())).to_csv(csv_path, index=False)

    residuals_frame = combined_finite[
        [c for c in ("galaxy", "cig_index", "sample_origin") if c in combined_finite.columns]
    ].copy()
    residuals_frame["log_D25"] = log_x
    residuals_frame["log_DHI"] = log_y
    residuals_frame["log_D25_err"] = log_x_err
    residuals_frame["log_DHI_err"] = log_y_err
    for method, payload in method_results.items():
        col = f"resid_{method.replace('(', '').replace(')', '').replace('|', '').replace(' ', '_').replace('-', '_')}"
        residuals_frame[col] = log_y - (payload["intercept"] + payload["slope"] * log_x)
    residuals_path = products_dir / f"amiga_residuals_per_galaxy{suffix}.csv"
    residuals_frame.to_csv(residuals_path, index=False)

    report_path = products_dir / f"amiga_residual_trend_audit{suffix}.txt"
    n_resolved = int(
        np.sum(combined_finite.get("sample_origin", pd.Series(dtype=str)) == "resolved")
    )
    n_inferred = int(
        np.sum(combined_finite.get("sample_origin", pd.Series(dtype=str)) == "inferred_jones")
    )
    banner = "=" * 72
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Residual-trend audit on the enlarged AMIGA sample\n")
        handle.write(banner + "\n")
        handle.write(
            f"Sample: N = {sample_stats['n']} "
            f"(resolved = {n_resolved}, Jones-inferred = {n_inferred})\n"
        )
        handle.write(
            f"Pearson  r(logD25, logDHI) = "
            f"{sample_stats['pearson_r_logD25_logDHI']:+.4f}  "
            f"(r^2 = {sample_stats['pearson_r_squared_logD25_logDHI']:.4f}, "
            f"p = {sample_stats['pearson_p_logD25_logDHI']:.3g})\n"
        )
        handle.write(
            f"Spearman rho(logD25, logDHI) = "
            f"{sample_stats['spearman_rho_logD25_logDHI']:+.4f}  "
            f"(p = {sample_stats['spearman_p_logD25_logDHI']:.3g})\n"
        )
        handle.write(
            f"log(D25) range: [{sample_stats['log_d25_min']:.3f}, "
            f"{sample_stats['log_d25_max']:.3f}], "
            f"median = {sample_stats['log_d25_median']:.3f}\n"
        )
        handle.write(
            f"log(DHI) range: [{sample_stats['log_dhi_min']:.3f}, "
            f"{sample_stats['log_dhi_max']:.3f}], "
            f"median = {sample_stats['log_dhi_median']:.3f}\n\n"
        )
        header = (
            f"{'Method':24s} {'slope':>8s} {'s_err':>7s} {'intercept':>10s} {'i_err':>7s} "
            f"{'sigma_obs':>10s} {'sigma_int':>10s} {'rhoS':>8s} {'pS':>10s} "
            f"{'<low>':>8s} {'<high>':>8s}\n"
        )
        handle.write(header)
        handle.write("-" * len(header) + "\n")
        for method, payload in method_results.items():
            handle.write(
                f"{method:24s} "
                f"{payload['slope']:>8.4f} "
                f"{payload.get('slope_err_1sigma', float('nan')):>7.4f} "
                f"{payload['intercept']:>+10.4f} "
                f"{payload.get('intercept_err_1sigma', float('nan')):>7.4f} "
                f"{payload['scatter']:>10.4f} "
                f"{payload.get('sigma_int', float('nan')):>10.4f} "
                f"{payload['residual_spearman_rho']:>+8.4f} "
                f"{payload['residual_spearman_p']:>10.3g} "
                f"{payload['low_bin_mean']:>+8.4f} "
                f"{payload['high_bin_mean']:>+8.4f}\n"
            )
        handle.write(
            "\nColumn notes:\n"
            "  slope, intercept: fitted coefficients of log(D_HI) = slope * log(D_25) + intercept\n"
            "  s_err / i_err:    1-sigma uncertainty = 0.5 * (p84 - p16); NaN when the estimator\n"
            "                    returns only a point estimate (OLS, Theil-Sen, bisector, ODR,\n"
            "                    BCES, York, HYPER-FIT)\n"
            "  sigma_obs:        empirical std dev of residuals from the fit (dex)\n"
            "  sigma_int:        posterior median intrinsic scatter (Bayesian only; NaN otherwise)\n"
            "  rhoS, pS:         residual Spearman rank correlation with log(D_25) and its p-value\n"
            "  <low>, <high>:    mean residual in the low-/high-D_25 halves of the sample (dex)\n"
        )

    table_path, eq_path = _write_trend_test_latex_fragments(
        method_results=method_results,
        sample_stats=sample_stats,
        latex_dir=PROJECT_ROOT / "latex" / "autogen",
    )

    print(f"Residual-trend audit figure: {figure_output_path}")
    print(f"Residual-trend audit summary JSON: {summary_path}")
    print(f"Residual-trend audit summary CSV:  {csv_path}")
    print(f"Residual-trend audit text report: {report_path}")
    print(f"Per-galaxy residuals CSV:          {residuals_path}")
    print(f"LaTeX table fragment:              {table_path}")
    print(f"LaTeX equation fragment:           {eq_path}")
    print(
        "Sample correlation: "
        f"Pearson r={sample_stats['pearson_r_logD25_logDHI']:+.3f} "
        f"(r^2={sample_stats['pearson_r_squared_logD25_logDHI']:.3f}), "
        f"Spearman rho={sample_stats['spearman_rho_logD25_logDHI']:+.3f}"
    )

    return method_results


_TABLE_METHOD_LABELS = {
    "OLS(Y|X)": r"OLS($Y|X$)",
    "Bayesian": r"\textbf{Bayesian (adopted)}",
    "Theil-Sen": r"Theil--Sen",
    "Bisector": r"Bisector",
    "ODR": r"ODR",
    "BCES(Y|X)": r"BCES($Y|X$)",
    "York": r"York",
    "linmix (Kelly 2007)": r"\texttt{linmix}",
    "HYPER-FIT": r"\textsc{hyper-fit}",
}


def _fmt_p_value(p: float) -> str:
    """Render a p-value as either a plain decimal (p >= 1e-3) or scientific."""
    if not np.isfinite(p):
        return "$\\mathrm{nan}$"
    if p >= 1e-3:
        return f"${p:.3g}$"
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / 10**exponent
    return f"${mantissa:.1f}\\times 10^{{{exponent}}}$"


def _render_row(method: str, result: dict, bold: bool) -> str:
    label = _TABLE_METHOD_LABELS.get(method, method)
    wrap = (lambda s: rf"$\mathbf{{{s}}}$") if bold else (lambda s: f"${s}$")
    return (
        " & ".join(
            [
                label,
                wrap(f"{result['slope']:.3f}"),
                wrap(f"{result['intercept']:+.3f}"),
                wrap(f"{result['scatter']:.3f}"),
                wrap(f"{result['residual_spearman_rho']:+.3f}"),
                (
                    rf"$\mathbf{{{_fmt_p_value(result['residual_spearman_p'])[1:-1]}}}$"
                    if bold
                    else _fmt_p_value(result["residual_spearman_p"])
                ),
                wrap(f"{result['low_bin_mean']:+.3f}"),
                wrap(f"{result['high_bin_mean']:+.3f}"),
            ]
        )
        + r" \\"
    )


def _write_trend_test_latex_fragments(
    method_results: dict[str, dict],
    sample_stats: dict,
    latex_dir: Path,
) -> tuple[Path, Path]:
    """Write LaTeX fragments for table:trend_test and eq:baseline.

    The fragments are intended to be \\input into the main paper so that
    the table body and the adopted baseline equation stay in sync with
    the outputs of this script automatically.  Only the contents of the
    tabular (from \\toprule through \\bottomrule) and the equation body
    are auto-generated; the caption, tablefoot and surrounding prose
    remain under manual control in hi_disk_size_environments.tex.
    """
    latex_dir.mkdir(parents=True, exist_ok=True)

    passing = sorted(
        [
            (name, method_results[name])
            for name in method_results
            if method_results[name]["residual_spearman_p"] >= 0.05
        ],
        key=lambda kv: abs(kv[1]["residual_spearman_rho"]),
    )
    failing = sorted(
        [
            (name, method_results[name])
            for name in method_results
            if method_results[name]["residual_spearman_p"] < 0.05
        ],
        key=lambda kv: abs(kv[1]["residual_spearman_rho"]),
    )

    table_lines: list[str] = []
    table_lines.append(r"% Auto-generated by scripts/measure_size_residuals.py.")
    table_lines.append(r"% Regenerate by running that script; manual edits will be overwritten.")
    table_lines.append(
        r"% This file is a full tabular block; hi_disk_size_environments.tex wraps it with caption+tablefoot only."
    )
    table_lines.append(r"\begin{tabular}{lccccccc}")
    table_lines.append(r"\toprule \toprule")
    table_lines.append(
        r"Method & $\alpha$ & $\beta$ & $\sigma_{\rm obs}$ & $\rho_{\rm S}$ & "
        r"$p_{\rm S}$ & $\langle \Delta \rangle_{\rm low}$ & $\langle \Delta \rangle_{\rm high}$ \\"
    )
    table_lines.append(r"\midrule")
    table_lines.append(r"\multicolumn{8}{l}{\emph{Consistent with zero $D_{25}$ trend}}\\")
    for method, result in passing:
        table_lines.append(_render_row(method, result, bold=(method == "Bayesian")))
    table_lines.append(r"\midrule")
    table_lines.append(r"\multicolumn{8}{l}{\emph{Noticeable trend with $D_{25}$; not adopted}}\\")
    for method, result in failing:
        table_lines.append(_render_row(method, result, bold=False))
    table_lines.append(r"\bottomrule")
    table_lines.append(r"\end{tabular}")

    table_path = latex_dir / "table_trend_test.tex"
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(table_lines) + "\n")

    bayesian = method_results.get("Bayesian", {})
    slope = bayesian["slope"]
    slope_up = bayesian["slope_p84"] - slope
    slope_lo = slope - bayesian["slope_p16"]
    intercept = bayesian["intercept"]
    intercept_up = bayesian["intercept_p84"] - intercept
    intercept_lo = intercept - bayesian["intercept_p16"]
    eq_lines = [
        "% Auto-generated by scripts/measure_size_residuals.py.",
        "% Regenerate by running that script; manual edits will be overwritten.",
        rf"\log(D_{{\rm HI}}) = {slope:.3f}^{{+{slope_up:.3f}}}_{{-{slope_lo:.3f}}}\,\log(D_{{25}})",
        rf"                 {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}^{{+{intercept_up:.3f}}}_{{-{intercept_lo:.3f}}},",
    ]
    eq_path = latex_dir / "eq_baseline.tex"
    with open(eq_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(eq_lines) + "\n")

    return table_path, eq_path


# Surveys with a precisely characterised environment / interaction state, used
# for the "well-defined" sample comparison plot. AMIGA = isolated baseline,
# Pairs = Bok+20 ALFALFA-selected pairs (D_HI inferred from M_HI), HCGs =
# compact groups, VIVA = Virgo cluster, Ursa Major = group, Hydra I = cluster
# (Reynolds+22 WALLABY pilot).
WELL_DEFINED_ENVIRONMENT_SURVEYS = (
    "AMIGA",
    "Pairs (Bok+20)",
    "HCGs",
    "VIVA",
    "Ursa Major",
    "Hydra I",
)


def register_hydra_i_survey(analyzer, csv_path: Path) -> int:
    """Register Reynolds+22 Hydra I HI detections as a 'Hydra I' survey.

    Reads the matched-only CSV (HyperLEDA cross-match required for D_25).
    Unmatched Reynolds rows have no D_25 and are skipped.
    """
    df = pd.read_csv(csv_path)
    d_hi = df["reynolds_diameter_hi_kpc"].to_numpy(dtype=float)
    d_25 = df["hyperleda_d25_kpc"].to_numpy(dtype=float)
    mask = np.isfinite(d_hi) & np.isfinite(d_25) & (d_hi > 0) & (d_25 > 0)
    if not hasattr(analyzer, "surveys") or analyzer.surveys is None:
        analyzer.surveys = {}
    analyzer.surveys["Hydra I"] = {
        "D_HI": d_hi[mask],
        "D_25": d_25[mask],
    }
    n = int(np.sum(mask))
    print(f"Loaded Hydra I (Reynolds+22): {n} galaxies from {csv_path.name}")
    return n


def register_bok_pairs_survey(
    analyzer,
    csv_path: Path,
    mass_size_fit: dict,
    survey_name: str = "Pairs (Bok+20)",
) -> int:
    """Register the Bok+2020 ALFALFA pair sample with M_HI-inferred D_HI.

    The Bok+20 catalogue does not provide a measured HI diameter, so D_HI is
    inferred from log M_HI via the Bayesian HI mass-size relation calibrated
    on AMIGA + HCGs + MIGHTEE + Wang+16 -- the same recipe that produces
    inferred D_HI for the Jones-only AMIGA single-dish detections. D_25 comes
    from the HyperLEDA cross-match (rows with no logD25 are skipped).
    """
    df = pd.read_csv(csv_path)
    df = df[df["match_status"] == "matched"].copy()

    log_mhi = df["log_himass"].to_numpy(dtype=float)
    d_25 = df["hyperleda_d25_kpc"].to_numpy(dtype=float)
    mask = np.isfinite(log_mhi) & np.isfinite(d_25) & (log_mhi > 0) & (d_25 > 0)

    log_dhi = mass_size_fit["intercept"] + mass_size_fit["slope"] * log_mhi[mask]
    d_hi = 10**log_dhi

    if not hasattr(analyzer, "surveys") or analyzer.surveys is None:
        analyzer.surveys = {}
    analyzer.surveys[survey_name] = {
        "D_HI": d_hi,
        "D_25": d_25[mask],
    }
    n = int(np.sum(mask))
    print(
        f"Loaded {survey_name}: {n} galaxies from {csv_path.name} "
        "(D_HI inferred from M_HI via Bayesian mass-size relation)"
    )
    return n


def _build_hydra_environment_subsets(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Build per-environment (cluster / infall / field) Hydra I subsets from
    the Reynolds matched CSV. Same column conventions as register_hydra_i_survey.
    """
    df = pd.read_csv(csv_path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for env in ("cluster", "infall", "field"):
        sub = df[df["environment"] == env]
        d_hi = sub["reynolds_diameter_hi_kpc"].to_numpy(dtype=float)
        d_25 = sub["hyperleda_d25_kpc"].to_numpy(dtype=float)
        mask = np.isfinite(d_hi) & np.isfinite(d_25) & (d_hi > 0) & (d_25 > 0)
        out[env] = {"D_HI": d_hi[mask], "D_25": d_25[mask]}
    return out


HYDRA_SPLIT_WELL_DEFINED_SURVEYS = (
    "AMIGA",
    "Pairs (Bok+20)",
    "HCGs",
    "VIVA",
    "Ursa Major",
    "Hydra I (combined)",
    "Hydra I (cluster)",
    "Hydra I (infall)",
    "Hydra I (field)",
)

SURVEY_RESIDUALS_TABLE_REFS: dict[str, str] = {
    "AMIGA": r"this work",
    "Pairs (Bok+20)": r"\citet{2020MNRAS.499.3193B}",
    "HCGs": r"this work",
    "Ursa Major": r"\citet{2001A-A...370..765V}",
    "VIVA": r"\citet{2009AJ....138.1741C}",
    "Hydra I (combined)": r"\citet{2022MNRAS.510.1716R}",
    "Hydra I (cluster)": r"\citet{2022MNRAS.510.1716R}",
    "Hydra I (infall)": r"\citet{2022MNRAS.510.1716R}",
    "Hydra I (field)": r"\citet{2022MNRAS.510.1716R}",
}


def _write_survey_residuals_latex_fragment(
    survey_stats: dict,
    survey_order: tuple[str, ...],
    survey_refs: dict[str, str],
    output_path: Path,
) -> None:
    """Write the auto-generated body of table:survey_residuals.

    Rows for surveys present in ``survey_stats`` are sorted by mean residual
    in descending order (least truncated at the top, most truncated at the
    bottom), to match the ordering convention of the original aanda table.
    Surveys listed in ``survey_order`` but absent from ``survey_stats`` are
    silently skipped.
    """
    rows = []
    for name in survey_order:
        s = survey_stats.get(name)
        if s is None:
            continue
        rows.append(
            (
                name,
                int(s["N"]),
                float(s["median"]),
                float(s["scatter"]),
                float(s["f_severe_trunc"]),
                float(s["f_extended"]),
                survey_refs.get(name, ""),
            )
        )
    rows.sort(key=lambda r: -r[2])

    name_w = max((len(r[0]) for r in rows), default=20)
    lines = [
        r"% Auto-generated by scripts/measure_size_residuals.py.",
        r"% Regenerate by running that script; manual edits will be overwritten.",
        r"% This file is a full tabular block; hi_disk_size_environments.tex wraps it with caption+tablefoot only.",
        r"\begin{tabular}{lcccccc}",
        r"\toprule \toprule",
        r"Sample & $\mathrm{N_{gal}}$ & Median $\Delta$ & $\sigma$ & "
        r"$f_{\Delta < -\sigma}$ & $f_{\Delta > +\sigma}$ & Ref \\",
        r" & & [dex] & [dex] & [\%] & [\%] & \\",
        r"\midrule",
    ]
    for name, n, median, scatter, f_lo, f_hi, ref in rows:
        sign = "+" if median >= 0 else "-"
        lines.append(
            f"{name:<{name_w}s} & {n:>3d} & ${sign}{abs(median):.3f}$ & "
            f"{scatter:.3f} & {f_lo:5.1f} & {f_hi:5.1f} & {ref} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote survey-residuals fragment to {output_path}")


SURVEY_PAIRWISE_SHORT_LABELS = {
    "AMIGA": "AMIGA",
    "Pairs (Bok+20)": "Pairs",
    "HCGs": "HCGs",
    "Ursa Major": "UMa",
    "VIVA": "VIVA",
    "Hydra I (combined)": r"H\,I (comb)",
    "Hydra I (cluster)": r"H\,I (cl)",
    "Hydra I (infall)": r"H\,I (inf)",
    "Hydra I (field)": r"H\,I (fld)",
}


def _benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR correction. Returns (adjusted_p, reject_mask)."""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return [], []
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    raw = ranked * n / np.arange(1, n + 1)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(n)
    adj[order] = adj_sorted
    reject = adj < alpha
    return adj.tolist(), reject.tolist()


def _compute_pairwise_residual_tests(
    survey_stats: dict,
    survey_order: tuple[str, ...],
    n_boot: int = 5000,
    rng_seed: int = 12345,
) -> list[dict]:
    """For every (i, j) pair of surveys present in `survey_stats`, compute:
        * Δmedian = median_i - median_j (i taken from row, j from column)
        * Bootstrap 95% percentile CI on Δmedian (resample within sample)
        * Mann--Whitney U statistic and two-sided p-value
        * Cliff's δ effect size
    Returns one dict per pair, ordered (i < j) by survey_order index.
    """
    rng = np.random.default_rng(rng_seed)
    names = [n for n in survey_order if n in survey_stats and "residuals" in survey_stats[n]]
    results: list[dict] = []
    for ii, ni in enumerate(names):
        xi = np.asarray(survey_stats[ni]["residuals"], dtype=float)
        xi = xi[np.isfinite(xi)]
        for jj in range(ii + 1, len(names)):
            nj = names[jj]
            xj = np.asarray(survey_stats[nj]["residuals"], dtype=float)
            xj = xj[np.isfinite(xj)]

            med_diff = float(np.median(xi) - np.median(xj))

            try:
                u_stat, p = stats.mannwhitneyu(xi, xj, alternative="two-sided")
            except Exception:
                u_stat, p = float("nan"), float("nan")

            gt = float(np.sum(xi[:, None] > xj[None, :]))
            lt = float(np.sum(xi[:, None] < xj[None, :]))
            cliff = (gt - lt) / (len(xi) * len(xj)) if len(xi) and len(xj) else float("nan")

            boot = np.empty(n_boot)
            for k in range(n_boot):
                bi = rng.choice(xi, size=len(xi), replace=True)
                bj = rng.choice(xj, size=len(xj), replace=True)
                boot[k] = np.median(bi) - np.median(bj)
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

            results.append(
                {
                    "i": ni,
                    "j": nj,
                    "n_i": int(len(xi)),
                    "n_j": int(len(xj)),
                    "median_i": float(np.median(xi)),
                    "median_j": float(np.median(xj)),
                    "median_diff": med_diff,
                    "median_diff_ci_lo": float(ci_lo),
                    "median_diff_ci_hi": float(ci_hi),
                    "mannwhitney_U": float(u_stat),
                    "mannwhitney_p": float(p),
                    "cliff_delta": float(cliff),
                }
            )
    return results


def _write_pairwise_residual_outputs(
    pair_results: list[dict],
    survey_order: tuple[str, ...],
    json_path: Path,
    matrix_tex_path: Path,
) -> None:
    """Persist pair-test results as JSON and as a compact LaTeX matrix.

    The matrix is a 9x9 block: upper triangle holds the BH-adjusted p-value with
    a significance marker, lower triangle holds Δmedian = (row - col), diagonal
    is dashed.
    """
    raw_p = [r["mannwhitney_p"] for r in pair_results]
    adj_p, reject = _benjamini_hochberg(raw_p, alpha=0.05)
    for r, padj, rej in zip(pair_results, adj_p, reject):
        r["mannwhitney_p_bh"] = float(padj)
        r["bh_reject"] = bool(rej)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(pair_results, handle, indent=2)

    names_present = []
    seen: set[str] = set()
    for r in pair_results:
        for k in ("i", "j"):
            if r[k] not in seen:
                names_present.append(r[k])
                seen.add(r[k])
    names = [n for n in survey_order if n in seen]

    diff = {(n1, n2): None for n1 in names for n2 in names}
    p_bh = {(n1, n2): None for n1 in names for n2 in names}
    for r in pair_results:
        ni, nj = r["i"], r["j"]
        diff[(ni, nj)] = r["median_diff"]
        diff[(nj, ni)] = -r["median_diff"]
        p_bh[(ni, nj)] = r["mannwhitney_p_bh"]
        p_bh[(nj, ni)] = r["mannwhitney_p_bh"]

    def _p_cell(p: float) -> str:
        if p is None or not np.isfinite(p):
            return "---"
        if p < 1e-3:
            text = r"$<\!10^{-3}$"
            star = r"^{\ast\ast\ast}"
        elif p < 1e-2:
            text = f"${p:.3f}$"
            star = r"^{\ast\ast}"
        elif p < 0.05:
            text = f"${p:.3f}$"
            star = r"^{\ast}"
        elif p < 0.10:
            text = f"${p:.3f}$"
            star = r"^{\dagger}"
        else:
            text = f"${p:.2f}$"
            star = ""
        return text + (rf" ${star}$" if star else "")

    short = SURVEY_PAIRWISE_SHORT_LABELS
    n = len(names)
    col_spec = "l" + "c" * n
    lines = [
        r"% Auto-generated by scripts/measure_size_residuals.py.",
        r"% Regenerate by running that script; manual edits will be overwritten.",
        r"% This file is a full tabular block (with \resizebox); the wrapping document supplies caption+tablefoot.",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule \toprule",
        " & ".join([""] + [short.get(nm, nm) for nm in names]) + r" \\",
        r"\midrule",
    ]
    for i, ni in enumerate(names):
        cells = [short.get(ni, ni)]
        for j, nj in enumerate(names):
            if i == j:
                cells.append(r"---")
            elif i < j:
                cells.append(_p_cell(p_bh[(ni, nj)]))
            else:
                d = diff[(ni, nj)]
                if d is None:
                    cells.append("---")
                else:
                    sign = "+" if d >= 0 else "-"
                    cells.append(f"${sign}{abs(d):.3f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")

    matrix_tex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(matrix_tex_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote pairwise JSON to {json_path}")
    print(f"Wrote pairwise matrix fragment to {matrix_tex_path}")


def plot_hydra_split_well_defined_residual(
    analyzer,
    hydra_csv_path: Path,
    figure_output_file: str,
    products_pdf_path: Path,
    figures_dir: Path,
) -> None:
    """Generate a well-defined-environment residual plot in which the Hydra I
    sample is split into combined / cluster / infall / field subsets.

    The figure is saved through the analyzer's standard plot machinery (i.e.
    in ``figures_dir``), and a copy is placed at ``products_pdf_path`` so it
    can be embedded directly in the standalone consistency-test report.
    """
    full_surveys = analyzer.surveys
    full_stats = getattr(analyzer, "survey_stats", None)

    if "Hydra I" not in full_surveys:
        print("  [warn] 'Hydra I' not in analyzer.surveys; skipping split plot.")
        return

    subsets = _build_hydra_environment_subsets(hydra_csv_path)
    derived = {
        "Hydra I (combined)": full_surveys["Hydra I"],
        "Hydra I (cluster)": subsets["cluster"],
        "Hydra I (infall)": subsets["infall"],
        "Hydra I (field)": subsets["field"],
    }
    selected: dict[str, dict] = {}
    for name in HYDRA_SPLIT_WELL_DEFINED_SURVEYS:
        if name in full_surveys:
            selected[name] = full_surveys[name]
        elif name in derived:
            selected[name] = derived[name]

    analyzer.surveys = selected
    try:
        analyzer.compute_survey_residuals()
        analyzer.plot_survey_mean_residual(
            rank_metric="median",
            output_file=figure_output_file,
            show=False,
        )
        # While the well-defined-plus-Hydra-split survey_stats are alive,
        # write the LaTeX fragment that backs table:survey_residuals.
        _write_survey_residuals_latex_fragment(
            analyzer.survey_stats,
            HYDRA_SPLIT_WELL_DEFINED_SURVEYS,
            SURVEY_RESIDUALS_TABLE_REFS,
            ANALYSIS_LATEX_DIR / "autogen" / "table_survey_residuals.tex",
        )
        # Pairwise statistical separation between the well-defined samples.
        pair_results = _compute_pairwise_residual_tests(
            analyzer.survey_stats,
            HYDRA_SPLIT_WELL_DEFINED_SURVEYS,
        )
        _write_pairwise_residual_outputs(
            pair_results,
            HYDRA_SPLIT_WELL_DEFINED_SURVEYS,
            ANALYSIS_PRODUCTS_DIR / "well_defined_pairwise_residuals.json",
            ANALYSIS_LATEX_DIR / "autogen" / "table_pairwise_residuals.tex",
        )
        # Frac-truncated and frac-extended bar plots over the same
        # well-defined-plus-Hydra-split subset; filenames derived from
        # the median figure_output_file by swapping the metric stem.
        out_name = Path(figure_output_file).name
        frac_trunc_file = out_name.replace("median_residual", "frac_truncated")
        frac_ext_file = out_name.replace("median_residual", "frac_extended")
        analyzer.plot_survey_frac_truncated(output_file=frac_trunc_file, show=False)
        analyzer.plot_survey_frac_extended(output_file=frac_ext_file, show=False)
    finally:
        analyzer.surveys = full_surveys
        if full_stats is not None:
            analyzer.survey_stats = full_stats

    src = figures_dir / Path(figure_output_file).name
    products_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(src, products_pdf_path)
    print(f"Copied Hydra-split well-defined plot to {products_pdf_path}")


def plot_well_defined_subset_median_residual(
    analyzer,
    output_file: str,
    survey_names: tuple[str, ...] = WELL_DEFINED_ENVIRONMENT_SURVEYS,
) -> None:
    """Generate the median-residual plot for a well-defined-environment subset.

    Snapshots the analyzer's full surveys/survey_stats, restricts to the
    requested subset, recomputes residuals, calls plot_survey_mean_residual,
    then restores the originals.
    """
    full_surveys = analyzer.surveys
    full_stats = getattr(analyzer, "survey_stats", None)

    available = [name for name in survey_names if name in full_surveys]
    missing = [name for name in survey_names if name not in full_surveys]
    if missing:
        print(f"  [warn] missing surveys for well-defined subset: {missing}")
    if not available:
        print("  [warn] no well-defined surveys available, skipping plot.")
        return

    analyzer.surveys = {name: full_surveys[name] for name in available}
    try:
        analyzer.compute_survey_residuals()
        analyzer.plot_survey_mean_residual(
            rank_metric="median",
            output_file=output_file,
            show=False,
        )
        # Also produce frac_truncated and frac_extended bar plots over the
        # same well-defined-environment subset.  Filenames are derived from
        # the median-residual output_file by swapping the metric stem.
        out_name = Path(output_file).name
        frac_trunc_file = out_name.replace("median_residual", "frac_truncated")
        frac_ext_file = out_name.replace("median_residual", "frac_extended")
        analyzer.plot_survey_frac_truncated(output_file=frac_trunc_file, show=False)
        analyzer.plot_survey_frac_extended(output_file=frac_ext_file, show=False)
    finally:
        analyzer.surveys = full_surveys
        if full_stats is not None:
            analyzer.survey_stats = full_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the larger-sample AMIGA baseline analysis."
    )
    parser.add_argument(
        "--amiga-file",
        default=str(DATA_DIR / "isolated_galaxies_results.csv"),
        help="Resolved AMIGA CSV file.",
    )
    parser.add_argument(
        "--hcg-file",
        default=str(DATA_DIR / "interacting_galaxies_results.csv"),
        help="HCG CSV file.",
    )
    parser.add_argument(
        "--jones-mass-file",
        default=str(DATA_DIR / "jones-detections.txt"),
        help="Jones detections file containing log HI masses.",
    )
    parser.add_argument(
        "--jones-d25-file",
        default=str(DATA_DIR / "jones-detectionsd25.txt"),
        help="Jones detections file containing D25 and distance information.",
    )
    parser.add_argument(
        "--jones-cds-tableb1",
        default=str(DATA_DIR / "tableb1.dat"),
        help="Optional CDS tableb1.dat for Jones detection and quality flags.",
    )
    parser.add_argument(
        "--jones-cds-tableb3",
        default=str(DATA_DIR / "tableb3.dat"),
        help="Optional CDS tableb3.dat for Jones isolation and completeness flags.",
    )
    parser.add_argument(
        "--bayesian-fit-summary",
        default=str(ANALYSIS_PRODUCTS_DIR / "mass_size_relation_all_surveys_summary.json"),
        help=(
            "Path to the Bayesian mass-size fit summary JSON produced by "
            "scripts/plot_size_mass_all_surveys.py on the combined "
            "AMIGA + HCGs + MIGHTEE + Wang+16 sample."
        ),
    )
    parser.add_argument(
        "--report-file",
        default=str(ANALYSIS_PRODUCTS_DIR / "analysis_report_kelley_larger_sample.txt"),
        help="Text report path.",
    )
    parser.add_argument(
        "--summary-file",
        default=str(ANALYSIS_PRODUCTS_DIR / "analysis_summary_kelley_larger_sample.json"),
        help="JSON summary path.",
    )
    parser.add_argument(
        "--font-dir",
        action="append",
        default=[d for d in [os.environ.get("GALAXYDISKSIZE_FONT_DIR")] if d],
        help="Custom font directory. Default matches the original script.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Do not echo the report to the console.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    original_module = load_module(
        ORIGINAL_SCRIPT,
        "residual_analysis_with_surveys_star_final_original_for_larger_sample",
    )
    pipeline_module = load_module(
        PIPELINE_SCRIPT,
        "measure_hi_disk_sizes_for_larger_sample",
    )

    original_module.FIGURE_OUTPUT_DIR = str(ANALYSIS_FIGURES_DIR)

    with original_module._redirect_output(
        args.report_file,
        echo_to_console=(not args.no_console),
    ):
        analyzer = original_module.CorrelationAnalysis(font_dirs=args.font_dir)
        analyzer.load_data(amiga_file=args.amiga_file, hcg_file=args.hcg_file)

        combined_df, mass_size_fit, d25_mode, sample_metadata = build_larger_amiga_sample(
            analyzer=analyzer,
            jones_mass_path=Path(args.jones_mass_file),
            jones_d25_path=Path(args.jones_d25_file),
            cds_tableb1_path=Path(args.jones_cds_tableb1),
            cds_tableb3_path=Path(args.jones_cds_tableb3),
            hi_module=pipeline_module,
            bayesian_fit_summary_path=Path(args.bayesian_fit_summary),
        )

        print("=" * 60)
        print("KELLEY LARGER-SAMPLE ANALYSIS")
        print("=" * 60)
        print(f"Resolved AMIGA sample: {np.sum(combined_df['sample_origin'] == 'resolved')}")
        print(
            f"Jones-inferred AMIGA sample: {np.sum(combined_df['sample_origin'] == 'inferred_jones')}"
        )
        print(f"Combined larger AMIGA baseline sample: {len(combined_df)}")
        print(f"Jones filter mode: {sample_metadata['jones_filter_mode']}")
        print(f"Jones mass rows before filter: {sample_metadata['jones_mass_input_n']}")
        print(f"Jones mass rows after filter: {sample_metadata['jones_mass_filtered_n']}")
        if sample_metadata.get("jones_cds_overlap_n") is not None:
            print(f"Jones rows with CDS flags: {sample_metadata['jones_cds_overlap_n']}")
        print(f"Jones rows after D25 merge: {sample_metadata['jones_after_d25_merge_n']}")
        print(
            f"Jones rows overlapping resolved sample: {sample_metadata['jones_overlap_with_resolved_n']}"
        )
        print(f"Jones D25 interpretation used: {d25_mode}")
        print(
            "\nMass-size calibration source: "
            f"{mass_size_fit.get('source_label', 'combined surveys')}"
        )
        print(f"Calibration method: {mass_size_fit['method']}")
        print(
            "  log(D_HI) = "
            f"{mass_size_fit['slope']:.3f} * log(M_HI) + {mass_size_fit['intercept']:.3f}"
        )
        print(
            "  Slope  16/50/84: "
            f"{mass_size_fit['slope_p16']:.4f}, "
            f"{mass_size_fit['slope']:.4f}, "
            f"{mass_size_fit['slope_p84']:.4f}"
        )
        print(
            "  Intercept  16/50/84: "
            f"{mass_size_fit['intercept_p16']:.4f}, "
            f"{mass_size_fit['intercept']:.4f}, "
            f"{mass_size_fit['intercept_p84']:.4f}"
        )
        print(
            "  Scatter  16/50/84: "
            f"{mass_size_fit['scatter_p16']:.4f}, "
            f"{mass_size_fit['scatter']:.4f}, "
            f"{mass_size_fit['scatter_p84']:.4f} dex"
        )
        print(
            f"  Calibration sample: N={mass_size_fit['n_total']} "
            f"(AMIGA={mass_size_fit['n_amiga']}, HCG={mass_size_fit['n_hcg']}, "
            f"MIGHTEE={mass_size_fit['n_mightee']}, Wang+16={mass_size_fit['n_wang16']})"
        )

        analyzer.load_cig_stellar_masses(str(DATA_DIR / "angmom_final.csv"))
        results = analyzer.run_full_analysis(method="ols")

        # Replace the analyzer's OLS baseline with the adopted Bayesian emcee
        # fit (same prescription as plot_size_mass_all_surveys.py) so every
        # downstream plot and diagnostic uses the adopted baseline. Residuals
        # for AMIGA and HCG are recomputed against the Bayesian fit.
        print("\n" + "=" * 60)
        print("OVERRIDING BASELINE WITH ADOPTED BAYESIAN FIT")
        print("=" * 60)
        _amiga_d25 = np.asarray(analyzer.amiga_data["D_25"], dtype=float)
        _amiga_dhi = np.asarray(analyzer.amiga_data["D_HI"], dtype=float)
        _finite_mask = (
            np.isfinite(_amiga_d25) & np.isfinite(_amiga_dhi) & (_amiga_d25 > 0) & (_amiga_dhi > 0)
        )
        _log_d25 = np.log10(_amiga_d25[_finite_mask])
        _log_dhi = np.log10(_amiga_dhi[_finite_mask])
        _log_dhi_err_arr = analyzer.amiga_data.get("D_HI_err", None)
        if _log_dhi_err_arr is not None:
            _log_dhi_err_arr = np.asarray(_log_dhi_err_arr, dtype=float)[_finite_mask]
            _log_dhi_err = _fractional_to_log_error(_amiga_dhi[_finite_mask], _log_dhi_err_arr)
        else:
            _log_dhi_err = np.full_like(_log_d25, 0.05)
        # x-errors (log D25) for the Jones+2018 Eq. 14 errors-in-variables term,
        # so the adopted residual baseline matches the Table 4 Bayesian estimator.
        _d25_err_arr = analyzer.amiga_data.get("D_25_err", None)
        if _d25_err_arr is not None:
            _d25_err_arr = np.asarray(_d25_err_arr, dtype=float)[_finite_mask]
            _log_d25_err = _fractional_to_log_error(_amiga_d25[_finite_mask], _d25_err_arr)
        else:
            _log_d25_err = None
        _bayes = _fit_bayesian_emcee(
            _log_d25, _log_dhi, _log_dhi_err, seed=42, log_x_err=_log_d25_err
        )
        _bayes_residuals_amiga = _log_dhi - (_bayes["intercept"] + _bayes["slope"] * _log_d25)
        _bayes_scatter_obs = float(np.std(_bayes_residuals_amiga, ddof=2))
        _pearson = stats.pearsonr(_log_d25, _log_dhi)
        analyzer.fit_results["amiga"] = {
            "method": "Bayesian (emcee)",
            "slope": float(_bayes["slope"]),
            "intercept": float(_bayes["intercept"]),
            "scatter": _bayes_scatter_obs,
            "sigma_int": float(_bayes["sigma_int"]),
            "slope_p16": float(_bayes["slope_p16"]),
            "slope_p84": float(_bayes["slope_p84"]),
            "slope_err_1sigma": float(0.5 * (_bayes["slope_p84"] - _bayes["slope_p16"])),
            "intercept_p16": float(_bayes["intercept_p16"]),
            "intercept_p84": float(_bayes["intercept_p84"]),
            "intercept_err_1sigma": float(
                0.5 * (_bayes["intercept_p84"] - _bayes["intercept_p16"])
            ),
            "sigma_int_p16": float(_bayes["sigma_int_p16"]),
            "sigma_int_p84": float(_bayes["sigma_int_p84"]),
            "sigma_int_err_1sigma": float(
                0.5 * (_bayes["sigma_int_p84"] - _bayes["sigma_int_p16"])
            ),
            "r_value": float(_pearson.statistic),
            "r_squared": float(_pearson.statistic) ** 2,
            "p_value": float(_pearson.pvalue),
        }
        analyzer.fit_results["residuals_amiga"] = analyzer.compute_residuals(
            analyzer.amiga_data["D_HI"],
            analyzer.amiga_data["D_25"],
            analyzer.fit_results["amiga"],
        )
        analyzer.fit_results["residuals_hcg"] = analyzer.compute_residuals(
            analyzer.hcg_data["D_HI"],
            analyzer.hcg_data["D_25"],
            analyzer.fit_results["amiga"],
        )
        results["amiga"] = analyzer.fit_results["amiga"]
        results["residuals_amiga"] = analyzer.fit_results["residuals_amiga"]
        results["residuals_hcg"] = analyzer.fit_results["residuals_hcg"]
        print(
            f"Bayesian baseline adopted: "
            f"slope = {_bayes['slope']:.4f} ± {0.5 * (_bayes['slope_p84'] - _bayes['slope_p16']):.4f}, "
            f"intercept = {_bayes['intercept']:+.4f} ± "
            f"{0.5 * (_bayes['intercept_p84'] - _bayes['intercept_p16']):.4f}, "
            f"σ_obs = {_bayes_scatter_obs:.4f} dex, "
            f"σ_int = {_bayes['sigma_int']:.4f} dex"
        )
        print(f"Pearson r (log D_25, log D_HI) on cleaned AMIGA: r = {_pearson.statistic:+.4f}")

        print("\n" + "=" * 60)
        print("GENERATING LARGER-SAMPLE SPECIFIC FIGURES")
        print("=" * 60)
        plot_mass_size_calibration(
            original_module,
            analyzer,
            combined_df,
            mass_size_fit,
            "amiga_mass_size_calibration_kelley_larger_sample.pdf",
        )
        plot_larger_sample_baseline(
            original_module,
            analyzer,
            combined_df,
            "diameter_correlation_larger_sample_overview_kelley_larger_sample.pdf",
        )

        print("\n" + "=" * 60)
        print("GENERATING CORE PLOTS (KELLEY LARGER SAMPLE)")
        print("=" * 60)
        correlation_output_file = "diameter_correlation_kelley_larger_sample.pdf"
        correlation_fig, correlation_ax = analyzer.plot_correlation(
            output_file=correlation_output_file, show=False
        )
        d25_all = np.concatenate(
            [
                analyzer.amiga_data["D_25"],
                analyzer.hcg_data["D_25"],
            ]
        )
        dhi_all = np.concatenate(
            [
                analyzer.amiga_data["D_HI"],
                analyzer.hcg_data["D_HI"],
            ]
        )
        d25_all[np.isfinite(d25_all) & (d25_all > 0)]
        dhi_finite = dhi_all[np.isfinite(dhi_all) & (dhi_all > 0)]
        # correlation_ax.set_xlim(0.6 * d25_finite.min(), 1.4 * d25_finite.max())
        correlation_ax.set_xlim(4, 100)
        correlation_ax.set_ylim(
            min(1.0, 0.5 * dhi_finite.min()),
            max(1000.0, 1.8 * dhi_finite.max()),
        )

        import matplotlib.collections as _mcoll

        for line in list(correlation_ax.lines):
            line.remove()
        for coll in list(correlation_ax.collections):
            if isinstance(coll, _mcoll.PolyCollection):
                coll.remove()

        new_xlim = correlation_ax.get_xlim()
        amiga_fit = analyzer.fit_results["amiga"]
        fit_x = np.logspace(np.log10(new_xlim[0]), np.log10(new_xlim[1]), 300)
        fit_logx = np.log10(fit_x)
        fit_y = 10 ** (amiga_fit["intercept"] + amiga_fit["slope"] * fit_logx)
        sigma = amiga_fit["scatter"]
        fit_y_1up = 10 ** (amiga_fit["intercept"] + amiga_fit["slope"] * fit_logx + sigma)
        fit_y_1lo = 10 ** (amiga_fit["intercept"] + amiga_fit["slope"] * fit_logx - sigma)
        fit_y_3up = 10 ** (amiga_fit["intercept"] + amiga_fit["slope"] * fit_logx + 3 * sigma)
        fit_y_3lo = 10 ** (amiga_fit["intercept"] + amiga_fit["slope"] * fit_logx - 3 * sigma)
        correlation_ax.fill_between(fit_x, fit_y_1lo, fit_y_1up, color="gray", alpha=0.2, zorder=1)
        correlation_ax.plot(fit_x, fit_y_3up, "k--", linewidth=1, alpha=0.5, zorder=1)
        correlation_ax.plot(fit_x, fit_y_3lo, "k--", linewidth=1, alpha=0.5, zorder=1)
        correlation_ax.plot(fit_x, fit_y, "k-", linewidth=2.5, zorder=2)
        correlation_ax.plot(fit_x, fit_x, "k:", linewidth=1, alpha=0.5, zorder=1)

        correlation_fig.tight_layout()
        correlation_fig.savefig(
            original_module._figure_output_path(correlation_output_file),
            bbox_inches="tight",
            dpi=150,
        )
        original_module.plt.close(correlation_fig)

        print("\n" + "=" * 60)
        print("AMIGA GALAXIES BEYOND ±3σ OF THE ADOPTED BAYESIAN BASELINE")
        print("=" * 60)
        _baseline_sigma = float(analyzer.fit_results["amiga"]["scatter"])
        _baseline_method = analyzer.fit_results["amiga"].get("method", "Bayesian (emcee)")
        amiga_names_arr = np.asarray(
            analyzer.amiga_data.get("name", [""] * len(analyzer.amiga_data["D_25"])),
            dtype=str,
        )
        amiga_cig_arr = np.asarray(
            analyzer.amiga_data.get("cig_index", [-1] * len(analyzer.amiga_data["D_25"])),
            dtype=int,
        )
        amiga_origin_arr = np.asarray(
            analyzer.amiga_data.get("sample_origin", [""] * len(analyzer.amiga_data["D_25"])),
            dtype=str,
        )
        amiga_residuals_baseline = np.asarray(analyzer.fit_results["residuals_amiga"], dtype=float)
        amiga_d25 = np.asarray(analyzer.amiga_data["D_25"], dtype=float)
        amiga_dhi = np.asarray(analyzer.amiga_data["D_HI"], dtype=float)
        outlier_mask = np.abs(amiga_residuals_baseline) > 3 * _baseline_sigma
        n_outliers = int(np.sum(outlier_mask))
        print(
            f"±3σ threshold against the adopted {_baseline_method} baseline: "
            f"|Δ| > {3 * _baseline_sigma:.3f} dex "
            f"(σ_obs = {_baseline_sigma:.3f} dex)."
        )
        print(
            f"Number of AMIGA galaxies beyond ±3σ: {n_outliers}  "
            f"(below −3σ: {int(np.sum(amiga_residuals_baseline < -3 * _baseline_sigma))}, "
            f"above +3σ: {int(np.sum(amiga_residuals_baseline > 3 * _baseline_sigma))})"
        )
        outliers_df = pd.DataFrame(
            {
                "galaxy": amiga_names_arr[outlier_mask],
                "cig_index": amiga_cig_arr[outlier_mask],
                "sample_origin": amiga_origin_arr[outlier_mask],
                "D_25_kpc": amiga_d25[outlier_mask],
                "D_HI_kpc": amiga_dhi[outlier_mask],
                "residual_dex": amiga_residuals_baseline[outlier_mask],
                "residual_in_sigma": amiga_residuals_baseline[outlier_mask] / _baseline_sigma,
            }
        ).sort_values("residual_dex")
        outliers_path = ANALYSIS_PRODUCTS_DIR / "amiga_beyond_3sigma_kelley_larger_sample.csv"
        outliers_df.to_csv(outliers_path, index=False)
        if n_outliers:
            for _, row in outliers_df.iterrows():
                print(
                    f"  {row['galaxy']:>8s}  "
                    f"origin={row['sample_origin']:<14s}  "
                    f"D_25={row['D_25_kpc']:>7.2f} kpc  "
                    f"D_HI={row['D_HI_kpc']:>7.2f} kpc  "
                    f"Δ={row['residual_dex']:+.3f} dex  "
                    f"({row['residual_in_sigma']:+.2f} σ)"
                )
        print(f"Saved: {outliers_path}")

        print("\n" + "=" * 60)
        print("HCG RESIDUAL STATISTICS VS AMIGA BASELINE (LARGER SAMPLE)")
        print("=" * 60)
        hcg_residuals = np.asarray(analyzer.fit_results["residuals_hcg"], dtype=float)
        hcg_phases = np.asarray(analyzer.hcg_data["phase"], dtype=str)
        try:
            _hcg_csv = pd.read_csv(args.hcg_file)
            _hcg_mask = (
                np.isfinite(_hcg_csv["hi_diameter_kpc"].to_numpy(dtype=float))
                & np.isfinite(_hcg_csv["optical_diameter_kpc"].to_numpy(dtype=float))
                & (_hcg_csv["hi_diameter_kpc"].to_numpy(dtype=float) > 0)
                & (_hcg_csv["optical_diameter_kpc"].to_numpy(dtype=float) > 0)
            )
            hcg_names = _hcg_csv.loc[_hcg_mask, "galaxy"].astype(str).to_numpy()
            if hcg_names.size != hcg_residuals.size:
                hcg_names = np.asarray([f"HCG[{i}]" for i in range(hcg_residuals.size)])
        except Exception:
            hcg_names = np.asarray([f"HCG[{i}]" for i in range(hcg_residuals.size)])
        analyzer.hcg_data["name"] = hcg_names
        baseline_sigma = float(analyzer.fit_results["amiga"]["scatter"])
        hcg_stats: dict[str, object] = {
            "baseline_slope": float(analyzer.fit_results["amiga"]["slope"]),
            "baseline_intercept": float(analyzer.fit_results["amiga"]["intercept"]),
            "baseline_sigma": baseline_sigma,
            "baseline_method": analyzer.fit_results["amiga"]["method"],
        }

        def _summarise_residuals(mask: np.ndarray, label: str) -> dict[str, float | int]:
            subset = hcg_residuals[mask]
            n = int(subset.size)
            if n == 0:
                return {"label": label, "n": 0}
            below_zero = int(np.sum(subset < 0))
            below_1s = int(np.sum(subset < -baseline_sigma))
            below_3s = int(np.sum(subset < -3 * baseline_sigma))
            above_1s = int(np.sum(subset > baseline_sigma))
            return {
                "label": label,
                "n": n,
                "mean": float(np.mean(subset)),
                "median": float(np.median(subset)),
                "std": float(np.std(subset, ddof=1)) if n > 1 else float("nan"),
                "below_zero_n": below_zero,
                "below_zero_frac": below_zero / n,
                "below_1sigma_n": below_1s,
                "below_1sigma_frac": below_1s / n,
                "above_1sigma_n": above_1s,
                "above_1sigma_frac": above_1s / n,
                "below_3sigma_n": below_3s,
                "below_3sigma_frac": below_3s / n,
            }

        hcg_stats["overall"] = _summarise_residuals(
            np.ones_like(hcg_residuals, dtype=bool), "all HCG"
        )
        hcg_stats["by_phase"] = {
            phase: _summarise_residuals(hcg_phases == phase, f"Phase {phase}")
            for phase in sorted(set(hcg_phases.tolist()))
        }

        print(
            f"AMIGA baseline: log(D_HI) = {hcg_stats['baseline_slope']:.3f} * "
            f"log(D_25) + {hcg_stats['baseline_intercept']:.3f}  "
            f"[sigma = {baseline_sigma:.3f} dex, method = {hcg_stats['baseline_method']}]"
        )
        overall = hcg_stats["overall"]
        print(
            f"\nAll HCG  (N = {overall['n']}): "
            f"mean residual = {overall['mean']:+.3f} dex, "
            f"median = {overall['median']:+.3f} dex"
        )
        print(
            f"  below baseline    : {overall['below_zero_n']}/{overall['n']} "
            f"= {100 * overall['below_zero_frac']:.1f}%"
        )
        print(
            f"  below -1 sigma    : {overall['below_1sigma_n']}/{overall['n']} "
            f"= {100 * overall['below_1sigma_frac']:.1f}%"
        )
        print(
            f"  above +1 sigma    : {overall['above_1sigma_n']}/{overall['n']} "
            f"= {100 * overall['above_1sigma_frac']:.1f}%"
        )
        print(
            f"  below -3 sigma    : {overall['below_3sigma_n']}/{overall['n']} "
            f"= {100 * overall['below_3sigma_frac']:.1f}%"
        )
        print("\nBy phase:")
        print(
            f"  {'Phase':>6s} {'N':>4s} {'mean':>8s} {'median':>8s} "
            f"{'<0':>10s} {'<-1s':>10s} {'<-3s':>10s}"
        )
        for phase, entry in hcg_stats["by_phase"].items():
            if entry["n"] == 0:
                continue
            print(
                f"  {phase:>6s} {entry['n']:>4d} "
                f"{entry['mean']:>+8.3f} {entry['median']:>+8.3f} "
                f"{entry['below_zero_n']:>3d}/{entry['n']:<3d} "
                f"({100 * entry['below_zero_frac']:>4.1f}%) "
                f"{entry['below_1sigma_n']:>3d}/{entry['n']:<3d} "
                f"({100 * entry['below_1sigma_frac']:>4.1f}%) "
                f"{entry['below_3sigma_n']:>3d}/{entry['n']:<3d} "
                f"({100 * entry['below_3sigma_frac']:>4.1f}%)"
            )

        from scipy.stats import kruskal as _kruskal  # noqa: WPS433 -- local import
        from scipy.stats import ks_2samp as _ks_2samp
        from scipy.stats import pearsonr as _pearsonr
        from scipy.stats import spearmanr as _spearmanr
        from scipy.stats import ttest_1samp as _ttest_1samp

        phase_samples = [
            hcg_residuals[hcg_phases == phase]
            for phase in hcg_stats["by_phase"]
            if hcg_stats["by_phase"][phase]["n"] > 1
        ]
        if len(phase_samples) >= 2:
            kw = _kruskal(*phase_samples)
            hcg_stats["kruskal_between_phases"] = {
                "statistic": float(kw.statistic),
                "p_value": float(kw.pvalue),
                "null": "all phases have the same residual distribution",
            }
            print(
                f"\nKruskal-Wallis across phases: H = {kw.statistic:.3f}, "
                f"p = {kw.pvalue:.3g} "
                f"(null: no phase-dependent difference in residuals)"
            )

        amiga_residuals = np.asarray(analyzer.fit_results["residuals_amiga"], dtype=float)
        amiga_residuals = amiga_residuals[np.isfinite(amiga_residuals)]

        tt = _ttest_1samp(hcg_residuals, 0.0)
        hcg_stats["ttest_mean_vs_zero"] = {
            "statistic": float(tt.statistic),
            "p_value": float(tt.pvalue),
            "null": "HCG mean residual = 0",
        }
        print(
            f"\nOne-sample t-test (HCG mean residual vs 0): "
            f"t = {tt.statistic:+.3f}, p = {tt.pvalue:.3g} "
            f"(null: HCG residuals have zero mean)"
        )

        ks = _ks_2samp(hcg_residuals, amiga_residuals)
        hcg_stats["ks_vs_amiga"] = {
            "statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "null": "HCG and AMIGA residuals drawn from the same distribution",
            "amiga_n": int(amiga_residuals.size),
        }
        print(f"KS test HCG vs AMIGA residuals:   D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")

        from scipy.stats import anderson_ksamp as _anderson_ksamp
        from scipy.stats import mannwhitneyu as _mannwhitneyu

        mw = _mannwhitneyu(amiga_residuals, hcg_residuals, alternative="two-sided")
        hcg_stats["mannwhitney_vs_amiga"] = {
            "statistic": float(mw.statistic),
            "p_value": float(mw.pvalue),
            "null": "AMIGA and HCG residuals have the same location",
        }
        print(f"Mann-Whitney U (AMIGA vs HCG):    U = {mw.statistic:.1f}, p = {mw.pvalue:.3g}")

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            ad_result = _anderson_ksamp([amiga_residuals, hcg_residuals])
        hcg_stats["anderson_darling_vs_amiga"] = {
            "statistic": float(ad_result.statistic),
            "p_value_floored": float(getattr(ad_result, "significance_level", np.nan)),
            "null": "AMIGA and HCG residuals drawn from the same distribution",
            "note": "p-value is floored at 0.001 by scipy; see p_value_floored.",
        }
        print(
            f"Anderson-Darling (AMIGA vs HCG):  "
            f"A^2 = {ad_result.statistic:.3f}, "
            f"p ~ {getattr(ad_result, 'significance_level', float('nan')):.4f}"
        )

        _gt = 0
        _lt = 0
        for value in amiga_residuals:
            _gt += int(np.sum(value > hcg_residuals))
            _lt += int(np.sum(value < hcg_residuals))
        _n_pairs = int(len(amiga_residuals) * len(hcg_residuals))
        cliff_delta = float((_gt - _lt) / _n_pairs) if _n_pairs else float("nan")
        cliff_prob_amiga_gt_hcg = float(_gt / _n_pairs) if _n_pairs else float("nan")
        hcg_stats["cliffs_delta_amiga_vs_hcg"] = {
            "delta": cliff_delta,
            "prob_amiga_greater_than_hcg": cliff_prob_amiga_gt_hcg,
            "note": (
                "Cliff's delta = P(A > H) - P(A < H). Positive delta means AMIGA "
                "residuals tend to exceed HCG residuals."
            ),
        }
        print(
            f"Cliff's delta (AMIGA vs HCG):     "
            f"delta = {cliff_delta:+.3f}  (P(A>H) = {cliff_prob_amiga_gt_hcg:.3f})"
        )

        hcg_std = float(np.std(hcg_residuals, ddof=1))
        amiga_std = float(np.std(amiga_residuals, ddof=1))
        hcg_stats["scatter_inflation"] = {
            "hcg_std_dex": hcg_std,
            "amiga_std_dex": amiga_std,
            "ratio_hcg_over_amiga": hcg_std / amiga_std,
        }
        print(
            f"Residual std HCG / AMIGA: {hcg_std:.3f} / {amiga_std:.3f} = {hcg_std / amiga_std:.2f}"
        )

        hcg_log_d25 = np.log10(np.asarray(analyzer.hcg_data["D_25"], dtype=float))
        finite_pair = np.isfinite(hcg_log_d25) & np.isfinite(hcg_residuals)
        if int(np.sum(finite_pair)) >= 3:
            hcg_pr = _pearsonr(hcg_log_d25[finite_pair], hcg_residuals[finite_pair])
            hcg_sp = _spearmanr(hcg_log_d25[finite_pair], hcg_residuals[finite_pair])
            hcg_stats["hcg_residual_vs_logD25"] = {
                "pearson_r": float(hcg_pr.statistic),
                "pearson_p": float(hcg_pr.pvalue),
                "spearman_rho": float(hcg_sp.statistic),
                "spearman_p": float(hcg_sp.pvalue),
            }
            print(
                f"HCG residuals vs log(D_25): "
                f"Pearson r = {hcg_pr.statistic:+.3f} (p = {hcg_pr.pvalue:.3g}), "
                f"Spearman rho = {hcg_sp.statistic:+.3f} (p = {hcg_sp.pvalue:.3g})"
            )

        outlier_thresholds_dex = [-0.3, -0.5, -1.0]
        outlier_counts = {
            f"below_{abs(t):.1f}dex": int(np.sum(hcg_residuals < t)) for t in outlier_thresholds_dex
        }
        outlier_counts.update(
            {
                "extreme_list": [
                    {
                        "galaxy": str(analyzer.hcg_data.get("name", [""] * len(hcg_residuals))[i])
                        if "name" in analyzer.hcg_data
                        else "",
                        "phase": str(hcg_phases[i]),
                        "residual_dex": float(hcg_residuals[i]),
                        "D_25_kpc": float(analyzer.hcg_data["D_25"][i]),
                        "D_HI_kpc": float(analyzer.hcg_data["D_HI"][i]),
                    }
                    for i in np.argsort(hcg_residuals)[:5]  # 5 most truncated
                ],
            }
        )
        hcg_stats["outliers"] = outlier_counts
        print(
            "Strong-truncation outliers: "
            f"<-0.3 dex: {outlier_counts['below_0.3dex']}, "
            f"<-0.5 dex: {outlier_counts['below_0.5dex']}, "
            f"<-1.0 dex: {outlier_counts['below_1.0dex']}"
        )
        print("Five most HI-truncated HCG galaxies:")
        for row in outlier_counts["extreme_list"]:
            print(
                f"  {row['galaxy']:>10s} (phase {row['phase']:>2s}): "
                f"residual = {row['residual_dex']:+.3f} dex, "
                f"D_25 = {row['D_25_kpc']:.1f} kpc, "
                f"D_HI = {row['D_HI_kpc']:.1f} kpc"
            )

        rng_boot = np.random.default_rng(42)
        n_boot = 5000
        below_zero_boot = np.empty(n_boot)
        below_1sig_boot = np.empty(n_boot)
        mean_boot = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng_boot.choice(hcg_residuals, size=hcg_residuals.size, replace=True)
            below_zero_boot[i] = np.mean(sample < 0)
            below_1sig_boot[i] = np.mean(sample < -baseline_sigma)
            mean_boot[i] = np.mean(sample)
        hcg_stats["bootstrap_ci_68"] = {
            "mean_residual_dex": [
                float(np.percentile(mean_boot, 16)),
                float(np.percentile(mean_boot, 84)),
            ],
            "below_baseline_frac": [
                float(np.percentile(below_zero_boot, 16)),
                float(np.percentile(below_zero_boot, 84)),
            ],
            "below_1sigma_frac": [
                float(np.percentile(below_1sig_boot, 16)),
                float(np.percentile(below_1sig_boot, 84)),
            ],
            "n_boot": n_boot,
        }
        print(
            "Bootstrap 68% CIs (5000 resamples):\n"
            f"  mean residual:           "
            f"[{hcg_stats['bootstrap_ci_68']['mean_residual_dex'][0]:+.3f}, "
            f"{hcg_stats['bootstrap_ci_68']['mean_residual_dex'][1]:+.3f}] dex\n"
            f"  below baseline fraction: "
            f"[{100 * hcg_stats['bootstrap_ci_68']['below_baseline_frac'][0]:.1f}%, "
            f"{100 * hcg_stats['bootstrap_ci_68']['below_baseline_frac'][1]:.1f}%]\n"
            f"  below -1 sigma fraction: "
            f"[{100 * hcg_stats['bootstrap_ci_68']['below_1sigma_frac'][0]:.1f}%, "
            f"{100 * hcg_stats['bootstrap_ci_68']['below_1sigma_frac'][1]:.1f}%]"
        )

        hcg_size_fraction = 10 ** hcg_stats["overall"]["mean"]
        hcg_stats["hcg_size_fraction_of_baseline"] = float(hcg_size_fraction)
        print(
            f"\nEffective HCG HI-size ratio at fixed D_25: "
            f"D_HI(HCG) / D_HI(baseline) = 10^({hcg_stats['overall']['mean']:+.3f}) "
            f"= {hcg_size_fraction:.2f}  "
            f"(HCG disks are on average {100 * (1 - hcg_size_fraction):.0f}% smaller than the AMIGA "
            f"baseline predicts at fixed D_25)"
        )

        hcg_residuals_path = (
            ANALYSIS_PRODUCTS_DIR / "hcg_residual_statistics_kelley_larger_sample.json"
        )
        with open(hcg_residuals_path, "w", encoding="utf-8") as handle:
            json.dump(hcg_stats, handle, indent=2)
        hcg_per_galaxy_path = (
            ANALYSIS_PRODUCTS_DIR / "hcg_residuals_per_galaxy_kelley_larger_sample.csv"
        )
        pd.DataFrame(
            {
                "galaxy": np.asarray(analyzer.hcg_data.get("name", [""] * len(hcg_residuals))),
                "phase": hcg_phases,
                "D_25_kpc": np.asarray(analyzer.hcg_data["D_25"], dtype=float),
                "D_HI_kpc": np.asarray(analyzer.hcg_data["D_HI"], dtype=float),
                "residual_dex": hcg_residuals,
                "residual_in_sigma": hcg_residuals / baseline_sigma,
                "below_baseline": hcg_residuals < 0,
                "below_1sigma": hcg_residuals < -baseline_sigma,
                "below_3sigma": hcg_residuals < -3 * baseline_sigma,
            }
        ).to_csv(hcg_per_galaxy_path, index=False)

        hcg_latex_dir = PROJECT_ROOT / "latex" / "autogen"
        hcg_latex_dir.mkdir(parents=True, exist_ok=True)

        ad_p = hcg_stats["anderson_darling_vs_amiga"].get("p_value_floored", float("nan"))
        ad_p_cell = f"$< {ad_p:.3f}$" if np.isfinite(ad_p) and ad_p <= 0.001 else f"${ad_p:.3f}$"

        def _p_cell(value: float) -> str:
            if not np.isfinite(value):
                return "$\\mathrm{nan}$"
            if value >= 1e-3:
                return f"${value:.3g}$"
            exponent = int(np.floor(np.log10(value)))
            mantissa = value / 10**exponent
            return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"

        stat_table_lines = [
            r"% Auto-generated by scripts/measure_size_residuals.py.",
            r"% Regenerate by running that script; manual edits will be overwritten.",
            r"% This file is a full tabular block (with \resizebox); hi_disk_size_environments.tex wraps it with caption+tablefoot only.",
            r"\resizebox{0.47\textwidth}{!}{%",
            r"\begin{tabular}{lcc}",
            r"\toprule \toprule",
            r"Test & Statistic & $p$-value \\",
            r"\midrule",
            (
                rf"Mann--Whitney $U$         & ${hcg_stats['mannwhitney_vs_amiga']['statistic']:.0f}$   "
                rf"& {_p_cell(hcg_stats['mannwhitney_vs_amiga']['p_value'])} \\"
            ),
            (
                rf"Kolmogorov--Smirnov       & $D = {hcg_stats['ks_vs_amiga']['statistic']:.3f}$ "
                rf"& {_p_cell(hcg_stats['ks_vs_amiga']['p_value'])} \\"
            ),
            (
                rf"Anderson--Darling          & ${hcg_stats['anderson_darling_vs_amiga']['statistic']:.3f}$ "
                rf"& {ad_p_cell} \\"
            ),
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Effect sizes}} \\",
            r"\midrule",
            (
                rf"Cliff's $\delta$          & \multicolumn{{2}}{{c}}"
                rf"{{${cliff_delta:+.3f}$ "
                rf"($P(\mathrm{{AMIGA}} > \mathrm{{HCG}}) = {cliff_prob_amiga_gt_hcg:.3f}$)}} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
        stat_table_path = hcg_latex_dir / "table_stat_tests.tex"
        with open(stat_table_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(stat_table_lines) + "\n")

        hcg_size_fraction_pct = float(round(100 * (10 ** hcg_stats["overall"]["mean"])))
        hcg_size_reduction_pct = float(round(100 * (1 - 10 ** hcg_stats["overall"]["mean"])))
        below_baseline_pct = float(round(100 * hcg_stats["overall"]["below_zero_frac"]))
        below_1sigma_pct = float(round(100 * hcg_stats["overall"]["below_1sigma_frac"]))
        cliff_prob_pct = float(round(100 * cliff_prob_amiga_gt_hcg))
        mean_residual_dex = float(hcg_stats["overall"]["mean"])

        macros_lines = [
            r"% Auto-generated by scripts/measure_size_residuals.py.",
            r"% Regenerate by running that script; manual edits will be overwritten.",
            r"% Inline quantities for the HCG-vs-AMIGA comparison paragraph in hi_disk_size_environments.tex.",
            rf"\providecommand{{\HCGSizeFractionPercent}}{{{hcg_size_fraction_pct:.0f}}}",
            rf"\providecommand{{\HCGSizeReductionPercent}}{{{hcg_size_reduction_pct:.0f}}}",
            rf"\providecommand{{\HCGDeficitPercent}}{{{hcg_size_reduction_pct:.0f}}}",
            rf"\providecommand{{\HCGBelowBaselinePercent}}{{{below_baseline_pct:.0f}}}",
            rf"\providecommand{{\HCGBelowOneSigmaPercent}}{{{below_1sigma_pct:.0f}}}",
            rf"\providecommand{{\HCGMeanResidualDex}}{{{mean_residual_dex:+.3f}}}",
            rf"\providecommand{{\CliffProbAmigaGreaterHCGPercent}}{{{cliff_prob_pct:.0f}}}",
            rf"\providecommand{{\HCGSampleSize}}{{{hcg_stats['overall']['n']}}}",
            rf"\providecommand{{\HCGBelowBaselineCount}}{{{hcg_stats['overall']['below_zero_n']}}}",
            rf"\providecommand{{\AmigaSigmaDex}}{{{baseline_sigma:.3f}}}",
        ]
        _phase_macro_names = {
            "1": "PhaseOne",
            "2": "PhaseTwo",
            "3a": "PhaseThreeA",
            "3c": "PhaseThreeC",
        }
        for _phase, _macro_stem in _phase_macro_names.items():
            _entry = hcg_stats["by_phase"].get(_phase)
            if not _entry or _entry.get("n", 0) < 2:
                continue
            _median_delta = float(_entry["median"])
            _size_pct = round(100 * (10**_median_delta))
            _reduction_pct = round(100 - _size_pct)
            macros_lines.append(
                rf"\providecommand{{\{_macro_stem}MedianSizePercent}}{{{_size_pct:.0f}}}"
            )
            macros_lines.append(
                rf"\providecommand{{\{_macro_stem}ReductionPercent}}{{{_reduction_pct:.0f}}}"
            )
            macros_lines.append(
                rf"\providecommand{{\{_macro_stem}DeficitPercent}}{{{_reduction_pct:.0f}}}"
            )
            macros_lines.append(rf"\providecommand{{\{_macro_stem}N}}{{{_entry['n']}}}")
            macros_lines.append(
                rf"\providecommand{{\{_macro_stem}MedianDeltaDex}}{{{_median_delta:+.3f}}}"
            )
        macros_path = hcg_latex_dir / "macros_hcg_comparison.tex"
        with open(macros_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(macros_lines) + "\n")

        # Phase-by-phase residual statistics table (table:phase_stats).
        # Phases with N < 2 (e.g., a lone Phase 3a galaxy) are dropped to match
        # the figure, which also excludes them.
        phase_table_lines = [
            r"% Auto-generated by scripts/measure_size_residuals.py.",
            r"% Regenerate by running that script; manual edits will be overwritten.",
            r"% This file is a full tabular block; hi_disk_size_environments.tex wraps it with caption+tablefoot only.",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Phase & $n$ & Mean $\Delta$ & Median $\Delta$ & Median $D_{\rm{HI}}$  \\",
            r"      &     &               &                 & deficit \\",
            r"      &     & [dex]         & [dex]           & [\%]    \\",
            r"\midrule",
        ]
        for phase in ("1", "2", "3a", "3c"):
            entry = hcg_stats["by_phase"].get(phase, {})
            if not entry or entry.get("n", 0) < 2:
                continue
            median_delta = float(entry["median"])
            median_deficit_pct = 100 * (1 - 10**median_delta)
            phase_table_lines.append(
                rf"{phase}     & {entry['n']:<3d} "
                rf"& ${entry['mean']:+.3f}$      "
                rf"& ${median_delta:+.3f}$        "
                rf"& {median_deficit_pct:.0f} \\"
            )
        phase_table_lines.append(r"\bottomrule")
        phase_table_lines.append(r"\end{tabular}")
        phase_table_path = hcg_latex_dir / "table_phase_stats.tex"
        with open(phase_table_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(phase_table_lines) + "\n")

        print(f"\nHCG residual statistics JSON:    {hcg_residuals_path}")
        print(f"HCG per-galaxy residuals CSV:    {hcg_per_galaxy_path}")
        print(f"LaTeX stat-tests fragment:       {stat_table_path}")
        print(f"LaTeX HCG macros fragment:       {macros_path}")
        print(f"LaTeX phase-stats fragment:      {phase_table_path}")

        analyzer.plot_residuals_histogram(
            output_file="diameter_residuals_hist_kelley_larger_sample.pdf", show=False
        )

        # Replace the count-based residual histogram with a density-normalised
        # version so that AMIGA (N=407) and HCG (N=54) can be compared on
        # equal footing. Sample sizes are reported in the legend; everything
        # else (phase stacking, hatching, mean lines, offset annotation)
        # mirrors the count-based histogram produced by the analyzer.
        from matplotlib.patches import Patch as _Patch

        hist_res_amiga = np.asarray(analyzer.fit_results["residuals_amiga"], dtype=float)
        hist_res_amiga = hist_res_amiga[np.isfinite(hist_res_amiga)]
        hist_res_hcg = np.asarray(analyzer.fit_results["residuals_hcg"], dtype=float)
        hist_res_hcg = hist_res_hcg[np.isfinite(hist_res_hcg)]
        hist_phases = np.asarray(analyzer.hcg_data["phase"], dtype=str)
        # Trim phases to align with finite residuals
        hist_phases = hist_phases[: hist_res_hcg.size]

        all_res_arr = np.concatenate([hist_res_amiga, hist_res_hcg])
        bins = np.linspace(np.min(all_res_arr) - 0.1, np.max(all_res_arr) + 0.1, 20)
        bin_width = bins[1] - bins[0]

        n_amiga = int(hist_res_amiga.size)
        n_hcg = int(hist_res_hcg.size)
        1.0 / (n_amiga * bin_width)
        density_norm_hcg = 1.0 / (n_hcg * bin_width)

        phase_order = ["1", "2", "3a", "3c"]
        phase_colors = {
            "1": "#a6cee3",
            "2": "#8fbc8f",
            "3a": "#c4a6d6",
            "3c": "#f4b07c",
        }
        phase_labels = {
            "1": "Phase 1",
            "2": "Phase 2",
            "3a": "Phase 3a",
            "3c": "Phase 3c",
        }

        hist_fig, hist_ax = original_module.plt.subplots(figsize=(10, 8))

        bottom = np.zeros(len(bins) - 1)
        phases_present = []
        for phase in phase_order:
            mask = hist_phases == phase
            if int(np.sum(mask)) == 0:
                continue
            counts, _ = np.histogram(hist_res_hcg[mask], bins=bins)
            heights = counts * density_norm_hcg
            hist_ax.bar(
                bins[:-1],
                heights,
                width=bin_width,
                bottom=bottom,
                align="edge",
                color=phase_colors[phase],
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            bottom += heights
            phases_present.append(phase)

        hist_ax.hist(
            hist_res_hcg,
            bins=bins,
            density=True,
            histtype="stepfilled",
            facecolor="none",
            edgecolor="none",
            hatch=".",
            linewidth=0,
            zorder=3,
        )
        hist_ax.hist(
            hist_res_hcg,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.5,
            edgecolor="blue",
            zorder=4,
        )

        hist_ax.hist(
            hist_res_amiga,
            bins=bins,
            density=True,
            histtype="stepfilled",
            facecolor="white",
            edgecolor="none",
            hatch="/",
            linewidth=0,
            zorder=5,
            alpha=0.7,
        )
        hist_ax.hist(
            hist_res_amiga,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.5,
            edgecolor="black",
            zorder=6,
        )

        mean_amiga = float(np.mean(hist_res_amiga))
        mean_hcg = float(np.mean(hist_res_hcg))
        hist_ax.axvline(mean_amiga, color="black", linestyle="--", linewidth=2, zorder=7)
        hist_ax.axvline(mean_hcg, color="blue", linestyle="--", linewidth=2, zorder=7)
        hist_ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, zorder=1)

        hist_ax.set_xlabel(r"$\Delta \log(D_{\rm HI})$ [dex]", fontsize=22, labelpad=15)
        hist_ax.set_ylabel(r"Probability density [dex$^{-1}$]", fontsize=22, labelpad=15)
        hist_ax.minorticks_on()
        hist_ax.tick_params(which="both", direction="in", top=True, right=True)
        hist_ax.tick_params(which="major", length=8, width=1.2, pad=10)
        hist_ax.tick_params(which="minor", length=4, width=1, pad=10)

        hcg_handle = _Patch(
            facecolor="none",
            edgecolor="blue",
            hatch=".",
            linewidth=1.2,
            label=f"Galaxies in HCGs (N={n_hcg})",
        )
        amiga_handle = _Patch(
            facecolor="white",
            edgecolor="black",
            hatch="/",
            linewidth=1.2,
            label=f"AMIGA galaxies (N={n_amiga})",
        )
        mean_handles = [
            original_module.plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"AMIGA mean: {mean_amiga:+.2f}",
            ),
            original_module.plt.Line2D(
                [0],
                [0],
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"HCG mean: {mean_hcg:+.2f}",
            ),
        ]
        main_legend = hist_ax.legend(
            handles=[hcg_handle, amiga_handle, *mean_handles],
            loc="upper left",
            fontsize=15,
            frameon=True,
        )
        hist_ax.add_artist(main_legend)

        phase_handles = [
            _Patch(
                facecolor=phase_colors[p],
                edgecolor="blue",
                linewidth=1.2,
                label=phase_labels[p],
            )
            for p in phases_present
        ]
        phase_leg = hist_ax.legend(
            handles=phase_handles,
            title="HCG phases",
            title_fontsize=17,
            fontsize=15,
            frameon=True,
            framealpha=0.9,
            edgecolor="0.6",
            loc="center right",
            bbox_to_anchor=(0.99, 0.35),
        )
        hist_ax.add_artist(phase_leg)

        offset = mean_hcg - mean_amiga
        hist_ax.annotate(
            f"Offset: {offset:+.2f} dex",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            fontsize=15,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        hist_fig.tight_layout()
        hist_fig.savefig(
            original_module._figure_output_path("diameter_residuals_hist_kelley_larger_sample.pdf"),
            bbox_inches="tight",
            dpi=150,
        )
        original_module.plt.close(hist_fig)

        residuals_vs_d25_output_file = "diameter_residuals_vs_D25_kelley_larger_sample.pdf"
        resid_fig, resid_ax = analyzer.plot_residuals_vs_D25(
            output_file=residuals_vs_d25_output_file, show=False
        )
        resid_fig.set_size_inches(8, 8)
        for line in list(resid_ax.lines):
            line.remove()
        for coll in list(resid_ax.collections):
            if isinstance(coll, _mcoll.PolyCollection):
                coll.remove()

        resid_ax.set_xlim(4, 100)
        resid_ax.set_ylim(-1.5, 1.5)
        resid_sigma = analyzer.fit_results["amiga"]["scatter"]
        resid_x = np.logspace(np.log10(4), np.log10(100), 300)
        resid_ax.fill_between(
            resid_x,
            -resid_sigma,
            resid_sigma,
            color="gray",
            alpha=0.2,
            zorder=1,
        )
        resid_ax.plot(
            resid_x,
            np.full_like(resid_x, 3 * resid_sigma),
            "k--",
            linewidth=1,
            alpha=0.5,
            zorder=1,
        )
        resid_ax.plot(
            resid_x,
            np.full_like(resid_x, -3 * resid_sigma),
            "k--",
            linewidth=1,
            alpha=0.5,
            zorder=1,
        )
        resid_ax.axhline(0, color="black", linestyle="-", linewidth=2, zorder=2)

        old_legend = resid_ax.get_legend()
        if old_legend is not None:
            old_legend.remove()
        resid_ax.legend(loc="upper right", fontsize=14, frameon=True, framealpha=0.9)

        for text_obj in resid_ax.texts:
            text_obj.set_linespacing(1.8)

        resid_fig.tight_layout()
        resid_fig.savefig(
            original_module._figure_output_path(residuals_vs_d25_output_file),
            bbox_inches="tight",
            dpi=150,
        )
        original_module.plt.close(resid_fig)
        analyzer.plot_residuals_by_phase(
            output_file="diameter_residuals_by_phase_kelley_larger_sample.pdf", show=False
        )

        analyzer.load_wang_table(str(DATA_DIR / "wang-surveys-table.txt"))
        analyzer.load_broeils_rhee(str(DATA_DIR / "broelis-rhee.txt"), survey_name="B97")
        analyzer.register_surveys()
        register_hydra_i_survey(
            analyzer,
            DATA_DIR / "reynolds_hyperleda_detection_matches.csv",
        )
        register_bok_pairs_survey(
            analyzer,
            DATA_DIR / "bok_hyperleda_pair_matches.csv",
            mass_size_fit,
        )
        analyzer.compute_survey_residuals()

        print("\n" + "=" * 60)
        print("GENERATING SURVEY COMPARISON PLOTS (KELLEY LARGER SAMPLE)")
        print("=" * 60)
        analyzer.plot_survey_mean_residual(
            rank_metric="median",
            output_file="survey_median_residual_kelley_larger_sample.pdf",
            show=False,
        )
        plot_well_defined_subset_median_residual(
            analyzer,
            output_file="survey_median_residual_kelley_larger_well_defined_sample.pdf",
        )
        # This call also emits table_survey_residuals.tex, table_pairwise_residuals.tex,
        # and well_defined_pairwise_residuals.json, so it must run. Its FIGURE,
        # however, is the older red-star design; the production Figure 8 (top) is
        # owned by scripts/plot_survey_residual_forest.py (the fig_survey_forest
        # rule), which draws the HCG point as a marker with a directional arrow.
        # We therefore send this figure to a throwaway filename so it cannot
        # overwrite the production Figure 8.
        plot_hydra_split_well_defined_residual(
            analyzer,
            hydra_csv_path=DATA_DIR / "reynolds_hyperleda_detection_matches.csv",
            figure_output_file="survey_median_residual_legacy_hydra_split.pdf",
            products_pdf_path=ANALYSIS_PRODUCTS_DIR
            / "mass_size_consistency_test_standalone_hydra_split.pdf",
            figures_dir=ANALYSIS_FIGURES_DIR,
        )
        analyzer.plot_survey_frac_extended(
            output_file="survey_frac_extended_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_survey_frac_truncated(
            output_file="survey_frac_truncated_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_correlation_with_all_surveys(
            output_file="diameter_correlation_with_all_surveys_kelley_larger_sample.pdf",
            show=False,
        )

        print("\n" + "=" * 60)
        print("GENERATING MASS-RELATED DIAGNOSTICS (KELLEY LARGER SAMPLE)")
        print("=" * 60)
        analyzer.plot_residuals_vs_stellar_mass(
            output_file="residuals_vs_stellar_mass_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_residuals_vs_hi_mass(
            output_file="residuals_vs_hi_mass_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_residuals_vs_gas_fraction(
            output_file="residuals_vs_gas_fraction_kelley_larger_sample.pdf", show=False
        )

        focused = analyzer.run_focused_comparison()
        results["focused"] = focused

        print("\n" + "=" * 60)
        print("GENERATING FOCUSED COMPARISON PLOTS (KELLEY LARGER SAMPLE)")
        print("=" * 60)
        analyzer.plot_truncation_index_by_phase(
            output_file="truncation_index_by_phase_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_size_matched_comparison(
            output_file="size_matched_comparison_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_binned_residuals_comparison(
            output_file="binned_residuals_comparison_kelley_larger_sample.pdf", show=False
        )
        analyzer.plot_bootstrap_shift_distribution(
            output_file="bootstrap_shift_distribution_kelley_larger_sample.pdf", show=False
        )

        print("\n" + "=" * 60)
        print("RESIDUAL-TREND AUDIT (9 FIT METHODS, BAYESIAN REPLACES SIGMA-CLIPPED OLS)")
        print("=" * 60)
        print(
            "Data: combined AMIGA resolved + Jones-inferred single-dish "
            f"(N={len(combined_df)}), shown with a single symbol per panel."
        )
        trend_fig_path = (
            ANALYSIS_FIGURES_DIR / "amiga_residual_trends_kelley_larger_sample_dictionary.pdf"
        )
        run_residual_trend_audit(
            combined_df=combined_df,
            plt_module=original_module.plt,
            style_fn=analyzer._style_axes,
            figure_output_path=trend_fig_path,
            products_dir=ANALYSIS_PRODUCTS_DIR,
            suffix="_kelley_larger_sample_dictionary",
        )

        combined_df.to_csv(
            ANALYSIS_PRODUCTS_DIR / "amiga_combined_larger_sample_kelley_larger_sample.csv",
            index=False,
        )
        inferred_only = combined_df[combined_df["sample_origin"] == "inferred_jones"].copy()
        inferred_only.to_csv(
            ANALYSIS_PRODUCTS_DIR / "amiga_inferred_jones_kelley_larger_sample.csv",
            index=False,
        )
        write_summary(
            Path(args.summary_file),
            results,
            combined_df,
            mass_size_fit,
            d25_mode,
            sample_metadata,
        )

        print("\n" + "=" * 60)
        print("KELLEY LARGER-SAMPLE ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"  - Larger-sample AMIGA baseline: D_HI ∝ D_25^{results['amiga']['slope']:.2f}")
        print(f"  - Larger-sample AMIGA scatter: {results['amiga']['scatter']:.3f} dex")
        print(
            f"  - HCG offset from larger AMIGA baseline: {results['comparison']['offset']:.3f} dex"
        )
        print(
            "  - This means HCG galaxies have HI disks that are "
            f"{10 ** results['comparison']['offset']:.1%} the size expected at fixed D_25"
        )
        print(f"Figures saved to: {ANALYSIS_FIGURES_DIR}")
        print(f"Summary JSON saved to: {args.summary_file}")

    print(f"Full text report saved to: {args.report_file}")


if __name__ == "__main__":
    main()
