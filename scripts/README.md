# Analysis scripts

These are the canonical scripts that reproduce the figures, tables, and numbers
of the manuscript. Each is invoked by exactly one Snakemake rule
(`workflow/rules/*.smk`), and each manuscript output has a single producing
script. The reusable, documented re-implementations of the core methods live in
the `galaxydisksize` package under `src/`; these scripts are being progressively
slimmed to thin command-line front-ends over that package.

Names describe what each script does. The original, cryptically-named versions
are preserved untouched under `../../amiga-aastex/scripts` as the cross-check
reference.

## Catalogue tier (moment-0 maps -> measured CSVs)

| Script | Produces |
|---|---|
| `measure_hi_disk_sizes.py` | ellipse fits, HI diameters/masses, moment-0 postage stamps |
| `build_hcg_catalogue.py` | `interacting_galaxies_results.csv` |
| `crossmatch_bok_hyperleda.py`, `crossmatch_reynolds_hyperleda.py` | optical-diameter cross-matches |
| `export_amiga_optical_diameters.py` | AMIGA `log D_25` catalogue |
| `flag_beam_upper_limits.py` | augmented upper-limit table (`..._bmaj.csv`) |
| `image_cutouts.py` | shared optical-cutout / unit-conversion helpers |

## Inference tier (CSVs -> fits, residuals, survival, autogen tables)

| Script | Produces |
|---|---|
| `fit_size_mass_relation.py` | per-sample size-mass fit summary JSON (Table 3 input) |
| `plot_size_mass_all_surveys.py` | combined size-mass fit + figures |
| `plot_size_mass_amiga_hcgs.py` | AMIGA-vs-HCG size-mass figure |
| `write_size_mass_macros.py` | `macros_mass_size.tex`, `table_consistency_fits.tex` |
| `measure_size_residuals.py` | residual products, six autogen tables, trend figure (main driver) |
| `size_residual_baseline_engine.py` | the `CorrelationAnalysis` baseline/residual engine imported by the driver |
| `survival_analysis.py` | censored tables/macros, upper-limit appendix table |

## Figure tier

| Script | Produces |
|---|---|
| `plot_diameter_correlation.py` | correlation and residual-vs-D_25 panels |
| `plot_residual_histogram.py` | residual histogram (Fig. 6) |
| `plot_residuals_by_phase.py` | residuals by phase (Fig. 7) |
| `plot_survey_residual_forest.py` | survey forest plot (Fig. 8 top) |
| `plot_pairwise_clique_euler.py` | pairwise-clique Euler diagram (Fig. 8 bottom) |

## Notes on the two residual scripts

`measure_size_residuals.py` is the driver you run. It imports
`size_residual_baseline_engine.py` (the `CorrelationAnalysis` class implementing
the D_HI-D_25 baseline and residual method) and extends the AMIGA sample by
inferring D_HI for the single-dish galaxies from the combined size-mass
calibration. The engine is a library for the driver, not an alternative entry
point.
