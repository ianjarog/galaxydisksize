# Usage

## The two tiers

| Tier | Inputs | Cost | Default? |
|---|---|---|---|
| **Statistics** (fits, residuals, survival, figures, paper) | committed measurement CSVs | minutes on a laptop | yes |
| **Catalogue** (ellipse fitting of the moment-0 maps) | moment-0 maps + SoFiA masks | expensive | no — opt-in |

Most users only need the statistics tier. The catalogue tier
(`workflow/rules/catalogue.smk`) re-derives the measurement CSVs from the
moment-0 maps; its outputs are committed so the rest of the pipeline never
depends on re-running it.

## Reproducing the analysis (Snakemake)

```bash
snakemake -n paper            # dry run: inspect the DAG
snakemake --cores 4 figures   # build the analysis figures
snakemake --cores 4 paper     # build the full manuscript PDF
snakemake --report report.html  # HTML provenance report
```

Useful named targets include `figures`, `paper`, `survival_analysis`,
`residual_analysis`, `mass_size_all_surveys`, and `mass_size_amiga_hcgs`
(see `workflow/Snakefile` and `workflow/rules/`).

Each output file is produced by **exactly one** rule, so two scripts can never
overwrite the same figure.

### Running on a cluster

The same workflow scales to HPC via a Snakemake execution profile:

```bash
snakemake --profile workflow/profiles/<your-profile> --jobs 100 paper
```

## Using the library directly

The package is usable on any sample of HI masses and diameters:

```python
import galaxydisksize as gds

fit = gds.fit_mass_size(log_hi_mass, log_hi_diameter, seed=42)
print(fit.slope, fit.intercept, fit.scatter)   # slope ~ 0.5 => constant Sigma_HI

delta = gds.size_residual(log_hi_diameter, log_d25, slope, intercept)
km = gds.kaplan_meier_left_censored(delta, is_upper_limit)
print(km.median, km.fraction_below(0.0))
```

A guided tour is in [`notebooks/demo.ipynb`](../notebooks/demo.ipynb), runnable in
the browser via [Binder](../binder/README.md).

## Command-line interface

The package installs a `galaxydisksize` entry point (see
`src/galaxydisksize/cli.py`):

```bash
galaxydisksize --help
```
