# galaxydisksize

[![CI](https://github.com/ianjarog/galaxydisksize/actions/workflows/ci.yml/badge.svg)](https://github.com/ianjarog/galaxydisksize/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)


Measure HI disk diameters of galaxies through direct ellipse fitting of their 
\(1\ M_\odot\,\mathrm{pc}^{-2}\) HI surface-density contour. Use isolated galaxies 
as a reference population to establish a scaling relation between HI and optical 
disk sizes (\(D_{\mathrm{HI}}\)–\(D_{25}\)), and quantify how galaxies across 
different environments deviate from this adopted baseline.

This repository contains both a reusable Python package and a fully reproducible
Snakemake workflow behind the study of HI disc truncation across different 
environments relative to isolated AMIGA galaxies. From the moment-0 maps it
measures HI diameters, fits the HI size-mass relation, computes size residuals
about the HI-to-optical baseline, runs left-censored survival statistics for the
non-detections, and compiles the manuscript PDF — every figure, table, and inline
number flowing from the same committed inputs.

The intended audience is researchers in extragalactic HI astronomy who want to
reproduce the accompanying paper or apply the size-mass / truncation analysis to
their own data.

## Status

- **Version:** 0.1.0 (see [`CHANGELOG.md`](CHANGELOG.md)).
- **Development stage:** Beta — research software accompanying a manuscript in
  preparation. The science is complete; the pipeline is being finalised for
  release.
- **Reproducibility:** the default statistics tier rebuilds every figure, table,
  and number from committed inputs.

## Features

- Measure HI diameters from moment-0 maps (deterministic contour fit) with two
  uncertainty modes (beam-correlated Monte-Carlo, or a mask-free bootstrap).
- Fit the HI size-mass relation with nine linear estimators.
- Compute size residuals about the HI-to-optical (D25) baseline.
- Left-censored survival statistics (Kaplan-Meier, Gehan) for the non-detections.
- One-command reproduction of the manuscript PDF from committed inputs.
- A reusable, tested library with golden-number regression guards.

## Why this layout

The analysis is split into a **library** and a **workflow**:

- `src/galaxydisksize/` — an installable, documented, tested package with the
  reusable science (size-mass fitting, size residuals, surface-density
  conversions, survival statistics). Use it on your own data.
- `workflow/` + `scripts/` — a Snakemake pipeline that reproduces the paper. Each
  output file is produced by exactly one rule, so two scripts can never overwrite
  the same figure (the problem this repository was created to eliminate).

## Requirements

- **Python ≥ 3.10** (the conda environment pins 3.12).
- Scientific stack: NumPy, SciPy, pandas, Astropy, Matplotlib, emcee, PyYAML.
- **Snakemake ≥ 8** for the workflow.
- A LaTeX engine for the `paper` rule (`environment.yml` installs **tectonic**;
  any TeX distribution works).
- Optional: conda/mamba (recommended), Docker or Apptainer for containers.

Full details and alternatives are in [`docs/installation.md`](docs/installation.md).

## Quickstart

```bash
# 1. Create the environment and install the package.
conda env create -f environment.yml
conda activate galaxydisksize
pip install -e .

# 2. Reproduce the analysis figures and the manuscript PDF.
snakemake --cores 4 figures      # the analysis figures
snakemake --cores 4 paper        # the full manuscript PDF
snakemake -n paper               # dry run: inspect the DAG without building

# 3. Run the tests.
pytest
```

Use the package directly on any sample of HI masses and diameters:

```python
import numpy as np
import galaxydisksize as gds

fit = gds.fit_mass_size(log_hi_mass, log_hi_diameter, seed=42)
print(fit.slope, fit.intercept, fit.scatter)        # slope ~ 0.5 => constant Sigma_HI

delta = gds.size_residual(log_hi_diameter, log_d25, slope, intercept)
km = gds.kaplan_meier_left_censored(delta, is_upper_limit)
print(km.median, km.fraction_below(0.0))
```

An interactive tour is in [`notebooks/demo.ipynb`](notebooks/demo.ipynb), runnable
in the browser through [Binder](binder/README.md).

## The two tiers

| Tier | Inputs | Environment | Default? |
|---|---|---|---|
| **Statistics** (fits, residuals, survival, figures, paper) | the committed measurement CSVs | light, pure-Python (`environment.yml`) | yes |
| **Catalogue** (ellipse fitting of the moment-0 maps) | moment-0 maps + SoFiA masks | heavier (pyspeckit, cutout services) | no — opt-in |

Most users only need the statistics tier, which runs from the committed CSVs in
minutes on a laptop. The catalogue tier (`workflow/rules/catalogue.smk`) re-derives
those CSVs from the moment-0 maps and is opt-in; its outputs are committed so the
rest of the pipeline does not depend on re-running it.

## Repository layout

```
galaxydisksize/
  src/galaxydisksize/    installable package (the reusable science)
  scripts/               canonical analysis scripts (see scripts/README.md)
  workflow/              Snakefile + rules (the reproducible DAG)
  config/                workflow configuration and external data sources
  data/                  committed measurement tables, configs, demo maps
  products/              persisted fit summaries (regenerated by the workflow)
  figures/               figures (regenerated by the workflow)
  latex/                 manuscript sources + autogen/ (generated fragments)
  notebooks/             demonstration notebook (Binder)
  containers/            Dockerfile and Apptainer definition
  binder/                Binder environment
  tests/                 unit tests and golden-number regression guards
```

## Data and provenance

- The **moment-0 maps** that the catalogue tier measures are **committed** under
  `data/moment0_maps/` (the AMIGA set plus the small SoFiA noise catalogues,
  kinematic-centre files, and the `RESULTS_OPT` distances table). Re-running the
  measurement from these reproduces the published **HI diameters exactly**.
- The **SoFiA mask cubes** are large (multi-GB) and **not** committed. They are
  required only for the Monte-Carlo **error bars** (the diameter itself needs only
  the map). Obtain them from
  [Zenodo 6909872](https://zenodo.org/records/6909872) (Jones et al. 2023) and
  place them so the pipeline finds them at the default paths:
  - AMIGA: `data/moment0_maps/sofiamasks/{galaxy}_sofiamask.fits`
  - HCG:   `data/moment0_maps/SoFiA_masks/`

  (or point `GALAXYDISKSIZE_CIG_MASKS` / `GALAXYDISKSIZE_HCG_MASKS` at an existing
  copy). These directories are git-ignored.
- The HI **data cubes** are also archived in the same Zenodo record and are cited,
  not redistributed.
- Reproducibility note: the published MC error bars predate the current
  measurement code, so they are kept as-is in the committed CSVs; a re-run
  reproduces the diameters but produces fresh (stochastic) error bars.

### Two diameter-error modes

The HI **diameter** is the deterministic fit to the central 1 M⊙ pc⁻² contour and
needs only the moment-0 map — it reproduces exactly with or without the masks. Only
the **uncertainty** depends on the masks:

- **With masks (default).** When the SoFiA mask is present the diameter error is the
  beam-correlated Monte-Carlo estimate (perturb the map with mask-derived noise,
  re-contour, re-fit). The mask-derived HI-mass error and column-density limit are
  computed as well. This is the published method.
- **Mask-free (fallback).** When the mask is absent — or `GALAXYDISKSIZE_NO_MASKS=1`
  is set — the diameter error falls back to a vertex bootstrap of the fitted contour,
  which needs no mask. It is reported in the same column; the mask-derived HI-mass
  error and column-density limit are then `NaN`. This lets users without the Zenodo
  masks still reproduce the diameters and obtain a usable (if simpler) uncertainty.

## Configuration (paths and fonts)

The repository contains no machine-specific absolute paths. Locations are
resolved at run time from environment variables with repository-relative
defaults (`scripts/project_config.py`), so the code runs unchanged on any
machine:

| Variable | Default | Purpose |
|---|---|---|
| `GALAXYDISKSIZE_EXTERNAL` | `data/external` | Root for the large external inputs |
| `GALAXYDISKSIZE_HCG_MASKS` | `<external>/SoFiA_masks` | HCG SoFiA masks |
| `GALAXYDISKSIZE_KINPARS`, `GALAXYDISKSIZE_MOM_MAPS`, `GALAXYDISKSIZE_CIG_MASKS`, `GALAXYDISKSIZE_NOISE_CUBES`, `GALAXYDISKSIZE_DISTANCES` | sub-dirs of `<external>` | AMIGA catalogue-tier inputs |
| `GALAXYDISKSIZE_NO_MASKS` | unset | Set to `1` to force the mask-free error path (see below) even when masks are present |
| `GALAXYDISKSIZE_FONT_DIR` | unset | TeX Gyre Heros fonts; if unset, Matplotlib's default fonts are used |

Only the opt-in catalogue tier needs the external data; the default statistics
tier and the figures run from the committed inputs with no configuration. Set a
variable only to point at data you have downloaded, for example:

```bash
export GALAXYDISKSIZE_EXTERNAL=/scratch/me/hcg_data
export GALAXYDISKSIZE_FONT_DIR=/usr/share/fonts/tex-gyre   # optional, for the paper's fonts
```

### Credentials

The repository contains no credentials, and the workflow needs none. It does not
connect to any database or external service: every input, including the AMIGA/CIG
optical-diameter catalogue, is provided directly as a committed machine-readable
file under `data/` (see [Data and provenance](docs/data-and-provenance.md)). Just
clone and run.

## Reproducibility and FAIR

- **Environment capture:** `environment.yml`; regenerate an exact lock with
  `conda-lock` for byte-level reproducibility. Container recipes are in
  `containers/` (Docker and Apptainer/Singularity for HPC).
- **Provenance:** the Snakemake DAG records the input/output lineage of every
  figure and table; `snakemake --report` produces an HTML provenance report.
- **Regression guards:** `tests/test_golden_numbers.py` pins the published
  numbers (combined size-mass slope 0.508, scatter 0.065 dex, baseline scatter
  0.153 dex) so accidental drift fails the test suite.
- **Scaling:** the same workflow runs on a laptop, a workstation, or an HPC
  cluster via a Snakemake execution profile (e.g. SLURM under
  `workflow/profiles/`).

## First-run reconciliation (developers)

The workflow rules wrap the original analysis scripts, several of which build
their output paths dynamically. The declared `output:` lists capture every file
consumed downstream; the first end-to-end run in the conda environment is the
point to tighten any remaining input/output lists that Snakemake flags. This is
expected and is the one place where the DAG is finalised against real execution.

## Testing

```bash
pytest                     # unit tests + golden-number regression guards
./quality-gate.sh          # the full gate CI runs (ruff format + ruff check + pytest)
```

The golden-number guards (`tests/test_golden_numbers.py`) pin the published
results, so accidental drift fails the suite.

## Documentation

Detailed documentation is in [`docs/`](docs/):

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Data and provenance](docs/data-and-provenance.md)
- [Development](docs/development.md)

## Development workflow

- `main` is the protected, releasable branch; work on topic branches
  (`type/short-description`) and open pull requests.
- CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.10–3.12.
- Install the git hooks with `pre-commit install` to run the same checks locally
  (including secret scanning).
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

## Roadmap

- Commit a `conda-lock` file for byte-level environment reproducibility.
- Publish a Zenodo record for the moment-0 map set and wire a DOI-based fetch
  rule into the workflow.
- Tag a `v0.1.0` release and archive the software via the Zenodo–GitHub
  integration on publication of the paper.
- Optionally publish the `docs/` to Read the Docs.

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report security issues privately as
described in [`SECURITY.md`](SECURITY.md).

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite both this software and the
accompanying paper, and the upstream data (Jones et al. 2023, Zenodo 6909872).

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).

## Acknowledgements

- Built on the scientific Python ecosystem: NumPy, SciPy, pandas, Astropy,
  Matplotlib, and emcee.
- Orchestrated with [Snakemake](https://snakemake.github.io/); the manuscript is
  compiled with [tectonic](https://tectonic-typesetting.github.io/).
- Upstream MeerKAT HI data cubes and SoFiA masks of Hickson compact groups are
  from Jones et al. (2023), archived at
  [Zenodo 6909872](https://zenodo.org/records/6909872).
- The isolated-galaxy comparison sample is from the AMIGA project.
