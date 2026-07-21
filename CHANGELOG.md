# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository hygiene for release: CI workflow (ruff + pytest), Dependabot,
  issue/PR templates, `CODEOWNERS`, `SECURITY.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `.editorconfig`, `.pre-commit-config.yaml`,
  `.env.example`, and `docs/`.

### Changed
- The AMIGA/CIG optical-diameter catalogue is now documented purely as a set of
  committed machine-readable files (`data/amiga_full_catalogue_logd25.csv`,
  `data/cig-d25-w-error.txt`). README and `docs/data-and-provenance.md` describe
  their provenance without referencing any database access.

### Removed
- `scripts/export_amiga_optical_diameters.py` and the `AMIGA_DB_*` scaffolding in
  `.env.example`. The workflow never queried the internal IAA AMIGA database — it
  reads the frozen catalogue files above — so no database connection code, host
  names, or credentials ship with the public repository.

### Security
- The public repository no longer contains any database connection code or
  credential scaffolding (host, user, password, or port).

## [0.1.0] - 2026-06-27

### Added
- Initial release of the `galaxydisksize` package and reproducible Snakemake
  workflow accompanying the study of HI disc truncation in Hickson compact
  groups relative to isolated AMIGA galaxies.
- Reusable library: HI size-mass fitting (nine linear estimators), size
  residuals about the HI-to-optical baseline, surface-density conversions, and
  left-censored survival statistics.
- Workflow that reproduces every figure, table, and inline number of the
  manuscript from committed inputs, plus an opt-in catalogue tier that
  re-derives the measurement CSVs from the moment-0 maps.
- Committed moment-0 maps and measurement tables; container recipes
  (Docker, Apptainer); Binder configuration; unit tests and golden-number
  regression guards.

[Unreleased]: https://github.com/ianjarog/galaxydisksize/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ianjarog/galaxydisksize/releases/tag/v0.1.0
