# galaxydisksize documentation

This directory holds the detailed documentation. The top-level
[`README.md`](../README.md) is the entry point and quickstart; the pages here go
deeper.

## Contents

- [Installation](installation.md) — environment, package, and the LaTeX toolchain.
- [Usage](usage.md) — the two analysis tiers, Snakemake targets, and the library API.
- [Data and provenance](data-and-provenance.md) — what is committed, what comes
  from Zenodo, and the diameter-error modes.
- [Development](development.md) — workflow rules, the quality gate, and how the
  golden-number guards protect published results.

## What this project is

`galaxydisksize` is both a reusable Python package and a reproducible Snakemake
workflow behind a study of HI disc truncation in Hickson compact groups (HCGs)
relative to isolated AMIGA galaxies. From committed moment-0 maps it measures HI
diameters, fits the HI size-mass relation, computes size residuals about the
HI-to-optical baseline, runs left-censored survival statistics for the
non-detections, and compiles the manuscript PDF — every figure, table, and inline
number flowing from the same committed inputs.
