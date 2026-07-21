# Data and provenance

## What is committed

- **Moment-0 maps** under `data/moment0_maps/` (the AMIGA set plus small SoFiA
  noise catalogues, kinematic-centre files, and the `RESULTS_OPT` distances
  table). Re-running the measurement from these reproduces the published **HI
  diameters exactly**.
- The **measurement CSVs** that the statistics tier consumes (so the paper
  rebuilds without re-running the expensive ellipse fitting).

## What is not committed (obtain from Zenodo)

- The **SoFiA mask cubes** are large (multi-GB) and are **not** committed. They
  are needed only for the Monte-Carlo **error bars** — the diameter itself needs
  only the moment-0 map. Obtain them from
  [Zenodo 6909872](https://zenodo.org/records/6909872) (Jones et al. 2023) and
  place them at the default paths:
  - AMIGA: `data/moment0_maps/sofiamasks/{galaxy}_sofiamask.fits`
  - HCG:   `data/moment0_maps/SoFiA_masks/`

  or point `GALAXYDISKSIZE_CIG_MASKS` / `GALAXYDISKSIZE_HCG_MASKS` at an existing
  copy. These directories are git-ignored.
- The HI **data cubes** are archived in the same Zenodo record; they are cited
  for provenance, not redistributed here.

## Two diameter-error modes

The HI **diameter** is the deterministic fit to the central 1 M⊙ pc⁻² contour and
reproduces exactly with or without the masks. Only the **uncertainty** depends on
the masks:

- **With masks (default).** When the SoFiA mask is present, the diameter error is
  the beam-correlated Monte-Carlo estimate (perturb the map with mask-derived
  noise, re-contour, re-fit). The mask-derived HI-mass error and column-density
  limit are computed as well. This is the published method.
- **Mask-free (fallback).** When the mask is absent — or `GALAXYDISKSIZE_NO_MASKS=1`
  is set — the diameter error falls back to a vertex bootstrap of the fitted
  contour, which needs no mask. The mask-derived HI-mass error and column-density
  limit are then `NaN`.

## Reproducibility note

The published MC error bars predate the current measurement code, so they are
kept as-is in the committed CSVs; a re-run reproduces the diameters but produces
fresh (stochastic) error bars. The deterministic estimators are bit-identical
across rebuilds.

## Optical-diameter catalogue

The AMIGA/CIG optical diameters are provided directly as committed,
machine-readable files — the workflow reads them from disk and **never contacts
any database**:

- `data/amiga_full_catalogue_logd25.csv` — the full AMIGA/CIG catalogue, one row
  per galaxy with columns `CIG`, `logd25` (log of the optical D25 diameter),
  `E_logd25` (its uncertainty), and `PHYS_DISTANCE_decimal` (physical distance).
- `data/cig-d25-w-error.txt` — the `log D25` and error values the size
  measurement consumes (`scripts/measure_hi_disk_sizes.py`).

These values were extracted once from the internal AMIGA `CIG_RELEASE_2012`
optical database (the LEDA `logd25` field, joined to the CIG coordinate and
`RESULTS_OPT` distance tables) and frozen into the files above. That database is
an IAA-internal service with no public access, so no credentials, connection
settings, or query code ship with this repository; the frozen files are the
authoritative, reproducible input.
