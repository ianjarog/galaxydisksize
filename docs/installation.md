# Installation

## Requirements

- **Python ≥ 3.10** (the conda environment pins 3.12).
- Core scientific stack: NumPy, SciPy, pandas, Astropy, Matplotlib, emcee, PyYAML.
- **Snakemake ≥ 8** to run the workflow.
- A LaTeX engine to compile the manuscript: the conda environment installs
  **tectonic**; any standard TeX distribution also works.
- Optional: conda/mamba (recommended), Docker or Apptainer for containers.

## Recommended: conda environment

```bash
conda env create -f environment.yml
conda activate galaxydisksize
pip install -e .
```

This installs the package, the workflow tooling, and the LaTeX engine in one
environment.

## Alternative: pip only

If you manage your own environment and TeX:

```bash
python -m pip install -e ".[workflow,plots]"
```

Optional dependency groups (see `pyproject.toml`):

- `plots` — corner plots for the size-mass posterior.
- `cutouts` — per-galaxy postage-stamp figures (pulls cutouts from image servers).
- `workflow` — Snakemake.
- `dev` — pytest, pytest-cov, ruff.

## Containers

Reproducible images are defined in [`containers/`](../containers):

```bash
# Docker
docker build -t galaxydisksize -f containers/Dockerfile .

# Apptainer / Singularity (HPC)
apptainer build galaxydisksize.sif containers/Apptainer.def
```

## Verify the installation

```bash
pytest          # unit tests + golden-number regression guards
snakemake -n    # dry run: prints the DAG without building anything
```
