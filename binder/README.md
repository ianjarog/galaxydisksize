# Binder configuration

These files let [mybinder.org](https://mybinder.org) build a ready-to-run
environment for the demonstration notebook in a web browser, with no local
installation.

- `environment.yml` -- the conda environment (Python plus the scientific
  dependencies and JupyterLab).
- `postBuild` -- installs `galaxydisksize` from the cloned repository.

Launch badge (update the URL once the repository is on GitHub):

```
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ianjamasimanana/galaxydisksize/HEAD?labpath=notebooks/demo.ipynb)
```

The Binder session covers the interactive demonstration only. The full,
checkpointed pipeline and the manuscript PDF build are driven by Snakemake in the
conda or container environment described in the top-level `README.md`.
