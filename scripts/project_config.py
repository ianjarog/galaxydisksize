"""Shared, machine-independent path resolution for the analysis scripts.

The catalogue-tier scripts read large external inputs (HI cubes, SoFiA masks,
moment-0 maps, kinematic-parameter files) that are not shipped with the
repository. Rather than hard-code absolute paths, those locations are resolved
here from environment variables, with repository-relative defaults under
``data/external``. Override any of them per machine, for example::

    export GALAXYDISKSIZE_EXTERNAL=/scratch/me/hcg_data      # root for all of them
    export GALAXYDISKSIZE_SOFIA_MASKS=/scratch/me/SoFiA_masks  # a single one

See ``config/data_sources.yaml`` for what each directory should contain and
where to obtain it (Zenodo).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
PRODUCTS_DIR = PROJECT_ROOT / "products"


def external_root() -> Path:
    """Root directory for the large external inputs.

    ``$GALAXYDISKSIZE_EXTERNAL`` if set, else ``<repo>/data/external``.
    """
    value = os.environ.get("GALAXYDISKSIZE_EXTERNAL")
    return Path(value).expanduser() if value else DATA_DIR / "moment0_maps"


def external_dir(env_var: str, default_subdir: str) -> Path:
    """Resolve one external-input directory.

    Parameters
    ----------
    env_var : str
        Environment variable that, if set, gives the directory directly.
    default_subdir : str
        Sub-directory of :func:`external_root` used when ``env_var`` is unset.

    Returns
    -------
    pathlib.Path
        The resolved directory (not guaranteed to exist).
    """
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else external_root() / default_subdir


def font_dirs() -> list[str]:
    """Font search directories from ``$GALAXYDISKSIZE_FONT_DIR``.

    Returns an empty list when the variable is unset, so Matplotlib falls back
    to its default fonts.
    """
    value = os.environ.get("GALAXYDISKSIZE_FONT_DIR")
    return [value] if value else []
