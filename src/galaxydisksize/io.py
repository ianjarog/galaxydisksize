"""Loaders for the measurement tables and group configuration files.

These helpers wrap :func:`pandas.read_csv` and a YAML reader with light
validation so that downstream code fails early and clearly when a column or
field is missing, rather than deep inside an analysis step.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns expected in a per-galaxy measurement table. Additional columns are
# preserved; only these are required.
REQUIRED_MEASUREMENT_COLUMNS = (
    "galaxy",
    "hi_diameter_kpc",
    "hi_mass",
    "optical_diameter_kpc",
    "distance_mpc",
)


def load_measurements(path: str | Path, *, require_upper_limit_flag: bool = False) -> pd.DataFrame:
    """Load a per-galaxy HI measurement table.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a CSV file with one row per galaxy.
    require_upper_limit_flag : bool, optional
        If ``True``, also require an ``is_upper_limit`` column (present in the
        augmented tables that include beam-size upper limits). Default ``False``.

    Returns
    -------
    pandas.DataFrame
        The measurement table.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a required column is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"measurement table not found: {path}")
    table = pd.read_csv(path)

    required = list(REQUIRED_MEASUREMENT_COLUMNS)
    if require_upper_limit_flag:
        required.append("is_upper_limit")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {missing}")
    return table


def load_group_config(path: str | Path) -> dict:
    """Load a YAML configuration file describing the galaxy groups.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a YAML file (for example ``config_hcg_galaxies.yaml``).

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ImportError
        If :mod:`pyyaml` is not installed.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
        raise ImportError(
            "load_group_config requires PyYAML; install it with `pip install pyyaml`."
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)
