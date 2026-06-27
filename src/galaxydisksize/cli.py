"""Command-line front-end to the :mod:`galaxydisksize` library.

This is a thin convenience wrapper for quick, interactive use on a measurement
table; the reproducible paper pipeline is driven by Snakemake, not by this CLI.

Examples
--------
Print the package version::

    galaxydisksize version

Fit the size-mass relation on a CSV with ``hi_mass`` and ``hi_diameter_kpc``
columns::

    galaxydisksize fit-mass-size measurements.csv --seed 42
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .io import load_measurements
from .masssize import fit_mass_size


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_fit_mass_size(args: argparse.Namespace) -> int:
    table = load_measurements(args.csv)
    log_mass = np.log10(table["hi_mass"].to_numpy(dtype=float))
    log_diameter = np.log10(table["hi_diameter_kpc"].to_numpy(dtype=float))
    fit = fit_mass_size(log_mass, log_diameter, seed=args.seed)
    pct = fit.percentiles()
    print(f"n_data   = {fit.n_data}")
    print(f"slope    = {fit.slope:.4f}  (16-84: {pct['slope'][0]:.4f} .. {pct['slope'][2]:.4f})")
    print(
        f"intercept= {fit.intercept:.4f}  "
        f"(16-84: {pct['intercept'][0]:.4f} .. {pct['intercept'][2]:.4f})"
    )
    print(
        f"scatter  = {fit.scatter:.4f}  (16-84: {pct['scatter'][0]:.4f} .. {pct['scatter'][2]:.4f})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``version`` and ``fit-mass-size`` subcommands.
    """
    parser = argparse.ArgumentParser(prog="galaxydisksize", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="print the package version")
    version_parser.set_defaults(func=_cmd_version)

    fit_parser = subparsers.add_parser(
        "fit-mass-size", help="fit the HI size-mass relation on a measurement CSV"
    )
    fit_parser.add_argument("csv", help="CSV with hi_mass and hi_diameter_kpc columns")
    fit_parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    fit_parser.set_defaults(func=_cmd_fit_mass_size)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``galaxydisksize`` command.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
