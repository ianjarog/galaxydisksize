#!/usr/bin/env bash
#
# Quality gate for galaxydisksize.
#
# Runs the checks the project standardises on. Set up the dev tooling first:
#
#     pip install -e ".[dev]"        # ruff, pytest, pytest-cov
#
# Design notes
# ------------
# * Formatting and linting are BOTH done with ruff (see [tool.ruff] in
#   pyproject.toml: line length 100, numpydoc convention). We deliberately do
#   not also run black -- black defaults to an 88-column style and would fight
#   ruff format, reformatting the whole tree back and forth.
# * The library (src/galaxydisksize) is held to the full rule set, including
#   numpydoc docstrings; the driver scripts (scripts/) are held to every *code*
#   rule but not the docstring-style rules. The archive/ of superseded one-off
#   scripts is excluded. All of that lives in pyproject.toml, so `ruff check .`
#   here enforces exactly that policy.
# * Heavier static-analysis tools (mypy --strict, bandit, vulture, pip-audit,
#   radon) are NOT part of the required gate: the scientific code is not fully
#   type-annotated, so mypy --strict would drown real issues in noise, and the
#   others are not project dependencies. They are run as optional, advisory
#   steps below *only if* they happen to be installed, and never fail the gate.
#
# Exit status is non-zero if any required check fails, so this is safe to call
# from CI or a pre-push hook.

set -euo pipefail
cd "$(dirname "$0")"

fail=0
step() { printf '\n=== %s ===\n' "$1"; }

# --- Required checks (blocking) -------------------------------------------- #

step "ruff format --check  (formatting)"
ruff format --check . || fail=1

step "ruff check  (lint: code rules everywhere, docstrings on the library)"
ruff check . || fail=1

step "pytest  (unit tests + golden-number regressions)"
if python -c "import pytest_cov" >/dev/null 2>&1; then
    # Coverage of the library only; the scripts are integration-tested by the
    # Snakemake pipeline, not by unit tests. 65% leaves headroom over the
    # current ~70% so a small change does not trip the gate; raise over time.
    python -m pytest --cov=galaxydisksize --cov-report=term-missing --cov-fail-under=65 || fail=1
else
    echo "(pytest-cov not installed -- running without coverage)"
    python -m pytest || fail=1
fi

# --- Optional advisory checks (never fail the gate) ------------------------ #

step "optional static analysis  (advisory; only runs if the tool is installed)"
if command -v mypy >/dev/null 2>&1; then
    echo "[mypy] (library only, non-strict)"
    mypy src || true
else
    echo "mypy not installed -- skip"
fi
if command -v vulture >/dev/null 2>&1; then
    echo "[vulture] (dead-code hints)"
    vulture src scripts --min-confidence 80 || true
else
    echo "vulture not installed -- skip"
fi

# --- Verdict --------------------------------------------------------------- #

if [ "$fail" -ne 0 ]; then
    printf '\nQUALITY GATE FAILED\n'
    exit 1
fi
printf '\nQUALITY GATE PASSED\n'
