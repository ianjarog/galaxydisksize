"""Regression guards on the published numbers.

These tests read the committed fit summaries under ``products/`` and assert the
key values quoted in the manuscript. They protect against silent drift when the
analysis scripts or their inputs change: if a refit moves one of these numbers,
a test fails and the change must be reviewed deliberately.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "products"

pytestmark = pytest.mark.skipif(
    not (PRODUCTS / "mass_size_consistency_test_summary.json").exists(),
    reason="committed products/ summaries not present",
)


def test_combined_size_mass_fit():
    """Combined-sample size-mass relation used to infer D_HI for AMIGA."""
    summary = json.loads((PRODUCTS / "mass_size_consistency_test_summary.json").read_text())
    combined = summary["table_consistency_fits"]["Combined sample"]
    assert combined["combined_sample_slope"] == pytest.approx(0.5078, abs=5e-3)
    assert combined["combined_sample_intercept"] == pytest.approx(-3.304, abs=2e-2)
    assert combined["combined_sample_scatter"] == pytest.approx(0.0650, abs=5e-3)
    assert combined["combined_sample_n"] == 727


def test_residual_baseline():
    """HI-to-optical baseline slope, intercept, and scatter."""
    stats = json.loads((PRODUCTS / "hcg_residual_statistics.json").read_text())
    assert stats["baseline_slope"] == pytest.approx(0.704, abs=1e-2)
    assert stats["baseline_intercept"] == pytest.approx(0.691, abs=2e-2)
    assert stats["baseline_sigma"] == pytest.approx(0.153, abs=5e-3)
