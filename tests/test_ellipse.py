"""Tests for :mod:`galaxydisksize.ellipse`."""

import numpy as np
import pytest

from galaxydisksize.ellipse import (
    EllipseFitter,
    EllipseParameters,
    conic_to_geometric,
    ellipse_points,
    fit_ellipse_conic,
)


def _sample_ellipse(x0, y0, semi_major, semi_minor, angle, n=400):
    """Return points lying exactly on a known ellipse."""
    params = EllipseParameters(x0, y0, semi_major, semi_minor, 0.0, angle)
    return ellipse_points(params, n_points=n)


def test_conic_fit_recovers_known_ellipse():
    """Fitting points on a known ellipse recovers its parameters."""
    x, y = _sample_ellipse(12.0, -5.0, 9.0, 4.0, np.deg2rad(30.0))
    fitted = conic_to_geometric(fit_ellipse_conic(x, y))
    assert fitted.x0 == pytest.approx(12.0, abs=1e-6)
    assert fitted.y0 == pytest.approx(-5.0, abs=1e-6)
    assert fitted.semi_major == pytest.approx(9.0, abs=1e-6)
    assert fitted.semi_minor == pytest.approx(4.0, abs=1e-6)
    assert fitted.angle == pytest.approx(np.deg2rad(30.0), abs=1e-6)


def test_conic_to_geometric_orders_axes():
    """The semi-major axis is always the larger of the two."""
    x, y = _sample_ellipse(0.0, 0.0, 3.0, 7.0, 0.0)  # "minor" given larger
    fitted = conic_to_geometric(fit_ellipse_conic(x, y))
    assert fitted.semi_major >= fitted.semi_minor
    assert fitted.semi_major == pytest.approx(7.0, abs=1e-6)


def test_conic_to_geometric_rejects_non_ellipse():
    """A hyperbola-like conic raises a clear error."""
    # a x^2 - c y^2 = 1  ->  b^2 - a c > 0 (not an ellipse).
    with pytest.raises(ValueError, match="ellipse"):
        conic_to_geometric([1.0, 0.0, -1.0, 0.0, 0.0, -1.0])


def test_ellipse_fitter_on_synthetic_map():
    """EllipseFitter recovers the axes of a filled elliptical source."""
    ny, nx = 121, 121
    yy, xx = np.mgrid[0:ny, 0:nx]
    x0, y0, semi_major, semi_minor = 60.0, 60.0, 30.0, 18.0
    # Axis-aligned filled ellipse, value 2 inside so the level-1 contour sits on
    # the boundary.
    inside = ((xx - x0) / semi_major) ** 2 + ((yy - y0) / semi_minor) ** 2 <= 1.0
    surface_density = np.where(inside, 2.0, 0.0)

    fitter = EllipseFitter(surface_density, level=1.0)
    fitted = fitter.fit()
    assert fitted.x0 == pytest.approx(x0, abs=1.0)
    assert fitted.y0 == pytest.approx(y0, abs=1.0)
    assert fitted.semi_major == pytest.approx(semi_major, rel=0.05)
    assert fitted.semi_minor == pytest.approx(semi_minor, rel=0.05)
