"""Ellipse fitting for HI surface-density contours.

The HI diameter of a galaxy is measured by fitting an ellipse to the
1 M_sun pc^-2 contour of its moment-0 (surface-density) map. This module
separates the reusable geometry from the map-specific measurement:

* :func:`fit_ellipse_conic` and :func:`conic_to_geometric` implement the direct
  least-squares conic fit of Halir & Flusser (1998) and its conversion to
  geometric parameters. They are pure NumPy and independently testable.
* :class:`EllipseFitter` wraps a surface-density map: it extracts the relevant
  contour, fits an ellipse, and estimates uncertainties by either resampling the
  contour vertices or running beam-correlated Monte-Carlo perturbations.

Per-galaxy contour-selection choices (which contour to take, how aggressively to
trim outliers) are passed in as options rather than hard-coded, so the geometry
stays general.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class EllipseParameters:
    """Geometric parameters of a fitted ellipse, in pixel units.

    Attributes
    ----------
    x0, y0 : float
        Centre of the ellipse.
    semi_major, semi_minor : float
        Semi-major and semi-minor axis lengths (``semi_major >= semi_minor``).
    eccentricity : float
        Eccentricity ``sqrt(1 - (semi_minor / semi_major)**2)``.
    angle : float
        Position angle of the major axis in radians, in ``[0, pi)``.
    """

    x0: float
    y0: float
    semi_major: float
    semi_minor: float
    eccentricity: float
    angle: float


def fit_ellipse_conic(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Fit a general conic to points by the direct least-squares method.

    Implements the numerically stable ellipse-specific fit of Halir & Flusser
    (1998), which guarantees an elliptical solution.

    Parameters
    ----------
    x, y : array_like
        Coordinates of the points to fit. At least six points are required for
        a well-determined fit.

    Returns
    -------
    numpy.ndarray
        The six conic coefficients ``(a, b, c, d, f, g)`` of
        ``a x^2 + b x y + c y^2 + d x + f y + g = 0``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the design matrices are singular (e.g. collinear points).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design_quadratic = np.vstack([x**2, x * y, y**2]).T
    design_linear = np.vstack([x, y, np.ones(len(x))]).T
    scatter_qq = design_quadratic.T @ design_quadratic
    scatter_ql = design_quadratic.T @ design_linear
    scatter_ll = design_linear.T @ design_linear
    linear_from_quadratic = -np.linalg.inv(scatter_ll) @ scatter_ql.T
    reduced = scatter_qq + scatter_ql @ linear_from_quadratic
    constraint = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
    reduced = np.linalg.inv(constraint) @ reduced
    _, eigenvectors = np.linalg.eig(reduced)
    # The elliptical solution is the eigenvector with a positive conic constant.
    # np.linalg.eig may return complex dtype with negligible imaginary parts, so
    # select on the real part and cast the (physically real) solution to float.
    conic_constant = (4 * eigenvectors[0] * eigenvectors[2] - eigenvectors[1] ** 2).real
    quadratic_terms = eigenvectors[:, np.nonzero(conic_constant > 0)[0]]
    coeffs = np.concatenate((quadratic_terms, linear_from_quadratic @ quadratic_terms)).ravel()
    return np.real(coeffs).astype(float)


def conic_to_geometric(coeffs: ArrayLike) -> EllipseParameters:
    """Convert conic coefficients to geometric ellipse parameters.

    Parameters
    ----------
    coeffs : array_like
        The six conic coefficients ``(a, b, c, d, f, g)`` from
        :func:`fit_ellipse_conic`.

    Returns
    -------
    EllipseParameters
        Centre, axis lengths, eccentricity, and position angle.

    Raises
    ------
    ValueError
        If the coefficients do not describe an ellipse (``b^2 - a c >= 0``).
    """
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    denominator = b**2 - a * c
    if denominator > 0:
        raise ValueError("coefficients do not describe an ellipse (b^2 - a c must be < 0)")

    x0 = (c * d - b * f) / denominator
    y0 = (a * f - b * d) / denominator

    numerator = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    axis_factor = np.sqrt((a - c) ** 2 + 4 * b**2)
    semi_major = np.sqrt(numerator / denominator / (axis_factor - a - c))
    semi_minor = np.sqrt(numerator / denominator / (-axis_factor - a - c))

    major_is_first = semi_major >= semi_minor
    if not major_is_first:
        semi_major, semi_minor = semi_minor, semi_major
    eccentricity = np.sqrt(1 - (semi_minor / semi_major) ** 2)

    if b == 0:
        angle = 0.0 if a < c else np.pi / 2
    else:
        angle = np.arctan(2.0 * b / (a - c)) / 2
        if a > c:
            angle += np.pi / 2
    if not major_is_first:
        angle += np.pi / 2
    angle = angle % np.pi

    return EllipseParameters(
        x0=float(x0),
        y0=float(y0),
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        eccentricity=float(eccentricity),
        angle=float(angle),
    )


def ellipse_points(
    params: EllipseParameters, n_points: int = 5000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample points along the outline of an ellipse.

    Parameters
    ----------
    params : EllipseParameters
        The ellipse to sample.
    n_points : int, optional
        Number of points around the ellipse, by default 5000.

    Returns
    -------
    x, y : numpy.ndarray
        Coordinates tracing the ellipse outline.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    cos_t, sin_t = np.cos(t), np.sin(t)
    cos_phi, sin_phi = np.cos(params.angle), np.sin(params.angle)
    x = params.x0 + params.semi_major * cos_t * cos_phi - params.semi_minor * sin_t * sin_phi
    y = params.y0 + params.semi_major * cos_t * sin_phi + params.semi_minor * sin_t * cos_phi
    return x, y


def _contour_segments(image: NDArray[np.float64], level: float) -> list[NDArray[np.float64]]:
    """Return the contour segments of ``image`` at ``level``.

    Uses Matplotlib's contour generator without creating a visible figure.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure()
    try:
        contour_set = plt.contour(image, [level])
    finally:
        plt.close(figure)
    if not contour_set.allsegs or not contour_set.allsegs[0]:
        return []
    return [segment for segment in contour_set.allsegs[0] if segment.shape[0] > 10]


@dataclass
class _ContourSelection:
    """Configuration for choosing and trimming the contour to fit."""

    prefer_outer_contour: bool = False
    percentile_range: tuple[float, float] = (1.0, 99.0)
    fit_all_segments: bool = False


class EllipseFitter:
    """Fit an ellipse to a surface-density contour and estimate its uncertainty.

    Parameters
    ----------
    surface_density : array_like
        The 2-D moment-0 / surface-density map.
    level : float, optional
        Contour level to fit, by default 1.0 (the 1 M_sun pc^-2 isophote when the
        map is in those units).
    prefer_outer_contour : bool, optional
        If ``True`` select the contour furthest from the image centre instead of
        the closest; used for a few galaxies whose central source is not the one
        of interest. Default ``False``.
    percentile_range : tuple of float, optional
        Radial percentile range used to trim contour outliers before fitting,
        by default ``(1.0, 99.0)``.
    fit_all_segments : bool, optional
        If ``True`` fit the union of all candidate segments without trimming;
        used for a few galaxies with broken contours. Default ``False``.
    sigma_map : array_like, optional
        Per-pixel noise map, required for beam-correlated Monte-Carlo errors.
    beam_major_pix, beam_minor_pix : float, optional
        Beam FWHM in pixels, required for beam-correlated Monte-Carlo errors.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible resampling.

    Raises
    ------
    RuntimeError
        If no usable contour is found at ``level``.
    """

    def __init__(
        self,
        surface_density: ArrayLike,
        level: float = 1.0,
        *,
        prefer_outer_contour: bool = False,
        percentile_range: tuple[float, float] = (1.0, 99.0),
        fit_all_segments: bool = False,
        sigma_map: ArrayLike | None = None,
        beam_major_pix: float | None = None,
        beam_minor_pix: float | None = None,
        rng: int | np.random.Generator | None = None,
    ):
        self.surface_density = np.asarray(surface_density, dtype=float)
        self.level = float(level)
        self.selection = _ContourSelection(
            prefer_outer_contour=prefer_outer_contour,
            percentile_range=percentile_range,
            fit_all_segments=fit_all_segments,
        )
        self.sigma_map = None if sigma_map is None else np.asarray(sigma_map, dtype=float)
        self.beam_major_pix = beam_major_pix
        self.beam_minor_pix = beam_minor_pix
        self.rng = np.random.default_rng(rng)

        segments = _contour_segments(self.surface_density, self.level)
        if not segments:
            raise RuntimeError(f"no usable contour found at level {self.level}")
        self.segments = segments
        self.vertices = self._select_vertices(segments)
        self.x = self.vertices[:, 0]
        self.y = self.vertices[:, 1]

    def _select_vertices(self, segments: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Pick and trim the contour segment to fit, per the selection options."""
        if self.selection.fit_all_segments:
            return np.vstack(segments)

        image_centre = np.array(self.surface_density.shape) / 2
        distances = [
            np.linalg.norm(segment.mean(axis=0) - image_centre[::-1]) for segment in segments
        ]
        pick = np.argmax if self.selection.prefer_outer_contour else np.argmin
        chosen = segments[int(pick(distances))]

        centre_x, centre_y = chosen.mean(axis=0)
        radius = np.hypot(chosen[:, 0] - centre_x, chosen[:, 1] - centre_y)
        low, high = np.percentile(radius, self.selection.percentile_range)
        keep = (radius >= low) & (radius <= high)
        return chosen[keep]

    def fit(self) -> EllipseParameters:
        """Fit an ellipse to the selected contour vertices.

        Returns
        -------
        EllipseParameters
            The best-fit ellipse in pixel coordinates.
        """
        return conic_to_geometric(fit_ellipse_conic(self.x, self.y))

    def bootstrap_errors(
        self, n_bootstrap: int = 500, *, mc_noise: bool = False, resample_points: bool = True
    ) -> dict[str, float]:
        """Estimate axis-length uncertainties by resampling or Monte-Carlo.

        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap or Monte-Carlo realizations, by default 500.
        mc_noise : bool, optional
            If ``True`` perturb the map with beam-correlated noise and re-contour
            each realization (requires ``sigma_map`` and the beam size); otherwise
            resample the fixed contour vertices. Default ``False``.
        resample_points : bool, optional
            Whether to also resample vertices within each Monte-Carlo realization,
            by default ``True``. Ignored when ``mc_noise`` is ``False``.

        Returns
        -------
        dict
            Means and standard deviations of the semi-major and semi-minor axes:
            ``{"semi_major_mean", "semi_major_std", "semi_minor_mean",
            "semi_minor_std", "n_success"}``.
        """
        if mc_noise:
            if self.sigma_map is None:
                raise ValueError("mc_noise=True requires a sigma_map")
            base_signal = self._smoothed_signal()
            majors, minors = self._mc_realizations(base_signal, n_bootstrap, resample_points)
            if len(majors) < max(10, n_bootstrap // 20):
                # Fall back to the original (unsmoothed) signal if too few fits succeeded.
                majors, minors = self._mc_realizations(
                    self.surface_density, n_bootstrap, resample_points
                )
        else:
            majors, minors = self._vertex_bootstrap(n_bootstrap)

        return {
            "semi_major_mean": float(np.nanmean(majors)) if majors else np.nan,
            "semi_major_std": float(np.nanstd(majors)) if majors else np.nan,
            "semi_minor_mean": float(np.nanmean(minors)) if minors else np.nan,
            "semi_minor_std": float(np.nanstd(minors)) if minors else np.nan,
            "n_success": len(majors),
        }

    def _vertex_bootstrap(self, n_bootstrap: int) -> tuple[list[float], list[float]]:
        """Resample the fixed contour vertices with replacement and refit."""
        majors: list[float] = []
        minors: list[float] = []
        n_vertices = len(self.vertices)
        for _ in range(n_bootstrap):
            index = self.rng.integers(0, n_vertices, size=n_vertices)
            try:
                params = conic_to_geometric(
                    fit_ellipse_conic(self.vertices[index, 0], self.vertices[index, 1])
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            majors.append(params.semi_major)
            minors.append(params.semi_minor)
        return majors, minors

    def _mc_realizations(
        self, base_signal: NDArray[np.float64], n_bootstrap: int, resample_points: bool
    ) -> tuple[list[float], list[float]]:
        """Add beam-correlated noise, re-contour, and refit for each realization."""
        majors: list[float] = []
        minors: list[float] = []
        for _ in range(n_bootstrap):
            noisy = base_signal + self._beam_correlated_noise()
            segments = _contour_segments(noisy, self.level)
            if not segments:
                continue
            try:
                vertices = self._select_vertices(segments)
            except (ValueError, IndexError):
                continue
            if len(vertices) < 6:
                continue
            x, y = vertices[:, 0], vertices[:, 1]
            if resample_points:
                index = self.rng.integers(0, len(x), size=len(x))
                x, y = x[index], y[index]
            try:
                params = conic_to_geometric(fit_ellipse_conic(x, y))
            except (ValueError, np.linalg.LinAlgError):
                continue
            majors.append(params.semi_major)
            minors.append(params.semi_minor)
        return majors, minors

    def _beam_correlated_noise(self) -> NDArray[np.float64]:
        """Generate one beam-correlated noise realization scaled by ``sigma_map``."""
        from astropy.convolution import Gaussian2DKernel, convolve

        if self.beam_major_pix is None or self.beam_minor_pix is None:
            raise ValueError("beam_major_pix and beam_minor_pix are required for MC noise")
        white_noise = self.rng.standard_normal(self.surface_density.shape)
        kernel = Gaussian2DKernel(
            x_stddev=self.beam_minor_pix / 2.355, y_stddev=self.beam_major_pix / 2.355
        )
        correlated = convolve(white_noise, kernel, boundary="wrap")
        noise_std = np.nanstd(correlated)
        if noise_std > 0:
            return self.sigma_map * correlated / noise_std
        return np.zeros_like(correlated)

    def _smoothed_signal(self, kernel_beams: float = 3.0) -> NDArray[np.float64]:
        """Median-smooth the source so MC realizations do not double-count noise."""
        from scipy.ndimage import generic_filter

        if self.beam_major_pix is None or self.beam_minor_pix is None:
            raise ValueError("beam_major_pix and beam_minor_pix are required to smooth the signal")
        kernel_size = int(np.ceil(kernel_beams * max(self.beam_major_pix, self.beam_minor_pix)))
        kernel_size = max(3, kernel_size + (kernel_size % 2 == 0))

        source_mask = np.isfinite(self.surface_density) & (self.surface_density > 0)
        if not np.any(source_mask):
            raise ValueError("no positive finite source pixels to smooth")
        masked = self.surface_density.copy()
        masked[~source_mask] = np.nan
        smoothed = generic_filter(
            masked, function=np.nanmedian, size=kernel_size, mode="constant", cval=np.nan
        )
        missing = ~np.isfinite(smoothed) & source_mask
        smoothed[missing] = self.surface_density[missing]
        smoothed[~source_mask] = 0.0
        return smoothed
