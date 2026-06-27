#!/usr/bin/env python3
"""
HI Diameter and Mass Analysis Pipeline
======================================

This script measures HI diameters and masses for isolated (CIG) and interacting
(HCG) galaxies using moment-0 maps and SoFiA masks.

The uncertainty on HI diameters is estimated using a Monte Carlo approach that
accounts for spatially correlated noise imposed by the synthesized beam.

Author: Roger Ianjamasimanana
Date: 27-06-2026
"""

import glob
import json
import logging
import math
import os
from dataclasses import dataclass

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import font_manager
from scipy.ndimage import generic_filter

# Optional, only needed for specific steps; imported lazily so the moment-0 ->
# diameter measurement runs without them:
#   - aplpy: per-galaxy postage-stamp plotting (imported inside the plot helper)
#   - analysis_tools.delheader / image_cutouts.cut_fits: HCG cube handling
try:
    from analysis_tools.functions import delheader
except ModuleNotFoundError:
    delheader = None

try:
    from image_cutouts import cut_fits
except ModuleNotFoundError:
    cut_fits = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set to DEBUG for more verbose output during troubleshooting
# logger.setLevel(logging.DEBUG)

import project_config  # noqa: E402  (sibling module; scripts/ is on sys.path)

PROJECT_ROOT = project_config.PROJECT_ROOT
DATA_DIR = project_config.DATA_DIR
FIGURES_DIR = project_config.FIGURES_DIR


# =============================================================================
# Configuration and Constants
# =============================================================================


@dataclass
class DirectoryConfig:
    """Directory paths for data files.

    The large external inputs default to sub-directories of ``data/external``
    and are overridable via the environment variables named below (see
    ``scripts/project_config.py`` and ``config/data_sources.yaml``).
    """

    kinpars: str = str(project_config.external_dir("GALAXYDISKSIZE_KINPARS", "KINPARS"))
    distances: str = str(
        project_config.external_dir("GALAXYDISKSIZE_DISTANCES", "MOMMAPS-BAROLO-DHI")
    )
    mom_maps: str = str(project_config.external_dir("GALAXYDISKSIZE_MOM_MAPS", "CIG-MOMMAPS-SOFIA"))
    noise_cubes: str = str(project_config.external_dir("GALAXYDISKSIZE_NOISE_CUBES", "sofia2_cig"))
    sofia_masks: str = str(project_config.external_dir("GALAXYDISKSIZE_CIG_MASKS", "sofiamasks"))
    hcg_base: str = str(project_config.external_dir("GALAXYDISKSIZE_HCG_MASKS", "SoFiA_masks"))
    output_plots: str = str(FIGURES_DIR)
    config_dir: str = str(DATA_DIR)
    optical_positions: str = str(DATA_DIR / "cig_revised_positions.txt")
    optical_data_dir: str = str(DATA_DIR)
    intermediate_products: str = str(PROJECT_ROOT / "products" / "output_data")


# Physical constants for HI analysis
HI_REST_FREQ_MHZ = 1420.405751
COLUMN_DENSITY_FACTOR = 1.104e21  # N_HI conversion factor
SURFACE_DENSITY_FACTOR = 1.248e20  # Sigma_HI conversion factor
MASS_CONVERSION = 2.356e2  # M_HI = 2.356e5 * D^2 * S_int


def masks_enabled() -> bool:
    """Whether to use the SoFiA masks for beam-correlated Monte-Carlo errors.

    The masks are used by default: with a mask present the diameter uncertainty
    comes from the beam-correlated MC (the published method) and the mask-derived
    HI-mass error and column-density limit are computed as well.

    Set ``GALAXYDISKSIZE_NO_MASKS=1`` to force the mask-free path even when masks
    are present. The diameter itself is unaffected (it is the deterministic fit to
    the central contour); only the uncertainty falls back to the vertex bootstrap.

    Returns
    -------
    bool
        ``True`` to use masks (default), ``False`` to force the mask-free path.
    """
    return os.environ.get("GALAXYDISKSIZE_NO_MASKS", "").lower() not in ("1", "true", "yes")


# =============================================================================
# Configuration Loading
# =============================================================================


def load_yaml_config(filepath: str) -> dict:
    """Load configuration from YAML file."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str) -> tuple[dict, dict, dict]:
    """
    Load all configuration files.

    Returns
    -------
    isolated_config : dict
        Configuration for isolated (CIG) galaxies
    hcg_config : dict
        Configuration for HCG galaxies
    positions_config : dict
        Galaxy positions and velocities
    """
    isolated_config = load_yaml_config(os.path.join(config_dir, "config_isolated_galaxies.yaml"))
    hcg_config = load_yaml_config(os.path.join(config_dir, "config_hcg_galaxies.yaml"))
    positions_config = load_yaml_config(os.path.join(config_dir, "config_hcg_positions.yaml"))
    return isolated_config, hcg_config, positions_config


# =============================================================================
# Font and Plot Setup
# =============================================================================


class FontManager:
    """Configure matplotlib to use custom fonts."""

    def __init__(self, font_dirs: list[str] = None, font_name: str = "tex gyre heros"):
        if font_dirs is None:
            font_dirs = [d for d in [os.environ.get("GALAXYDISKSIZE_FONT_DIR")] if d]
        self.font_dirs = font_dirs
        self.font_name = font_name
        self.setup_fonts()

    def setup_fonts(self):
        """Register custom fonts with matplotlib."""
        for font_dir in self.font_dirs:
            if os.path.exists(font_dir):
                font_files = font_manager.findSystemFonts(fontpaths=[font_dir])
                for font_file in font_files:
                    font_manager.fontManager.addfont(font_file)

        mpl.rcParams["font.sans-serif"] = self.font_name
        mpl.rc("mathtext", fontset="custom", it=f"{self.font_name}:italic")


def setup_plot_style():
    """Configure matplotlib plot style."""
    try:
        FontManager()
    except Exception as e:
        logger.warning(f"Could not set up custom fonts: {e}")


# =============================================================================
# Unit Conversions
# =============================================================================


def logd25_to_arcmin(logd25: float) -> float:
    """Convert RC2 logD25 value to diameter in arcminutes."""
    return (10**logd25) * 0.1


def arcmin_to_logd25(diameter_arcmin: float) -> float:
    """Convert diameter in arcminutes to RC2 logD25 format."""
    if diameter_arcmin <= 0:
        raise ValueError("Diameter must be positive")
    return math.log10(diameter_arcmin / 0.1)


def arcsec_to_kpc(arcsec: float, distance_mpc: float) -> float:
    """Convert angular size in arcseconds to physical size in kpc."""
    return arcsec * 4.848 * distance_mpc / 1000


# =============================================================================
# HI Sensitivity Calculations
# =============================================================================


def hi_column_density_sensitivity(
    rms_mjy: float,
    beam_major_arcsec: float,
    beam_minor_arcsec: float,
    channel_width_kms: float,
    line_width_kms: float | None = None,
    n_sigma: float = 3.0,
    freq_mhz: float = HI_REST_FREQ_MHZ,
) -> float:
    """
    Calculate HI column density sensitivity limit.

    Parameters
    ----------
    rms_mjy : float
        RMS noise per channel in mJy/beam
    beam_major_arcsec, beam_minor_arcsec : float
        Beam axes in arcseconds
    channel_width_kms : float
        Velocity channel width in km/s
    line_width_kms : float, optional
        Total line width for integration (km/s)
    n_sigma : float
        Detection significance level
    freq_mhz : float
        Observing frequency in MHz

    Returns
    -------
    float
        Column density sensitivity in cm^-2
    """
    if line_width_kms is None:
        line_width_kms = channel_width_kms

    # Wavelength in cm
    c_cm_s = 2.99792458e10
    wavelength_cm = c_cm_s / (freq_mhz * 1e6)

    # Brightness temperature RMS
    # T_b [K] = 1.36 * S [mJy/beam] * (lambda [cm])^2 / (theta_maj * theta_min)
    t_b_rms = 1.36 * rms_mjy * wavelength_cm**2 / (beam_major_arcsec * beam_minor_arcsec)

    # Column density: N_HI = 1.823e18 * T_b * sqrt(delta_v * W)
    n_hi = 1.823e18 * t_b_rms * np.sqrt(channel_width_kms * line_width_kms) * n_sigma

    return n_hi


# =============================================================================
# FITS File Operations
# =============================================================================


@dataclass
class FitsInfo:
    """Container for FITS file information."""

    data: np.ndarray
    header: fits.Header
    wcs: WCS
    pixscale: float  # arcsec/pixel
    bmaj: float | None  # beam major axis in arcsec
    bmin: float | None  # beam minor axis in arcsec
    nx: int
    ny: int


def load_fits_image(
    filepath: str, galaxy_name: str = None, samples_config: dict = None
) -> FitsInfo:
    """
    Load a FITS image and extract metadata.

    Parameters
    ----------
    filepath : str
        Path to FITS file
    galaxy_name : str, optional
        Galaxy identifier for beam lookup
    samples_config : dict, optional
        Configuration with beam information

    Returns
    -------
    FitsInfo
        Container with image data and metadata
    """
    with fits.open(filepath) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header.copy()

    # Handle negative/zero values
    data = data.astype(float)
    data[data <= 0] = np.nan

    pixscale = abs(header.get("CDELT1", 1.0)) * 3600  # Convert deg to arcsec

    # Get beam parameters
    bmaj = header.get("BMAJ")
    bmin = header.get("BMIN")

    if bmaj is not None:
        bmaj *= 3600  # Convert deg to arcsec
    if bmin is not None:
        bmin *= 3600

    # Fallback to config if beam not in header
    if (bmaj is None or bmin is None) and samples_config and galaxy_name:
        if galaxy_name in samples_config:
            bmaj = samples_config[galaxy_name].get("BMAJ", bmaj)
            bmin = samples_config[galaxy_name].get("BMIN", bmin)

    ny, nx = data.shape[-2:]
    wcs = WCS(header).celestial if data.ndim >= 2 else WCS(header)

    return FitsInfo(
        data=data, header=header, wcs=wcs, pixscale=pixscale, bmaj=bmaj, bmin=bmin, nx=nx, ny=ny
    )


def reduce_fits_image(
    filepath: str, galaxy_name: str, crop_pixels: int, samples_config: dict = None
) -> tuple[str, FitsInfo]:
    """
    Crop a FITS image by removing border pixels.

    Parameters
    ----------
    filepath : str
        Path to FITS file
    galaxy_name : str
        Galaxy identifier
    crop_pixels : int
        Number of pixels to remove from each edge
    samples_config : dict, optional
        Configuration with beam information

    Returns
    -------
    output_path : str
        Path to cropped FITS file
    fits_info : FitsInfo
        Information about the cropped image
    """
    fits_info = load_fits_image(filepath, galaxy_name, samples_config)

    ny, nx = fits_info.ny, fits_info.nx

    # Validate crop dimensions
    if crop_pixels <= 0:
        # No cropping needed, return original
        return filepath, fits_info

    # Check if cropping would leave a valid image
    new_ny = ny - 2 * crop_pixels
    new_nx = nx - 2 * crop_pixels

    if new_ny < 10 or new_nx < 10:
        logger.warning(
            f"{galaxy_name}: Requested crop ({crop_pixels}px) too large for "
            f"image size ({nx}x{ny}). Reducing crop to preserve data."
        )
        # Reduce crop to leave at least 50% of the image
        max_crop = min(ny, nx) // 4
        crop_pixels = max(0, max_crop)
        new_ny = ny - 2 * crop_pixels
        new_nx = nx - 2 * crop_pixels

        if new_ny < 10 or new_nx < 10:
            logger.warning(f"{galaxy_name}: Skipping crop entirely, image too small")
            return filepath, fits_info

    cropped_data = fits_info.data[crop_pixels : ny - crop_pixels, crop_pixels : nx - crop_pixels]

    # Verify cropped data is valid
    if cropped_data.size == 0 or np.all(np.isnan(cropped_data)):
        logger.warning(f"{galaxy_name}: Cropped image is empty, using original")
        return filepath, fits_info

    # Update header
    new_header = fits_info.header.copy()
    new_header["CRPIX1"] -= crop_pixels
    new_header["CRPIX2"] -= crop_pixels

    output_path = filepath.replace(".fits", "_smaller.fits")
    fits.writeto(output_path, cropped_data, new_header, overwrite=True)

    logger.debug(f"{galaxy_name}: Cropped from {nx}x{ny} to {new_nx}x{new_ny}")

    return output_path, load_fits_image(output_path, galaxy_name, samples_config)


def extract_medians_from_cube(filepath: str) -> np.ndarray:
    """Extract median values from each channel of a data cube."""
    with fits.open(filepath) as hdul:
        cube = np.squeeze(hdul[0].data)

    if cube.ndim != 3:
        raise ValueError("FITS file must contain a 3D data cube")

    return np.nanmedian(cube, axis=(1, 2))


def find_emission_channels(mask_cube: np.ndarray) -> tuple[int, int]:
    """
    Find the range of channels containing emission.

    Parameters
    ----------
    mask_cube : ndarray
        3D mask array

    Returns
    -------
    chan_start, chan_end : int
        Start and end channel indices
    """
    num_channels = mask_cube.shape[0]
    chan_start = None
    chan_end = None

    for channel in range(num_channels):
        if not np.isnan(mask_cube[channel]).all():
            if chan_start is None:
                chan_start = channel
            chan_end = channel

    if chan_start is None or chan_end is None:
        raise ValueError("No emission found in data cube")

    # Add padding
    chan_start = max(0, chan_start - 3)
    chan_end = min(num_channels - 1, chan_end + 3)

    return int(chan_start), int(chan_end)


# =============================================================================
# Noise Estimation
# =============================================================================


def median_noise_from_sofia_catalog(
    catalog_path: str, score_col: int = 12, value_col: int = 23
) -> float:
    """
    Extract noise value from SoFiA catalog for the best source.

    Selects the row with highest score and returns the noise value.
    """
    data = np.genfromtxt(
        catalog_path, dtype=str, comments="#", delimiter=None, autostrip=True, ndmin=2
    )

    if data.size == 0:
        raise ValueError(f"Empty catalog file: {catalog_path}")

    ncols = data.shape[1]
    if score_col >= ncols or value_col >= ncols:
        raise IndexError(f"Catalog has {ncols} columns, need cols {score_col} and {value_col}")

    # Parse score column
    scores = []
    for x in data[:, score_col]:
        try:
            scores.append(float(x))
        except ValueError:
            scores.append(np.nan)
    scores = np.array(scores)

    if np.all(np.isnan(scores)):
        raise ValueError(f"No valid scores in catalog: {catalog_path}")

    best_row = int(np.nanargmax(scores))
    return float(data[best_row, value_col])


def load_cig_optical_positions(filepath: str) -> dict:
    """
    Load CIG optical positions from the revised positions file.

    Returns dictionary mapping galaxy name to (RA, Dec) in HMS/DMS format.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Optical positions file not found: {filepath}")
        return {}

    data = np.loadtxt(filepath, dtype=str)
    result = {}

    for row in data:
        key = f"{row[0]}{row[1]}"  # Combine "CIG" and its number
        # Format RA and Dec with leading zeros
        ra_parts = [f"{int(part):02}" if part.isdigit() else part for part in row[8:11]]
        dec_parts = [f"{int(part):02}" if part.isdigit() else part for part in row[11:14]]

        ra = ":".join(ra_parts)
        dec = ":".join(dec_parts)
        result[key] = (ra, dec)

    return result


def get_kinematic_position(
    galaxy_name: str, kinpars_dir: str, crop_pixels: int = 0
) -> tuple[float, float] | None:
    """
    Load kinematic center position from Barolo output.

    Parameters
    ----------
    galaxy_name : str
        Galaxy identifier
    kinpars_dir : str
        Directory containing kinematic parameter files
    crop_pixels : int
        Number of pixels cropped from each edge (to adjust coordinates)

    Returns
    -------
    x_kin, y_kin : float or None
        Kinematic center in pixel coordinates (adjusted for cropping)
    """
    pattern = os.path.join(kinpars_dir, galaxy_name, "*final2.txt")
    matches = glob.glob(pattern)

    if not matches:
        return None

    try:
        kinpars = np.loadtxt(matches[0])
        x_kin = kinpars[:, 9][0] - crop_pixels
        y_kin = kinpars[:, 10][0] - crop_pixels
        return x_kin, y_kin
    except Exception as e:
        logger.debug(f"Could not load kinematic parameters for {galaxy_name}: {e}")
        return None


def get_noise_level(
    galaxy_name: str,
    noise_cube_dir: str,
    group_id: str = None,
    wrong_unit_galaxies: list[str] = None,
    telescope: str = None,
) -> float:
    """
    Get the median noise level for a galaxy.

    Attempts to read from SoFiA catalog first, falls back to cube statistics.
    """
    if group_id is None:
        group_id = galaxy_name

    # Try SoFiA catalog first
    try:
        pattern = os.path.join(noise_cube_dir, group_id, f"{galaxy_name}*_SOFIA_cat.txt")
        matches = glob.glob(pattern)
        if not matches:
            pattern = os.path.join(noise_cube_dir, group_id, f"{group_id}_SOFIA_cat.txt")
            matches = glob.glob(pattern)

        if matches:
            median_noise = median_noise_from_sofia_catalog(matches[0])
            logger.info(f"{galaxy_name}: Noise from SoFiA catalog = {median_noise:.6f} Jy/beam")
        else:
            raise FileNotFoundError("No SoFiA catalog found")

    except Exception as e:
        # Fallback to noise cube statistics
        logger.warning(f"{galaxy_name}: Catalog read failed ({e}), using cube median")
        noise_cube_path = os.path.join(noise_cube_dir, group_id, f"{galaxy_name}_SOFIA_noise.fits")
        if not os.path.exists(noise_cube_path):
            noise_cube_path = os.path.join(noise_cube_dir, group_id, f"{group_id}_SOFIA_noise.fits")

        noise_per_channel = extract_medians_from_cube(noise_cube_path)
        median_noise = np.nanmedian(noise_per_channel)

        # Apply WSRT scaling if needed
        if telescope == "WSRT":
            median_noise *= 5e-3

    # Correct unit for known problematic galaxies
    if wrong_unit_galaxies and galaxy_name in wrong_unit_galaxies:
        median_noise /= 1000.0
        logger.info(f"{galaxy_name}: Applied unit correction (divided by 1000)")

    return median_noise


# =============================================================================
# Surface Density Noise Map
# =============================================================================


def create_surface_density_noise_map(
    mask_cube_path: str,
    rms_jy: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    crop_pixels: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Create a 2D map of per-pixel surface density uncertainty.

    Implements Equation 26 from the paper:
    σ_Σ(x,y) = K * σ_ch * Δv * sqrt(N_ch(x,y)) / (B_maj * B_min)

    Parameters
    ----------
    mask_cube_path : str
        Path to 3D mask or masked cube
    rms_jy : float
        Per-channel RMS noise in Jy/beam
    bmaj_arcsec, bmin_arcsec : float
        Beam axes in arcseconds
    crop_pixels : tuple
        (size1, size2) pixels to crop from edges

    Returns
    -------
    sigma_sd : ndarray
        2D array of surface density uncertainties in M_sun/pc^2
    """
    with fits.open(mask_cube_path) as hdul:
        cube = np.squeeze(hdul[0].data)
        header = hdul[0].header

        # Get channel width in km/s
        cdelt3 = abs(header.get("CDELT3", 1.0))
        dv_kms = cdelt3 / 1000.0 if cdelt3 > 1000 else cdelt3

    if cube.ndim != 3:
        raise ValueError("Mask must be a 3D cube")

    # Count channels contributing at each pixel
    valid = np.isfinite(cube) & (cube != 0)
    nchan_map = valid.sum(axis=0)

    # Apply cropping
    size1, size2 = crop_pixels
    if size1 > 0 or size2 > 0:
        ny, nx = nchan_map.shape

        # Validate crop dimensions
        new_ny = ny - 2 * size2
        new_nx = nx - 2 * size1

        if new_ny < 2 or new_nx < 2:
            logger.warning(
                f"Noise map crop ({size1}, {size2}) too large for shape ({nx}, {ny}). "
                f"Skipping crop."
            )
        else:
            nchan_map = nchan_map[size2 : ny - size2, size1 : nx - size1]

    # Check for valid data
    if nchan_map.size == 0:
        raise ValueError("Channel count map is empty after cropping")

    # Conversion factor: K = 1.104e21 / 1.248e20 ≈ 8.85
    K = COLUMN_DENSITY_FACTOR / SURFACE_DENSITY_FACTOR

    # Integrated intensity noise (Jy km/s per beam)
    sigma_intensity = rms_jy * dv_kms * np.sqrt(nchan_map.astype(float))

    # Convert to surface density noise (M_sun/pc^2)
    beam_area = bmaj_arcsec * bmin_arcsec
    sigma_sd = np.zeros_like(nchan_map, dtype=float)
    valid_pix = nchan_map > 0
    sigma_sd[valid_pix] = K * sigma_intensity[valid_pix] / beam_area

    return sigma_sd


# =============================================================================
# Distance Calculations
# =============================================================================


def get_distance_from_velocity(
    ra_deg: float, dec_deg: float, cz_kms: float, calculator: str = "CF3"
) -> float:
    """
    Get distance from Cosmicflows calculator.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Coordinates in degrees
    cz_kms : float
        Recession velocity in km/s
    calculator : str
        'NAM' or 'CF3'

    Returns
    -------
    float
        Distance in Mpc
    """
    query = {
        "coordinate": [float(ra_deg), float(dec_deg)],
        "system": "equatorial",
        "parameter": "velocity",
        "value": float(cz_kms),
    }
    headers = {"Content-type": "application/json"}
    api_url = f"http://edd.ifa.hawaii.edu/{calculator}calculator/api.php"

    try:
        response = requests.get(api_url, data=json.dumps(query), headers=headers)
        output = json.loads(response.text)
        return int(np.round(output["observed"]["distance"][0]))
    except Exception as e:
        logger.warning(f"Distance calculation failed: {e}")
        return np.nan


# =============================================================================
# Mass Error Calculation
# =============================================================================


def calculate_hi_mass_error(
    mask_path: str,
    rms_jy: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    pix_arcsec: float,
    distance_mpc: float,
    correlation_factor: float = 1.0,
) -> float:
    """
    Calculate HI mass uncertainty using error propagation.

    σ_M_HI = 2.356e5 * D^2 * σ_ch * Δv * sqrt(N_eff)

    where N_eff is the effective number of independent beam-channel elements.
    """
    with fits.open(mask_path) as hdul:
        mask3d = np.squeeze(hdul[0].data) > 0
        header = hdul[0].header

        cdelt3 = abs(header.get("CDELT3", 1.0))
        dv_kms = cdelt3 / 1000.0 if cdelt3 > 1000 else cdelt3

    # Per-pixel channel counts
    nchan_map = mask3d.sum(axis=0)
    spatial_mask = nchan_map > 0

    if spatial_mask.sum() == 0:
        raise ValueError("Mask has no valid pixels")

    # Beam area in pixels
    beam_area_pix = (1.133 * bmaj_arcsec * bmin_arcsec) / (pix_arcsec**2)

    # Effective number of independent elements
    n_eff = nchan_map[spatial_mask].sum() / beam_area_pix

    # Integrated flux uncertainty
    sigma_flux = correlation_factor * rms_jy * dv_kms * np.sqrt(n_eff)

    # Mass uncertainty
    sigma_mass = MASS_CONVERSION * (distance_mpc**2) * sigma_flux

    return sigma_mass


# =============================================================================
# Ellipse Fitting with Beam-Correlated Noise Monte Carlo
# =============================================================================


class FitEllipse:
    def __init__(
        self, sd, gal=None, threshold=1.0, sigma_sd_map=None, bmaj_pix=None, bmin_pix=None, rng=None
    ):
        self.gal = gal
        self.level = float(threshold)
        self.sd = np.asarray(sd, dtype=float)
        self.sigma_sd_map = sigma_sd_map  # 2D array or None
        self.bmaj_pix = bmaj_pix
        self.bmin_pix = bmin_pix
        cs = plt.contour(sd, [1])
        contour_segments = cs.allsegs[0]
        self.rng = np.random.default_rng(rng)
        if not contour_segments:
            raise RuntimeError("No contour found at level 1.")

        # Compute image center
        img_center = np.array(sd.shape) / 2

        # Find the contour closest to image center
        def centroid(seg):
            return np.mean(seg, axis=0)

        [np.linalg.norm(centroid(seg) - img_center[::-1]) for seg in contour_segments]
        # select contour closest to center, with enough points
        main_candidates = [seg for seg in contour_segments if seg.shape[0] > 10]
        if not main_candidates:
            raise RuntimeError("No good contour found.")
        if gal in ["HCG96a", "CIG85"]:
            idx = np.argmax(
                [np.linalg.norm(centroid(seg) - img_center[::-1]) for seg in main_candidates]
            )
        else:
            idx = np.argmin(
                [np.linalg.norm(centroid(seg) - img_center[::-1]) for seg in main_candidates]
            )
        largest = main_candidates[idx]

        all_verts = largest
        cx, cy = all_verts.mean(axis=0)
        # Distances
        d = np.hypot(all_verts[:, 0] - cx, all_verts[:, 1] - cy)
        # Garder l'intervalle 5–95%
        if gal in ["CIG147"]:
            low, high = np.percentile(d, [25, 99])
        else:
            low, high = np.percentile(d, [1, 99])
        sel = (d >= low) & (d <= high)
        if gal in ["HCG25b", "CIG587"]:
            self.segments = main_candidates  # just for contour plots.
            self.largest_vertices = np.vstack(main_candidates)  # fit all points, no selection
        else:
            self.largest_vertices = all_verts[sel]
            self.segments = [all_verts[sel]]  # just for contour plots
        self.x, self.y = self.largest_vertices[:, 0], self.largest_vertices[:, 1]

    @staticmethod
    def fit_ellipse_from_points(x, y):
        """
        Fit an ellipse given arrays of x and y points.
        Returns the concatenated coefficients.
        """
        D1 = np.vstack([x**2, x * y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = np.dot(D1.T, D1)
        S2 = np.dot(D1.T, D2)
        S3 = np.dot(D2.T, D2)
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        # Select the eigenvector that produces a valid ellipse (conic constant positive).
        # np.linalg.eig may return complex dtype with negligible imaginary parts, so
        # select on the real part and cast the (physically real) solution to float.
        con = (4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2).real
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.real(np.concatenate((ak, T @ ak)).ravel()).astype(float)

    def _main_contour_vertices(self, Z):
        """Return vertices for the main contour at self.level on image Z, or None."""
        fig = plt.figure()
        cs = plt.contour(Z, [self.level])
        plt.close(fig)
        if not cs.allsegs or not cs.allsegs[0]:
            return None
        segs = [seg for seg in cs.allsegs[0] if seg.shape[0] > 10]
        if not segs:
            return None
        img_center = np.array(Z.shape) / 2

        def centroid(seg):
            return np.mean(seg, axis=0)

        if self.gal in ["HCG96a", "CIG85"]:
            idx = np.argmax([np.linalg.norm(centroid(s) - img_center[::-1]) for s in segs])
        else:
            idx = np.argmin([np.linalg.norm(centroid(s) - img_center[::-1]) for s in segs])
        largest = segs[idx]
        cx, cy = largest.mean(axis=0)
        d = np.hypot(largest[:, 0] - cx, largest[:, 1] - cy)
        low, high = np.percentile(d, [1, 99])
        sel = (d >= low) & (d <= high)
        return largest[sel]

    def _generate_beam_correlated_noise(self):
        """Generate a beam-correlated noise realization scaled by sigma_sd_map."""
        if self.sigma_sd_map is None:
            raise ValueError("sigma_sd_map is required for beam-correlated noise generation.")
        if self.bmaj_pix is None or self.bmin_pix is None:
            raise ValueError(
                "Beam size in pixels is required for beam-correlated noise generation."
            )

        ny, nx = self.sd.shape
        white_noise = self.rng.standard_normal((ny, nx))

        sigma_maj = self.bmaj_pix / 2.355
        sigma_min = self.bmin_pix / 2.355

        kernel = Gaussian2DKernel(x_stddev=sigma_min, y_stddev=sigma_maj)
        correlated_noise = convolve(white_noise, kernel, boundary="wrap")

        noise_std = np.nanstd(correlated_noise)
        if noise_std > 0:
            return self.sigma_sd_map * correlated_noise / noise_std
        return np.zeros_like(correlated_noise)

    def _get_smoothed_signal(self, kernel_beams=3.0):
        """Estimate the underlying signal so MC realizations do not double-count noise."""
        if self.bmaj_pix is None or self.bmin_pix is None:
            raise ValueError("Beam size in pixels is required to smooth the signal map.")

        kernel_size = int(np.ceil(kernel_beams * max(self.bmaj_pix, self.bmin_pix)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        data = self.sd.copy()
        source_mask = np.isfinite(data) & (data > 0)

        if not np.any(source_mask):
            raise ValueError(
                "Cannot smooth signal map because no positive finite source pixels were found."
            )

        # Ignore zero-valued exterior pixels when estimating the underlying signal.
        # This preserves the source contour better for compact/asymmetric HCG maps.
        masked_data = data.copy()
        masked_data[~source_mask] = np.nan
        smoothed = generic_filter(
            masked_data, function=np.nanmedian, size=kernel_size, mode="constant", cval=np.nan
        )

        # Keep the original source values where the local filtered window has no valid pixels.
        missing = ~np.isfinite(smoothed) & source_mask
        smoothed[missing] = data[missing]
        smoothed[~source_mask] = 0.0
        return smoothed

    def _run_mc_realizations(self, base_signal, n_bootstrap, resample_points):
        """Run MC contour perturbations and return ellipse parameter samples."""
        aps, bps, phis, x0s, y0s = [], [], [], [], []
        rng = self.rng

        for _ in range(n_bootstrap):
            noisy = base_signal + self._generate_beam_correlated_noise()
            verts = self._main_contour_vertices(noisy)
            if verts is None or len(verts) < 6:
                continue
            x, y = verts[:, 0], verts[:, 1]
            if resample_points:
                idx = rng.integers(0, len(x), size=len(x), endpoint=False)
                x, y = x[idx], y[idx]
            try:
                coeffs = FitEllipse.fit_ellipse_from_points(x, y)
                x0, y0, ap, bp, e, phi = self.cart_to_pol(coeffs)
            except Exception:
                continue
            aps.append(ap)
            bps.append(bp)
            phis.append(phi)
            x0s.append(x0)
            y0s.append(y0)

        return aps, bps, phis, x0s, y0s

    def bootstrap_errors(self, n_bootstrap=500, mc_noise=False, resample_points=True):
        """
        If mc_noise=True and sigma_sd_map is provided, do Monte-Carlo:
          sd_i = sd + N(0, sigma_sd_map); re-contour at level; refit ellipse.
        Else: fall back to vertex-resampling bootstrap on the fixed contour.
        """
        aps, bps, phis, x0s, y0s = [], [], [], [], []
        rng = self.rng

        if mc_noise:
            if self.sigma_sd_map is None:
                raise ValueError("sigma_sd_map is required for mc_noise=True.")
            smoothed_signal = self._get_smoothed_signal()
            aps, bps, phis, x0s, y0s = self._run_mc_realizations(
                smoothed_signal, n_bootstrap, resample_points
            )
            self.mc_mode = "smoothed_signal"
            self.mc_success_count = len(aps)

            min_successes = max(10, n_bootstrap // 20)
            if self.mc_success_count < min_successes:
                logger.warning(
                    f"{self.gal}: only {self.mc_success_count}/{n_bootstrap} successful "
                    "beam-correlated MC fits using the smoothed signal; retrying with the "
                    "original signal map."
                )
                aps, bps, phis, x0s, y0s = self._run_mc_realizations(
                    self.sd, n_bootstrap, resample_points
                )
                self.mc_mode = "original_signal"
                self.mc_success_count = len(aps)

            logger.info(
                f"{self.gal}: beam-correlated MC successful fits = "
                f"{self.mc_success_count}/{n_bootstrap} (mode={self.mc_mode})"
            )
        else:
            # original: bootstrap the fixed contour vertices
            X = self.largest_vertices
            N = len(X)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, N, size=N, endpoint=False)
                x_bs, y_bs = X[idx, 0], X[idx, 1]
                try:
                    coeffs = FitEllipse.fit_ellipse_from_points(x_bs, y_bs)
                    x0, y0, ap, bp, e, phi = self.cart_to_pol(coeffs)
                except Exception:
                    continue
                aps.append(ap)
                bps.append(bp)
                phis.append(phi)
                x0s.append(x0)
                y0s.append(y0)
            self.mc_mode = "fixed_contour"
            self.mc_success_count = len(aps)

        # Save stats
        self.ap_mean = np.nanmean(aps) if aps else np.nan
        self.ap_std = np.nanstd(aps) if aps else np.nan
        self.bp_mean = np.nanmean(bps) if bps else np.nan
        self.bp_std = np.nanstd(bps) if bps else np.nan
        self.phi_mean = np.nanmean(phis) if phis else np.nan
        self.phi_std = np.nanstd(phis) if phis else np.nan
        self.x0_mean = np.nanmean(x0s) if x0s else np.nan
        self.x0_std = np.nanstd(x0s) if x0s else np.nan
        self.y0_mean = np.nanmean(y0s) if y0s else np.nan
        self.y0_std = np.nanstd(y0s) if y0s else np.nan

    def fit_ellipse(self):
        D1 = np.vstack([self.x**2, self.x * self.y, self.y**2]).T
        D2 = np.vstack([self.x, self.y, np.ones(len(self.x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        # See fit_ellipse_from_points: take the real part of the complex eig output.
        con = (4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2).real
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.real(np.concatenate((ak, T @ ak)).ravel()).astype(float)

    def cart_to_pol(self, coeffs):
        a = coeffs[0]
        b = coeffs[1] / 2
        c = coeffs[2]
        d = coeffs[3] / 2
        f = coeffs[4] / 2
        g = coeffs[5]
        den = b**2 - a * c
        if den > 0:
            raise ValueError("coeffs do not represent an ellipse: b^2 - 4ac must be negative!")
        x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den
        num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
        fac = np.sqrt((a - c) ** 2 + 4 * b**2)
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))
        width_gt_height = ap >= bp
        if not width_gt_height:
            ap, bp = bp, ap
        e = np.sqrt(1 - (bp / ap) ** 2)
        if b == 0:
            phi = 0 if a < c else np.pi / 2
        else:
            phi = np.arctan((2.0 * b) / (a - c)) / 2
            if a > c:
                phi += np.pi / 2
        if not width_gt_height:
            phi += np.pi / 2
        phi = phi % np.pi
        return x0, y0, ap, bp, e, phi

    def get_ellipse_pts(self, params, npts=5000, tmin=0, tmax=2 * np.pi):
        x0, y0, ap, bp, e, phi = params
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return x, y


def skycoord_to_pixels(ra: str, dec: str, wcs: WCS, unit: str = "hourangle") -> tuple[float, float]:
    """Convert RA/Dec string to pixel coordinates."""
    if unit == "hourangle":
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")

    return wcs.world_to_pixel(coord)


def skycoord_degrees_to_pixels(ra_deg: float, dec_deg: float, wcs: WCS) -> tuple[float, float]:
    """Convert RA/Dec in degrees to pixel coordinates."""
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return wcs.world_to_pixel(coord)


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_surface_density_map(
    fits_path: str,
    galaxy_name: str,
    ellipse_x: np.ndarray = None,
    ellipse_y: np.ndarray = None,
    center_x: float = None,
    center_y: float = None,
    optical_x: float = None,
    optical_y: float = None,
    kinematic_x: float = None,
    kinematic_y: float = None,
    # Green contour (the one that is fitted) can be passed either as world arrays
    # OR as pixel segments + WCS (same behaviour as the original code).
    fitted_verts_ra: np.ndarray = None,
    fitted_verts_dec: np.ndarray = None,
    fitted_segments_pix: list[np.ndarray] = None,
    output_path: str = None,
    contour_fits_path: str = None,
    contour_levels: list[float] = None,
):
    """
    Create surface density map with:
    - yellow contours (display-only)
    - lime contour (the extracted contour actually fitted)
    - black fitted ellipse
    - three crosses: optical (yellow), kinematic (black), ellipse centre (lime)

    Notes
    -----
    The fit is NEVER performed on the yellow contours. They are only for display.
    """
    import aplpy  # lazy: only the postage-stamp plotting needs it

    # Load header/data (for vmin/vmax + WCS)
    with fits.open(fits_path) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header

    v_min = np.nanmin(data[data > 0]) if np.any(data > 0) else 0
    v_max = np.nanmax(data)

    fig = plt.figure(figsize=(5, 5))
    f1 = aplpy.FITSFigure(fits_path, figure=fig)
    f1.show_colorscale(vmin=v_min, vmax=v_max, aspect="auto", cmap=plt.cm.RdBu_r)

    # Yellow contours (display-only)
    contour_src = contour_fits_path or fits_path
    levels = contour_levels if contour_levels is not None else [1.0]
    try:
        f1.show_contour(contour_src, levels=levels, colors="yellow", linewidths=1.3)
    except Exception:
        # If contour fails for any reason, keep the plot instead of crashing
        pass

    ax = plt.gca()

    # Crosses (pixel coordinates; matches original workflow)
    if optical_x is not None and optical_y is not None:
        ax.plot(optical_x, optical_y, "x", ms=14, color="yellow")
    if kinematic_x is not None and kinematic_y is not None:
        ax.plot(kinematic_x, kinematic_y, "x", ms=14, color="black")
    if center_x is not None and center_y is not None:
        ax.plot(center_x, center_y, "x", ms=14, color="lime")

    # Plot the fitted contour (lime) in WORLD coordinates
    if fitted_verts_ra is not None and fitted_verts_dec is not None:
        ax.plot(
            fitted_verts_ra,
            fitted_verts_dec,
            "-",
            color="lime",
            lw=2,
            transform=ax.get_transform("world"),
        )
    elif fitted_segments_pix is not None:
        wcs = WCS(header).celestial
        for seg in fitted_segments_pix:
            if seg is None or len(seg) == 0:
                continue
            ra_w, dec_w = wcs.wcs_pix2world(seg[:, 0], seg[:, 1], 0)
            ax.plot(ra_w, dec_w, "-", color="lime", lw=2, transform=ax.get_transform("world"))

    # Fitted ellipse (black) in pixel coordinates
    if ellipse_x is not None and ellipse_y is not None:
        ax.plot(ellipse_x, ellipse_y, "-", color="black", lw=1.3)

    # Label
    ax.text(
        0.08,
        0.9,
        galaxy_name,
        color="black",
        fontsize=15,
        transform=ax.transAxes,
        fontweight="light",
    )

    # Tick formatting
    ax.tick_params(direction="in", length=8.7, width=1.3, pad=10)
    ax.tick_params(which="minor", length=5)

    # Colorbar
    try:
        f1.add_colorbar()
        f1.colorbar.show()
        f1.colorbar.set_pad(0.0)
        f1.colorbar.set_axis_label_text(r"$\mathrm{\Sigma_{HI}(M_{\odot}~pc^{-2})}$")
        f1.colorbar.set_axis_label_pad(15)
        f1.colorbar.set_location("top")
        ax_cbar = plt.gca()
        ax_cbar.tick_params(direction="in", length=3, width=1)
    except Exception:
        pass

    # Beam
    try:
        f1.add_beam()
        f1.beam.set_color("white")
        f1.beam.set_edgecolor("black")
    except Exception:
        pass

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=400)

    plt.close(fig)


# =============================================================================
# Results Summary
# =============================================================================


def summarize_mass_errors(fractional_errors: list[float]) -> dict:
    """
    Compute summary statistics for HI mass uncertainties.

    Parameters
    ----------
    fractional_errors : list
        Fractional uncertainties (e.g., 0.03 = 3%)

    Returns
    -------
    dict
        Statistics and formatted text summary
    """
    errors = np.asarray(fractional_errors, dtype=float)

    # Auto-detect percent vs fraction
    if np.nanmax(errors) > 1.5:
        errors = errors / 100.0

    stats = {
        "median": np.nanmedian(errors),
        "p16": np.nanpercentile(errors, 16),
        "p25": np.nanpercentile(errors, 25),
        "p75": np.nanpercentile(errors, 75),
        "p84": np.nanpercentile(errors, 84),
        "p90": np.nanpercentile(errors, 90),
        "p95": np.nanpercentile(errors, 95),
        "n": np.sum(np.isfinite(errors)),
    }

    stats["latex_text"] = (
        f"Across the sample, the median fractional HI mass uncertainty is "
        f"{stats['median']:.2%}; the central 68% interval spans "
        f"{stats['p16']:.2%}--{stats['p84']:.2%}, and 95% of the sample "
        f"has uncertainties $\\leq$ {stats['p95']:.2%}. "
        f"The interquartile range is {stats['p25']:.2%}--{stats['p75']:.2%}."
    )

    return stats


# =============================================================================
# Main Processing Functions
# =============================================================================


def process_isolated_galaxy(
    galaxy_name: str, config: dict, positions_data: dict, dirs: DirectoryConfig
) -> dict:
    """
    Process a single isolated (CIG) galaxy.

    Returns dictionary with HI diameter, mass, and uncertainties.
    """
    logger.info(f"Processing isolated galaxy: {galaxy_name}")

    samples = config["samples"]
    wrong_unit_galaxies = config.get("median_noise_wrong_unit", [])

    gal_config = samples[galaxy_name]
    crop_pixels = gal_config["size"]
    telescope = gal_config["telescope"]
    bmaj = gal_config["BMAJ"]
    bmin = gal_config["BMIN"]

    # Find moment-0 map
    mom0_pattern = os.path.join(dirs.mom_maps, f"{galaxy_name}*mom0.fits")
    mom0_files = glob.glob(mom0_pattern)
    if not mom0_files:
        raise FileNotFoundError(f"No moment-0 map found matching: {mom0_pattern}")
    mom0_path = mom0_files[0]
    logger.debug(f"  Found moment-0 map: {mom0_path}")

    # Decide the error mode. With the SoFiA mask present (and not disabled) the
    # uncertainty comes from the beam-correlated Monte-Carlo; otherwise it falls
    # back to the mask-free vertex bootstrap on the fixed contour. The diameter is
    # identical in either case.
    sofia_mask_path = os.path.join(dirs.sofia_masks, f"{galaxy_name}_sofiamask.fits")
    use_mask = masks_enabled() and os.path.exists(sofia_mask_path)
    if not use_mask:
        reason = (
            "GALAXYDISKSIZE_NO_MASKS set"
            if masks_enabled() is False
            else (f"SoFiA mask not found ({sofia_mask_path})")
        )
        logger.warning(
            f"{galaxy_name}: {reason}; using vertex-bootstrap diameter errors. "
            "Mask-derived HI-mass error and column-density limit will be NaN."
        )

    # Load and reduce FITS
    fits_info = load_fits_image(mom0_path, galaxy_name, samples)
    logger.debug(f"  Original image size: {fits_info.nx}x{fits_info.ny}")

    if crop_pixels > 0:
        mom0_reduced_path, fits_info_reduced = reduce_fits_image(
            mom0_path, galaxy_name, crop_pixels, samples
        )
        logger.debug(f"  Reduced image size: {fits_info_reduced.nx}x{fits_info_reduced.ny}")
    else:
        fits_info_reduced = fits_info

    # Apply telescope-specific flux scaling
    if telescope == "WSRT":
        flux_data = fits_info_reduced.data * 5e-3
    else:
        flux_data = fits_info_reduced.data

    # Convert to column density and surface density
    nhi = COLUMN_DENSITY_FACTOR * flux_data / (bmaj * bmin)
    logger.debug(f"{galaxy_name}: column-density map shape={nhi.shape}")
    sd = nhi / SURFACE_DENSITY_FACTOR

    # Save surface density map (use reduced header since sd was computed from reduced data)
    sd_path = os.path.join(dirs.intermediate_products, f"{galaxy_name}_sd.fits")
    fits.writeto(sd_path, sd, fits_info_reduced.header, overwrite=True)

    # Load the surface density info (no additional cropping needed - already cropped)
    sd_info = load_fits_image(sd_path, galaxy_name, samples)
    logger.debug(f"  Surface density map size: {sd_info.nx}x{sd_info.ny}")

    # Get noise level (drives the mask-based error pathways only). In the
    # mask-free path a missing noise estimate is non-fatal.
    if use_mask:
        noise_rms = get_noise_level(
            galaxy_name,
            dirs.noise_cubes,
            wrong_unit_galaxies=wrong_unit_galaxies,
            telescope=telescope,
        )
    else:
        try:
            noise_rms = get_noise_level(
                galaxy_name,
                dirs.noise_cubes,
                wrong_unit_galaxies=wrong_unit_galaxies,
                telescope=telescope,
            )
        except Exception as e:
            logger.debug(f"{galaxy_name}: noise unavailable in mask-free mode ({e})")
            noise_rms = np.nan

    # Build the per-pixel surface-density noise map from the SoFiA mask. Only
    # needed for the beam-correlated MC; skipped entirely in the mask-free path.
    sigma_sd_map = None
    if use_mask:
        # First check the SoFiA mask dimensions
        with fits.open(sofia_mask_path) as hdul:
            mask_shape = np.squeeze(hdul[0].data).shape
            logger.debug(f"  SoFiA mask shape: {mask_shape}")

        # Verify crop is appropriate for mask dimensions
        if len(mask_shape) == 3:
            mask_ny, mask_nx = mask_shape[1], mask_shape[2]
        else:
            mask_ny, mask_nx = mask_shape[0], mask_shape[1]

        effective_crop = crop_pixels
        if mask_nx - 2 * crop_pixels < 10 or mask_ny - 2 * crop_pixels < 10:
            logger.warning(
                f"{galaxy_name}: Mask dimensions ({mask_nx}x{mask_ny}) too small for "
                f"crop={crop_pixels}. Adjusting crop."
            )
            effective_crop = min(crop_pixels, (min(mask_nx, mask_ny) - 10) // 2)
            effective_crop = max(0, effective_crop)
            logger.info(f"  Adjusted crop from {crop_pixels} to {effective_crop}")

        sigma_sd_map = create_surface_density_noise_map(
            sofia_mask_path, noise_rms, bmaj, bmin, crop_pixels=(effective_crop, effective_crop)
        )

        logger.debug(f"  Sigma SD map shape: {sigma_sd_map.shape}")

        # Ensure sigma_sd_map matches sd_info dimensions
        if sigma_sd_map.shape != sd_info.data.shape:
            logger.warning(
                f"{galaxy_name}: Shape mismatch - SD map {sd_info.data.shape} vs "
                f"noise map {sigma_sd_map.shape}. Resizing noise map."
            )
            # Resize sigma_sd_map to match sd_info
            from scipy.ndimage import zoom

            zoom_factors = (
                sd_info.data.shape[0] / sigma_sd_map.shape[0],
                sd_info.data.shape[1] / sigma_sd_map.shape[1],
            )
            sigma_sd_map = zoom(sigma_sd_map, zoom_factors, order=1)

    # Fit ellipse with Monte Carlo errors
    bmaj_pix = bmaj / sd_info.pixscale
    bmin_pix = bmin / sd_info.pixscale

    # Validate surface density map has valid data
    valid_data = sd_info.data[np.isfinite(sd_info.data) & (sd_info.data > 0)]
    if valid_data.size == 0:
        raise ValueError("Surface density map has no valid positive values")

    logger.debug(
        f"  SD map stats: shape={sd_info.data.shape}, "
        f"min={np.nanmin(sd_info.data):.2f}, max={np.nanmax(sd_info.data):.2f}, "
        f"n_valid={valid_data.size}"
    )

    # Check if there's emission above the threshold
    above_threshold = np.sum(sd_info.data > 1.0)
    if above_threshold < 10:
        raise ValueError(
            f"Only {above_threshold} pixels above 1 M_sun/pc^2 threshold. "
            f"Cannot fit reliable contour."
        )

    # Keep the SD map exactly like the original fitter expects (zeros outside).
    # Do NOT introduce NaNs here, because NaNs fragment the contour into tiny segments.
    sd_data_fit = np.squeeze(sd_info.data).astype(float).copy()
    # Important: keep the original behaviour for fitting: finite values only, zero outside.
    # NaNs/inf fragment contours into many tiny segments.
    sd_data_fit[~np.isfinite(sd_data_fit)] = 0.0
    sd_data_fit[sd_data_fit <= 0] = 0.0
    fitter = FitEllipse(
        sd_data_fit,
        galaxy_name,
        threshold=1.0,
        sigma_sd_map=sigma_sd_map,
        bmaj_pix=bmaj_pix,
        bmin_pix=bmin_pix,
        rng=42,
    )

    coeffs = fitter.fit_ellipse()
    x0, y0, semi_major, semi_minor, eccentricity, phi = fitter.cart_to_pol(coeffs)
    # mc_noise=use_mask: beam-correlated MC when the mask is available, else the
    # mask-free vertex bootstrap on the fixed contour.
    fitter.bootstrap_errors(n_bootstrap=500, mc_noise=use_mask, resample_points=True)

    # Get distance
    cig_index = int(galaxy_name[3:])
    results_df = pd.read_csv(os.path.join(dirs.distances, "RESULTS_OPT.csv"))
    distance_mpc = results_df["PHYS_DISTANCE_decimal"][cig_index - 1]
    optical_path = os.path.join(dirs.optical_data_dir, "cig-d25-w-error.txt")
    optical_diameter = np.loadtxt(optical_path)
    optical_dict = {int(row[0]): float(row[1]) for row in optical_diameter}  # log D25
    optical_dict_error = {int(row[0]): float(row[2]) for row in optical_diameter}  # e_log D25 (dex)
    # cig-d25-w-error.txt stores log D25 (RC2/HyperLEDA convention) and its error in dex.
    # D25[arcmin] = 0.1 * 10**logD25  (logd25_to_arcmin); previously logD25 was wrongly
    # used directly as arcmin, making the AMIGA diameters too small.
    optical_diameter_arcmin = logd25_to_arcmin(optical_dict[cig_index])
    optical_diameter_kpc = optical_diameter_arcmin * 60 * 4.848 * distance_mpc / 1000
    # Error on a log quantity: sigma(D25) = D25 * ln(10) * e_logD25.
    optical_diameter_kpc_err = optical_diameter_kpc * np.log(10.0) * optical_dict_error[cig_index]
    # Calculate diameter (with beam deconvolution)
    diameter_arcsec = 2 * semi_major * sd_info.pixscale
    diameter_arcsec_err = (
        2
        * (fitter.ap_std if np.isfinite(getattr(fitter, "ap_std", np.nan)) else 0.0)
        * sd_info.pixscale
    )

    beam_product = bmaj * bmin
    if diameter_arcsec**2 > beam_product:
        diameter_corrected = np.sqrt(diameter_arcsec**2 - beam_product)
    else:
        diameter_corrected = np.sqrt(beam_product)  # Upper limit

    diameter_kpc = arcsec_to_kpc(diameter_corrected, distance_mpc)
    diameter_kpc_err = arcsec_to_kpc(diameter_arcsec_err, distance_mpc) * (
        diameter_arcsec / diameter_corrected if diameter_arcsec**2 > beam_product else 1.0
    )

    # Calculate HI mass
    beam_area_pix = 1.133 * bmaj * bmin / (sd_info.pixscale**2)
    integrated_flux = flux_data / beam_area_pix
    hi_mass = MASS_CONVERSION * distance_mpc**2 * np.nansum(integrated_flux)

    # Calculate mass error and column-density sensitivity. Both are derived from
    # the SoFiA mask (channel counts / channel width), so they are only available
    # in the mask-based path; otherwise they are reported as NaN.
    if use_mask:
        mass_error = calculate_hi_mass_error(
            sofia_mask_path, noise_rms, bmaj, bmin, sd_info.pixscale, distance_mpc
        )

        with fits.open(sofia_mask_path) as hdul:
            chan_width = abs(hdul[0].header.get("CDELT3", 1.0)) / 1000.0

        coldens_limit = hi_column_density_sensitivity(
            noise_rms * 1000, bmaj, bmin, chan_width, line_width_kms=20, n_sigma=3
        )
    else:
        mass_error = np.nan
        coldens_limit = np.nan

    # Linear resolution
    linear_res = np.sqrt(bmaj**2 + bmin**2) * 4.848 * distance_mpc / 1000

    # Get optical position
    optical_positions = load_cig_optical_positions(dirs.optical_positions)
    opt_x, opt_y = None, None
    if galaxy_name in optical_positions:
        ra, dec = optical_positions[galaxy_name]
        try:
            opt_x, opt_y = skycoord_to_pixels(ra, dec, sd_info.wcs)
        except Exception as e:
            logger.debug(f"Could not convert optical position for {galaxy_name}: {e}")

    # Get kinematic position
    kin_pos = get_kinematic_position(galaxy_name, dirs.kinpars, crop_pixels)
    kin_x, kin_y = kin_pos if kin_pos else (None, None)

    # Generate plot
    # Eccentricity (fallback if not returned)
    e = float(np.sqrt(max(0.0, 1.0 - (semi_minor / semi_major) ** 2))) if semi_major != 0 else 0.0
    ellipse_x, ellipse_y = fitter.get_ellipse_pts((x0, y0, semi_major, semi_minor, e, phi))

    # Convert to NaN version for plotting
    sd_nan_path = sd_path.replace(".fits", "_nan.fits")
    sd_data = sd_info.data.copy()
    sd_data[sd_data <= 0] = np.nan
    fits.writeto(sd_nan_path, sd_data, sd_info.header, overwrite=True)

    plot_path = os.path.join(dirs.output_plots, f"{galaxy_name}_mom0_onesolar.pdf")
    plot_surface_density_map(
        sd_nan_path,
        galaxy_name,
        ellipse_x=ellipse_x,
        ellipse_y=ellipse_y,
        center_x=x0,
        center_y=y0,
        optical_x=opt_x,
        optical_y=opt_y,
        kinematic_x=kin_x,
        kinematic_y=kin_y,
        fitted_segments_pix=fitter.segments,
        output_path=plot_path,
    )

    return {
        "galaxy": galaxy_name,
        "cig_index": cig_index,
        "hi_diameter_kpc": diameter_kpc,
        "hi_diameter_err_kpc": diameter_kpc_err,
        "optical_diameter_kpc": optical_diameter_kpc,
        "optical_diameter_err_kpc": optical_diameter_kpc_err,
        "hi_mass": hi_mass,
        "hi_mass_err": mass_error,
        "distance_mpc": distance_mpc,
        "linear_resolution_kpc": linear_res,
        "coldens_limit": coldens_limit,
        "fractional_mass_error": 100 * mass_error / hi_mass,
    }


def extract_galaxy_names_from_members(hcgs_members: dict) -> list[str]:
    """Extract individual galaxy names from HCG member dictionary."""
    names = []
    for members in hcgs_members.values():
        for member in members:
            name = member.split("_")[0]
            names.append(name)
    return names


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main processing function."""
    setup_plot_style()

    # Configuration
    dirs = DirectoryConfig()

    # Load configurations
    isolated_config, hcg_config, positions_config = load_all_configs(dirs.config_dir)

    # Create output directory
    os.makedirs(dirs.output_plots, exist_ok=True)
    os.makedirs(dirs.intermediate_products, exist_ok=True)

    # Process isolated galaxies
    isolated_results = []
    galaxy_list = list(isolated_config["samples"].keys())

    logger.info(f"Processing {len(galaxy_list)} isolated galaxies...")

    for galaxy_name in galaxy_list:
        try:
            result = process_isolated_galaxy(galaxy_name, isolated_config, positions_config, dirs)
            isolated_results.append(result)

            # Handle potential NaN in error
            err_str = (
                f"± {result['hi_diameter_err_kpc']:.2f}"
                if result["hi_diameter_err_kpc"] and not np.isnan(result["hi_diameter_err_kpc"])
                else "(no error)"
            )
            logger.info(
                f"{galaxy_name}: D_HI = {result['hi_diameter_kpc']:.2f} {err_str} kpc, "
                f"M_HI = {result['hi_mass']:.2e} M_sun"
            )
        except FileNotFoundError as e:
            logger.error(f"SKIPPED {galaxy_name}: Missing file - {e}")
            continue
        except Exception as e:
            logger.error(f"FAILED {galaxy_name}: {type(e).__name__} - {e}")
            import traceback

            logger.debug(traceback.format_exc())
            continue

    logger.info(f"Successfully processed {len(isolated_results)}/{len(galaxy_list)} galaxies")

    # Save results
    if isolated_results:
        df = pd.DataFrame(isolated_results)
        results_csv_path = os.path.join(dirs.config_dir, "isolated_galaxies_results.csv")
        df.to_csv(results_csv_path, index=False)

        # Summary statistics
        fractional_errors = [r["fractional_mass_error"] for r in isolated_results]
        stats = summarize_mass_errors(fractional_errors)
        logger.info(f"Mass error summary: {stats['latex_text']}")

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
