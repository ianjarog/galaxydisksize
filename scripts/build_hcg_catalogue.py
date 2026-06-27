#!/usr/bin/env python3
"""
HCG (Interacting Galaxies) Processing Module
=============================================

This module processes Hickson Compact Group galaxies for HI diameter
and mass measurements using the beam-correlated noise Monte Carlo method.

This is designed to be used alongside measure_hi_disk_sizes.py
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from measure_hi_disk_sizes import (
    COLUMN_DENSITY_FACTOR,
    MASS_CONVERSION,
    SURFACE_DENSITY_FACTOR,
    DirectoryConfig,
    FitEllipse,
    arcsec_to_kpc,
    calculate_hi_mass_error,
    create_surface_density_noise_map,
    find_emission_channels,
    get_distance_from_velocity,
    get_noise_level,
    hi_column_density_sensitivity,
    load_all_configs,
    load_fits_image,
    logd25_to_arcmin,
    masks_enabled,
    plot_surface_density_map,
    setup_plot_style,
    skycoord_degrees_to_pixels,
    summarize_mass_errors,
)

# Optional: only needed by create_hcg_moment0_maps(), which regenerates the
# moment-0 maps from the group cubes. The default flow reads the pre-made
# moment-0 maps, so these are not required to measure HI diameters.
try:
    from analysis_tools.functions import delheader
except ModuleNotFoundError:
    delheader = None

try:
    from image_cutouts import cut_fits
except ModuleNotFoundError:
    cut_fits = None

logger = logging.getLogger(__name__)


# =============================================================================
# FITS Manipulation for HCG Data
# =============================================================================


def remove_history_entries(header: fits.Header) -> fits.Header:
    """Remove HISTORY entries related to third dimension."""
    history_entries = header.get("HISTORY", [])
    patterns_to_remove = ["PC1_3", "PC2_3", "PC3_3", "CUNIT3", "PC3_1", "PC3_2"]

    cleaned_history = [
        entry
        for entry in history_entries
        if not any(pattern in str(entry) for pattern in patterns_to_remove)
    ]

    header.remove("HISTORY", ignore_missing=True, remove_all=True)
    for entry in cleaned_history:
        header["HISTORY"] = entry

    return header


def regrid_mask_to_surface_density(
    mask_cube_path: str,
    sd_header: fits.Header,
    sd_shape: tuple[int, int],
    output_path: str,
    chan_start: int = None,
    chan_end: int = None,
    fill_value: float = 0,
) -> str:
    """
    Reproject 3D mask cube spatial grid onto surface density image grid.

    Uses nearest-neighbour sampling, preserves spectral axis.
    """
    with fits.open(mask_cube_path) as hdul:
        header_mask = hdul[0].header.copy()
        cube = np.squeeze(hdul[0].data)

    if cube.ndim != 3:
        raise ValueError("Mask must be a 3D cube")

    nz, ny_mask, nx_mask = cube.shape

    # Spectral trim
    if chan_start is None or chan_end is None:
        chan_start, chan_end = find_emission_channels(cube)

    cube = cube[chan_start : chan_end + 1, :, :]
    nz = cube.shape[0]

    # WCS transformations (spatial only)
    wcs_sd = WCS(sd_header).celestial
    wcs_mask = WCS(header_mask).celestial

    ny_sd, nx_sd = sd_shape
    x_sd = np.arange(nx_sd)
    y_sd = np.arange(ny_sd)
    xx_sd, yy_sd = np.meshgrid(x_sd, y_sd)

    # SD pixels -> world -> mask pixels
    ra, dec = wcs_sd.wcs_pix2world(xx_sd, yy_sd, 0)
    x_mask, y_mask = wcs_mask.wcs_world2pix(ra, dec, 0)

    xi = np.rint(x_mask).astype(int)
    yi = np.rint(y_mask).astype(int)

    inside = (xi >= 0) & (xi < nx_mask) & (yi >= 0) & (yi < ny_mask)

    # Allocate output cube
    out_cube = np.full((nz, ny_sd, nx_sd), fill_value, dtype=cube.dtype)
    for k in range(nz):
        layer = np.full((ny_sd, nx_sd), fill_value, dtype=cube.dtype)
        layer[inside] = cube[k, yi[inside], xi[inside]]
        out_cube[k] = layer

    # Build output header
    out_header = sd_header.copy()
    out_header["NAXIS"] = 3
    out_header["NAXIS1"] = nx_sd
    out_header["NAXIS2"] = ny_sd
    out_header["NAXIS3"] = nz

    for key in [
        "CTYPE3",
        "CRVAL3",
        "CDELT3",
        "CUNIT3",
        "CRPIX3",
        "PC3_3",
        "PC1_3",
        "PC2_3",
        "PC3_1",
        "PC3_2",
        "CD3_3",
    ]:
        if key in header_mask:
            out_header[key] = header_mask[key]

    # Adjust CRPIX3 for spectral crop
    if "CRPIX3" in out_header:
        out_header["CRPIX3"] = out_header["CRPIX3"] - chan_start

    fits.writeto(output_path, out_cube, out_header, overwrite=True)
    return output_path


def create_hcg_moment0_maps(
    hcg_config: dict, positions_config: dict, base_dir: str = "../SoFiA_masks/"
):
    """
    Create moment-0 maps for all HCG member galaxies.

    This function cuts out individual galaxy regions from group cubes
    and creates moment-0 maps from the masked data.
    """
    hcgs_members = hcg_config["hcgs_members"]
    cut_sizes = hcg_config.get("cut_sizes", {})
    galaxies = positions_config["galaxies"]

    for group_cube, members in hcgs_members.items():
        group_id = group_cube.split("_")[0]
        group_dir = os.path.join(base_dir, group_id)

        for member_file in members:
            member_id = member_file.split("_")[0]

            mask_path = os.path.join(group_dir, member_file)
            cube_path = os.path.join(group_dir, group_cube)

            if not os.path.exists(mask_path) or not os.path.exists(cube_path):
                logger.warning(f"Missing files for {member_id}")
                continue

            # Get galaxy position
            if member_id not in galaxies:
                logger.warning(f"No position data for {member_id}")
                continue

            gal_data = galaxies[member_id]
            ra_center = gal_data["ra"]
            dec_center = gal_data["dec"]

            # Determine cutout size
            rect_size = cut_sizes.get(member_id, 5)

            # Find emission channels
            with fits.open(mask_path) as hdul:
                mask_cube = np.squeeze(hdul[0].data)
            chan_start, chan_end = find_emission_channels(mask_cube)

            # Cut out regions
            mask_cut_path = mask_path.replace(".fits", f"{member_id}_cut.fits")
            cube_cut_path = cube_path.replace(".fits", f"{member_id}_cut.fits")

            cut_fits(
                cube_path,
                {"ra": ra_center, "dec": dec_center},
                rect_size,
                f_out=None,
                f_out3d=cube_cut_path,
                chanstart=chan_start,
                chanend=chan_end,
            )

            cut_fits(
                mask_path,
                {"ra": ra_center, "dec": dec_center},
                rect_size,
                f_out=None,
                f_out3d=mask_cut_path,
                chanstart=chan_start,
                chanend=chan_end,
            )

            # Create moment-0 map
            with fits.open(cube_cut_path) as hdul:
                cube_data = hdul[0].data
                cube_header = hdul[0].header

            with fits.open(mask_cut_path) as hdul:
                mask_data = hdul[0].data

            # Apply mask
            cube_data[np.isnan(mask_data)] = np.nan

            # Integrate along spectral axis
            cdelt3 = abs(cube_header.get("CDELT3", 1.0))
            mom0 = np.nansum(cube_data, axis=0) * cdelt3

            # Create 2D header
            mom0_header = cube_header.copy()
            mom0_header = delheader(mom0_header)
            mom0_header = delheader(mom0_header, "3")
            mom0_header["NAXIS"] = 2
            mom0_header = remove_history_entries(mom0_header)

            # Save files
            cubemask_path = mask_cut_path.replace(".fits", "_cubemask.fits")
            mom0_path = cube_cut_path.replace(".fits", "_mom0.fits")

            fits.writeto(cubemask_path, cube_data, cube_header, overwrite=True)
            fits.writeto(mom0_path, mom0, mom0_header, overwrite=True)

            logger.info(f"Created moment-0 map: {mom0_path}")


# =============================================================================
# HCG Galaxy Processing
# =============================================================================


def process_interacting_galaxy(
    galaxy_name: str, group_id: str, hcg_config: dict, positions_config: dict, dirs: DirectoryConfig
) -> dict | None:
    """
    Process a single interacting (HCG) galaxy.

    Parameters
    ----------
    galaxy_name : str
        Individual galaxy identifier (e.g., 'HCG2a')
    group_id : str
        Parent group identifier (e.g., 'HCG2')
    hcg_config : dict
        HCG configuration dictionary
    positions_config : dict
        Galaxy positions and velocities
    dirs : DirectoryConfig
        Directory paths

    Returns
    -------
    dict or None
        Results dictionary or None if processing fails
    """
    logger.info(f"Processing interacting galaxy: {galaxy_name} in {group_id}")

    galaxies = positions_config["galaxies"]
    logd25_dict = hcg_config["logd25"]
    logd25_err_dict = hcg_config["logd25_err"]
    phases = hcg_config["phases"]
    logm_star_dict = hcg_config.get("logm_star", {})

    # Get galaxy data
    if galaxy_name not in galaxies:
        logger.error(f"No position data for {galaxy_name}")
        return None

    gal_data = galaxies[galaxy_name]
    ra_deg = gal_data["ra"]
    dec_deg = gal_data["dec"]
    cz = gal_data["cz"]

    # Get distance
    distance_mpc = get_distance_from_velocity(ra_deg, dec_deg, cz)
    if np.isnan(distance_mpc):
        logger.error(f"Could not determine distance for {galaxy_name}")
        return None

    # Find moment-0 map
    group_dir = os.path.join(dirs.hcg_base, group_id)
    mom0_pattern = os.path.join(group_dir, f"*{galaxy_name}*mom0.fits")
    mom0_files = glob.glob(mom0_pattern)

    if not mom0_files:
        logger.error(f"No moment-0 map found for {galaxy_name}")
        return None

    mom0_path = mom0_files[0]
    fits_info = load_fits_image(mom0_path, galaxy_name)

    bmaj = fits_info.bmaj
    bmin = fits_info.bmin
    pixscale = fits_info.pixscale
    if bmaj is None or bmin is None:
        logger.error(f"No beam information for {galaxy_name}")
        return None

    # Convert to column and surface density
    flux_data = fits_info.data
    nhi = COLUMN_DENSITY_FACTOR * flux_data / (bmaj * bmin)
    sd = nhi / SURFACE_DENSITY_FACTOR

    # Save surface density map
    sd_path = os.path.join(dirs.hcg_base, f"{galaxy_name}_cut_sd.fits")
    fits.writeto(sd_path, sd, fits_info.header, overwrite=True)
    sd_info = load_fits_image(sd_path, galaxy_name)

    # Find the SoFiA cube mask. Its presence (and the GALAXYDISKSIZE_NO_MASKS
    # switch) decides the error mode: with a mask the uncertainty is the
    # beam-correlated MC; otherwise it is the mask-free vertex bootstrap. The
    # diameter is identical either way.
    mask_pattern = os.path.join(group_dir, f"*{galaxy_name}*cubemask.fits")
    mask_files = glob.glob(mask_pattern)
    use_mask = masks_enabled() and bool(mask_files)
    mask_path = mask_files[0] if mask_files else None
    if not use_mask:
        reason = (
            "GALAXYDISKSIZE_NO_MASKS set"
            if masks_enabled() is False
            else (f"no cube mask found ({mask_pattern})")
        )
        logger.warning(
            f"{galaxy_name}: {reason}; using vertex-bootstrap diameter errors. "
            "Mask-derived HI-mass error and column-density limit will be NaN."
        )

    # Get noise level (drives the mask-based error pathways only). A missing
    # noise estimate is fatal only in the mask-based path.
    try:
        noise_rms = get_noise_level(galaxy_name, dirs.noise_cubes, group_id=group_id)
    except Exception as e:
        if use_mask:
            logger.error(f"Failed to get noise for {galaxy_name}: {e}")
            return None
        logger.debug(f"{galaxy_name}: noise unavailable in mask-free mode ({e})")
        noise_rms = np.nan

    # Build the per-pixel surface-density noise map from the SoFiA mask. Only the
    # beam-correlated MC needs it; skipped in the mask-free path.
    chan_width = np.nan
    sigma_sd_map = None
    if use_mask:
        # Get channel width
        with fits.open(mask_path) as hdul:
            header = hdul[0].header
            chan_width = abs(header.get("CDELT3", 1.0)) / 1000.0  # to km/s

        # Regrid mask to SD grid
        mask_regridded_path = os.path.join(
            dirs.sofia_masks, f"{galaxy_name}_sofiamask_int_ON_SD.fits"
        )

        with fits.open(mask_path) as hdul:
            mask_cube = np.squeeze(hdul[0].data)
        chan_start, chan_end = find_emission_channels(mask_cube)

        regrid_mask_to_surface_density(
            mask_path, sd_info.header, sd_info.data.shape, mask_regridded_path, chan_start, chan_end
        )

        # Create surface density noise map
        sigma_sd_map = create_surface_density_noise_map(
            mask_regridded_path, noise_rms, bmaj, bmin, crop_pixels=(0, 0)
        )

    # Fit ellipse with Monte Carlo
    bmaj_pix = bmaj / pixscale
    bmin_pix = bmin / pixscale

    try:
        # Fit in pixel space on a finite SD map (match original code assumptions).
        sd_data_fit = np.squeeze(sd_info.data).astype(float).copy()
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

        # Uncertainties (stored on fitter as ap_std/bp_std/phi_std/x0_std/y0_std).
        # mc_noise=use_mask: beam-correlated MC with the mask, else vertex bootstrap.
        fitter.bootstrap_errors(n_bootstrap=500, mc_noise=use_mask, resample_points=True)
        semi_major_err = getattr(fitter, "ap_std", np.nan)
        # Ellipse points for plotting
        x_el, y_el = fitter.get_ellipse_pts(
            (x0, y0, semi_major, semi_minor, eccentricity, phi), npts=5000
        )
        fitted_segments_pix = getattr(fitter, "segments", None)
    except Exception as e:
        logger.error(f"Ellipse fitting failed for {galaxy_name}: {e}")
        return None

    # Calculate diameter
    diameter_arcsec = 2 * semi_major * pixscale
    diameter_arcsec_err = 2 * semi_major_err * pixscale
    beam_product = bmaj * bmin

    if diameter_arcsec**2 > beam_product:
        diameter_corrected = np.sqrt(diameter_arcsec**2 - beam_product)
        diameter_kpc = arcsec_to_kpc(diameter_corrected, distance_mpc)
        diameter_kpc_err = arcsec_to_kpc(diameter_arcsec_err, distance_mpc) * (
            diameter_arcsec / diameter_corrected
        )
    else:
        # Upper limit: use beam size
        diameter_corrected = np.sqrt(beam_product)
        diameter_kpc = arcsec_to_kpc(diameter_corrected, distance_mpc)
        diameter_kpc_err = np.nan
        logger.warning(f"{galaxy_name}: Diameter smaller than beam, using upper limit")

    # Calculate HI mass
    beam_area_pix = 1.133 * bmaj * bmin / (pixscale**2)
    integrated_flux = flux_data / beam_area_pix
    hi_mass = MASS_CONVERSION * distance_mpc**2 * np.nansum(integrated_flux)

    # Mass error and column-density sensitivity are derived from the SoFiA mask
    # (channel counts / channel width), so they are only available in the
    # mask-based path; otherwise they are reported as NaN.
    if use_mask:
        mass_error = calculate_hi_mass_error(
            mask_path, noise_rms, bmaj, bmin, pixscale, distance_mpc
        )

        coldens_limit = hi_column_density_sensitivity(
            noise_rms * 1000, bmaj, bmin, chan_width, line_width_kms=20, n_sigma=3
        )
    else:
        mass_error = np.nan
        coldens_limit = np.nan

    # Linear resolution
    linear_res = np.sqrt(bmaj**2 + bmin**2) * 4.848 * distance_mpc / 1000

    # Optical diameter
    if galaxy_name in logd25_dict:
        logd25_val = logd25_dict[galaxy_name]
        logd25_err = logd25_err_dict.get(galaxy_name, 0.1)

        if logd25_val < 10:  # Valid measurement
            opt_diam_arcmin = logd25_to_arcmin(logd25_val)
            opt_diam_kpc = arcsec_to_kpc(opt_diam_arcmin * 60, distance_mpc)
            opt_diam_err_kpc = arcsec_to_kpc(logd25_to_arcmin(logd25_err) * 60, distance_mpc)
        else:
            opt_diam_kpc = np.nan
            opt_diam_err_kpc = np.nan
    else:
        opt_diam_kpc = np.nan
        opt_diam_err_kpc = np.nan

    # Evolutionary phase
    phase = phases.get(group_id, "unknown")

    # Stellar mass
    stellar_mass = logm_star_dict.get(galaxy_name, 0.0)

    # Generate plot
    # Generate plot
    ellipse_x, ellipse_y = x_el, y_el

    sd_nan_path = sd_path.replace(".fits", "_nan.fits")
    sd_data = sd_info.data.copy()
    sd_data[sd_data <= 0] = np.nan
    fits.writeto(sd_nan_path, sd_data, sd_info.header, overwrite=True)

    wcs = WCS(sd_info.header).celestial
    opt_x, opt_y = skycoord_degrees_to_pixels(ra_deg, dec_deg, wcs)

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
        fitted_segments_pix=fitted_segments_pix,
        output_path=plot_path,
    )

    return {
        "galaxy": galaxy_name,
        "group": group_id,
        "phase": phase,
        "hi_diameter_kpc": diameter_kpc,
        "hi_diameter_err_kpc": diameter_kpc_err,
        "hi_mass": hi_mass,
        "hi_mass_err": mass_error,
        "optical_diameter_kpc": opt_diam_kpc,
        "optical_diameter_err_kpc": opt_diam_err_kpc,
        "log_stellar_mass": stellar_mass,
        "distance_mpc": distance_mpc,
        "linear_resolution_kpc": linear_res,
        "coldens_limit": coldens_limit,
        "fractional_mass_error": 100 * mass_error / hi_mass,
    }


def extract_galaxy_group_mapping(hcgs_members: dict) -> dict[str, str]:
    """Create mapping from galaxy name to parent group."""
    mapping = {}
    for group_cube, members in hcgs_members.items():
        group_id = group_cube.split("_")[0]
        for member in members:
            galaxy_name = member.split("_")[0]
            mapping[galaxy_name] = group_id
    return mapping


def process_all_hcg_galaxies(
    hcg_config: dict,
    positions_config: dict,
    dirs: DirectoryConfig,
    selected_galaxy: str | None = None,
) -> list[dict]:
    """
    Process all HCG interacting galaxies.

    Returns
    -------
    list
        List of result dictionaries
    """
    hcgs_members = hcg_config["hcgs_members"]
    galaxy_group_map = extract_galaxy_group_mapping(hcgs_members)

    results = []
    galaxy_items = list(galaxy_group_map.items())
    if selected_galaxy is not None:
        if selected_galaxy not in galaxy_group_map:
            raise ValueError(f"Galaxy {selected_galaxy} not found in HCG configuration")
        galaxy_items = [(selected_galaxy, galaxy_group_map[selected_galaxy])]

    for galaxy_name, group_id in galaxy_items:
        try:
            result = process_interacting_galaxy(
                galaxy_name, group_id, hcg_config, positions_config, dirs
            )

            if result is not None:
                results.append(result)
                logger.info(
                    f"{galaxy_name}: D_HI = {result['hi_diameter_kpc']:.2f} kpc, "
                    f"M_HI = {result['hi_mass']:.2e} M_sun, "
                    f"Phase = {result['phase']}"
                )
        except Exception as e:
            logger.error(f"Failed to process {galaxy_name}: {e}")
            continue
    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main processing function for HCG galaxies."""
    parser = argparse.ArgumentParser(description="Process HCG galaxies")
    parser.add_argument("--galaxy", help="Process only a single HCG galaxy (for faster testing).")
    args = parser.parse_args()

    setup_plot_style()

    # Configuration
    dirs = DirectoryConfig()

    # Load configurations
    isolated_config, hcg_config, positions_config = load_all_configs(dirs.config_dir)

    # Create output directory
    os.makedirs(dirs.output_plots, exist_ok=True)

    # Optionally create moment-0 maps (uncomment if needed)
    # create_hcg_moment0_maps(hcg_config, positions_config, dirs.hcg_base)

    # Process all HCG galaxies
    results = process_all_hcg_galaxies(
        hcg_config, positions_config, dirs, selected_galaxy=args.galaxy
    )

    # Save results
    if results:
        df = pd.DataFrame(results)
        results_csv_path = os.path.join(dirs.config_dir, "interacting_galaxies_results.csv")
        df.to_csv(results_csv_path, index=False)

        # Also save in original format for compatibility (using manual formatting)
        results_txt_path = os.path.join(dirs.config_dir, "diameter_mass_interacting_star.txt")
        with open(results_txt_path, "w") as f:
            f.write(
                "# hcg_index diameter diameter_err mass mass_err phase "
                "optical_diameter optical_diameter_err logm_star "
                "linear_res coldens_lim\n"
            )
            for r in results:
                # Handle NaN values for error columns
                diam_err = r["hi_diameter_err_kpc"]
                diam_err_str = f"{diam_err:.4e}" if not np.isnan(diam_err) else "nan"
                opt_err = r["optical_diameter_err_kpc"]
                opt_err_str = f"{opt_err:.4e}" if not np.isnan(opt_err) else "nan"
                opt_diam = r["optical_diameter_kpc"]
                opt_diam_str = f"{opt_diam:.4e}" if not np.isnan(opt_diam) else "nan"

                line = (
                    f"{r['galaxy']} {r['hi_diameter_kpc']:.4e} {diam_err_str} "
                    f"{r['hi_mass']:.4e} {r['hi_mass_err']:.4e} {r['phase']} "
                    f"{opt_diam_str} {opt_err_str} "
                    f"{r['log_stellar_mass']:.4e} {r['linear_resolution_kpc']:.4e} "
                    f"{r['coldens_limit']:.4e}\n"
                )
                f.write(line)

        # Summary statistics
        fractional_errors = [r["fractional_mass_error"] for r in results]
        stats = summarize_mass_errors(fractional_errors)
        logger.info(f"HCG mass error summary: {stats['latex_text']}")

    logger.info("HCG processing complete!")


if __name__ == "__main__":
    main()
