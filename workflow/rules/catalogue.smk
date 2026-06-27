# Catalogue (radio/measurement) tier -- OPT-IN.
#
# These rules re-derive the measured per-galaxy tables from the moment-0 maps
# and SoFiA masks. They are NOT part of the default `all` target because they
# require:
#   * the moment-0 maps and SoFiA masks (Jones et al. 2023; Zenodo 6909872),
#     which are downloaded into data/moment0_maps/ -- see config/data_sources.yaml;
#   * a heavier environment (pyspeckit, the analysis_tools package, and the
#     cutout libraries listed under the "cutouts" extra in pyproject.toml).
#
# Their outputs (the measurement CSVs) are committed to the repository so that
# the statistics tier runs without re-executing this tier. Build them explicitly
# with, for example:
#
#     snakemake --use-conda --cores 4 data/interacting_galaxies_results.csv
#
# Inputs are intentionally coarse (a directory of maps); tighten them to the
# exact per-galaxy file lists when running this tier against a local data copy.
#
# All inputs below are wrapped in ancient(): the measurement CSVs are committed
# and authoritative, so editing a measurement script or refreshing the maps must
# NOT silently invalidate them and drag the expensive catalogue tier into a
# `snakemake paper` build. To deliberately re-derive a CSV, request it explicitly
# and force it, e.g.  `snakemake --force data/interacting_galaxies_results.csv`.

MOMENT0_DIR = "data/moment0_maps"
SOFIA_MASK_DIR = "data/moment0_maps/SoFiA_masks"


rule measure_interacting:
    """Ellipse-fit the HCG moment-0 maps to measure HI diameters and masses."""
    input:
        script=ancient("scripts/build_hcg_catalogue.py"),
        pipeline=ancient("scripts/measure_hi_disk_sizes.py"),
        config_galaxies=ancient("data/config_hcg_galaxies.yaml"),
        config_positions=ancient("data/config_hcg_positions.yaml"),
        maps=ancient(MOMENT0_DIR),
        masks=ancient(SOFIA_MASK_DIR),
    output:
        "data/interacting_galaxies_results.csv",
    shell:
        "{PYTHON} {input.script}"


rule measure_isolated:
    """Measure HI diameters of the AMIGA (isolated) moment-0 maps."""
    input:
        script=ancient("scripts/measure_hi_disk_sizes.py"),
        config=ancient("data/config_isolated_galaxies.yaml"),
        maps=ancient(MOMENT0_DIR),
    output:
        "data/isolated_galaxies_results.csv",
    shell:
        "{PYTHON} {input.script}"


rule augment_upper_limits:
    """Reclassify beam-unresolved members as B_maj upper limits.

    Adds the three beam-limited members (HCG 15f, 40d, 58e) to the upper-limit
    set and writes the augmented table consumed by the statistics tier.
    """
    input:
        script=ancient("scripts/flag_beam_upper_limits.py"),
        interacting=ancient("data/interacting_galaxies_results_with_upperlimits.csv"),
        masks=ancient(SOFIA_MASK_DIR),
    output:
        augmented="data/interacting_galaxies_results_with_upperlimits_bmaj.csv",
        provenance="data/upperlimits_bmaj_provenance.csv",
    shell:
        "{PYTHON} {input.script}"
