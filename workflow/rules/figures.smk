# Figure tier: the adopted-design residual figures and the overlay panels.
#
# The three "promote" figures below were the ones overwritten in the original
# tree because two different scripts wrote the same filenames. Here each figure
# has a single owning rule, so Snakemake will not allow that to happen again.

INT_UL = config["measurements"]["interacting_upper_limits"]

# Inputs shared by the promote figures: they import survival_analysis.load(),
# which reads the augmented upper-limit CSV plus the residual products.
_SURVIVAL_INPUTS = [
    INT_UL,
    "products/hcg_residual_statistics_kelley_larger_sample.json",
    "products/amiga_residuals_per_galaxy_kelley_larger_sample_dictionary.csv",
]


rule fig_residuals_hist:
    """Probability-density residual histogram, colour-coded by HCG phase."""
    input:
        script="scripts/plot_residual_histogram.py",
        survival=_SURVIVAL_INPUTS,
    output:
        "figures/diameter_residuals_hist_kelley_larger_sample.pdf",
    shell:
        "{PYTHON} {input.script} --promote"


rule fig_residuals_by_phase:
    """Box-and-whisker residuals by phase with Kaplan-Meier median bars."""
    input:
        script="scripts/plot_residuals_by_phase.py",
        survival=_SURVIVAL_INPUTS,
    output:
        "figures/diameter_residuals_by_phase_kelley_larger_sample.pdf",
    shell:
        "{PYTHON} {input.script} --promote"


rule fig_survey_forest:
    """Forest plot of the per-survey median residual (Hydra split out)."""
    input:
        script="scripts/plot_survey_residual_forest.py",
        survival=_SURVIVAL_INPUTS,
        survey_table="latex/autogen/table_survey_residuals.tex",
    output:
        "figures/survey_median_residual_kelley_larger_well_defined_sample_hydra_split.pdf",
    shell:
        "{PYTHON} {input.script} --promote"


rule fig_overlays:
    """D_HI-D_25 correlation and residual-vs-D_25 panels with upper limits."""
    input:
        script="scripts/plot_diameter_correlation.py",
        interacting_ul=INT_UL,
        hcg_stats="products/hcg_residual_statistics_kelley_larger_sample.json",
    output:
        correlation="figures/diameter_correlation_kelley_larger_sample.pdf",
        residuals_vs_d25="figures/diameter_residuals_vs_D25_kelley_larger_sample.pdf",
    shell:
        "{PYTHON} {input.script}"


rule fig_pairwise_euler:
    """Euler diagram of the well-defined pairwise residual cliques."""
    input:
        script="scripts/plot_pairwise_clique_euler.py",
        pairwise="products/well_defined_pairwise_residuals.json",
    output:
        "figures/pairwise_clique_euler_vertical.pdf",
    shell:
        "{PYTHON} {input.script} --promote"
