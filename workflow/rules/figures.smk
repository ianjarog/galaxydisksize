# Figure tier: the adopted-design residual figures and the overlay panels.
#
# Each figure below has exactly one owning rule (and each script writes only
# its declared outputs), so two rules can never overwrite the same file.

INT_UL = config["measurements"]["interacting_upper_limits"]

# Inputs shared by the promote figures: they import survival_analysis.load(),
# which reads the augmented upper-limit CSV plus the residual products.
_SURVIVAL_INPUTS = [
    INT_UL,
    "products/hcg_residual_statistics.json",
    "products/amiga_residuals_per_galaxy.csv",
]


rule fig_residuals_hist:
    """Probability-density residual histogram, colour-coded by HCG phase."""
    input:
        script="scripts/plot_residual_histogram.py",
        survival=_SURVIVAL_INPUTS,
    output:
        "figures/diameter_residuals_hist.pdf",
    shell:
        "{PYTHON} {input.script}"


rule fig_residuals_by_phase:
    """Box-and-whisker residuals by phase with Kaplan-Meier median bars."""
    input:
        script="scripts/plot_residuals_by_phase.py",
        survival=_SURVIVAL_INPUTS,
    output:
        "figures/diameter_residuals_by_phase.pdf",
    shell:
        "{PYTHON} {input.script}"


rule fig_survey_forest:
    """Forest plot of the per-survey median residual (Hydra split out)."""
    input:
        script="scripts/plot_survey_residual_forest.py",
        survival=_SURVIVAL_INPUTS,
        survey_table="latex/autogen/table_survey_residuals.tex",
    output:
        "figures/survey_median_residual.pdf",
    shell:
        "{PYTHON} {input.script}"


rule fig_overlays:
    """D_HI-D_25 correlation and residual-vs-D_25 panels with upper limits."""
    input:
        script="scripts/plot_diameter_correlation.py",
        interacting_ul=INT_UL,
        hcg_stats="products/hcg_residual_statistics.json",
        amiga_combined="products/amiga_combined_larger_sample.csv",
        hcg_residuals="products/hcg_residuals_per_galaxy.csv",
    output:
        correlation="figures/diameter_correlation.pdf",
        residuals_vs_d25="figures/diameter_residuals_vs_D25.pdf",
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
        "{PYTHON} {input.script}"
