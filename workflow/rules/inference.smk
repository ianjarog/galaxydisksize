# Statistics tier: fit the size-mass relation, compute size residuals, and run
# the censored survival analysis. Every rule runs from the committed measurement
# CSVs, so this tier needs only a light pure-Python environment.
#
# Output ownership is exclusive: each file below is written by exactly one rule.

ISO = config["measurements"]["isolated"]
INT = config["measurements"]["interacting"]
INT_UL = config["measurements"]["interacting_upper_limits"]


rule mass_size_consistency:
    """Per-sample size-mass fits (AMIGA, HCGs, MIGHTEE, Wang16, combined)."""
    input:
        script="scripts/fit_size_mass_relation.py",
        isolated=ISO,
        interacting=INT,
        mightee="data/MIGHTEE_D_HI_M_HI_rajohnson22.txt",
        wang="data/wang-surveys-table_original.txt",
    output:
        "products/mass_size_consistency_test_summary.json",
    shell:
        "{PYTHON} {input.script}"


rule mass_size_all_surveys:
    """Combined size-mass relation used to infer D_HI for unresolved AMIGA."""
    input:
        script="scripts/plot_size_mass_all_surveys.py",
        isolated=ISO,
        interacting=INT,
        mightee="data/MIGHTEE_D_HI_M_HI_rajohnson22.txt",
        wang="data/wang-surveys-table_original.txt",
    output:
        summary="products/mass_size_relation_all_surveys_summary.json",
        figure="figures/mass_size_relation_all_surveys.pdf",
        corner="figures/mass_size_relation_all_surveys_corner.pdf",
    shell:
        "{PYTHON} {input.script}"


rule mass_size_amiga_hcgs:
    """Two-sample (AMIGA vs HCG) size-mass comparison figure and fit."""
    input:
        script="scripts/plot_size_mass_amiga_hcgs.py",
        isolated=ISO,
        interacting=INT,
    output:
        summary="products/mass_size_relation_amiga_hcgs_summary.json",
        figure="figures/mass_size_relation_amiga_hcgs.pdf",
    shell:
        "{PYTHON} {input.script}"


rule emit_mass_size_macros:
    """Render the size-mass fit JSON into the manuscript's macros and Table 3."""
    input:
        script="scripts/write_size_mass_macros.py",
        summary="products/mass_size_consistency_test_summary.json",
    output:
        macros="latex/autogen/macros_mass_size.tex",
        table="latex/autogen/table_consistency_fits.tex",
    shell:
        "{PYTHON} {input.script}"


rule residual_analysis:
    """Size residuals about the D_HI-D_25 baseline: products, tables, and the
    AMIGA residual-trend figure.

    This is the largest analysis step. It requires the combined size-mass fit
    (rule mass_size_all_surveys) to have run first.
    """
    input:
        script="scripts/measure_size_residuals.py",
        baseline_script="scripts/size_residual_baseline_engine.py",
        all_surveys="products/mass_size_relation_all_surveys_summary.json",
        isolated=ISO,
        interacting=INT,
        interacting_ul=INT_UL,
    output:
        hcg_stats="products/hcg_residual_statistics_kelley_larger_sample.json",
        amiga_residuals="products/amiga_residuals_per_galaxy_kelley_larger_sample_dictionary.csv",
        hcg_residuals="products/hcg_residuals_per_galaxy_kelley_larger_sample.csv",
        pairwise="products/well_defined_pairwise_residuals.json",
        eq_baseline="latex/autogen/eq_baseline.tex",
        phase_stats="latex/autogen/table_phase_stats.tex",
        stat_tests="latex/autogen/table_stat_tests.tex",
        survey_residuals="latex/autogen/table_survey_residuals.tex",
        trend_test="latex/autogen/table_trend_test.tex",
        pairwise_table="latex/autogen/table_pairwise_residuals.tex",
        hcg_comparison="latex/autogen/macros_hcg_comparison.tex",
        trends_figure="figures/amiga_residual_trends_kelley_larger_sample_dictionary.pdf",
    shell:
        "{PYTHON} {input.script}"


rule survival_analysis:
    """Kaplan-Meier and Gehan statistics for the beam-size upper limits."""
    input:
        script="scripts/survival_analysis.py",
        interacting_ul=INT_UL,
        provenance="data/upperlimits_bmaj_provenance.csv",
        hcg_stats="products/hcg_residual_statistics_kelley_larger_sample.json",
        amiga_residuals="products/amiga_residuals_per_galaxy_kelley_larger_sample_dictionary.csv",
    output:
        macros="latex/autogen/macros_hcg_censored.tex",
        phase_stats="latex/autogen/table_phase_stats_censored.tex",
        stat_tests="latex/autogen/table_stat_tests_censored.tex",
        upper_limits="latex/autogen/table_upperlimits.tex",
    shell:
        "{PYTHON} {input.script} --emit"
