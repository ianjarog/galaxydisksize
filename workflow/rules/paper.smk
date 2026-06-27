# Paper tier: compile the manuscript once every figure and auto-generated
# fragment it depends on is up to date.
#
# The per-galaxy moment-0 postage-stamp figures (HCG*/CIG*_mom0_onesolar.pdf)
# are products of the catalogue tier (or fetched from Zenodo); they are treated
# as ambient inputs in the figures/ directory rather than tracked here, because
# regenerating them requires the radio environment and the full data set.

rule paper:
    """Compile the manuscript PDF from the LaTeX sources and built figures."""
    input:
        manuscript=config["manuscript"],
        bib="latex/reference.bib",
        cls="latex/aa.cls",
        bst="latex/aa.bst",
        figures=ANALYSIS_FIGURES,
        autogen=AUTOGEN,
    output:
        "latex/hi_disk_size_environments.pdf",
    shell:
        # Tectonic resolves and caches LaTeX packages itself and runs BibTeX as
        # needed, which keeps the build self-contained and reproducible. To use
        # a classic TeX Live toolchain instead, replace this line with:
        #   latexmk -pdf -outdir=latex {input.manuscript}
        "tectonic --chatter minimal --outdir latex {input.manuscript}"
