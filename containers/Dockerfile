# Container image for the galaxydisksize statistics tier and manuscript build.
#
# Build:
#     docker build -t galaxydisksize:0.1.0 -f containers/Dockerfile .
# Run the workflow:
#     docker run --rm -v "$PWD":/work -w /work galaxydisksize:0.1.0 \
#         snakemake --cores 4 paper
#
# This image covers the light pure-Python tier (fits, residuals, survival,
# figures) and the LaTeX build via Tectonic. The radio catalogue tier
# (pyspeckit, analysis_tools, cutout services) is intentionally out of scope.

FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Install the package itself into the environment.
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /opt/galaxydisksize
COPY . /opt/galaxydisksize
RUN pip install --no-deps -e .

WORKDIR /work
CMD ["snakemake", "--cores", "1", "paper"]
