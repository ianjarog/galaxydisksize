# Development

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full contributor guide. This
page summarises the moving parts.

## Layout

- `src/galaxydisksize/` — the reusable, documented, tested library.
- `scripts/` — canonical analysis drivers / thin CLIs (see `scripts/README.md`).
- `workflow/` — the Snakemake `Snakefile` and `rules/` that define the DAG.
- `tests/` — unit tests and golden-number regression guards.
- `config/` — workflow configuration and external data sources.

## Quality gate

```bash
./quality-gate.sh          # ruff format --check + ruff check + pytest (with coverage)
```

Formatting and linting are both ruff (line length 100, numpy docstring
convention). The library is held to the full rule set including docstrings; the
`scripts/` are held to every code rule but the docstring-style rules are relaxed
(rationale is documented in `pyproject.toml`).

Install the git hooks to run the same checks automatically:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Protecting published results

`tests/test_golden_numbers.py` pins the published numbers (e.g. the combined
size-mass slope, scatter, and baseline scatter). Any change expected to move a
result must update these guards in the same pull request and justify the change;
changes not expected to move results must leave them passing.

## The one-output-one-rule invariant

Each workflow output is produced by exactly one rule. Never add a second rule or
script that writes an existing output file — Snakemake treats two rules writing
one file as an error, which is the structural fix for the figure-overwrite class
of bug. Verify with:

```bash
snakemake -n paper
```

## Continuous integration

`.github/workflows/ci.yml` runs ruff (format + lint) and pytest with coverage on
Python 3.10–3.12 for every push and pull request.
